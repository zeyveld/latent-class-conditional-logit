"""Tests for rank diagnostics, sandwich conventions, and constraint pullbacks.

These cover the inference-side behaviours that are easy to get silently wrong:
a pseudo-inverse absorbing a rank deficiency without complaint, two robust
covariance branches disagreeing about whether to centre scores, a chain rule
that ignores the constraint it was configured with, and a test statistic
reported for a parameter whose null is excluded by construction.
"""

import logging

import jax.numpy as jnp
import numpy as onp
import polars as pl
import pytest

import lcl  # noqa: F401  (enables x64)
from lcl._struct import InferenceOptions
from lcl.conditional_logit import ConditionalLogit
from lcl.constraints import (
    NegativeCoefficient,
    pullback_negative_derivatives,
    pullback_negative_gradient,
    pullback_negative_hessian,
    pullback_negative_score_rows,
)
from lcl.utils import _invert_information, _robust_covariance


def _choice_frame(
    seed: int = 0, num_cases: int = 120, num_alts: int = 3, *, collinear: bool = False
) -> pl.DataFrame:
    """Build a small long-format choice dataset."""
    rng = onp.random.default_rng(seed)
    records = []
    for case in range(num_cases):
        design = rng.normal(size=(num_alts, 2))
        utility = design @ onp.array([-1.0, 0.5])
        probs = onp.exp(utility - utility.max())
        probs /= probs.sum()
        chosen = rng.choice(num_alts, p=probs)
        for alt in range(num_alts):
            row = {
                "panel": case // 4,
                "case": case,
                "alt": alt,
                "price": float(design[alt, 0]),
                "quality": float(design[alt, 1]),
                "choice": int(alt == chosen),
            }
            if collinear:
                # Exactly collinear with price, so one direction is unidentified.
                row["price_copy"] = row["price"]
            records.append(row)
    return pl.DataFrame(records)


# --------------------------------------------------------------------------
# Rank deficiency is reported rather than absorbed
# --------------------------------------------------------------------------
def test_invert_information_reports_full_rank_and_conditioning() -> None:
    """A well-conditioned matrix inverts cleanly and reports its condition."""
    matrix = onp.diag([4.0, 2.0, 1.0])
    inverse, diagnostics = _invert_information(matrix)

    onp.testing.assert_allclose(onp.asarray(inverse), onp.diag([0.25, 0.5, 1.0]))
    assert diagnostics.rank == 3
    assert not diagnostics.rank_deficient
    assert diagnostics.positive_definite
    assert diagnostics.condition_number == pytest.approx(4.0)
    assert diagnostics.smallest_eigenvalue == pytest.approx(1.0)


def test_invert_information_flags_rank_deficiency(caplog) -> None:
    """A singular information matrix warns instead of silently truncating."""
    # Third direction carries no curvature at all.
    matrix = onp.diag([3.0, 1.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="lcl.utils"):
        inverse, diagnostics = _invert_information(matrix, label="test matrix")

    assert diagnostics.rank == 2
    assert diagnostics.num_params == 3
    assert diagnostics.rank_deficient
    assert not diagnostics.positive_definite
    assert diagnostics.condition_number == onp.inf
    # Pseudo-inverse still zeroes the null direction rather than exploding.
    assert onp.asarray(inverse)[2, 2] == 0.0
    assert "rank deficient" in caplog.text
    assert "test matrix" in caplog.text


def test_invert_information_flags_saddle_point(caplog) -> None:
    """A negative eigenvalue means a saddle point, not a maximum."""
    matrix = onp.diag([2.0, -1.0])
    with caplog.at_level(logging.WARNING, logger="lcl.utils"):
        _, diagnostics = _invert_information(matrix)

    assert not diagnostics.positive_definite
    assert not diagnostics.rank_deficient
    assert diagnostics.smallest_eigenvalue == pytest.approx(-1.0)
    assert "not positive definite" in caplog.text


def test_invert_information_matches_pinv_on_full_rank() -> None:
    """The eigen-based inverse agrees with ``pinv`` where both are valid."""
    rng = onp.random.default_rng(5)
    factor = rng.normal(size=(6, 6))
    matrix = factor @ factor.T + 6.0 * onp.eye(6)
    inverse, _ = _invert_information(matrix)
    onp.testing.assert_allclose(
        onp.asarray(inverse), onp.linalg.pinv(matrix), rtol=1e-9, atol=1e-12
    )


def test_collinear_design_surfaces_rank_diagnostics() -> None:
    """An exactly collinear covariate leaves a detectable rank deficiency."""
    df = _choice_frame(collinear=True)
    results = ConditionalLogit().fit(
        df,
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality", "price_copy"],
    )
    diagnostics = results.information_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_params == 3
    assert diagnostics.rank_deficient
    assert diagnostics.rank < diagnostics.num_params


# --------------------------------------------------------------------------
# Sandwich conventions are uniform
# --------------------------------------------------------------------------
def test_robust_covariance_does_not_centre_scores() -> None:
    """Scores enter the meat uncentred, matching the clustered branches."""
    bread = onp.eye(2)
    scores = onp.array([[1.0, 0.0], [3.0, 2.0], [-1.0, 4.0]])
    n = scores.shape[0]

    observed = onp.asarray(_robust_covariance(bread, jnp.asarray(scores)))
    expected = (n / (n - 1)) * (scores.T @ scores)
    onp.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    centred = scores - scores.mean(axis=0)
    assert not onp.allclose(observed, (n / (n - 1)) * (centred.T @ centred))


def test_robust_covariance_honours_finite_sample_flag() -> None:
    """Disabling the correction drops the ``n / (n - 1)`` multiplier."""
    bread = onp.eye(2)
    scores = jnp.asarray([[1.0, 0.0], [3.0, 2.0], [-1.0, 4.0]])
    with_correction = onp.asarray(_robust_covariance(bread, scores, True))
    without = onp.asarray(_robust_covariance(bread, scores, False))
    onp.testing.assert_allclose(with_correction, without * (3 / 2), rtol=1e-12)


def test_clustered_and_unclustered_branches_share_conventions() -> None:
    """Panel clustering changes only the level scores are summed at."""
    df = _choice_frame(seed=3)
    inference = InferenceOptions(covariance="clustered")
    clustered = ConditionalLogit().fit(
        df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["price", "quality"],
        inference=inference,
    )
    unclustered = ConditionalLogit().fit(
        df,
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
        inference=inference,
    )

    # Rebuild each meat by hand under the shared (uncentred) convention.
    bread = onp.asarray(clustered.hess_inv)
    case_scores = onp.asarray(clustered.grad_n)
    panels_of_cases = onp.asarray(clustered.data.panels_of_cases)
    num_panels = int(clustered.data.num_panels)
    panel_scores = onp.zeros((num_panels, case_scores.shape[1]))
    onp.add.at(panel_scores, panels_of_cases, case_scores)

    expected_clustered = (
        (num_panels / (num_panels - 1))
        * bread
        @ (panel_scores.T @ panel_scores)
        @ bread
    )
    onp.testing.assert_allclose(
        onp.asarray(clustered.covariance), expected_clustered, rtol=1e-9, atol=1e-12
    )

    n = case_scores.shape[0]
    expected_unclustered = (
        (n / (n - 1))
        * onp.asarray(unclustered.hess_inv)
        @ (case_scores.T @ case_scores)
        @ onp.asarray(unclustered.hess_inv)
    )
    onp.testing.assert_allclose(
        onp.asarray(unclustered.covariance), expected_unclustered, rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------
# Constraint pullbacks honour the caller's min_abs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("min_abs", [1e-8, 1e-5, 1e-2, 0.5])
def test_pullbacks_accept_and_use_min_abs(min_abs) -> None:
    """Each pullback takes ``min_abs`` and applies the matching constraint."""
    rng = onp.random.default_rng(11)
    raw = jnp.asarray(rng.normal(size=4))
    grad = jnp.asarray(rng.normal(size=4))
    score_rows = jnp.asarray(rng.normal(size=(7, 4)))
    hessian = jnp.asarray(rng.normal(size=(4, 4)))
    index = 2
    constraint = NegativeCoefficient(min_abs=min_abs)
    jacobian = float(constraint.jacobian_diag(raw[index]))
    curvature = float(constraint.hessian_diag(raw[index]))

    pulled_grad = pullback_negative_gradient(raw, index, grad, min_abs)
    assert float(pulled_grad[index]) == pytest.approx(float(grad[index]) * jacobian)

    pulled_rows = pullback_negative_score_rows(raw, index, score_rows, min_abs)
    onp.testing.assert_allclose(
        onp.asarray(pulled_rows[:, index]),
        onp.asarray(score_rows[:, index]) * jacobian,
        rtol=1e-12,
    )

    pulled_hess = pullback_negative_hessian(raw, index, grad, hessian, min_abs)
    expected_diagonal = (
        float(hessian[index, index]) * jacobian**2 + float(grad[index]) * curvature
    )
    assert float(pulled_hess[index, index]) == pytest.approx(expected_diagonal)

    combined = pullback_negative_derivatives(
        raw, index, grad, score_rows, hessian, min_abs
    )
    onp.testing.assert_allclose(onp.asarray(combined[0]), onp.asarray(pulled_grad))
    onp.testing.assert_allclose(onp.asarray(combined[1]), onp.asarray(pulled_rows))
    onp.testing.assert_allclose(onp.asarray(combined[2]), onp.asarray(pulled_hess))


def test_pullback_min_abs_defaults_are_backward_compatible() -> None:
    """Omitting ``min_abs`` reproduces the previous default behaviour."""
    rng = onp.random.default_rng(13)
    raw = jnp.asarray(rng.normal(size=3))
    grad = jnp.asarray(rng.normal(size=3))
    score_rows = jnp.asarray(rng.normal(size=(5, 3)))
    hessian = jnp.asarray(rng.normal(size=(3, 3)))

    without = pullback_negative_derivatives(raw, 1, grad, score_rows, hessian)
    with_default = pullback_negative_derivatives(
        raw, 1, grad, score_rows, hessian, 1e-5
    )
    for left, right in zip(without, with_default):
        onp.testing.assert_array_equal(onp.asarray(left), onp.asarray(right))


# --------------------------------------------------------------------------
# No test statistic for the constrained numeraire
# --------------------------------------------------------------------------
def test_numeraire_z_and_p_values_are_suppressed() -> None:
    """The softplus null is excluded by construction, so no statistic is shown."""
    df = _choice_frame(seed=7)
    results = ConditionalLogit(numeraire="price").fit(
        df,
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
    )
    idx = results.model.numeraire_idx
    assert idx is not None
    assert onp.isnan(results.zvalues[idx])
    assert onp.isnan(results.pvalues[idx])

    # Unconstrained coefficients keep their statistics.
    other = 1 - idx
    assert onp.isfinite(results.zvalues[other])
    assert onp.isfinite(results.pvalues[other])

    table = results.coefficient_table()
    numeraire_row = table.filter(pl.col("variable") == "price")
    assert onp.isnan(numeraire_row["z_value"][0])
    assert onp.isnan(numeraire_row["p_value"][0])
    # The estimate and standard error remain reportable.
    assert onp.isfinite(numeraire_row["estimate"][0])
    assert onp.isfinite(numeraire_row["std_error"][0])
