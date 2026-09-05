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
from lcl.options import FitOptions, InferenceOptions
from lcl.conditional_logit import ConditionalLogit
from lcl.constraints import (
    NegativeCoefficient,
    pullback_negative_derivatives,
    pullback_negative_gradient,
    pullback_negative_hessian,
    pullback_negative_score_rows,
)
from lcl._inference import _invert_information, _robust_covariance


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
    with caplog.at_level(logging.WARNING, logger="lcl._inference"):
        inverse, diagnostics = _invert_information(matrix, label="test matrix")

    assert diagnostics.rank == 2
    assert diagnostics.num_params == 3
    assert diagnostics.rank_deficient
    assert not diagnostics.positive_definite
    assert diagnostics.condition_number == onp.inf
    # No covariance is exposed for an unidentified direction.
    assert onp.isnan(onp.asarray(inverse)).all()
    assert "rank deficient" in caplog.text
    assert "test matrix" in caplog.text


def test_invert_information_flags_saddle_point(caplog) -> None:
    """A negative eigenvalue means a saddle point, not a maximum."""
    matrix = onp.diag([2.0, -1.0])
    with caplog.at_level(logging.WARNING, logger="lcl._inference"):
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
    with pytest.raises(ValueError, match="rank deficient"):
        ConditionalLogit().fit(
            df,
            alts_col="alt",
            cases_col="case",
            choice_col="choice",
            case_varnames=["price", "quality", "price_copy"],
        )


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
        onp.asarray(clustered.cov_matrix), expected_clustered, rtol=1e-9, atol=1e-12
    )

    n = case_scores.shape[0]
    expected_unclustered = (
        (n / (n - 1))
        * onp.asarray(unclustered.hess_inv)
        @ (case_scores.T @ case_scores)
        @ onp.asarray(unclustered.hess_inv)
    )
    onp.testing.assert_allclose(
        onp.asarray(unclustered.cov_matrix), expected_unclustered, rtol=1e-9, atol=1e-12
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


def test_a_zero_gradient_block_reports_convergence_immediately() -> None:
    """A flat block is stationary, not a case for escalating diagonal shifts.

    Class padding on a multi-device mesh feeds the solver all-zero weights, and a
    collapsed latent class can underflow to the same state.  Requiring strict
    descent there would leave the decrement infinite and burn the whole iteration
    budget on zero-length steps.
    """
    from lcl._optimize import exact_newton_minimize

    def value_fn(params):
        return jnp.zeros(())

    def value_grad_hess_fn(params):
        return (
            jnp.zeros(()),
            jnp.zeros(params.shape),
            jnp.zeros((params.size, params.size)),
        )

    start = jnp.array([0.3, -1.2])
    state = exact_newton_minimize(
        value_fn, value_grad_hess_fn, start, tol=1e-6, maxiter=50
    )
    assert float(state.error) == 0.0
    assert int(state.step_num) == 0
    onp.testing.assert_allclose(onp.asarray(state.params), onp.asarray(start))


def test_a_nonzero_gradient_is_never_mistaken_for_a_stationary_point() -> None:
    """The stationarity shortcut fires only when the gradient really has vanished."""
    from lcl._optimize import exact_newton_minimize

    def value_fn(params):
        return jnp.sum(params)

    def value_grad_hess_fn(params):
        # No curvature, but a slope: diagonal shifts turn this into a descent
        # direction, so the solver must keep stepping rather than declare victory.
        return (
            jnp.sum(params),
            jnp.ones(params.shape),
            jnp.zeros((params.size, params.size)),
        )

    state = exact_newton_minimize(
        value_fn, value_grad_hess_fn, jnp.zeros(2), tol=1e-6, maxiter=3
    )
    assert float(state.error) > 0.0
    assert int(state.step_num) == 3
    assert float(state.loss) < 0.0


def test_an_unusable_hessian_reports_an_infinite_decrement() -> None:
    """A direction that cannot be repaired is a failure, not a stationary point."""
    from lcl._optimize import exact_newton_minimize

    def value_fn(params):
        return jnp.sum(params)

    def value_grad_hess_fn(params):
        return (
            jnp.sum(params),
            jnp.ones(params.shape),
            jnp.full((params.size, params.size), jnp.nan),
        )

    state = exact_newton_minimize(
        value_fn, value_grad_hess_fn, jnp.zeros(2), tol=1e-6, maxiter=2
    )
    assert not onp.isfinite(float(state.error))


def test_between_class_spread_without_variation_has_no_standard_error() -> None:
    """A coefficient common to every class has an unidentified standard deviation.

    The square root of a variance is not differentiable at zero, so a floored
    square root would report a spuriously exact spread of zero.
    """
    from lcl import ChoiceIds, LCLSpec, NegativeCoefficient
    from lcl import fit as lcl_fit

    rng = onp.random.default_rng(11)
    rows = []
    for panel in range(80):
        for case in range(4):
            price = rng.uniform(0.5, 3.0, size=3)
            quality = rng.uniform(0.0, 5.0, size=3)
            utility = -1.2 * price + 1.0 * quality + rng.gumbel(size=3)
            chosen = int(onp.argmax(utility))
            for alt in range(3):
                rows.append(
                    {
                        "panel": panel,
                        "case": panel * 4 + case,
                        "alt": alt,
                        "choice": alt == chosen,
                        "price": float(price[alt]),
                        "quality": float(quality[alt]),
                    }
                )
    df = pl.DataFrame(rows)
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    results = lcl_fit(
        df,
        spec,
        fit_options=FitOptions(seed=1, max_em_iter=4, polish=False, num_devices=1),
        inference=InferenceOptions(covariance="unadjusted"),
    )
    # Force the two classes to coincide, which is the degenerate case.
    latent = results.em_res.latent_betas.at[:, 1].set(results.em_res.latent_betas[:, 0])
    results.em_res = results.em_res._replace(
        latent_betas=latent,
        structural_betas=results._param_packing.to_structural(latent),
    )
    results.flat_params = results._pack_params()
    summary = results.beta_summary()
    assert onp.all(onp.asarray(summary["sd"]) == 0.0)
    assert onp.all(onp.isnan(onp.asarray(summary["sd_se"])))


def test_the_polish_reaches_a_stationary_point_and_never_lowers_the_likelihood() -> None:
    """Newton steps on the observed-data likelihood finish what EM starts."""
    from lcl import ChoiceIds, LCLSpec, NegativeCoefficient
    from lcl import fit as lcl_fit

    rng = onp.random.default_rng(6)
    latent_class = rng.choice(2, size=100, p=[0.5, 0.5])
    price_by_class = onp.array([-1.7, -0.5])
    quality_by_class = onp.array([0.3, 1.5])
    rows = []
    for panel in range(100):
        for case in range(4):
            price = rng.uniform(0.5, 3.0, size=3)
            quality = rng.uniform(0.0, 5.0, size=3)
            utility = (
                price_by_class[latent_class[panel]] * price
                + quality_by_class[latent_class[panel]] * quality
                + rng.gumbel(size=3)
            )
            chosen = int(onp.argmax(utility))
            for alt in range(3):
                rows.append(
                    {
                        "panel": panel,
                        "case": panel * 4 + case,
                        "alt": alt,
                        "choice": alt == chosen,
                        "price": float(price[alt]),
                        "quality": float(quality[alt]),
                    }
                )
    df = pl.DataFrame(rows)
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    common = dict(inference=InferenceOptions(covariance="clustered"))
    # A deliberately short EM run leaves the score far from zero.
    unpolished = lcl_fit(
        df,
        spec,
        fit_options=FitOptions(seed=3, max_em_iter=6, polish=False, num_devices=1),
        **common,
    )
    polished = lcl_fit(
        df,
        spec,
        fit_options=FitOptions(seed=3, max_em_iter=6, polish=True, num_devices=1),
        **common,
    )

    assert unpolished.observed_score_max > polished.observed_score_max
    assert polished.observed_score_max <= polished.score_tol
    assert polished.converged
    assert not unpolished.converged
    assert float(polished.em_res.unconditional_loglik) >= float(
        unpolished.em_res.unconditional_loglik
    )
    report = polished.polish_report
    assert report is not None and report.accepted
    assert report.loglik_after >= report.loglik_before
    assert report.score_after < report.score_before
