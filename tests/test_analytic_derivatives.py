"""Equivalence tests for the analytic LCL score and Hessian.

The analytic derivatives in ``lcl._analytic_derivatives`` (Fisher identity for
the score; Louis/Oakes observed-information form for the Hessian) must agree
with automatic differentiation of the observed-data panel log likelihood to
machine precision for every supported configuration: with and without
demographics, with and without a numeraire constraint, ragged choice sets, and
extreme parameter scales.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
import pytest

from lcl import _scheduling as scheduling
from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._case_utils import _diff_unchosen_chosen
from lcl._demographics import _compute_grouped_data_loglik_grad_hess
from lcl._em_alg_steps import _compute_panel_logliks
from lcl._params import ParamPacking
from lcl.options import FitOptions, InferenceOptions
from lcl._struct import Data
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def _random_panel_data(rng, num_panels, num_alt_vars, num_dem_vars):
    """Random ragged panel choice data as a Data struct."""
    rows, y, cases, panels, alts, panels_of_cases, cases_per_panel = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    case_id = 0
    for panel in range(num_panels):
        n_cases = int(rng.integers(1, 5))
        cases_per_panel.append(n_cases)
        for _ in range(n_cases):
            n_alts = int(rng.integers(2, 6))
            X_case = rng.normal(size=(n_alts, num_alt_vars))
            chosen = int(rng.integers(n_alts))
            for j in range(n_alts):
                rows.append(X_case[j])
                y.append(j == chosen)
                cases.append(case_id)
                panels.append(panel)
                alts.append(j)
            panels_of_cases.append(panel)
            case_id += 1
    dems = rng.normal(size=(num_panels, num_dem_vars)) if num_dem_vars else None
    return Data(
        X=jnp.asarray(onp.array(rows)),
        dems=None if dems is None else jnp.asarray(dems),
        y=jnp.asarray(onp.array(y)),
        alts=jnp.asarray(onp.array(alts), dtype=jnp.uint32),
        cases=jnp.asarray(onp.array(cases), dtype=jnp.uint32),
        panels=jnp.asarray(onp.array(panels), dtype=jnp.uint32),
        panels_of_cases=jnp.asarray(onp.array(panels_of_cases), dtype=jnp.uint32),
        num_cases_per_panel=jnp.asarray(onp.array(cases_per_panel), dtype=jnp.uint32),
        num_cases=case_id,
        num_alt_vars=num_alt_vars,
        num_panels=num_panels,
        num_dem_vars=num_dem_vars,
    )


@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("num_dem_vars", [0, 3])
@pytest.mark.parametrize("numeraire_idx", [None, 0, 3])
@pytest.mark.parametrize("param_scale", [0.4, 20.0])
def test_analytic_scores_and_hessian_match_autodiff(
    num_classes, num_dem_vars, numeraire_idx, param_scale
) -> None:
    rng = onp.random.default_rng(
        hash((num_classes, num_dem_vars, numeraire_idx, param_scale)) % 2**32
    )
    num_alt_vars = 4
    data = _random_panel_data(rng, 20, num_alt_vars, num_dem_vars)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(
        num_alt_vars=num_alt_vars,
        num_classes=num_classes,
        num_dem_vars=num_dem_vars,
        numeraire_idx=numeraire_idx,
    )
    flat = jnp.asarray(rng.normal(size=packing.num_params) * param_scale)

    def panel_loglik(fp):
        latent_betas, thetas = packing.unpack(fp)
        structural_betas = packing.to_structural(latent_betas)
        class_probs = packing.class_probs(thetas, data.dems, data.num_panels)
        return _compute_panel_logliks(structural_betas, class_probs, diff, data)

    scores_ad = jax.jacfwd(panel_loglik)(flat)
    hessian_ad = jax.hessian(lambda fp: jnp.sum(panel_loglik(fp)))(flat)
    scores_an, hessian_an = _panel_scores_and_hessian(flat, diff, data, packing)

    score_scale = max(float(jnp.max(jnp.abs(scores_ad))), 1.0)
    hessian_scale = max(float(jnp.max(jnp.abs(hessian_ad))), 1.0)
    assert float(jnp.max(jnp.abs(scores_ad - scores_an))) <= 1e-9 * score_scale
    assert float(jnp.max(jnp.abs(hessian_ad - hessian_an))) <= 1e-9 * hessian_scale


def test_fitted_covariance_matches_autodiff_formulation() -> None:
    """End to end: both public covariances equal their autodiff formulations.

    ``latent_cov_matrix`` is the sandwich in the optimizer's unconstrained
    parameterization; ``cov_matrix`` is that matrix pushed through the softplus
    Jacobian so its rows match the coefficients ``parameter_names`` labels.
    """
    rng = onp.random.default_rng(3)
    num_panels, cases_per_panel, num_alts = 25, 4, 3
    betas_by_class = {0: onp.array([-1.5, 0.5]), 1: onp.array([-0.3, 1.5])}
    records = []
    for panel in range(num_panels):
        income = float(rng.normal())
        class_probability = 1.0 / (1.0 + onp.exp(-(0.5 + income)))
        latent_class = int(rng.random() < class_probability)
        for case in range(cases_per_panel):
            X_case = rng.normal(size=(num_alts, 2))
            utility = X_case @ betas_by_class[latent_class]
            probs = onp.exp(utility - utility.max())
            probs /= probs.sum()
            chosen = rng.choice(num_alts, p=probs)
            for alt in range(num_alts):
                records.append(
                    {
                        "panel": panel,
                        "case": f"{panel}_{case}",
                        "alt": alt,
                        "price": X_case[alt, 0],
                        "quality": X_case[alt, 1],
                        "income": income,
                        "choice": bool(alt == chosen),
                    }
                )
    df = pl.DataFrame(records)

    model = LatentClassConditionalLogit(num_classes=2, numeraire="price")
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["price", "quality"],
        fit_options=FitOptions(max_em_iter=30),
    )

    diff = _diff_unchosen_chosen(results.data)
    hessian_ad = jax.hessian(results._full_loglik_fn)(
        results.flat_params, diff, results.data
    )
    scores_ad = jax.jacfwd(results._panel_loglik_fn)(
        results.flat_params, diff, results.data
    )
    hessian_inverse = onp.linalg.pinv(onp.asarray(-hessian_ad))
    bread = hessian_inverse
    meat = onp.asarray(scores_ad).T @ onp.asarray(scores_ad)
    G = results.data.num_panels
    expected = bread @ meat @ bread * (G / (G - 1))
    expected = 0.5 * (expected + expected.T)

    onp.testing.assert_allclose(
        onp.asarray(results.latent_cov_matrix), expected, rtol=1e-8, atol=1e-10
    )

    jacobian = jax.jacfwd(results._structural_from_latent)(results.flat_params)
    structural_expected = onp.asarray(jacobian) @ expected @ onp.asarray(jacobian).T
    structural_expected = 0.5 * (structural_expected + structural_expected.T)
    onp.testing.assert_allclose(
        onp.asarray(results.cov_matrix), structural_expected, rtol=1e-8, atol=1e-10
    )
    # The reported standard errors are the ones a user reads off cov_matrix.
    onp.testing.assert_allclose(
        onp.sqrt(onp.diag(onp.asarray(results.cov_matrix)))[
            : results.model.num_classes
        ],
        results.class_coefficients()
        .filter(pl.col("variable") == "price")["std_error"]
        .to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )

    results.inference = InferenceOptions(covariance="unadjusted")
    unadjusted = results._compute_covariance()
    onp.testing.assert_allclose(onp.asarray(unadjusted), bread, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("num_classes", [2, 3, 5])
@pytest.mark.parametrize("num_dem_vars", [0, 4])
@pytest.mark.parametrize("numeraire_idx", [None, 0])
@pytest.mark.parametrize("param_scale", [0.3, 15.0])
def test_batched_and_sequential_schedules_agree(
    monkeypatch, num_classes, num_dem_vars, numeraire_idx, param_scale
) -> None:
    """Both contraction schedules must return the same score and Hessian.

    The batched and sequential paths in ``lcl._analytic_derivatives`` differ
    only in summation order, so forcing each one on identical inputs must give
    results that agree to machine precision.
    """
    rng = onp.random.default_rng(
        hash((num_classes, num_dem_vars, numeraire_idx, param_scale)) % 2**32
    )
    num_alt_vars = 4
    data = _random_panel_data(rng, 16, num_alt_vars, num_dem_vars)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(
        num_alt_vars=num_alt_vars,
        num_classes=num_classes,
        num_dem_vars=num_dem_vars,
        numeraire_idx=numeraire_idx,
    )
    flat = jnp.asarray(rng.normal(size=packing.num_params) * param_scale)

    monkeypatch.setattr(scheduling, "INFERENCE_THRESHOLD_BYTES", 2**62)
    scores_batched, hessian_batched = _panel_scores_and_hessian(
        flat, diff, data, packing
    )

    monkeypatch.setattr(scheduling, "INFERENCE_THRESHOLD_BYTES", 0)
    scores_scan, hessian_scan = _panel_scores_and_hessian(flat, diff, data, packing)

    score_scale = max(float(jnp.max(jnp.abs(scores_batched))), 1.0)
    hessian_scale = max(float(jnp.max(jnp.abs(hessian_batched))), 1.0)
    assert float(jnp.max(jnp.abs(scores_batched - scores_scan))) <= 1e-11 * score_scale
    assert (
        float(jnp.max(jnp.abs(hessian_batched - hessian_scan))) <= 1e-11 * hessian_scale
    )


def test_sequential_schedule_still_matches_autodiff(monkeypatch) -> None:
    """The sequential schedule must also reproduce autodiff of the likelihood."""
    rng = onp.random.default_rng(17)
    num_alt_vars, num_classes, num_dem_vars = 4, 3, 2
    data = _random_panel_data(rng, 20, num_alt_vars, num_dem_vars)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(
        num_alt_vars=num_alt_vars,
        num_classes=num_classes,
        num_dem_vars=num_dem_vars,
        numeraire_idx=1,
    )
    flat = jnp.asarray(rng.normal(size=packing.num_params))

    def panel_loglik(fp):
        latent_betas, thetas = packing.unpack(fp)
        structural_betas = packing.to_structural(latent_betas)
        class_probs = packing.class_probs(thetas, data.dems, data.num_panels)
        return _compute_panel_logliks(structural_betas, class_probs, diff, data)

    scores_ad = jax.jacfwd(panel_loglik)(flat)
    hessian_ad = jax.hessian(lambda fp: jnp.sum(panel_loglik(fp)))(flat)

    monkeypatch.setattr(scheduling, "INFERENCE_THRESHOLD_BYTES", 0)
    scores_an, hessian_an = _panel_scores_and_hessian(flat, diff, data, packing)

    score_scale = max(float(jnp.max(jnp.abs(scores_ad))), 1.0)
    hessian_scale = max(float(jnp.max(jnp.abs(hessian_ad))), 1.0)
    assert float(jnp.max(jnp.abs(scores_ad - scores_an))) <= 1e-9 * score_scale
    assert float(jnp.max(jnp.abs(hessian_ad - hessian_an))) <= 1e-9 * hessian_scale


@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("num_dem_vars", [1, 5])
def test_membership_mstep_schedules_agree(
    monkeypatch, num_classes, num_dem_vars
) -> None:
    """The fractional-logit M-step must agree across contraction schedules."""
    rng = onp.random.default_rng(hash((num_classes, num_dem_vars)) % 2**32)
    num_panels = 40
    data = _random_panel_data(rng, num_panels, 3, num_dem_vars)
    targets = jnp.asarray(rng.dirichlet(onp.ones(num_classes), size=num_panels))
    thetas = jnp.asarray(rng.normal(size=(num_dem_vars + 1) * (num_classes - 1)) * 1.5)

    monkeypatch.setattr(scheduling, "ITERATION_THRESHOLD_BYTES", 2**62)
    value_b, grad_b, hess_b = _compute_grouped_data_loglik_grad_hess(
        thetas, targets, data, num_classes
    )

    monkeypatch.setattr(scheduling, "ITERATION_THRESHOLD_BYTES", 0)
    value_s, grad_s, hess_s = _compute_grouped_data_loglik_grad_hess(
        thetas, targets, data, num_classes
    )

    onp.testing.assert_allclose(value_s, value_b, rtol=0, atol=0)
    onp.testing.assert_allclose(
        onp.asarray(grad_s), onp.asarray(grad_b), rtol=1e-12, atol=1e-12
    )
    onp.testing.assert_allclose(
        onp.asarray(hess_s), onp.asarray(hess_b), rtol=1e-12, atol=1e-12
    )
