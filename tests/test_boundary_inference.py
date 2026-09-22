"""Boundary uncertainty: analytical cone identities and real constrained fits."""

from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import polars as pl
import pytest
from scipy.optimize import minimize
from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
    fit,
)
from lcl._boundary_inference import (
    project_upper_gaussian,
    _normal_draws,
    _multiplier_statistics,
    _projected_moment_errors,
    _summary_jacobian,
)
from lcl._boundary import boundary_kkt_violation, structural_score
from lcl._boundary_types import CoefficientMoments
from lcl._case_utils import _diff_unchosen_chosen
from lcl._polish import em_vars_from_flat
from lcl._results import LCLResults


def test_half_normal_projection_moments_and_correlation():
    rng = np.random.default_rng(492)
    inverse = np.array([[1.0, 0.4], [0.4, 2.0]])
    z = rng.multivariate_normal(np.zeros(2), inverse, size=50000)
    actual = project_upper_gaussian(z, inverse, [0])
    expected = z - np.maximum(z[:, :1], 0) * inverse[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.var(actual[:, 0]) == pytest.approx(0.5 - 1 / (2 * np.pi), abs=0.008)
    assert np.mean(actual[:, 0]) == pytest.approx(-1 / np.sqrt(2 * np.pi), abs=0.008)
    np.testing.assert_allclose(project_upper_gaussian(z, inverse, []), z)


def test_multiple_boundaries_match_independent_constrained_optimizer():
    inverse = np.array([[2.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.5]])
    information = np.linalg.inv(inverse)
    draws = np.random.default_rng(49).normal(size=(20, 3))
    projected = project_upper_gaussian(draws, inverse, [0, 2])
    for z, actual in zip(draws, projected):
        solution = minimize(
            lambda h: 0.5 * (h - z) @ information @ (h - z),
            np.zeros(3),
            jac=lambda h: information @ (h - z),
            method="SLSQP",
            bounds=[(None, 0), (None, None), (None, 0)],
            options={"ftol": 1e-13, "maxiter": 100},
        )
        assert solution.success
        np.testing.assert_allclose(actual, solution.x, atol=1e-6)


def _mixed_result(first_price_effect=0.8):
    rng = np.random.default_rng(321)
    rows = []
    for panel in range(240):
        dem = rng.normal()
        group = int(rng.random() < 1 / (1 + np.exp(-dem)))
        for occasion in range(6):
            price = rng.uniform(1, 5, 4)
            quality = np.array([0.0, 0.0, 1.0, 1.0])
            utility = (first_price_effect if group == 0 else -1.2) * price + (
                -2.0 if group == 0 else 2.0
            ) * quality
            chosen = np.argmax(utility + rng.gumbel(size=4))
            for alt in range(4):
                rows.append(
                    dict(
                        panel=panel,
                        case=panel * 6 + occasion,
                        alt=alt,
                        price=price[alt],
                        quality=quality[alt],
                        dem=dem,
                        choice=alt == chosen,
                    )
                )
    return fit(
        pl.DataFrame(rows),
        LCLSpec(
            ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
            utility=("price", "quality"),
            membership=("dem",),
            classes=2,
            constraints={"price": NegativeCoefficient()},
        ),
        fit_options=FitOptions(seed=82, max_em_iter=300, num_devices=1),
        inference=InferenceOptions(boundary="projected"),
    )


@pytest.fixture(scope="module")
def mixed_boundary():
    return _mixed_result()


def test_zero_price_effect_retains_boundary_uncertainty():
    r = _mixed_result(0.0)
    assert r.converged and r.covariance_available
    assert r._boundary_summary_inputs is not None
    table = r.beta_summary()
    assert r.boundary_summary_diagnostics["strict_parameters"] == []
    assert len(r.boundary_summary_diagnostics["weak_parameters"]) >= 1
    price = table.filter(pl.col("variable") == "price").row(0, named=True)
    assert price["mean_se"] > 0 and price["sd_se"] > 0


def test_coarser_cluster_centering_matches_explicit_score_aggregation(mixed_boundary):
    from lcl._analytic_derivatives import _panel_scores_and_hessian

    r = mixed_boundary
    clusters = np.arange(r.data.num_panels) // 3
    clustered = LCLResults(
        r.model,
        r.em_res,
        r.data,
        r.total_recursions,
        converged=r.converged,
        inference=InferenceOptions(boundary="projected"),
        estim_time_sec=0.0,
        observed_score_max=r.observed_score_max,
        param_packing=r._param_packing,
        cluster_ids=clusters,
        num_clusters=int(clusters.max()) + 1,
    )
    structural = r._structural_from_latent(r.flat_params)
    scores, _ = _panel_scores_and_hessian(
        structural,
        _diff_unchosen_chosen(r.data),
        r.data,
        replace(r._param_packing, numeraire_idx=None),
    )
    scores = np.asarray(scores)
    centered = scores - scores.mean(axis=0)
    summed = np.stack(
        [centered[clusters == group].sum(axis=0) for group in np.unique(clusters)]
    )
    expected = summed.T @ summed * len(summed) / (len(summed) - 1)
    np.testing.assert_allclose(
        clustered._boundary_summary_inputs["meat"], expected, atol=1e-10
    )


def test_mixed_boundary_preserves_predictions_and_summary_uncertainty(mixed_boundary):
    r = mixed_boundary
    assert r.converged and r.covariance_available
    assert len(r.boundary_parameter_indices) == 1
    table = r.class_coefficients()
    assert table.filter(pl.col("boundary"))["std_error"].is_nan().all()
    assert table.filter(~pl.col("boundary"))["std_error"].is_finite().all()
    summary = r.beta_summary()
    assert summary["mean_se"].is_finite().all()
    assert summary["sd_se"].is_finite().all()
    assert summary["mean_se"].min() > 0
    assert summary["sd_se"].min() > 0
    assert summary["inference_status"].unique().to_list() == [
        "critical_cone_projection"
    ]
    np.testing.assert_equal(
        r.beta_summary()["mean_se"].to_numpy(), summary["mean_se"].to_numpy()
    )
    assert len(r.boundary_summary_diagnostics["strict_parameters"]) == 1


def test_summary_jacobian_includes_membership_and_matches_autodiff(mixed_boundary):
    r = mixed_boundary
    means, variances, _, dm, dv = _summary_jacobian(r)
    packing = replace(r._param_packing, numeraire_idx=None)
    structural = r._structural_from_latent(r.flat_params)

    def moments(p):
        beta, theta = packing.unpack(p)
        shares = packing.class_probs(theta, r.data.dems, r.data.num_panels).mean(axis=0)
        mean = beta @ shares
        variance = ((beta - mean[:, None]) ** 2) @ shares
        return jnp.concatenate((mean, variance))

    expected = np.asarray(jax.jacfwd(moments)(structural))
    np.testing.assert_allclose(np.vstack((dm, dv)), expected, atol=1e-12)
    np.testing.assert_allclose(np.r_[means, variances], moments(structural), atol=1e-12)
    assert np.max(abs(dm[:, r._param_packing.num_beta_params :])) > 0.01


@pytest.mark.parametrize("skip", [False, True])
def test_saturated_wrong_boundary_is_not_convergence_even_with_skipped_inference(
    mixed_boundary, skip,
):
    r = mixed_boundary
    beta, theta = r._param_packing.unpack(r.flat_params)
    beta = beta.at[0, :].set(-100.0)
    flat = jnp.concatenate((beta.ravel(), theta.ravel()))
    diff = _diff_unchosen_chosen(r.data)
    score = structural_score(flat, diff, r.data, r._param_packing)
    assert boundary_kkt_violation(score, [0, 1]) > 0.01
    state = em_vars_from_flat(flat, diff, r.data, r._param_packing)
    invalid = LCLResults(
        r.model,
        state,
        r.data,
        1,
        True,
        inference=InferenceOptions(skip=skip, boundary="projected"),
        estim_time_sec=0.0,
        observed_score_max=0.0,
        param_packing=r._param_packing,
    )
    assert not invalid.converged
    assert invalid.observed_score_max > invalid.score_tol
    assert invalid.boundary_summary_diagnostics["method"] == (
        "skipped" if skip else "unavailable"
    )


def test_non_pd_structural_information_uses_labelled_conditional_fallback(
    mixed_boundary,
):
    import copy

    r = copy.copy(mixed_boundary)
    r._boundary_summary_cache = None
    r._boundary_summary_inputs = dict(r._boundary_summary_inputs)
    matrix = r._boundary_summary_inputs["information"].copy()
    matrix[-1, -1] = -1e6
    r._boundary_summary_inputs["information"] = matrix
    summary = r.beta_summary()
    assert summary["inference_status"].unique().to_list() == [
        "conditional_on_boundary_fallback"
    ]
    assert summary["mean_se"].is_finite().all()
    assert r.boundary_summary_diagnostics["fallback_reason"]


def test_invalid_gaussian_covariance_is_not_silently_repaired():
    with pytest.raises(ValueError, match="not positive semidefinite"):
        _normal_draws(np.diag([1.0, -1.0]), 100, 0)


def test_zero_spread_directional_derivative_matches_folded_normal_law():
    moments = CoefficientMoments(
        means=np.array([0.0]),
        variances=np.array([0.0]),
        shares=np.array([0.5, 0.5]),
        mean_jacobian=np.array([[0.5, 0.5]]),
        variance_jacobian=np.zeros((1, 2)),
    )
    mean_se, sd_se, dimension = _projected_moment_errors(
        np.eye(2),
        np.eye(2),
        np.array([], dtype=int),
        np.array([0, 1]),
        moments,
        np.array([False]),
        50000,
        284,
    )
    # At equal tastes, d(SD) = abs(h_0-h_1)/2, rather than division by zero.
    assert mean_se[0] == pytest.approx(np.sqrt(0.5), abs=1e-14)
    assert sd_se[0] == pytest.approx(np.sqrt((1 - 2 / np.pi) / 2), abs=0.008)
    assert dimension == 2


def _synthetic_projection_problem():
    """Two variables by three classes; class 0 and 1 of variable 0 are weak."""
    rng = np.random.default_rng(71)
    root = rng.normal(size=(6, 6))
    information = root @ root.T + 2 * np.eye(6)
    inverse = np.linalg.inv(information)
    meat_root = rng.normal(size=(6, 6))
    covariance = inverse @ (meat_root @ meat_root.T + np.eye(6)) @ inverse
    moments = CoefficientMoments(
        means=np.zeros(2),
        variances=np.array([0.7, 0.0]),
        shares=np.array([0.5, 0.3, 0.2]),
        mean_jacobian=rng.normal(size=(2, 6)),
        variance_jacobian=rng.normal(size=(2, 6)),
    )
    return covariance, inverse, np.array([0, 1]), moments


def test_moment_errors_are_exact_delta_method_without_weak_constraints():
    covariance, inverse, _, moments = _synthetic_projection_problem()
    separated = np.array([True, True])
    moments = moments._replace(variances=np.array([0.7, 0.4]))
    first = _projected_moment_errors(
        covariance, inverse, np.array([], dtype=int), np.arange(6),
        moments, separated, 200, 0,
    )
    second = _projected_moment_errors(
        covariance, inverse, np.array([], dtype=int), np.arange(6),
        moments, separated, 200, 99,
    )
    jm, jv = moments.mean_jacobian, moments.variance_jacobian
    expected_mean = np.sqrt(np.einsum("ip,pq,iq->i", jm, covariance, jm))
    expected_sd = np.sqrt(np.einsum("ip,pq,iq->i", jv, covariance, jv)) / (
        2 * np.sqrt(moments.variances)
    )
    np.testing.assert_allclose(first[0], expected_mean, rtol=1e-13)
    np.testing.assert_allclose(first[1], expected_sd, rtol=1e-13)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2] == 0


def test_low_dimensional_simulation_matches_full_projection():
    covariance, inverse, weak, moments = _synthetic_projection_problem()
    separated = np.array([True, False])
    mean_se, sd_se, dimension = _projected_moment_errors(
        covariance, inverse, weak, np.arange(6), moments, separated, 40000, 3
    )
    # Brute force: simulate every coordinate and project whole draws.
    z = np.random.default_rng(8).multivariate_normal(
        np.zeros(6), covariance, size=400000
    )
    h = project_upper_gaussian(z, inverse, weak)
    shares = moments.shares
    spread = h[:, 3:] - (h[:, 3:] @ shares)[:, None]
    expected_mean = (h @ moments.mean_jacobian.T).std(axis=0)
    expected_sd = [
        (h @ moments.variance_jacobian[0]).std() / (2 * np.sqrt(0.7)),
        np.sqrt((spread**2) @ shares).std(),
    ]
    np.testing.assert_allclose(mean_se, expected_mean, rtol=0.01)
    np.testing.assert_allclose(sd_se, expected_sd, rtol=0.015)
    assert dimension == 5  # two weak prices plus three zero-spread classes


def test_unaffected_summary_is_exact_despite_correlated_weak_scores() -> None:
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    moments = CoefficientMoments(
        np.zeros(1), np.ones(1), np.ones(1),
        np.array([[0.0, 1.0]]), np.array([[0.0, 2.0]]),
    )
    # The metric leaves coordinate 1 untouched, even though z_0 and z_1 covary.
    for seed in (0, 99):
        mean_se, sd_se, _ = _projected_moment_errors(
            covariance, np.eye(2), np.array([0]), np.arange(2),
            moments, np.array([True]), 200, seed,
        )
        np.testing.assert_array_equal(mean_se, [1.0])
        np.testing.assert_array_equal(sd_se, [1.0])


@pytest.mark.parametrize("small_variance", [1.0, 1e-20])
def test_weak_covariance_regression_preserves_small_variance_directions(
    small_variance: float,
) -> None:
    moments = CoefficientMoments(
        np.zeros(1), np.ones(1), np.ones(1),
        np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]),
    )
    mean_se, sd_se, _ = _projected_moment_errors(
        np.diag([small_variance, 1.0]), np.eye(2), np.array([0, 1]),
        np.arange(2), moments, np.array([True]), 50000, 0,
    )
    expected = np.sqrt(small_variance * (0.5 - 1 / (2 * np.pi)))
    np.testing.assert_allclose(mean_se, [expected], rtol=0.02)
    np.testing.assert_allclose(sd_se, [expected], rtol=0.02)


def test_multiplier_adjustment_can_increase_robust_sampling_variance() -> None:
    information = np.array([[2.0, 1.0], [1.0, 1.0]])
    meat = np.array([[1.0, -0.5], [-0.5, 1.0]])
    statistic = _multiplier_statistics(
        np.array([1.0, 0.0]), information, meat,
        np.array([0]), np.array([1]), np.ones((1, 1)),
    )
    # The multiplier influence is s_0 - s_1, with variance 3, not raw variance 1.
    np.testing.assert_allclose(statistic, [1 / np.sqrt(3)])


@pytest.mark.parametrize("score_variance", [0.0, 1.0])
def test_singular_weak_score_covariance_retains_its_gaussian_support(
    score_variance: float,
) -> None:
    moments = CoefficientMoments(
        np.zeros(1), np.ones(1), np.array([0.5, 0.5]),
        np.array([[0.5, 0.5]]), np.array([[1.0, 1.0]]),
    )
    # Both coordinates are the same Gaussian variable, or identically zero.
    mean_se, sd_se, _ = _projected_moment_errors(
        score_variance * np.ones((2, 2)), np.eye(2), np.array([0, 1]),
        np.arange(2), moments, np.array([True]), 50000, 7,
    )
    expected = np.sqrt(score_variance * (0.5 - 1 / (2 * np.pi)))
    np.testing.assert_allclose(mean_se, [expected], rtol=0.02)
    np.testing.assert_allclose(sd_se, [expected], rtol=0.02)


def test_normal_draws_are_independent_and_reproducible() -> None:
    draws = _normal_draws(np.eye(2), 200, 17)
    np.testing.assert_array_equal(draws, _normal_draws(np.eye(2), 200, 17))
    # Pairing would duplicate every even directional functional, such as a norm.
    assert not np.allclose(draws[:100], -draws[100:])


def test_multiplier_statistic_uses_nuisance_adjusted_score_variance(mixed_boundary):
    from lcl._analytic_derivatives import _panel_scores_and_hessian

    r = mixed_boundary
    structural = r._structural_from_latent(r.flat_params)
    scores, _ = _panel_scores_and_hessian(
        structural,
        _diff_unchosen_chosen(r.data),
        r.data,
        replace(r._param_packing, numeraire_idx=None),
    )
    scores = np.asarray(scores)
    information = r._boundary_summary_inputs["information"]
    active = np.asarray(r.boundary_parameter_indices)
    free = np.setdiff1d(np.arange(r.num_params), active)
    loading = information[np.ix_(active, free)] @ np.linalg.inv(
        information[np.ix_(free, free)]
    )
    # Per-panel influence of each multiplier: s_A - I_AF I_FF^{-1} s_F.
    influence = scores[:, active] - scores[:, free] @ loading.T
    influence -= influence.mean(axis=0)
    groups = len(scores)
    variance = (influence**2).sum(axis=0) * groups / (groups - 1)
    expected = scores[:, active].sum(axis=0) / np.sqrt(variance)
    np.testing.assert_allclose(
        r._boundary_summary_inputs["multiplier_z"], expected, rtol=1e-8
    )


def test_strict_only_projection_equals_conditional_delta_method(mixed_boundary):
    r = mixed_boundary
    summary = r.beta_summary()
    assert r.boundary_summary_diagnostics["weak_parameters"] == []
    moments = _summary_jacobian(r)
    covariance = np.asarray(r.cov_matrix)
    jm = moments.mean_jacobian
    np.testing.assert_allclose(
        summary["mean_se"].to_numpy(),
        np.sqrt(np.einsum("ip,pq,iq->i", jm, covariance, jm)),
        rtol=1e-8,
    )


@pytest.mark.parametrize("boundary", ["conditional", "projected"])
def test_unadjusted_covariance_warns_when_price_strictly_binds(
    mixed_boundary: LCLResults, caplog: pytest.LogCaptureFixture, boundary: str,
):
    r = mixed_boundary
    with caplog.at_level("WARNING", logger="lcl._boundary_inference"):
        LCLResults(
            r.model,
            r.em_res,
            r.data,
            r.total_recursions,
            r.converged,
            InferenceOptions(covariance="unadjusted", boundary=boundary),
            0.0,
            observed_score_max=r.observed_score_max,
            param_packing=r._param_packing,
        )
    assert "information equality" in caplog.text


def test_interior_projected_mode_agrees_with_regular_covariance():
    projected = _mixed_result(-1.0)
    assert projected.converged and not projected.boundary_parameter_indices
    strict = LCLResults(
        projected.model,
        projected.em_res,
        projected.data,
        projected.total_recursions,
        projected.converged,
        InferenceOptions(boundary="strict"),
        0.0,
        observed_score_max=projected.observed_score_max,
        param_packing=projected._param_packing,
    )
    assert projected.inference_status == "regular"
    np.testing.assert_allclose(
        projected.cov_matrix, strict.cov_matrix, rtol=1e-7, atol=1e-10
    )


@pytest.mark.parametrize(
    "settings",
    [dict(boundary="unknown"), dict(boundary_draws=0), dict(boundary_seed=-1)],
)
def test_invalid_boundary_options_fail_before_estimation(settings):
    with pytest.raises(ValueError):
        InferenceOptions(**settings)
