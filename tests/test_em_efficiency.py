"""EM reuse, ascent, and executable-cache regressions."""

from dataclasses import replace
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcl._case_utils import _diff_unchosen_chosen
from lcl._demographics import _update_thetas
from lcl._em_alg_startup import (
    _fit_starting_betas,
    _get_starting_vals,
    _random_class_weights,
)
from lcl._analytic_derivatives import prepare_panel_chunks
from lcl._em_alg_steps import (
    _compiled_em_step,
    _compute_conditional_class_probs,
    _compute_unconditional_loglik,
    _em_step,
    _update_betas,
    class_mesh_sharding,
    place_em_vars,
)
from lcl._struct import EMVars
from lcl.constraints import NegativeCoefficientBound
from lcl.options import FitOptions, OptimizationOptions
from lcl._polish import aitken_extrapolated_gap, _compiled_polish, POLISH_DECREMENT_TOL
from lcl._params import ParamPacking
from tests.test_analytic_derivatives import _random_panel_data


@pytest.mark.parametrize("dem_vars", [0, 2])
@pytest.mark.parametrize("numeraire", [None, 0])
@pytest.mark.parametrize("devices", [1, 2])
@pytest.mark.parametrize("classes", [3, 5])
def test_reused_em_matches_recomputed_estep(dem_vars, numeraire, devices, classes):
    if devices > jax.device_count():
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=2")
    data = _random_panel_data(np.random.default_rng(701), 24, 3, dem_vars)
    diff = _diff_unchosen_chosen(data)
    fit = FitOptions(num_devices=devices)
    opt = OptimizationOptions()
    # Odd counts test dummy-class padding and the final partial likelihood block.
    state = place_em_vars(
        _get_starting_vals(
            diff, data, classes, fit, opt, NegativeCoefficientBound(numeraire)
        ),
        devices,
    )

    @eqx.filter_jit
    def reference(old):
        # Original mathematical EM order: independently recompute the E-step,
        # expand all case weights, and then perform both M-steps.
        posterior, weights = _compute_conditional_class_probs(
            old.betas, old.thetas, old.shares, diff, data
        )
        betas, _ = _update_betas(
            old.betas, weights, diff, opt, devices, NegativeCoefficientBound(numeraire)
        )
        if data.dems is None:
            shares = posterior.sum(axis=0) / posterior.sum()
            prior = jnp.broadcast_to(shares, posterior.shape)
            thetas = None
        else:
            thetas, prior, _ = _update_thetas(old.thetas, posterior, data, classes, opt)
            shares = prior.mean(axis=0)
        loglik = _compute_unconditional_loglik(betas, prior, diff, data)
        posterior, _ = _compute_conditional_class_probs(
            betas, thetas, shares, diff, data
        )
        return EMVars(betas, thetas, shares, loglik, posterior)

    # The cached posterior must already match the actual initialized prior.
    posterior, _ = _compute_conditional_class_probs(
        state.betas, state.thetas, state.shares, diff, data
    )
    np.testing.assert_allclose(state.class_probs_by_panel, posterior, atol=1e-13)
    for _ in range(4):
        expected = reference(state)
        updated, _ = _em_step(
            state, diff, data, classes, opt, fit, NegativeCoefficientBound(numeraire)
        )
        for actual, target in zip(jax.tree.leaves(updated), jax.tree.leaves(expected)):
            np.testing.assert_allclose(actual, target, rtol=1e-8, atol=1e-9)
        assert (
            float(updated.unconditional_loglik)
            >= float(state.unconditional_loglik) - 1e-9
        )
        state = updated


@pytest.mark.parametrize("numeraire", [None, 0])
def test_startup_and_em_reuse_compilations(caplog, numeraire):
    data = _random_panel_data(np.random.default_rng(318), 27, 2, 1)
    diff = _diff_unchosen_chosen(data)
    opt = OptimizationOptions(maxiter=13)
    fit = FitOptions(num_devices=1)
    state = place_em_vars(
        _get_starting_vals(
            diff, data, 2, fit, opt, NegativeCoefficientBound(numeraire)
        ),
        1,
    )
    _compiled_em_step.cache_clear()
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        for offset in (0.0, 0.1, 0.2):
            beta = _fit_starting_betas(
                _random_class_weights(data.num_panels, 2, 9),
                diff._replace(X=diff.X + offset),
                data,
                opt,
                1,
                NegativeCoefficientBound(numeraire),
            )
            jax.block_until_ready(beta)
        for seed in (31, 32):
            for _ in range(3):
                state, diagnostics = _em_step(
                    state,
                    diff,
                    data,
                    2,
                    opt,
                    replace(fit, seed=seed),
                    NegativeCoefficientBound(numeraire),
                )
                jax.block_until_ready((state, diagnostics))
    messages = [record.getMessage() for record in caplog.records]
    for name in ("_fit_starting_betas", "step"):
        # Older supported JAX versions omit the "jit(...)" wrapper in this log.
        assert (
            sum(
                f"Compiling jit({name})" in m or f"Compiling {name} " in m
                for m in messages
            )
            == 1
        )


@pytest.mark.parametrize(
    "devices,classes,dem_vars",
    [
        (1, 2, 0),
        (1, 3, 0),
        (1, 4, 0),
        (2, 2, 0),
        (2, 3, 0),
        (2, 4, 0),
        (2, 3, 14),
        (4, 2, 0),
        (4, 5, 14),
        (4, 6, 14),
        (4, 8, 0),
    ],
)
def test_em_step_returns_its_input_placement(caplog, devices, classes, dem_vars):
    if devices > jax.device_count():
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=5")
    # An even class split used to return class-sharded betas, so the second
    # iteration saw a new input sharding and compiled the step again. Two or
    # six classes across four devices also exposed unsupported partial output
    # shardings in JAX 0.9.2. Wide designs exercise batched membership curvature.
    data = _random_panel_data(
        np.random.default_rng(907), 600 if dem_vars else 25, 3, dem_vars
    )
    if dem_vars:
        assert data.num_panels * (dem_vars + 1) ** 2 * 8 >= 1024**2
    diff = _diff_unchosen_chosen(data)
    fit = FitOptions(num_devices=devices)
    opt = OptimizationOptions()
    state = place_em_vars(_get_starting_vals(diff, data, classes, fit, opt), devices)
    replicated = class_mesh_sharding(devices)
    _compiled_em_step.cache_clear()
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        for _ in range(3):
            state, diagnostics = _em_step(state, diff, data, classes, opt, fit)
            jax.block_until_ready((state, diagnostics))
            for leaf in jax.tree.leaves((state, diagnostics)):
                assert leaf.sharding.is_equivalent_to(replicated, leaf.ndim)
    messages = [record.getMessage() for record in caplog.records]
    assert (
        sum("Compiling jit(step)" in m or "Compiling step " in m for m in messages) == 1
    )


def test_sharded_em_step_with_batched_membership_curvature():
    if jax.device_count() < 2:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=2")
    # Wide demographic designs assemble the membership Hessian with a batched
    # map over class pairs once the dense intermediate reaches 1 MiB.  Under JAX
    # 0.5.3, a class-sharded posterior reaching that map failed HLO verification.
    data = _random_panel_data(np.random.default_rng(911), 600, 2, 14)
    num_dem = data.num_dem_vars + 1
    assert data.num_panels * num_dem**2 * 8 >= 1024**2 and 3 - 1 < num_dem
    diff = _diff_unchosen_chosen(data)
    opt = OptimizationOptions()
    states = []
    for devices in (1, 2):
        fit = FitOptions(num_devices=devices)
        state = place_em_vars(_get_starting_vals(diff, data, 3, fit, opt), devices)
        for _ in range(2):
            state, _ = _em_step(state, diff, data, 3, opt, fit)
        states.append(state)
    for sharded, single in zip(jax.tree.leaves(states[1]), jax.tree.leaves(states[0])):
        np.testing.assert_allclose(sharded, single, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    "logliks",
    [
        [-100.0, -90.0, -95.0],
        [-90.0, -100.0, -100.0],
        [-100.0, -90.0, np.nan],
        [-100.0, -90.0, -np.inf],
        [-np.inf, -90.0, -90.0],
    ],
)
def test_aitken_does_not_accept_invalid_likelihood_sequences(logliks):
    assert np.isinf(aitken_extrapolated_gap(logliks))


def test_aitken_allows_roundoff_at_a_plateau():
    assert aitken_extrapolated_gap([-100.0, -90.0, -90.0 - 1e-13]) == 0.0
    assert aitken_extrapolated_gap([-100.0, -90.0, -90.0]) == 0.0


def test_polish_honors_solver_controls(monkeypatch):
    import lcl._polish as polish
    from lcl._optimize import exact_newton_minimize

    captured = []

    def recording_solver(*args, **kwargs):
        captured.append(kwargs)
        return exact_newton_minimize(*args, **kwargs)

    monkeypatch.setattr(polish, "exact_newton_minimize", recording_solver)
    data = _random_panel_data(np.random.default_rng(21), 9, 2, 0)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(2, 2, 0, None)
    solve = _compiled_polish(
        packing,
        OptimizationOptions(
            maxiter=1,
            newton_decrement_tol=POLISH_DECREMENT_TOL,
            max_step_norm=50.0,
            line_search_maxiter=7,
            hessian_damping=0.02,
            initial_trust_radius=0.3,
            accept_any_decrease=True,
        ),
    )
    params, _ = solve(
        jnp.zeros(packing.num_params),
        diff,
        data,
        jnp.array(9.0),
        prepare_panel_chunks(diff, data),
    )
    jax.block_until_ready(params)
    assert len(captured) == 1
    assert captured[0] == dict(
        tol=POLISH_DECREMENT_TOL,
        maxiter=1,
        max_step_norm=50.0,
        line_search_maxiter=7,
        damping=0.02,
        initial_trust_radius=0.3,
        accept_any_decrease=True,
        upper_bounds=None,
    )


def test_observed_score_uses_the_documented_per_panel_scale():
    from lcl._polish import observed_score_max, _total_loglik_kernel

    data = _random_panel_data(np.random.default_rng(867), 19, 3, 2)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(3, 3, 2, 0)
    params = jnp.asarray(np.random.default_rng(12).normal(size=packing.num_params))
    gradient = jax.grad(_total_loglik_kernel)(params, diff, data, packing)
    expected = np.max(np.abs(gradient)) / data.num_panels
    np.testing.assert_allclose(
        observed_score_max(params, diff, data, packing), expected, rtol=1e-12
    )


@pytest.mark.parametrize("skip_inference", [False, True])
def test_reported_score_and_convergence_agree_after_class_ordering(skip_inference):
    import polars as pl
    from lcl._polish import _total_loglik_kernel
    from lcl import InferenceOptions, LatentClassConditionalLogit

    data = _random_panel_data(np.random.default_rng(931), 19, 2, 1)
    frame = pl.DataFrame(
        dict(
            panel=np.asarray(data.panels),
            case=np.asarray(data.cases),
            alt=np.asarray(data.alts),
            choice=np.asarray(data.y),
            x=np.asarray(data.X[:, 0]),
            z=np.asarray(data.X[:, 1]),
            dem=np.asarray(data.dems[data.panels, 0]),
        )
    )
    result = LatentClassConditionalLogit(num_classes=2).fit(
        frame,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x", "z"],
        dem_varnames=["dem"],
        fit_options=FitOptions(max_em_iter=1, polish=False, score_tol=0.05),
        inference=InferenceOptions(skip=skip_inference),
    )
    diff = _diff_unchosen_chosen(result.data)
    gradient = jax.grad(_total_loglik_kernel)(
        result.flat_params, diff, result.data, result._param_packing
    )
    expected = float(np.max(np.abs(gradient)) / result.data.num_panels)
    np.testing.assert_allclose(result.observed_score_max, expected, rtol=1e-11)
    assert result.converged == (expected <= result.score_tol)


@pytest.mark.parametrize("bad_loglik", [float("nan"), float("inf"), -1e6])
def test_em_rejects_nonfinite_or_decreasing_likelihood(monkeypatch, bad_loglik):
    import importlib
    from lcl._struct import EMStepDiagnostics
    from lcl import LatentClassConditionalLogit

    model_module = importlib.import_module("lcl.latent_class_conditional_logit")
    data = _random_panel_data(np.random.default_rng(87), 12, 2, 0)
    diff = _diff_unchosen_chosen(data)

    def invalid_step(state, *_args):
        return state._replace(
            unconditional_loglik=jnp.asarray(bad_loglik)
        ), EMStepDiagnostics(jnp.zeros(2), jnp.array(0.0))

    monkeypatch.setattr(model_module, "_em_step", invalid_step)
    model = LatentClassConditionalLogit(num_classes=2)
    with pytest.raises(RuntimeError, match="EM (produced|decreased)"):
        model._run_em(
            diff_unchosen_chosen=diff,
            data_struct=data,
            fit_options=FitOptions(num_devices=1),
            optimization_options=OptimizationOptions(),
            progress_callback=None,
        )


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_wide_demographic_hessian_matches_autodiff(scale):
    from lcl._demographics import (
        _compute_grouped_data_loglik_value,
        _compute_grouped_data_loglik_grad_hess,
    )

    rng = np.random.default_rng(518)
    data = _random_panel_data(rng, 2000, 2, 8)
    # Non-unit row totals verify that fractional weights are retained exactly.
    targets = jnp.asarray(rng.gamma(2.0, size=(2000, 5)))
    theta = jnp.asarray(rng.normal(size=9 * 4) * scale)

    def objective(params):
        return _compute_grouped_data_loglik_value(params, targets, data, 5)

    value, gradient, hessian = _compute_grouped_data_loglik_grad_hess(
        theta, targets, data, 5
    )
    np.testing.assert_allclose(value, objective(theta), rtol=1e-12)
    np.testing.assert_allclose(
        gradient, jax.grad(objective)(theta), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        hessian, jax.hessian(objective)(theta), rtol=1e-10, atol=1e-10
    )
