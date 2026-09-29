"""Compilation, lifetime, placement, and numerical equivalence regressions."""

import gc
import logging
import os
import subprocess
import sys
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcl._analytic_derivatives import (
    _panel_scores_and_hessian,
    prepare_panel_chunks,
    summed_derivatives,
)
from lcl._case_utils import _diff_unchosen_chosen, _loglik_gradient, _loglik_value
from lcl._delta import parametric_bootstrap_se
from lcl._em_alg_startup import _get_starting_vals, _random_class_weights
from lcl._em_alg_steps import (
    _compiled_em_step,
    _em_step,
    class_mesh_sharding,
    place_em_vars,
)
from lcl._inference import _invert_information
from lcl._jax_compat import cpu_device, device_put_array_leaves
from lcl._optimize import exact_newton_minimize, newton_kwargs, scaled_objective
from lcl._params import ParamPacking
from lcl._polish import _compiled_polish, _total_loglik_kernel
from lcl.constraints import NegativeCoefficientBound
from lcl.options import FitOptions, OptimizationOptions
from tests.test_analytic_derivatives import _random_panel_data


def _compile_count(caplog, name):
    return sum(
        f"Compiling jit({name})" in record.getMessage()
        or f"Compiling {name} " in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("devices,classes", [(1, 8), (2, 8), (4, 5)])
def test_unbalanced_multistart_compiles_once(caplog, devices, classes):
    if devices > jax.local_device_count():
        pytest.skip("Requires multiple local devices")
    data = _random_panel_data(np.random.default_rng(318), 101, 2, 1)
    diff = _diff_unchosen_chosen(data)
    placement = class_mesh_sharding(devices)
    data, diff = device_put_array_leaves((data, diff), placement)
    opt = OptimizationOptions(maxiter=13)
    jax.clear_caches()
    eqx.clear_caches()
    _compiled_em_step.cache_clear()
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        for seed in range(3):
            fit = FitOptions(seed=seed, num_devices=devices)
            state = place_em_vars(
                _get_starting_vals(diff, data, classes, fit, opt), devices
            )
            # Transfer guards include warm executions, not only the first trace.
            with jax.transfer_guard_device_to_device("disallow"):
                for _ in range(3):
                    state, diagnostics = _em_step(state, diff, data, classes, opt, fit)
                    jax.block_until_ready((state, diagnostics))
    assert _compile_count(caplog, "_fit_starting_betas") == 1
    assert _compile_count(caplog, "step") == 1


@pytest.mark.parametrize(
    "bound", [NegativeCoefficientBound(), NegativeCoefficientBound(0)]
)
def test_full_data_start_matches_independent_subset_fits(bound):
    data = _random_panel_data(np.random.default_rng(563), 53, 2, 0)
    diff = _diff_unchosen_chosen(data)
    opt = OptimizationOptions()
    fit = FitOptions(seed=11, num_devices=1)
    weights = np.asarray(_random_class_weights(data.num_panels, 4, fit.seed))
    actual = _get_starting_vals(diff, data, 4, fit, opt, bound)
    reference = []
    for class_index in range(4):
        mask = weights[np.asarray(diff.panels), class_index].astype(bool)
        _, cases = np.unique(np.asarray(diff.cases)[mask], return_inverse=True)
        _, panels = np.unique(np.asarray(diff.panels)[mask], return_inverse=True)
        subset = diff._replace(
            X=diff.X[mask],
            alts=diff.alts[mask],
            cases=jnp.asarray(cases, dtype=jnp.uint32),
            panels=jnp.asarray(panels, dtype=jnp.uint32),
            num_cases=int(cases.max()) + 1,
        )
        value, derivatives = scaled_objective(
            _loglik_value, _loglik_gradient, subset.num_cases
        )
        initial = jnp.zeros(2)
        result = exact_newton_minimize(
            value,
            derivatives,
            initial,
            subset,
            jnp.ones(subset.num_cases),
            **newton_kwargs(opt),
            upper_bounds=bound.upper_bounds(initial),
        )
        reference.append(np.asarray(result.params))
    np.testing.assert_allclose(
        actual.betas, np.column_stack(reference), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("dem_vars,classes", [(0, 2), (3, 5)])
@pytest.mark.parametrize("chunk_size", [1, 7, 256])
@pytest.mark.parametrize("coarser", [False, True])
def test_chunked_derivatives_match_full_matrix(dem_vars, classes, chunk_size, coarser):
    rng = np.random.default_rng(426)
    data = _random_panel_data(rng, 23, 3, dem_vars)
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(3, classes, dem_vars, 0)
    flat = jnp.asarray(rng.normal(size=packing.num_params))
    scores, hessian = _panel_scores_and_hessian(flat, diff, data, packing)
    ids = jnp.asarray(np.arange(23) % 4) if coarser else None
    chunks = prepare_panel_chunks(
        diff,
        data,
        chunk_size=chunk_size,
        cluster_ids=ids,
        num_clusters=4 if coarser else None,
    )
    for center in (False, True):
        summary = summed_derivatives(flat, chunks, packing, center=center)
        reference = np.asarray(scores)
        if center:
            reference = reference - reference.mean(axis=0)
        if coarser:
            grouped = np.zeros((4, packing.num_params))
            np.add.at(grouped, np.asarray(ids), reference)
            reference = grouped
        np.testing.assert_allclose(
            summary.score, scores.sum(axis=0), rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(summary.hessian, hessian, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            summary.meat, reference.T @ reference, rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("scale", [1e-8, 1e-7, 1e-6, 1.0, 1e7, 1e8])
def test_information_rank_and_inverse_are_invariant_to_units(scale, caplog):
    matrix = np.array([[1.0, 2 / 3], [2 / 3, 1.0]])
    units = np.array([scale, 1.0])
    scaled = matrix * units[:, None] * units[None, :]
    inverse, diagnostics = _invert_information(scaled)
    assert diagnostics.positive_definite and diagnostics.rank == 2
    assert diagnostics.condition_number == pytest.approx(5)
    np.testing.assert_allclose(
        np.asarray(inverse) * units[:, None] * units[None, :],
        np.linalg.inv(matrix),
        rtol=1e-12,
    )
    assert "ill conditioned" not in caplog.text
    singular = np.ones((2, 2)) * units[:, None] * units[None, :]
    _, diagnostics = _invert_information(singular)
    assert diagnostics.rank_deficient


@pytest.mark.parametrize("draws", [2, 32, 35, 500])
def test_bootstrap_batches_match_original_draws_and_use_cpu(draws, caplog):
    params = jnp.array([0.4, -0.3])
    covariance = jnp.array([[0.2, 0.05], [0.05, 0.1]])
    X = jnp.arange(15.0).reshape(5, 3) / 10

    def target(p, *, design):
        return jnp.sin(p[0] * design) + p[1] ** 2

    values, vectors = np.linalg.eigh(covariance)
    samples = (
        np.asarray(params)
        + np.random.default_rng(9).standard_normal((draws, 2))
        @ (vectors * np.sqrt(values)).T
    )
    expected = np.std(
        np.sin(samples[:, 0, None, None] * np.asarray(X))
        + samples[:, 1, None, None] ** 2,
        axis=0,
        ddof=1,
    )
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        result = parametric_bootstrap_se(
            target, params, covariance, draws=draws, seed=9, design=X
        )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.devices() == {jax.local_devices(backend="cpu")[0]}
    assert _compile_count(caplog, "_bootstrap_values") == 1


def test_missing_cpu_backend_falls_back_with_warning(monkeypatch, caplog):
    fallback = jax.local_devices()[0]

    def local_devices(*, backend=None):
        if backend == "cpu":
            raise RuntimeError("Backend disabled by JAX_PLATFORMS")
        return [fallback]

    monkeypatch.setattr(jax, "local_devices", local_devices)
    assert cpu_device() == fallback
    assert "CPU backend is unavailable" in caplog.text


def test_run_em_places_data_before_the_recursion(monkeypatch):
    import importlib
    from lcl import LatentClassConditionalLogit

    module = importlib.import_module("lcl.latent_class_conditional_logit")
    devices = min(2, jax.local_device_count())
    placement = class_mesh_sharding(devices)
    original = module._em_step
    calls = 0

    def checked_step(state, diff, data, *args):
        nonlocal calls
        calls += 1
        for array in jax.tree.leaves((state, diff, data)):
            if isinstance(array, jax.Array):
                assert array.sharding.is_equivalent_to(placement, array.ndim)
        with jax.transfer_guard_device_to_device("disallow"):
            return original(state, diff, data, *args)

    monkeypatch.setattr(module, "_em_step", checked_step)
    data = _random_panel_data(np.random.default_rng(352), 25, 2, 1)
    LatentClassConditionalLogit(num_classes=3)._run_em(
        diff_unchosen_chosen=_diff_unchosen_chosen(data),
        data_struct=data,
        fit_options=FitOptions(max_em_iter=3, em_tol=1e-12, num_devices=devices),
        optimization_options=OptimizationOptions(),
        progress_callback=None,
    )
    assert calls == 3


def test_skipped_covariance_does_not_lookup_cpu(monkeypatch):
    from lcl._results import LCLResults
    from lcl.options import InferenceOptions

    result = object.__new__(LCLResults)
    result.num_params = 3
    result.inference = InferenceOptions(skip=True)

    def forbidden():
        raise AssertionError("Skipped covariance must not request a backend")

    monkeypatch.setattr("lcl._results.cpu_device", forbidden)
    assert np.isnan(result._compute_covariance()).all()


def test_cached_factories_do_not_retain_dataset_arrays():
    def run():
        data = _random_panel_data(np.random.default_rng(417), 13, 2, 1)
        diff = _diff_unchosen_chosen(data)
        refs = [
            weakref.ref(array)
            for array in jax.tree.leaves((data, diff))
            if isinstance(array, jax.Array)
        ]
        fit, opt = FitOptions(num_devices=1), OptimizationOptions(maxiter=2)
        state = place_em_vars(_get_starting_vals(diff, data, 2, fit, opt), 1)
        jax.block_until_ready(_em_step(state, diff, data, 2, opt, fit))
        packing = ParamPacking(2, 2, 1, None)
        chunks = prepare_panel_chunks(diff, data)
        refs.extend(
            weakref.ref(array)
            for array in jax.tree.leaves(chunks)
            if isinstance(array, jax.Array)
        )
        solve = _compiled_polish(packing, opt)
        jax.block_until_ready(
            solve(
                packing.pack(state.betas, state.thetas, state.shares),
                diff,
                data,
                jnp.array(13.0),
                chunks,
            )
        )
        return refs

    refs = run()
    gc.collect()
    assert all(reference() is None for reference in refs)
    jax.clear_caches()
    eqx.clear_caches()
    gc.collect()
    assert all(reference() is None for reference in refs)


@pytest.mark.parametrize("bound_method", [False, True])
def test_bootstrap_does_not_retain_captured_prediction_arrays(bound_method):
    class Target:
        def __init__(self, design):
            self.design = design

        def evaluate(self, params):
            return jnp.sin(params[0] * self.design)

    def simulate():
        design = jax.device_put(jnp.arange(13.0), jax.local_devices()[-1])
        owner = Target(design)
        references = [weakref.ref(design), weakref.ref(owner)]

        def target(params):
            return jnp.sin(params[0] * design)

        result = parametric_bootstrap_se(
            owner.evaluate if bound_method else target,
            jnp.array([0.5]),
            jnp.array([[0.1]]),
            draws=35,
        )
        jax.block_until_ready(result)
        assert result.devices() == {jax.local_devices(backend="cpu")[0]}
        return references

    references = simulate()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_import_fit_and_prediction_preserve_callers_precision():
    script = """
import jax
assert not jax.config.x64_enabled
import lcl
assert not jax.config.x64_enabled
from tests.test_inference_hardening import _choice_frame
frame = _choice_frame(num_cases=24)
for model in (lcl.ConditionalLogit(), lcl.LatentClassConditionalLogit(num_classes=2)):
    result = model.fit(frame, alts_col="alt", cases_col="case", panels_col="panel",
        choice_col="choice", case_varnames=["price", "quality"])
    assert not jax.config.x64_enabled
    assert result.flat_params.dtype.name == "float64"
    result.coefficient_table() if hasattr(result, "coefficient_table") else result.beta_summary()
    result.loglik(frame)
    result.predict(frame).market_shares()
    assert not jax.config.x64_enabled
try:
    lcl.ConditionalLogit().fit(frame, alts_col="absent", cases_col="case",
        choice_col="choice", case_varnames=["price"])
except Exception:
    pass
assert not jax.config.x64_enabled
"""
    environment = dict(os.environ, JAX_ENABLE_X64="0")
    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_extreme_membership_logit_likelihood_matches_derivatives():
    data = _random_panel_data(np.random.default_rng(9), 1, 1, 0)
    # A single binary choice with a strongly unfavorable baseline class.
    data = data._replace(
        X=jnp.array([[0.0], [1.0]]),
        y=jnp.array([True, False]),
        alts=jnp.array([0, 1], dtype=jnp.uint32),
        cases=jnp.zeros(2, dtype=jnp.uint32),
        panels=jnp.zeros(2, dtype=jnp.uint32),
        panels_of_cases=jnp.zeros(1, dtype=jnp.uint32),
        num_cases_per_panel=jnp.ones(1, dtype=jnp.uint32),
        num_cases=1,
    )
    diff = _diff_unchosen_chosen(data)
    packing = ParamPacking(1, 2, 0, None)
    flat = jnp.array([1000.0, 0.0, -800.0])
    value = _total_loglik_kernel(flat, diff, data, packing)
    assert float(value) == pytest.approx(-800 - np.log(2))
    scores, hessian = _panel_scores_and_hessian(flat, diff, data, packing)
    np.testing.assert_allclose(
        jax.grad(_total_loglik_kernel)(flat, diff, data, packing),
        scores.sum(axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.hessian(_total_loglik_kernel)(flat, diff, data, packing),
        hessian,
        atol=1e-12,
    )
