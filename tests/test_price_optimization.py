"""Structural bound handling, KKT release, and compilation regressions."""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from lcl._case_utils import _loglik_gradient, _loglik_value
from lcl._delta import parametric_bootstrap_se
from lcl._em_alg_steps import _distributed_update
from lcl._optimize import _minimize, _minimize_kernel, exact_newton_minimize
from lcl._params import ParamPacking
from lcl._polish import _compiled_polish
from lcl._struct import Data, DiffUnchosenChosen
from lcl.constraints import NegativeCoefficientBound

from lcl.options import OptimizationOptions


def binary_diff():
    # Four choices of the cheaper alternative and one of the expensive one:
    # likelihood optimum log(1/4), independently of the optimizer.
    return DiffUnchosenChosen(
        X=jnp.array([[1.0], [1.0], [1.0], [1.0], [-1.0]]),
        alts=jnp.zeros(5, dtype=jnp.uint32),
        cases=jnp.arange(5, dtype=jnp.uint32),
        panels=jnp.arange(5, dtype=jnp.uint32),
        num_cases=5,
    )


@pytest.mark.parametrize("min_abs", [1e-8, 1e-5, 0.01, 0.5])
def test_binding_estimate_is_stored_exactly_and_has_zero_kkt_error(min_abs):
    diff = binary_diff()
    weights = jnp.array([[1.0, 1.0, 1.0, 1.0, 16.0]])
    betas, error = _distributed_update(
        jnp.array([[-1.0]]),
        weights,
        diff,
        NegativeCoefficientBound(0, min_abs),
        OptimizationOptions(maxiter=100),
    )
    assert float(betas[0, 0]) == -min_abs
    assert float(error[0]) == 0.0
    packing = ParamPacking(1, 1, 0, 0, min_abs)
    flat = packing.pack(betas.T, None, jnp.ones(1))
    np.testing.assert_array_equal(packing.unpack(flat)[0], betas.T)
    np.testing.assert_array_equal(packing.upper_bounds(), [-min_abs])


def test_parameter_simulation_screens_high_bound_crossing_probability():
    params = jnp.array([-0.01, 2.0])
    with pytest.raises(ValueError, match="bound-crossing probability exceeds"):
        parametric_bootstrap_se(
            lambda p: p[1] / -p[0],
            params,
            jnp.eye(2),
            draws=100,
            upper_bounds=NegativeCoefficientBound(0).upper_bounds(params),
        )


@pytest.mark.parametrize("start", [-1e-5, -1e-5 - 1e-12, -1.0, 0.0, 30.0])
@pytest.mark.parametrize("units", [0.001, 1.0, 1000.0])
def test_conditional_logit_leaves_boundary_for_interior_optimum(start, units):
    diff = binary_diff()
    diff = diff._replace(X=diff.X * units)
    result = _minimize(
        _loglik_value,
        _loglik_gradient,
        jnp.array([start / units]),
        args=(diff, jnp.ones(5)),
        negative_bound=NegativeCoefficientBound(0),
        optimization_options=OptimizationOptions(maxiter=100),
        objective_scale=5.0,
    )
    beta = result.params
    assert result.success
    assert 0 < result.nit < 100
    np.testing.assert_allclose(beta * units, [-np.log(4)], atol=3e-6)


def test_weighted_class_can_leave_previous_boundary_and_zero_mass_is_stationary(caplog):
    diff = binary_diff()
    options = OptimizationOptions(maxiter=100)

    @eqx.filter_jit
    def update(beta, weights):
        return _distributed_update(
            beta, weights, diff, NegativeCoefficientBound(0), options
        )

    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        # First class has a truly binding constraint; the second is padded.
        weights = jnp.array([[1.0, 1.0, 1.0, 1.0, 16.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        betas, error = update(jnp.zeros((2, 1)), weights)
        jax.block_until_ready((betas, error))
        assert float(betas[0][0]) == pytest.approx(-1e-5, abs=1e-15)
        assert float(error[1]) == 0.0
        assert np.all(np.isfinite(betas))
        # A changed E-step reverses the KKT sign at the same price boundary.
        for _ in range(3):
            betas, error = update(betas, jnp.ones((2, 5)))
            jax.block_until_ready((betas, error))
            for beta in betas:
                np.testing.assert_allclose(beta, [-np.log(4)], atol=3e-6)
    messages = [r.getMessage() for r in caplog.records]
    assert (
        sum("Compiling jit(update)" in m or "Compiling update " in m for m in messages)
        == 1
    )


def test_correlated_bounds_match_independent_solver_under_vmap():
    # Dense curvature is essential: clipping an unrestricted Newton direction
    # can fail when an outward-bound coordinate couples to an inward one.
    rng = np.random.default_rng(491)
    a = rng.normal(size=(7, 7))
    h = a.T @ a + np.eye(7) * 0.2
    upper = np.array([0.0, np.inf, 0.0, 0.0, np.inf, 0.0, np.inf])
    linear = rng.normal(size=(12, 7))

    def value(x, q):
        return 0.5 * x @ h @ x + q @ x

    def derivatives(x, q):
        return value(x, q), h @ x + q, jnp.asarray(h)

    def solve(q):
        return exact_newton_minimize(
            value,
            derivatives,
            jnp.zeros(7),
            q,
            upper_bounds=jnp.asarray(upper),
            tol=1e-8,
            maxiter=200,
        )

    states = jax.jit(jax.vmap(solve))(jnp.asarray(linear))
    assert not np.any(states.failed)
    assert np.all(states.error < 1e-8)
    assert np.all(states.params <= upper)
    for actual, q in zip(states.params, linear):
        reference = minimize(
            lambda x: 0.5 * x @ h @ x + q @ x,
            np.zeros(7),
            jac=lambda x: h @ x + q,
            method="L-BFGS-B",
            bounds=[(None, u) for u in upper],
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
        )
        np.testing.assert_allclose(actual, reference.x, atol=2e-6)


def test_polish_leaves_boundary_without_recompiling(caplog):
    # The marginal choice probability has optimum 0.2. Individual tastes and
    # shares are unidentified here, exercising Hessian regularization too.
    diff = binary_diff()
    data = Data(
        X=jnp.zeros((10, 1)),
        dems=None,
        y=jnp.zeros(10, dtype=bool),
        alts=jnp.tile(jnp.arange(2, dtype=jnp.uint32), 5),
        cases=jnp.repeat(jnp.arange(5, dtype=jnp.uint32), 2),
        panels=jnp.repeat(jnp.arange(5, dtype=jnp.uint32), 2),
        panels_of_cases=jnp.arange(5, dtype=jnp.uint32),
        num_cases_per_panel=jnp.ones(5, dtype=jnp.uint32),
        num_cases=5,
        num_alt_vars=1,
        num_panels=5,
        num_dem_vars=0,
    )
    packing = ParamPacking(1, 2, 0, 0)
    _compiled_polish.cache_clear()
    solve = _compiled_polish(
        packing, OptimizationOptions(maxiter=100, newton_decrement_tol=1e-10)
    )
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        for start in (-1e-5, -1e-5 - 1e-12, 0.0):
            result, steps = solve(
                jnp.array([start, start, 0.0]), diff, data, jnp.array(5.0)
            )
            jax.block_until_ready(result)
            beta, theta = packing.unpack(result)
            probability = jax.nn.sigmoid(beta[0]) @ jax.nn.softmax(
                jnp.array([0.0, theta[0, 0]])
            )
            np.testing.assert_allclose(probability, 0.2, atol=1e-7)
            assert 0 < int(steps) <= 100
    messages = [r.getMessage() for r in caplog.records]
    assert (
        sum("Compiling jit(run)" in m or "Compiling run " in m for m in messages) == 1
    )


def test_standalone_solver_reuses_compilation_for_changed_data_and_starts(caplog):
    # A distinct configuration ensures this test measures a cold compilation.
    options = OptimizationOptions(maxiter=97, newton_decrement_tol=1e-6)
    diff = binary_diff()
    caplog.set_level(logging.WARNING, logger="jax._src.interpreters.pxla")
    with jax.log_compiles(True):
        for start, units in [(-1e-5, 1.0), (0.0, 2.0), (-1.0, 0.5)]:
            state = _minimize_kernel(
                _loglik_value,
                _loglik_gradient,
                jnp.array([start]),
                (diff._replace(X=diff.X * units), jnp.ones(5)),
                options,
                NegativeCoefficientBound(0),
                jnp.array(5.0),
            )
            jax.block_until_ready(state)
            np.testing.assert_allclose(state.params * units, [-np.log(4)], atol=3e-6)
    messages = [r.getMessage() for r in caplog.records]
    assert (
        sum(
            "Compiling jit(_minimize_kernel)" in m or "Compiling _minimize_kernel " in m
            for m in messages
        )
        == 1
    )
