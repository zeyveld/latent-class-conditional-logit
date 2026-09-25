"""Local geometry and state reuse in the structural Newton solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcl._optimize import (
    curvature_step_norm,
    exact_newton_minimize,
    projected_active_set,
    projected_decrement,
    regularized_newton_direction,
    scaled_objective,
)
from lcl.constraints import NegativeCoefficientBound
from lcl._params import ParamPacking


def test_active_set_releases_inward_slope_and_shrinks_near_stationarity():
    bounds = jnp.zeros(4)
    params = jnp.array([0.0, 0.0, -0.005, -0.02])
    grad = jnp.array([-1.0, 1.0, -1.0, -1.0])
    scale = jnp.ones(4)
    np.testing.assert_array_equal(
        projected_active_set(params, grad, scale, bounds), [True, False, True, False]
    )
    np.testing.assert_array_equal(
        projected_active_set(params, grad * 1e-8, scale, bounds),
        [True, False, False, False],
    )
    # Positive unit changes leave the neighborhood and KKT sign unchanged.
    units = jnp.array([1e-3, 1e3, 0.2, 5.0])
    np.testing.assert_array_equal(
        projected_active_set(params / units, grad * units, scale * units, bounds),
        projected_active_set(params, grad, scale, bounds),
    )


def test_projected_decrement_uses_feasible_active_displacement():
    params = jnp.array([-0.005, -2.0])
    grad = jnp.array([-2.0, 3.0])
    direction = jnp.array([2.0, -1.0])
    active = jnp.array([True, False])
    assert float(
        projected_decrement(params, grad, direction, active, jnp.zeros(2))
    ) == pytest.approx(np.sqrt(0.01 + 3.0))
    assert (
        float(
            projected_decrement(
                params, jnp.zeros(2), jnp.full(2, jnp.nan), active, None
            )
        )
        == 0
    )
    assert np.isinf(projected_decrement(params, grad, -direction, active, None))


@pytest.mark.parametrize(
    "hessian",
    [
        [[2.0, 0.8], [0.8, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],  # Indefinite raw Hessian, as in mixture polishing.
    ],
)
@pytest.mark.parametrize("bounded", [False, True])
def test_step_metric_is_the_regularized_two_metric_norm(hessian, bounded):
    h = jnp.asarray(hessian)
    x = jnp.array([-0.001, -1.0])
    g = jnp.array([-2.0, 1.0])
    newton = regularized_newton_direction(x, g, h, jnp.zeros(2) if bounded else None)
    scale = np.asarray(newton.diagonal_scale)
    metric = np.asarray(h).copy()
    active = np.asarray(newton.active)
    metric[active, :] = 0
    metric[:, active] = 0
    metric += np.diag(active * scale**2 + float(newton.shift) * scale**2)
    assert np.linalg.eigvalsh(metric).min() > 0
    step = jnp.array([0.001, -0.2])
    assert float(curvature_step_norm(step, h, newton)) == pytest.approx(
        np.sqrt(step @ metric @ step)
    )
    if not bounded:
        assert float(curvature_step_norm(newton.direction, h, newton)) == pytest.approx(
            float(newton.decrement)
        )


@pytest.mark.parametrize("budget", [1, 2, 4, 8, 20])
def test_interior_bounds_preserve_the_entire_newton_path(budget):
    h = jnp.array([[1.0, 0.9], [0.9, 1.0]])
    optimum = jnp.array([-3.0, -2.0])

    def value(x):
        return 0.5 * (x - optimum) @ h @ (x - optimum)

    def derivatives(x):
        return value(x), h @ (x - optimum), h

    def solve(bounds):
        return exact_newton_minimize(
            value,
            derivatives,
            jnp.array([-0.1, -0.2]),
            upper_bounds=bounds,
            initial_trust_radius=0.3,
            maxiter=budget,
            tol=1e-12,
        )

    free, bounded = solve(None), solve(jnp.zeros(2))
    for field in ("params", "error", "trust_radius", "step_num", "num_fun_eval"):
        np.testing.assert_allclose(
            getattr(free, field), getattr(bounded, field), atol=1e-12
        )


@pytest.mark.parametrize("reject_step", [False, True])
def test_each_iterate_solves_for_a_direction_only_once(monkeypatch, reject_step):
    import lcl._optimize as optimizer

    evaluated_at = []
    original = optimizer.regularized_newton_direction

    def recording_direction(params, *args):
        jax.debug.callback(
            lambda p: evaluated_at.append(np.array(p)), params, ordered=True
        )
        return original(params, *args)

    monkeypatch.setattr(optimizer, "regularized_newton_direction", recording_direction)

    def value(x):
        return jnp.asarray(jnp.inf) if reject_step else jnp.sum(x**2) / 2

    def derivatives(x):
        return jnp.sum(x**2) / 2, x, jnp.eye(x.size)

    state = jax.jit(lambda x: exact_newton_minimize(value, derivatives, x, maxiter=10))(
        jnp.array([2.0, 1.0])
    )
    jax.block_until_ready(state)
    jax.effects_barrier()
    assert len(evaluated_at) == int(state.num_grad_hess_eval)
    assert len(evaluated_at) == (1 if reject_step else int(state.step_num) + 1)
    np.testing.assert_array_equal(evaluated_at[-1], state.params)


def test_scale_helper_normalizes_value_gradient_and_hessian_together():
    def value(p, weights):
        return weights.sum() * (p @ p)

    def derivatives(p, weights):
        return (
            (value(p, weights), jnp.zeros((3, 2))),
            2 * weights.sum() * p,
            2 * weights.sum() * jnp.eye(2),
        )

    p, w = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0])
    f, vgh = scaled_objective(value, derivatives, w.sum())
    loss, grad, hessian = vgh(p, w)
    np.testing.assert_allclose([f(p, w), loss], [5, 5])
    np.testing.assert_allclose(grad, [2, 4])
    np.testing.assert_allclose(hessian, 2 * np.eye(2))


def test_one_bound_constructor_covers_cl_and_packed_lcl_layouts():
    bound = NegativeCoefficientBound(1, 0.02)
    np.testing.assert_array_equal(
        bound.upper_bounds(jnp.zeros(3)), [np.inf, -0.02, np.inf]
    )
    packing = ParamPacking(3, 2, 1, 1, 0.02)
    expected = np.full(packing.num_params, np.inf)
    expected[2:4] = -0.02
    np.testing.assert_array_equal(packing.upper_bounds(), expected)
    assert packing.negative_bound == bound
    assert NegativeCoefficientBound().upper_bounds(jnp.zeros(3)) is None
