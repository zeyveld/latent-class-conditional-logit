"""Deterministic ratio screening without changing Gaussian parameter draws."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import norm

from lcl._delta import (
    GAUSSIAN_BOUND_PROBABILITY_LIMIT,
    gaussian_upper_tail_probability,
    parametric_bootstrap_se,
)


def test_crossing_probability_has_correct_tail_and_handles_fixed_coefficients():
    actual = gaussian_upper_tail_probability(
        [-2.0, -2.0, 0.0, 0.1, -1.0, -1.0],
        [1.0, 4.0, 0.0, 0.0, np.nan, -1.0],
        np.zeros(6),
    )
    np.testing.assert_allclose(actual, [norm.sf(2), norm.sf(1), 0, 1, np.nan, np.nan])
    # Positive changes of units leave a probability unchanged.
    np.testing.assert_allclose(
        gaussian_upper_tail_probability(np.array([-2.0]) * 1000, [1e6], [0.0]),
        actual[:1],
    )


@pytest.mark.parametrize("seed,draws", [(0, 2), (4, 500), (17, 20000)])
def test_ratio_screen_is_independent_of_draw_count_and_seed(seed, draws):
    with pytest.raises(ValueError, match="bound-crossing probability exceeds 0.001"):
        parametric_bootstrap_se(
            lambda p: 1 / -p[0],
            jnp.array([-2.0]),
            jnp.ones((1, 1)),
            upper_bounds=jnp.zeros(1),
            seed=seed,
            draws=draws,
        )


def test_passing_screen_keeps_even_rare_infeasible_draws_unmodified():
    # An artificial linear target isolates the simulation law from ratio moments.
    seed, draws = 0, 20000
    mean = ndtri(GAUSSIAN_BOUND_PROBABILITY_LIMIT / 2)
    raw = mean + np.random.default_rng(seed).standard_normal((draws, 1))
    assert np.any(raw > 0.0)
    actual = parametric_bootstrap_se(
        lambda p: p[0],
        jnp.array([mean]),
        jnp.ones((1, 1)),
        upper_bounds=jnp.zeros(1),
        seed=seed,
        draws=draws,
    )
    assert float(actual) == pytest.approx(raw.std(ddof=1), rel=1e-12)


def test_fixed_binding_denominator_keeps_conditional_gaussian_uncertainty():
    params, cov = jnp.array([-0.01, 1.0]), jnp.diag(jnp.array([0.0, 0.04]))
    target = lambda p: p[1] / -p[0]  # noqa: E731
    actual = parametric_bootstrap_se(
        target, params, cov, upper_bounds=jnp.array([-0.01, jnp.inf])
    )
    reference = parametric_bootstrap_se(target, params, cov)
    np.testing.assert_array_equal(actual, reference)
