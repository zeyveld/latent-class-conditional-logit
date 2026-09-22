"""Boundary diagnostics for negatively constrained LCL utility coefficients."""

from collections.abc import Sequence
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from equinox import filter_jit
from jaxtyping import Array, ArrayLike, Float64, Integer

from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._params import ParamPacking
from lcl._struct import Data, DiffUnchosenChosen

BOUNDARY_DISTANCE_TOL = 1e-8
"""Absolute structural distance to the upper bound used to flag binding prices."""


def boundary_indices(
    betas: Float64[ArrayLike, "alt_vars classes"],
    packing: ParamPacking,
) -> Integer[np.ndarray, "binding_prices"]:
    """Return flat indices of coefficients within numerical distance of the bound."""
    if packing.numeraire_idx is None:
        return np.array([], dtype=int)
    row = packing.numeraire_idx
    distance = -np.asarray(betas)[row] - packing.numeraire_min_abs
    return row * packing.num_classes + np.flatnonzero(distance <= BOUNDARY_DISTANCE_TOL)


@filter_jit
def structural_score(
    flat_params: Float64[Array, "all_params"],
    diff: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> Float64[Array, "all_params"]:
    """Evaluate the structural score without dividing by a saturated Jacobian."""
    beta, theta = packing.unpack(flat_params)
    structural = jnp.concatenate([packing.to_structural(beta).ravel(), theta.ravel()])
    scores, _ = _panel_scores_and_hessian(
        structural, diff, data, replace(packing, numeraire_idx=None)
    )
    return jnp.mean(scores, axis=0)


def boundary_kkt_violation(
    score: Float64[ArrayLike, "all_params"],
    indices: Integer[ArrayLike, "binding_prices"] | Sequence[int],
) -> float:
    """Measure feasible ascent at beta <= upper_bound for a maximized likelihood.

    A nonnegative score at the upper bound is valid. A negative score means
    moving into the feasible interior raises likelihood, even if softplus has
    saturated and the raw optimizer score has rounded to zero.
    """
    index_array = np.asarray(indices, dtype=int)
    values = np.asarray(score)[index_array]
    return float(np.max(np.maximum(-values, 0.0))) if index_array.size else 0.0
