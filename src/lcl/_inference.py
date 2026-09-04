"""Shared observed-information and covariance helpers."""

import logging
from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import numpy as onp
from jax.ops import segment_sum
from jax.typing import ArrayLike
from jaxtyping import Array, Float64

from lcl.options import _resolve_weight_type

logger = logging.getLogger(__name__)


def _symmetrize(
    matrix: Float64[Array, "params params"],
) -> Float64[Array, "params params"]:
    """Remove tiny numerical asymmetry from a covariance-style matrix."""
    return 0.5 * (matrix + matrix.T)


def _aggregate_scores(
    scores: ArrayLike,
    group_ids: ArrayLike,
    num_groups: int,
) -> Array:
    """Sum score rows within clusters.

    Parameters
    ----------
    scores : ArrayLike
        ``(rows, params)`` score contributions, already weighted if applicable.
    group_ids : ArrayLike
        Contiguous zero-indexed cluster identifier for each row.
    num_groups : int
        Number of clusters.

    Returns
    -------
    Array
        ``(num_groups, params)`` cluster-level scores.
    """
    ids = jnp.asarray(group_ids)
    if ids.shape[0] != jnp.asarray(scores).shape[0]:
        raise ValueError("Cluster identifiers must align one-to-one with score rows.")
    return cast(Array, segment_sum(jnp.asarray(scores), ids, num_segments=num_groups))


def _robust_covariance(
    hess_inv: ArrayLike,
    grad_n: ArrayLike,
    finite_sample_correction: bool = True,
    *,
    weights: ArrayLike | None = None,
    weight_type: str = "probability",
) -> Array:
    """Return an uncentered Huber-White sandwich covariance.

    The meat depends on how the weights are interpreted, which is the same
    distinction Stata draws between ``pweight`` and ``fweight``:

    * ``"probability"`` -- survey, sampling, or post-stratification weights.
      The score of the weighted objective for unit ``i`` is ``w_i s_i``, so the
      meat is ``sum_i w_i^2 s_i s_i'`` and the finite-sample multiplier uses the
      number of sampled units.
    * ``"frequency"`` -- replication counts for collapsed data.  Unit ``i``
      stands for ``w_i`` identical units, so the meat is ``sum_i w_i s_i s_i'``
      and the multiplier uses the implied total ``sum_i w_i``.

    The two coincide when every weight is one, which is the unweighted default.

    Parameters
    ----------
    hess_inv : ArrayLike
        Inverse observed information -- the bread.
    grad_n : ArrayLike
        ``(units, params)`` unweighted score contributions.
    finite_sample_correction : bool, default=True
        Apply the ``n / (n - 1)`` multiplier.
    weights : ArrayLike | None, optional
        Nonnegative weights aligned one-to-one with ``grad_n`` rows.
    weight_type : str, default="probability"
        Either ``"probability"`` or ``"frequency"``.

    Returns
    -------
    Array
        Sandwich covariance matrix.
    """
    scores = jnp.asarray(grad_n)
    resolved_type = _resolve_weight_type(weight_type)
    if weights is None:
        inner = scores.T @ scores
        n = jnp.asarray(scores.shape[0], dtype=scores.dtype)
    else:
        score_weights = jnp.asarray(weights, dtype=scores.dtype)
        if score_weights.shape != (scores.shape[0],):
            raise ValueError("weights must align one-to-one with score rows.")
        if not bool(jnp.all(jnp.isfinite(score_weights))):
            raise ValueError("weights must contain only finite values.")
        if not bool(jnp.all(score_weights >= 0.0)):
            raise ValueError("weights must be nonnegative.")
        if resolved_type == "frequency":
            inner = scores.T @ (scores * score_weights[:, None])
            n = jnp.sum(score_weights)
        else:
            weighted = scores * score_weights[:, None]
            inner = weighted.T @ weighted
            n = jnp.asarray(scores.shape[0], dtype=scores.dtype)
    if n < 2:
        raise ValueError("Robust covariance requires at least two score contributions.")
    correction = n / (n - 1) if finite_sample_correction else 1.0
    return cast(Array, correction * (hess_inv @ inner @ hess_inv))


class InformationDiagnostics(NamedTuple):
    """Rank and conditioning summary for an observed information matrix."""

    num_params: int
    rank: int
    rank_deficient: bool
    condition_number: float
    smallest_eigenvalue: float
    positive_definite: bool


def _invert_information(
    information: ArrayLike, label: str = "information matrix"
) -> tuple[Array, InformationDiagnostics]:
    """Invert a positive-definite symmetric information matrix with diagnostics."""
    matrix = onp.asarray(information, dtype=onp.float64)
    num_params = matrix.shape[0]
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = onp.linalg.eigvalsh(symmetric)
    largest = float(onp.max(onp.abs(eigenvalues))) if num_params else 0.0
    cutoff = num_params * onp.finfo(onp.float64).eps * largest
    keep = onp.abs(eigenvalues) > cutoff
    rank = int(keep.sum())
    smallest_abs = float(onp.min(onp.abs(eigenvalues[keep]))) if rank else 0.0
    diagnostics = InformationDiagnostics(
        num_params=num_params,
        rank=rank,
        rank_deficient=rank < num_params,
        condition_number=(largest / smallest_abs) if rank == num_params else onp.inf,
        smallest_eigenvalue=float(onp.min(eigenvalues)) if num_params else 0.0,
        positive_definite=bool(num_params and onp.all(eigenvalues > cutoff)),
    )
    if diagnostics.rank_deficient:
        logger.warning(
            "The %s is rank deficient: rank %d of %d. %d parameter direction(s) "
            "carry no curvature, so covariance and standard errors are unavailable. "
            "Common causes are a collapsed latent class, collinear columns, or a "
            "variable with no within-case variation.",
            label,
            diagnostics.rank,
            num_params,
            num_params - diagnostics.rank,
        )
    elif not diagnostics.positive_definite:
        logger.warning(
            "The %s is not positive definite (smallest eigenvalue %.3e). The "
            "estimate is a saddle point rather than a maximum, and standard "
            "errors are unavailable.",
            label,
            diagnostics.smallest_eigenvalue,
        )
    elif diagnostics.condition_number > 1e12:
        logger.warning(
            "The %s is severely ill conditioned (condition number %.3e). Standard "
            "errors may be unreliable; consider rescaling covariates or reducing "
            "the number of latent classes.",
            label,
            diagnostics.condition_number,
        )
    inverse: onp.ndarray[Any, Any]
    if not diagnostics.positive_definite:
        inverse = onp.full_like(symmetric, onp.nan)
    else:
        # Positive definiteness has already been certified above, so a Cholesky
        # solve is both faster than a pseudo-inverse and free of its silent
        # truncation of small singular values.
        try:
            factor = onp.linalg.cholesky(symmetric)
            identity = onp.eye(num_params, dtype=onp.float64)
            forward = onp.linalg.solve(factor, identity)
            inverse = onp.linalg.solve(factor.T, forward)
        except onp.linalg.LinAlgError:  # pragma: no cover - guarded by the check
            inverse = onp.linalg.pinv(symmetric, hermitian=True)
        inverse = 0.5 * (inverse + inverse.T)
    return jnp.asarray(inverse), diagnostics
