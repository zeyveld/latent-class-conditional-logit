"""General utilities for LCL library."""

import logging
from typing import NamedTuple, cast

import jax.numpy as jnp
import numpy as onp
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import Array

logger = logging.getLogger(__name__)


def _as_array_or_none(
    data: pl.DataFrame | pd.DataFrame | ArrayLike | None,
    dtype: DTypeLike | None = None,
) -> Array | None:
    """Safely convert Polars or Pandas DataFrames to raw JAX arrays.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | ArrayLike | None
        The input data structure.
    dtype : DTypeLike | None, optional
        Target numeric type (e.g., 'float64', 'uint32').

    Returns
    -------
    Array | None
        The resulting JAX tensor, or None if the input was null.
    """
    if data is None:
        return None
    else:
        return jnp.asarray(data, dtype=dtype)


def _robust_covariance(
    hess_inv: ArrayLike, grad_n: ArrayLike, finite_sample_correction: bool = True
) -> Array:
    """Apply the Huber/White heteroskedasticity correction to the covariance matrix.

    Utilizes the outer product of the gradients (BHHH estimator) to construct
    a sandwich estimator robust to general heteroskedasticity.

    The scores are used as-is rather than being centered.  At a maximum
    likelihood estimate they already sum to zero, so centering is a no-op there;
    away from one it masks non-convergence rather than correcting for it.  This
    matches both Stata's ``vce(robust)`` for maximum likelihood and the
    clustered branches elsewhere in this package.

    Parameters
    ----------
    hess_inv : ArrayLike
        ``(K, K)`` Inverse of the negative Hessian matrix.
    grad_n : ArrayLike
        ``(N, K)`` matrix of case-level contributions to the gradient.
    finite_sample_correction : bool, default=True
        Apply the ``N / (N - 1)`` multiplier.

    Returns
    -------
    ArrayLike
        ``(K, K)`` Huber-White robust covariance matrix.
    """
    n = jnp.shape(grad_n)[0]
    if n < 2:
        raise ValueError("Robust covariance requires at least two score contributions.")
    inner = jnp.transpose(grad_n) @ grad_n
    correction = n / (n - 1) if finite_sample_correction else 1.0
    return cast(Array, correction * (hess_inv @ inner @ hess_inv))


class InformationDiagnostics(NamedTuple):
    """Rank and conditioning summary for an inverted information matrix.

    Attributes
    ----------
    num_params : int
        Order of the matrix.
    rank : int
        Numerical rank at the pseudo-inverse cutoff.
    rank_deficient : bool
        Whether ``rank < num_params``, meaning at least one parameter direction
        carries no curvature and its standard error is not identified.
    condition_number : float
        Ratio of largest to smallest absolute eigenvalue, or ``inf`` when the
        matrix is singular.
    smallest_eigenvalue : float
        Smallest eigenvalue; negative values indicate a saddle point rather
        than an optimum.
    positive_definite : bool
        Whether every eigenvalue is strictly above the cutoff.
    """

    num_params: int
    rank: int
    rank_deficient: bool
    condition_number: float
    smallest_eigenvalue: float
    positive_definite: bool


def _invert_information(
    information: ArrayLike, label: str = "information matrix"
) -> tuple[Array, InformationDiagnostics]:
    """Invert a symmetric information matrix and report its conditioning.

    A bare ``pinv`` silently truncates singular values below its cutoff, so a
    genuinely unidentified parameter direction — a collapsed latent class, a
    collinear demographic — yields finite, small standard errors instead of an
    obvious failure.  This wrapper performs the same pseudo-inversion but
    inspects the spectrum first and emits a warning when the result should not
    be trusted.

    Parameters
    ----------
    information : ArrayLike
        ``(K, K)`` symmetric information matrix.  Expected to be positive
        definite at an interior optimum.
    label : str, default="information matrix"
        Name used in warning messages.

    Returns
    -------
    inverse : Array
        Moore-Penrose pseudo-inverse, symmetrized.
    diagnostics : InformationDiagnostics
        Rank and conditioning summary.
    """
    matrix = onp.asarray(information, dtype=onp.float64)
    num_params = matrix.shape[0]
    symmetric = 0.5 * (matrix + matrix.T)

    eigenvalues, eigenvectors = onp.linalg.eigh(symmetric)
    largest = float(onp.max(onp.abs(eigenvalues))) if num_params else 0.0

    # Same cutoff numpy.linalg.pinv applies, made explicit so the rank it
    # implies can be reported instead of silently absorbed.
    cutoff = num_params * onp.finfo(onp.float64).eps * largest
    keep = onp.abs(eigenvalues) > cutoff
    rank = int(keep.sum())

    inverse_eigenvalues = onp.where(keep, 1.0 / onp.where(keep, eigenvalues, 1.0), 0.0)
    inverse = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    inverse = 0.5 * (inverse + inverse.T)

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
            "carry no curvature, so the pseudo-inverse reports finite standard "
            "errors for quantities the data do not identify. Treat the affected "
            "standard errors as invalid; common causes are a collapsed latent "
            "class, collinear columns, or a variable with no within-case "
            "variation.",
            label,
            diagnostics.rank,
            num_params,
            num_params - diagnostics.rank,
        )
    elif not diagnostics.positive_definite:
        logger.warning(
            "The %s is not positive definite (smallest eigenvalue %.3e). The "
            "estimate is a saddle point rather than a maximum, and the reported "
            "standard errors are not valid.",
            label,
            diagnostics.smallest_eigenvalue,
        )
    elif diagnostics.condition_number > 1e12:
        logger.warning(
            "The %s is severely ill conditioned (condition number %.3e). "
            "Standard errors may be unreliable; consider rescaling covariates "
            "or reducing the number of latent classes.",
            label,
            diagnostics.condition_number,
        )

    return jnp.asarray(inverse), diagnostics
