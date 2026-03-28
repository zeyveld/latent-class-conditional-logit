"""General utilities for LCL library."""

import jax.numpy as jnp
import pandas as pd
import polars as pl
from jax import Array
from jax.typing import ArrayLike, DTypeLike


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
        return
    else:
        return jnp.asarray(data, dtype=dtype)


def _robust_covariance(hess_inv: ArrayLike, grad_n: ArrayLike) -> ArrayLike:
    """Apply the Huber/White heteroskedasticity correction to the covariance matrix.

    Utilizes the outer product of the gradients (BHHH estimator) to construct
    a sandwich estimator robust to general heteroskedasticity.

    Parameters
    ----------
    hess_inv : ArrayLike
        ``(K, K)`` Inverse of the negative Hessian matrix.
    grad_n : ArrayLike
        ``(N, K)`` matrix of case-level contributions to the gradient.

    Returns
    -------
    ArrayLike
        ``(K, K)`` Huber-White robust covariance matrix.
    """
    n = jnp.shape(grad_n)[0]
    grad_n_sub = grad_n - (
        jnp.sum(grad_n, axis=0) / n
    )  # Subtract the mean gradient value
    inner = jnp.transpose(grad_n_sub) @ grad_n_sub
    correction = (n) / (n - 1)
    return correction * (hess_inv @ inner @ hess_inv)


def _ensure_sequential(vals) -> Array:
    """Ensure vector IDs are contiguous and zero-indexed.

    Crucial for safely utilizing JAX's `segment_sum`, which requires group IDs
    to map perfectly to array indices.

    Parameters
    ----------
    vals : ArrayLike
        A 1D array of potentially non-sequential identifiers (e.g., [10, 10, 24, 24, 99]).

    Returns
    -------
    Array
        A 1D array of strictly sequential identifiers (e.g., [0, 0, 1, 1, 2]).
    """
    vals_change = vals != jnp.roll(vals, shift=1)
    return (jnp.cumsum(vals_change) - 1).astype("uint32")


# EOF
