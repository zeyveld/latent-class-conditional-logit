"""General utilities for xlogit library."""

import jax.numpy as jnp
import pandas as pd
import polars as pl
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def _as_array_or_none(
    data: pl.DataFrame | pd.DataFrame | ArrayLike | None,
    dtype: DTypeLike | None = None,
) -> Array | None:
    if data is None:
        return
    else:
        return jnp.asarray(data, dtype=dtype)


def _robust_covariance(hess_inv: ArrayLike, grad_n: ArrayLike) -> ArrayLike:
    """Apply the Huber/White heteroskedasticity correction."""
    n = jnp.shape(grad_n)[0]
    grad_n_sub = grad_n - (
        jnp.sum(grad_n, axis=0) / n
    )  # Subtract the mean gradient value
    inner = jnp.transpose(grad_n_sub) @ grad_n_sub
    correction = (n) / (n - 1)
    return correction * (hess_inv @ inner @ hess_inv)


def _ensure_sequential(vals) -> Array:
    """Ensure ids can also serve as indices"""
    vals_change = vals != jnp.roll(vals, shift=1)
    return (jnp.cumsum(vals_change) - 1).astype("uint32")


# EOF
