"""Delta-method and parametric-bootstrap inference for functions of the parameters.

Both fitted-model classes expose the same two routines, so a target function
written once -- a willingness-to-pay ratio, a market share, an aggregate
elasticity -- gets standard errors from either.  Everything here operates on the
*latent* parameter vector and its covariance: constrained coefficients reach
their structural scale inside the target function, so the softplus Jacobian is
picked up by differentiation rather than applied by hand at each call site.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as onp
from jax import jacfwd, jacrev
from jax.tree_util import Partial
from jaxtyping import Array, Float64

from lcl._jax_compat import cpu_device, device_put_array_leaves

logger = logging.getLogger(__name__)

NEGATIVE_VARIANCE_RTOL = 1e-8
"""Relative slack allowed on a negative delta-method variance before warning."""


def _jacobian(
    func: Callable[[Float64[Array, "all_params"]], Float64[Array, "..."]],
    flat_params: Float64[Array, "all_params"],
    num_outputs: int,
) -> Float64[Array, "..."]:
    """Differentiate ``func`` in whichever mode costs fewer passes.

    Reverse mode costs one pass per output and forward mode one per parameter, so
    a scalar target (a willingness-to-pay ratio) wants ``jacrev`` while a long
    vector target (per-alternative market shares) wants ``jacfwd``.
    """
    mode = jacrev if num_outputs < flat_params.size else jacfwd
    return mode(func)(flat_params)


def apply_delta_method(
    func: Callable[..., Float64[Array, "..."]],
    flat_params: Float64[Array, "all_params"],
    cov_matrix: Float64[Array, "all_params all_params"],
    label: str = "delta method",
    **kwargs: Any,
) -> tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """Return ``func(flat_params)`` and its delta-method standard errors.

    Parameters
    ----------
    func : Callable
        Target taking the flat latent parameter vector as its only positional
        argument and returning a scalar or array.
    flat_params : Float64[Array, "all_params"]
        Latent parameters at which to evaluate.
    cov_matrix : Float64[Array, "all_params all_params"]
        Covariance of ``flat_params``, in the same latent parameterization.
    label : str, default="delta method"
        Name used in diagnostics when a variance comes back negative.
    **kwargs
        Extra keyword arguments bound into ``func``.  They are keyword-only so a
        stray positional argument cannot silently displace ``flat_params``.

    Returns
    -------
    value : Float64[Array, "..."]
        ``func`` evaluated at ``flat_params``.
    standard_error : Float64[Array, "..."]
        Square roots of ``diag(J C J')``.  An entry is ``NaN`` when its variance
        is negative beyond round-off, which means the covariance is not positive
        semidefinite.
    """
    cpu = cpu_device()
    with jax.default_device(cpu):
        flat_params_cpu = device_put_array_leaves(flat_params, cpu)
        kwargs_cpu = device_put_array_leaves(kwargs, cpu)
        cov_cpu = device_put_array_leaves(jnp.asarray(cov_matrix), cpu)

        target_func = Partial(func, **kwargs_cpu)
        value = jnp.asarray(target_func(flat_params_cpu))
        jacobian = _jacobian(target_func, flat_params_cpu, value.size)

        jac_rows = jacobian.reshape((-1, flat_params_cpu.size))
        variance = jnp.einsum("ip,pq,iq->i", jac_rows, cov_cpu, jac_rows)

        # A negative quadratic form can only come from a covariance that is not
        # positive semidefinite.  Round-off is clamped; a real violation is
        # surfaced rather than presented as a standard error of zero.
        diagonal = jnp.clip(jnp.diag(cov_cpu), min=0.0)
        magnitude = jnp.abs(jac_rows) @ jnp.sqrt(diagonal)
        tolerance = NEGATIVE_VARIANCE_RTOL * magnitude**2
        invalid = variance < -tolerance
        if bool(jnp.any(invalid)):
            logger.warning(
                "%s produced %d negative variance(s); the covariance matrix is "
                "not positive semidefinite. Those standard errors are NaN.",
                label,
                int(jnp.sum(invalid)),
            )
        standard_error = jnp.where(
            invalid, jnp.nan, jnp.sqrt(jnp.clip(variance, min=0.0))
        )
        return value, standard_error.reshape(value.shape)


def parametric_bootstrap_se(
    func: Callable[..., Float64[Array, "..."]],
    flat_params: Float64[Array, "all_params"],
    cov_matrix: Float64[Array, "all_params all_params"],
    *,
    draws: int = 500,
    seed: int = 0,
    **kwargs: Any,
) -> Float64[Array, "..."]:
    """Estimate standard errors from asymptotic draws of the latent parameters.

    Draws are taken in the *latent* parameterization and passed through ``func``,
    which applies the structural transform.  Drawing structural coefficients
    directly would put mass on the sign-flipped region that the softplus
    parameterization excludes, and a ratio with such a denominator has no finite
    variance to estimate.

    Parameters
    ----------
    func : Callable
        Target taking the flat latent parameter vector.
    flat_params : Float64[Array, "all_params"]
        Latent parameter estimates.
    cov_matrix : Float64[Array, "all_params all_params"]
        Latent covariance.
    draws : int, default=500
        Number of draws.
    seed : int, default=0
        Reproducible seed.
    **kwargs
        Extra keyword arguments bound into ``func``.

    Returns
    -------
    Float64[Array, "..."]
        Sample standard deviation of ``func`` across draws.
    """
    if draws < 2:
        raise ValueError("bootstrap_draws must be at least 2.")
    covariance = onp.asarray(cov_matrix, dtype=onp.float64)
    if not onp.all(onp.isfinite(covariance)):
        raise ValueError("A finite covariance matrix is required for bootstrap SEs.")
    eigenvalues, eigenvectors = onp.linalg.eigh(0.5 * (covariance + covariance.T))
    tolerance = (
        onp.finfo(onp.float64).eps
        * max(1.0, float(onp.max(onp.abs(eigenvalues))))
        * covariance.shape[0]
    )
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("The covariance matrix is not positive semidefinite.")
    root = eigenvectors * onp.sqrt(onp.maximum(eigenvalues, 0.0))[None, :]
    rng = onp.random.default_rng(seed)
    parameters = onp.asarray(flat_params, dtype=onp.float64)
    standard_normal = rng.standard_normal((draws, parameters.size))
    parameter_draws = parameters + standard_normal @ root.T
    target = Partial(func, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = jax.vmap(target)(jnp.asarray(parameter_draws))
    return jnp.std(values, axis=0, ddof=1)


__all__ = [
    "NEGATIVE_VARIANCE_RTOL",
    "apply_delta_method",
    "parametric_bootstrap_se",
]
