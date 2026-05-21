"""Pure JAX numerical kernels for choice probabilities and class likelihoods."""

import jax.numpy as jnp
from equinox import filter_jit
from jax import lax
from jax.nn import softmax
from jax.ops import segment_max, segment_sum
from jaxtyping import Array, Float64, UInt


@filter_jit
def _choice_probabilities_and_logsum(
    X: Float64[Array, "rows alt_vars"],
    betas: Float64[Array, "alt_vars classes"],
    cases: UInt[Array, "rows"],
    num_cases: int,
) -> tuple[Float64[Array, "rows classes"], Float64[Array, "cases classes"]]:
    """Compute logit probabilities and inclusive values with a case-level shift."""
    V = X @ betas
    shift = segment_max(V, cases, num_segments=num_cases)
    shift = lax.stop_gradient(shift)
    e = jnp.exp(V - shift[cases])
    den = segment_sum(e, cases, num_segments=num_cases)
    probs = e / den[cases]
    logsum = shift + jnp.log(den)
    return probs, logsum


@filter_jit
def _diff_logit_components(
    X_diff: Float64[Array, "unchosen_rows alt_vars"],
    betas: Float64[Array, "alt_vars"],
    cases: UInt[Array, "unchosen_rows"],
    num_cases: int,
) -> tuple[Float64[Array, "cases"], Float64[Array, "unchosen_rows"]]:
    """Compute chosen log probabilities and unchosen probabilities from differenced X."""
    Vd = X_diff @ betas
    shift = jnp.maximum(0.0, segment_max(Vd, cases, num_segments=num_cases))
    shift = lax.stop_gradient(shift)
    e_shifted = jnp.exp(Vd - shift[cases])
    den_shifted = jnp.exp(-shift) + segment_sum(
        e_shifted, cases, num_segments=num_cases
    )
    log_chosen_probs = -shift - jnp.log(den_shifted)
    p_unchosen = e_shifted / den_shifted[cases]
    return log_chosen_probs, p_unchosen


@filter_jit
def _diff_log_kernels(
    X_diff: Float64[Array, "unchosen_rows alt_vars"],
    betas: Float64[Array, "alt_vars classes"],
    diff_cases: UInt[Array, "unchosen_rows"],
    num_cases: int,
    panels_of_cases: UInt[Array, "cases"],
    num_panels: int,
) -> Float64[Array, "panels classes"]:
    """Compute panel-level log likelihood kernels by latent class."""
    Vd = X_diff @ betas
    shift = jnp.maximum(0.0, segment_max(Vd, diff_cases, num_segments=num_cases))
    shift = lax.stop_gradient(shift)
    e_shifted = jnp.exp(Vd - shift[diff_cases])
    den_shifted = jnp.exp(-shift) + segment_sum(
        e_shifted, diff_cases, num_segments=num_cases
    )
    log_probs_by_class = -shift - jnp.log(den_shifted)
    return segment_sum(log_probs_by_class, panels_of_cases, num_segments=num_panels)


@filter_jit
def _class_membership_probs(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_panels: int,
) -> Float64[Array, "panels classes"]:
    """Compute class-membership probabilities from fractional-logit coefficients."""
    if dems is not None:
        V = thetas[None, 0] + dems @ thetas[1:]
    else:
        V = jnp.repeat(thetas, num_panels, axis=0)

    V_ref = jnp.zeros((num_panels, 1))
    return softmax(jnp.concatenate([V_ref, V], axis=1), axis=1)
