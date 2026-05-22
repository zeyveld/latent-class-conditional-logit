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
    """Compute logit probabilities and inclusive values with a stable case shift.

    Parameters
    ----------
    X : Float64[Array, "rows alt_vars"]
        Long-format design matrix.
    betas : Float64[Array, "alt_vars classes"]
        Taste parameters, one column per latent class.
    cases : UInt[Array, "rows"]
        Contiguous zero-indexed choice-situation IDs.
    num_cases : int
        Number of choice situations.

    Returns
    -------
    probs : Float64[Array, "rows classes"]
        Choice probabilities for each alternative row and class.
    logsum : Float64[Array, "cases classes"]
        Inclusive values for each case and class.
    """
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
    """Compute chosen log probabilities from differenced alternative features.

    Parameters
    ----------
    X_diff : Float64[Array, "unchosen_rows alt_vars"]
        Differences ``X_ij - X_iy`` for unchosen alternatives only.
    betas : Float64[Array, "alt_vars"]
        Taste parameters for one model or latent class.
    cases : UInt[Array, "unchosen_rows"]
        Choice-situation IDs for each unchosen row.
    num_cases : int
        Number of choice situations.

    Returns
    -------
    log_chosen_probs : Float64[Array, "cases"]
        Log probability of the observed chosen alternative in each case.
    p_unchosen : Float64[Array, "unchosen_rows"]
        Conditional probabilities assigned to the unchosen alternatives.
    """
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
    """Compute panel-level log-likelihood kernels by latent class.

    Parameters
    ----------
    X_diff : Float64[Array, "unchosen_rows alt_vars"]
        Differences ``X_ij - X_iy`` for unchosen alternatives only.
    betas : Float64[Array, "alt_vars classes"]
        Taste parameters for each latent class.
    diff_cases : UInt[Array, "unchosen_rows"]
        Choice-situation IDs for each differenced row.
    num_cases : int
        Number of choice situations.
    panels_of_cases : UInt[Array, "cases"]
        Panel ID associated with each choice situation.
    num_panels : int
        Number of panels.

    Returns
    -------
    Float64[Array, "panels classes"]
        Log likelihood of each panel's observed choice sequence by class.
    """
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
    """Compute class-membership probabilities from fractional-logit coefficients.

    Parameters
    ----------
    thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Class-membership coefficients. If ``dems`` is None, this is treated as
        already containing non-baseline logits to repeat across panels.
    dems : Float64[Array, "panels dem_vars"] | None
        Panel-level demographic matrix.
    num_panels : int
        Number of panels to score.

    Returns
    -------
    Float64[Array, "panels classes"]
        Class-membership probabilities, including the baseline class.
    """
    if dems is not None:
        V = thetas[None, 0] + dems @ thetas[1:]
    else:
        V = jnp.repeat(thetas, num_panels, axis=0)

    V_ref = jnp.zeros((num_panels, 1))
    return softmax(jnp.concatenate([V_ref, V], axis=1), axis=1)
