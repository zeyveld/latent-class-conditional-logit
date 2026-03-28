"""Matrix algebra operations for case data (as against demographic data)"""

import jax.numpy as jnp
from equinox import filter_jit
from jax.nn import softplus
from jax.ops import segment_sum
from jaxtyping import Array, Float64

from lcl._struct import Data, DiffUnchosenChosen


@filter_jit
def _loglik_gradient(
    structural_betas: Float64[Array, "alt_vars"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    weights: Float64[Array, "cases"],
) -> tuple[
    tuple[Float64[Array, ""], Float64[Array, "cases alt_vars"]],
    Float64[Array, "alt_vars"],
]:
    """Compute log-likelihood and (optionally) gradient and hessian.

    Parameters
    ----------
    betas
        Vector of coefficients on alternative-specific variables.
    weights
        Vector of importance weights for each case in computing the likelihood,
        gradient, and hessian.
    case_data
        Container for case-specific data. Must include the following:
            * X: Matrix of characteristics for each alternative.
            *

    Returns
    -------
    grad_n
        Vector of cases' contributions to the gradient. Used for conditional
        logit standar errors.
    """
    # Compute representative utility and choice probabilities
    Vd: Float64[Array, "unchosen_alts_by_case"] = jnp.clip(
        diff_unchosen_chosen.X.dot(structural_betas), a_max=700.0
    )
    eVd = jnp.exp(Vd)
    probs: Float64[Array, "cases"] = 1 / (
        1
        + segment_sum(
            eVd, diff_unchosen_chosen.cases, num_segments=diff_unchosen_chosen.num_cases
        )
    )

    # Log likelihood
    loglik_by_case = jnp.log(probs) * weights
    neg_loglik = -jnp.sum(loglik_by_case)

    # Calculate cases' contribution to the gradient
    grad_n: Float64[Array, "cases alt_vars"] = -segment_sum(
        diff_unchosen_chosen.X * eVd[:, None],
        diff_unchosen_chosen.cases,
        num_segments=diff_unchosen_chosen.num_cases,
    )
    grad_n = grad_n * probs[:, None] * weights[:, None]
    grad = jnp.sum(grad_n, axis=0)
    return (neg_loglik, grad_n), -grad


def _to_structural_betas(
    latent_betas: Float64[Array, "..."], numeraire_idx: int | None
) -> Float64[Array, "..."]:
    """Transform latent optimization parameters into structural model parameters."""
    if numeraire_idx is not None:
        transformed_col = softplus(latent_betas[numeraire_idx]) + 1e-5
        return latent_betas.at[numeraire_idx].set(transformed_col)
    return latent_betas


def _diff_unchosen_chosen(case_data: Data) -> DiffUnchosenChosen:
    """Compute differences between unchosen and chosen alternatives.

    Parameters
    ----------

    Returns
    -------

    """
    assert isinstance(case_data.y, Array)
    _, num_unchosen_per_id = jnp.unique(
        case_data.cases[~case_data.y], return_counts=True
    )
    if case_data.panels is not None:
        panels = case_data.panels[~case_data.y]
    else:
        panels = None

    # Compute difference between unchosen alts' characteristics and those of the
    # chosen alt. That is, X^d_{ij} := X_{ij} - X_{y_i}
    X_d = case_data.X[~case_data.y] - jnp.repeat(
        case_data.X[case_data.y], num_unchosen_per_id, axis=0
    )
    alts_d = case_data.alts[~case_data.y]  # alt ID for unchosen alts only
    cases_d = case_data.cases[~case_data.y]  # case ID for unchosen alts only

    return DiffUnchosenChosen(X_d, alts_d, cases_d, panels, case_data.num_cases)


# EOF
