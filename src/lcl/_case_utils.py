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
    """Compute the log-likelihood and analytic gradient for a conditional logit specification.

    Parameters
    ----------
    structural_betas : Float64[Array, "alt_vars"]
        Vector of structural taste parameters corresponding to alternative characteristics.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Struct containing the differenced design matrix :math:`X_{ij} - X_{iy_i}`.
    weights : Float64[Array, "cases"]
        Vector of importance weights (or class-assignment probabilities) for each
        choice situation.

    Returns
    -------
    objective_and_aux : tuple
        A tuple containing:
        * ``neg_loglik``: The scalar negative log-likelihood.
        * ``grad_n``: A ``Float64[Array, "cases alt_vars"]`` matrix of case-level score
          contributions utilized for robust sandwich covariance estimation.
    grad : Float64[Array, "alt_vars"]
        The analytic gradient of the negative log-likelihood with respect to ``structural_betas``.
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
    """Transform unconstrained optimization parameters into structural parameters.

    If a numeraire is specified (e.g., price or cost), its parameter is restricted
    to be strictly positive via a softplus transformation.

    Parameters
    ----------
    latent_betas : Float64[Array, "..."]
        Unconstrained parameters managed by the L-BFGS solver.
    numeraire_idx : int | None
        The column index of the numeraire variable, if applicable.

    Returns
    -------
    Float64[Array, "..."]
        Structural parameters suitable for utility calculation.
    """
    if numeraire_idx is not None:
        # Force the parameter to be strictly negative
        transformed_col = -(softplus(latent_betas[numeraire_idx]) + 1e-5)
        return latent_betas.at[numeraire_idx].set(transformed_col)
    return latent_betas


def _diff_unchosen_chosen(case_data: Data) -> DiffUnchosenChosen:
    """Compute the differences between unchosen and chosen alternatives.

    By transforming the design matrix to represent the difference from the chosen
    alternative, we streamline the denominator of the logit probability computation.

    Parameters
    ----------
    case_data : :class:`~lcl._struct.Data`
        The core estimation data container.

    Returns
    -------
    :class:`~lcl._struct.DiffUnchosenChosen`
        Container holding the differenced design matrix and reduced dimensionality IDs.
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
