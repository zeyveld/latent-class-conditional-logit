"""Matrix algebra operations for case data (as against demographic data)"""

import jax.numpy as jnp
from equinox import filter_jit
from jax.nn import softplus
from jax.ops import segment_sum
from jaxtyping import Array, Float64

from lcl._kernels import _diff_logit_components
from lcl._struct import Data, DiffUnchosenChosen


@filter_jit
def _loglik_gradient(
    structural_betas: Float64[Array, "alt_vars"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    weights: Float64[Array, "cases"],
) -> tuple[
    tuple[Float64[Array, ""], Float64[Array, "cases alt_vars"]],
    Float64[Array, "alt_vars"],
    Float64[Array, "alt_vars alt_vars"],
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
        * ``neg_loglik``: Scalar negative log-likelihood.
        * ``grad_n``: ``Float64[Array, "cases alt_vars"]`` matrix of case-level score
          contributions used for robust sandwich covariance estimation.
    grad : Float64[Array, "alt_vars"]
        The analytic gradient of the negative log-likelihood with respect to ``structural_betas``.
    hessian : Float64[Array, "alt_vars alt_vars"]
        Observed Hessian of the negative log-likelihood with respect to ``structural_betas``.
    """
    log_probs, p_unchosen = _diff_logit_components(
        diff_unchosen_chosen.X,
        structural_betas,
        diff_unchosen_chosen.cases,
        diff_unchosen_chosen.num_cases,
    )

    # Log likelihood
    loglik_by_case = log_probs * weights
    neg_loglik = -jnp.sum(loglik_by_case)

    x_bar_d = segment_sum(
        diff_unchosen_chosen.X * p_unchosen[:, None],
        diff_unchosen_chosen.cases,
        num_segments=diff_unchosen_chosen.num_cases,
    )  # (cases, alt_vars)

    grad_n = -x_bar_d * weights[:, None]
    grad = jnp.sum(grad_n, axis=0)

    row_weights = p_unchosen * weights[diff_unchosen_chosen.cases]

    second_moment_total = diff_unchosen_chosen.X.T @ (
        diff_unchosen_chosen.X * row_weights[:, None]
    )

    first_moment_total = x_bar_d.T @ (x_bar_d * weights[:, None])

    hessian = second_moment_total - first_moment_total

    return (neg_loglik, grad_n), -grad, hessian


@filter_jit
def _loglik_value(
    structural_betas: Float64[Array, "alt_vars"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    weights: Float64[Array, "cases"],
) -> Float64[Array, ""]:
    """Evaluate the conditional-logit negative log-likelihood.

    Parameters
    ----------
    structural_betas : Float64[Array, "alt_vars"]
        Structural taste parameters used in representative utility.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix, with one row for each unchosen alternative.
    weights : Float64[Array, "cases"]
        Case-level weights for the objective.

    Returns
    -------
    Float64[Array, ""]
        Scalar negative log-likelihood.
    """
    Vd = jnp.minimum(diff_unchosen_chosen.X.dot(structural_betas), 700.0)
    eVd = jnp.exp(Vd)
    probs = 1 / (
        1
        + segment_sum(
            eVd, diff_unchosen_chosen.cases, num_segments=diff_unchosen_chosen.num_cases
        )
    )
    return -jnp.sum(jnp.log(probs) * weights)


def _to_structural_betas(
    latent_betas: Float64[Array, "..."], numeraire_idx: int | None
) -> Float64[Array, "..."]:
    """Transform unconstrained optimization parameters into structural parameters.

    If a numeraire is specified (e.g., price or cost), its parameter is restricted
    to be strictly negative via a softplus transformation.

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
    if case_data.y is None:
        raise ValueError(
            "Choice indicators are required to difference chosen alternatives."
        )
    if not isinstance(case_data.y, Array):
        raise TypeError("case_data.y must be a JAX array.")
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
