import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import Array, Float64

from lcl._optimize import _minimize
from lcl._struct import Data


def _update_thetas(
    starting_thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> tuple[
    Float64[Array, "dem_vars_plus_one classes_minus_one"],
    Float64[Array, "panels classes"],
]:
    """Update class shares based on conditional choice probabilities and demographics.

    Parameters
    ----------
    starting_thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _grouped_data_loglik_fn : fun
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.
    _class_membership_probs_fn : fun
        Function that computes class membership probabilities given class membership
        coefficients and explanatory variables.

    Returns
    -------
    updated_thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    predicted_class_probs : array
        (Np, C) matrix of predicted class membership probabilities.

    """
    updated_thetas, convergence = _perform_frac_response_reg(
        starting_thetas, class_probs_by_panel, data, num_classes
    )
    if not convergence:
        print("Oops! Demographic regression failed to converge.")

    predicted_class_probs, *_ = _predict_class_membership_probs(updated_thetas, data)

    return updated_thetas, predicted_class_probs


def _perform_frac_response_reg(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> tuple[Float64[Array, "dem_vars_plus_one classes_minus_one"], bool]:
    """Perform fractional response regression of class membership probabilities.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _grouped_data_loglik_fn : fun
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.

    Returns
    -------
    updated_shares : array
        (C,) vector of class shares.

    """
    fargs = (class_probs_by_panel, data, num_classes)
    optim_res = _minimize(
        _compute_grouped_data_loglik_and_grad, thetas.ravel(), args=fargs
    )
    thetas = optim_res.params.reshape(data.num_dem_vars + 1, num_classes - 1)
    return thetas, optim_res.success


@filter_jit
def _compute_grouped_data_loglik_and_grad(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> tuple[
    tuple[Float64[Array, ""], Float64[Array, "panels theta_len"]],
    Float64[Array, "theta_len"],
]:
    """Compute grouped-data log likelihood.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _class_membership_probs_fn : fun, optional
        Function that computes predicted probabilities of class membership
        based on demographics.
    num_dem_vars : int
        Number of demographic variables.
    num_classes: int
        Number of latent classes.

    Returns
    -------
    neg_grouped_data_loglik : float
        Grouped-data log likelihood given current coefficients.
    grad : array, optional
        (D * C,) vector representing the gradient, which is actually (D, C).
    grad_n: array, optional
        (Np, D * C) matrix representing individual panels' contributions to
        the gradient. (Conceptualize as [Np, D, C].)

    """
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)  # (D + 1, C - 1)
    predicted_class_probs, exp_latent_class_vars, sum_exp_latent_class_vars = (
        _predict_class_membership_probs(thetas, data)
    )

    neg_loglik = -jnp.sum(class_probs_by_panel * jnp.log(predicted_class_probs))

    probs_times_quotient = (
        class_probs_by_panel[:, 1:] * sum_exp_latent_class_vars[:, None]
        - exp_latent_class_vars
    ) / sum_exp_latent_class_vars[:, None]  # (Np, C - 1)

    grad_n = jnp.concat(
        [
            probs_times_quotient[:, None, :],  # (Np, C - 1)
            probs_times_quotient[:, None, :] * data.dems[..., None],  # (Np, D, C - 1)
        ],
        axis=1,
    )  # (Np, D + 1, C - 1)

    grad = grad_n.sum(axis=0)  # (D, C - 1)

    grad_n = grad_n.reshape(
        -1, (data.num_dem_vars + 1) * (num_classes - 1)
    )  # (Np, (D + 1) * (C - 1))
    return (neg_loglik, grad_n), -grad.ravel()


@filter_jit
def _predict_class_membership_probs(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"], data: Data
) -> tuple[
    Float64[Array, "panels classes"],
    Float64[Array, "panels classes_minus_one"],
    Float64[Array, "panels"],
]:
    """Compute predicted probabilities of class membership based on demographics.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    dems : array
        (Np, D) matrix of demographic variables.
    num_dem_vars : int
        Number of demographic variables.
    num_classes: int
        Number of latent classes.
    return_grad_components : bool, default=False
        Return

    Returns
    -------
    predicted_class_probs : array
        (Np, C) matrix of conditional probabilities of class membership for
        each panel.
    exp_latent_class_vars : array, optional
        (Np, C - 1) matrix of exponentiated latent variables
    sum_exp_latent_class_vars : array, optional
        (Np,) vector of sum of exponentiated latent variables

    """
    V = thetas[None, 0] + data.dems @ thetas[1:]
    exp_latent_class_vars: Float64[Array, "panels classes-1"] = jnp.exp(
        jnp.clip(V, a_max=700.0)
    )
    sum_exp_latent_class_vars: Float64[Array, "panels"] = (
        1.0 + exp_latent_class_vars.sum(axis=1)
    )

    probs_identified_classes: Float64[Array, "panels classes-1"] = (
        exp_latent_class_vars / sum_exp_latent_class_vars[:, None]
    )  # (Np, C - 1)

    predicted_class_probs = jnp.concat(
        [
            (1.0 / sum_exp_latent_class_vars)[:, None],  # (Np, 1)
            probs_identified_classes,  # (Np, C - 1)
        ],
        axis=1,
    )  # (Np, C)

    return predicted_class_probs, exp_latent_class_vars, sum_exp_latent_class_vars


# EOF
