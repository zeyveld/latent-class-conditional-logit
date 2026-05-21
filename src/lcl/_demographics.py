import logging

import jax.numpy as jnp
from equinox import filter_jit
from jax.nn import softmax
from jaxtyping import Array, Float64

from lcl._optimize import exact_newton_minimize
from lcl._struct import Data, MleConfig


logger = logging.getLogger(__name__)


def _update_thetas(
    starting_thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
    mle_config: MleConfig | None = None,
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
        starting_thetas, class_probs_by_panel, data, num_classes, mle_config
    )
    if not convergence:
        logger.warning("Demographic regression failed to converge.")

    predicted_class_probs, *_ = _predict_class_membership_probs(updated_thetas, data)

    return updated_thetas, predicted_class_probs


def _perform_frac_response_reg(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
    mle_config: MleConfig | None = None,
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
    if mle_config is None:
        mle_config = MleConfig()
    optim_res = exact_newton_minimize(
        _compute_grouped_data_loglik_value_scaled,
        _compute_grouped_data_loglik_grad_hess_scaled,
        thetas.ravel(),
        class_probs_by_panel,
        data,
        num_classes,
        tol=mle_config.ftol,
        maxiter=mle_config.maxiter,
        damping=1e-8,
    )
    thetas = optim_res.params.reshape(data.num_dem_vars + 1, num_classes - 1)
    return thetas, float(optim_res.error) <= mle_config.ftol


@filter_jit
def _compute_grouped_data_loglik_value(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> Float64[Array, ""]:
    """Compute the fractional-response negative log likelihood."""
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    predicted_class_probs, *_ = _predict_class_membership_probs(thetas, data)

    return -jnp.sum(
        class_probs_by_panel
        * jnp.log(jnp.maximum(predicted_class_probs, 1e-250))
    )


@filter_jit
def _compute_grouped_data_loglik_grad_hess(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> tuple[
    Float64[Array, ""],
    Float64[Array, "theta_len"],
    Float64[Array, "theta_len theta_len"],
]:
    """Compute the fractional-response log likelihood, gradient, and Hessian.

    The class-membership model is a baseline-category multinomial logit. Given
    fractional targets ``w`` and predicted non-baseline probabilities ``p``, the
    negative-loglik gradient contribution for panel ``n`` is
    ``z_n * (sum(w_n) * p_n - w_n)``. The corresponding Hessian block for classes
    ``k`` and ``l`` is ``sum(w_n) * p_nk * (1[k=l] - p_nl) * z_n z_n'``.
    """
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    predicted_class_probs, *_ = _predict_class_membership_probs(thetas, data)
    predicted_nonbaseline = predicted_class_probs[:, 1:]

    neg_loglik = -jnp.sum(
        class_probs_by_panel
        * jnp.log(jnp.maximum(predicted_class_probs, 1e-250))
    )

    dem_design = _demographic_design_matrix(data)
    row_weights = class_probs_by_panel.sum(axis=1)
    grad_n = dem_design[:, :, None] * (
        row_weights[:, None, None] * predicted_nonbaseline[:, None, :]
        - class_probs_by_panel[:, None, 1:]
    )
    grad = grad_n.sum(axis=0).ravel()

    class_cov = predicted_nonbaseline[:, :, None] * (
        jnp.eye(num_classes - 1, dtype=thetas.dtype)[None, :, :]
        - predicted_nonbaseline[:, None, :]
    )
    class_cov = row_weights[:, None, None] * class_cov
    hess = jnp.einsum("ni,nj,nkl->ikjl", dem_design, dem_design, class_cov).reshape(
        thetas.size, thetas.size
    )

    return neg_loglik, grad, hess


@filter_jit
def _compute_grouped_data_loglik_value_scaled(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> Float64[Array, ""]:
    """Compute mean fractional-response negative log likelihood."""
    scale = jnp.maximum(class_probs_by_panel.sum(), 1.0)
    return (
        _compute_grouped_data_loglik_value(
            thetas, class_probs_by_panel, data, num_classes
        )
        / scale
    )


@filter_jit
def _compute_grouped_data_loglik_grad_hess_scaled(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> tuple[
    Float64[Array, ""],
    Float64[Array, "theta_len"],
    Float64[Array, "theta_len theta_len"],
]:
    """Compute mean fractional-response value, gradient, and Hessian."""
    neg_loglik, grad, hess = _compute_grouped_data_loglik_grad_hess(
        thetas, class_probs_by_panel, data, num_classes
    )
    scale = jnp.maximum(class_probs_by_panel.sum(), 1.0)
    return neg_loglik / scale, grad / scale, hess / scale


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
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    predicted_class_probs, *_ = _predict_class_membership_probs(thetas, data)
    predicted_nonbaseline = predicted_class_probs[:, 1:]

    neg_loglik = _compute_grouped_data_loglik_value(
        thetas.ravel(), class_probs_by_panel, data, num_classes
    )
    dem_design = _demographic_design_matrix(data)
    row_weights = class_probs_by_panel.sum(axis=1)
    score_n = dem_design[:, :, None] * (
        class_probs_by_panel[:, None, 1:]
        - row_weights[:, None, None] * predicted_nonbaseline[:, None, :]
    )
    grad_n = score_n.reshape(-1, (data.num_dem_vars + 1) * (num_classes - 1))
    grad = -score_n.sum(axis=0).ravel()

    return (neg_loglik, grad_n), grad


@filter_jit
def _demographic_design_matrix(data: Data) -> Float64[Array, "panels dem_vars_plus_one"]:
    """Return the intercept-augmented demographic design matrix."""
    return jnp.concatenate(
        [jnp.ones((data.dems.shape[0], 1), dtype=data.dems.dtype), data.dems],
        axis=1,
    )


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
    V_full = jnp.concatenate([jnp.zeros((V.shape[0], 1)), V], axis=1)
    predicted_class_probs = softmax(V_full, axis=1)

    exp_latent_class_vars: Float64[Array, "panels classes-1"] = jnp.exp(
        jnp.minimum(V, 700.0)
    )
    sum_exp_latent_class_vars: Float64[Array, "panels"] = (
        1.0 + exp_latent_class_vars.sum(axis=1)
    )

    return predicted_class_probs, exp_latent_class_vars, sum_exp_latent_class_vars


# EOF
