import logging

import jax.numpy as jnp
from equinox import filter_jit
from jax import lax
from jax.nn import log_softmax, softmax
from jaxtyping import Array, Float64

from lcl._optimize import exact_newton_minimize
from lcl._scheduling import ITERATION_THRESHOLD_BYTES, use_sequential
from lcl._struct import Data, OptimizationOptions

logger = logging.getLogger(__name__)


def _require_demographics(data: Data) -> Float64[Array, "panels dem_vars"]:
    """Return demographics, raising a clear error if the model was fit without them."""
    if data.dems is None:
        raise ValueError("Demographics are required for the class-membership model.")
    return data.dems


def _update_thetas(
    starting_thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
    optimization_options: OptimizationOptions | None = None,
) -> tuple[
    Float64[Array, "dem_vars_plus_one classes_minus_one"],
    Float64[Array, "panels classes"],
]:
    """Update the class-membership regression from posterior class probabilities.

    Parameters
    ----------
    starting_thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Initial coefficients for the baseline-category multinomial logit that maps
        an intercept and demographics to non-baseline latent class logits.
    class_probs_by_panel : Float64[Array, "panels classes"]
        Posterior class probabilities from the E-step, one row per panel.
    data : :class:`~lcl._struct.Data`
        Estimation data containing the panel-level demographic matrix.
    num_classes : int
        Total number of latent classes, including the baseline class.
    optimization_options : :class:`~lcl._struct.OptimizationOptions` | None, optional
        Newton optimizer configuration for the fractional-response M-step.

    Returns
    -------
    updated_thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Optimized class-membership coefficients.
    predicted_class_probs : Float64[Array, "panels classes"]
        Unconditional class probabilities implied by the optimized demographic model.
    """
    updated_thetas, convergence = _perform_frac_response_reg(
        starting_thetas,
        class_probs_by_panel,
        data,
        num_classes,
        optimization_options,
    )
    if not convergence:
        logger.warning("Demographic regression failed to converge.")

    predicted_class_probs = _predict_class_membership_probs(updated_thetas, data)

    return updated_thetas, predicted_class_probs


def _perform_frac_response_reg(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
    optimization_options: OptimizationOptions | None = None,
) -> tuple[Float64[Array, "dem_vars_plus_one classes_minus_one"], bool]:
    """Fit the fractional-response class-membership model.

    The objective is the cross-entropy between posterior class assignments from
    the E-step and demographic multinomial-logit predictions. Coefficients are
    optimized in flattened form for the Newton solver and reshaped before return.

    Parameters
    ----------
    thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Starting class-membership coefficients.
    class_probs_by_panel : Float64[Array, "panels classes"]
        Posterior class probabilities by panel.
    data : :class:`~lcl._struct.Data`
        Estimation data containing demographics and dimensional metadata.
    num_classes : int
        Total number of latent classes, including the baseline class.
    optimization_options : :class:`~lcl._struct.OptimizationOptions` | None, optional
        Optimizer settings. Defaults to :class:`~lcl._struct.OptimizationOptions`.

    Returns
    -------
    updated_thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Optimized class-membership coefficients.
    converged : bool
        Whether the final Newton error is within
        ``optimization_options.gradient_tol``.
    """
    if optimization_options is None:
        optimization_options = OptimizationOptions()
    optim_res = exact_newton_minimize(
        _compute_grouped_data_loglik_value_scaled,
        _compute_grouped_data_loglik_grad_hess_scaled,
        thetas.ravel(),
        class_probs_by_panel,
        data,
        num_classes,
        tol=optimization_options.gradient_tol,
        maxiter=optimization_options.maxiter,
        damping=optimization_options.hessian_damping,
        max_step_norm=optimization_options.max_step_norm,
        line_search_maxiter=optimization_options.line_search_maxiter,
        accept_any_decrease=optimization_options.accept_any_decrease,
    )
    thetas = optim_res.params.reshape(data.num_dem_vars + 1, num_classes - 1)
    return thetas, float(optim_res.error) <= optimization_options.gradient_tol


@filter_jit
def _compute_grouped_data_loglik_value(
    thetas: Float64[Array, "theta_len"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    data: Data,
    num_classes: int,
) -> Float64[Array, ""]:
    """Compute the fractional-response negative log likelihood."""
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    logits = _class_membership_logits(thetas, data)
    return -jnp.sum(class_probs_by_panel * log_softmax(logits, axis=1))


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

    The gradient is contracted as a single matrix product rather than by summing
    a per-panel tensor, and the Hessian follows whichever schedule
    :mod:`~lcl._scheduling` selects for the problem size.
    """
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    logits = _class_membership_logits(thetas, data)
    predicted_class_probs = softmax(logits, axis=1)
    predicted_nonbaseline = predicted_class_probs[:, 1:]

    neg_loglik = -jnp.sum(class_probs_by_panel * log_softmax(logits, axis=1))

    dem_design = _demographic_design_matrix(data)
    row_weights = class_probs_by_panel.sum(axis=1)

    # z_n (W_n p_n - w_n) summed over panels, as one GEMM.  Building the
    # (panels, dems + 1, classes - 1) tensor only to reduce it away is the
    # single largest avoidable allocation in this M-step.
    residual = (
        row_weights[:, None] * predicted_nonbaseline - class_probs_by_panel[:, 1:]
    )  # (panels, classes - 1)
    grad = (dem_design.T @ residual).ravel()

    hess = _class_covariance_gram(
        dem_design, row_weights, predicted_nonbaseline
    ).reshape(thetas.size, thetas.size)

    return neg_loglik, grad, hess


def _class_covariance_gram(
    Z: Float64[Array, "panels dem_vars_plus_one"],
    row_weights: Float64[Array, "panels"],
    probs: Float64[Array, "panels classes_minus_one"],
) -> Float64[
    Array, "dem_vars_plus_one classes_minus_one dem_vars_plus_one classes_minus_one"
]:
    """Contract ``sum_n Z_ni Z_nj W_n p_nk (d_kl - p_nl)`` into an ``(i,k,j,l)`` block.

    A three-operand ``einsum`` contracts pairwise and materializes a
    ``(panels, dem_vars_plus_one, dem_vars_plus_one)`` intermediate, which at
    large panel counts dominates peak memory for the whole M-step.  Above the
    :mod:`~lcl._scheduling` threshold the class pairs are scanned instead, each
    recomputing its scalar coefficient from the ``(panels, classes_minus_one)``
    probabilities and contributing one ``Z' diag(c_kl) Z`` Gram matrix.  Neither
    the dense per-panel covariance nor the ``(panels, dem, dem)`` intermediate
    is then built.  The two orders are mathematically identical.

    Parameters
    ----------
    Z : Float64[Array, "panels dem_vars_plus_one"]
        Intercept-augmented demographic design matrix.
    row_weights : Float64[Array, "panels"]
        Total fractional weight on each panel, ``sum_c w_nc``.
    probs : Float64[Array, "panels classes_minus_one"]
        Predicted class probabilities, non-baseline classes only.

    Returns
    -------
    Array
        Block with axes ordered ``(dem, class, dem, class)`` to match the
        row-major flattening of the coefficient matrix.
    """
    num_panels, num_dem = Z.shape
    num_tail = probs.shape[1]
    eye_tail = jnp.eye(num_tail, dtype=Z.dtype)

    if not use_sequential(
        num_panels, num_dem, num_dem, threshold=ITERATION_THRESHOLD_BYTES
    ):
        class_cov = (
            row_weights[:, None, None]
            * probs[:, :, None]
            * (eye_tail[None, :, :] - probs[:, None, :])
        )
        return jnp.einsum("ni,nj,nkl->ikjl", Z, Z, class_cov)

    pairs = jnp.stack(
        jnp.meshgrid(jnp.arange(num_tail), jnp.arange(num_tail), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)

    def one_block(carry: None, pair: Array) -> tuple[None, Array]:
        """Contract one (k, l) class pair into a (dem, dem) Gram matrix."""
        row, col = pair[0], pair[1]
        delta = jnp.where(row == col, 1.0, 0.0)
        coefficient = row_weights * probs[:, row] * (delta - probs[:, col])
        return carry, (Z * coefficient[:, None]).T @ Z

    _, blocks = lax.scan(one_block, None, pairs)  # (tail * tail, dem, dem)
    # (k, l, i, j) -> (i, k, j, l)
    return jnp.transpose(
        blocks.reshape(num_tail, num_tail, num_dem, num_dem), (2, 0, 3, 1)
    )


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
    """Compute the fractional-response objective and analytic gradient.

    Parameters
    ----------
    thetas : Float64[Array, "theta_len"]
        Flattened coefficient matrix with shape
        ``((data.num_dem_vars + 1) * (num_classes - 1),)``.
    class_probs_by_panel : Float64[Array, "panels classes"]
        Fractional class targets from the E-step.
    data : :class:`~lcl._struct.Data`
        Estimation data containing demographics and dimensional metadata.
    num_classes : int
        Total number of latent classes, including the baseline class.

    Returns
    -------
    objective_and_aux : tuple
        ``(neg_loglik, grad_n)`` where ``grad_n`` stores panel-level score
        contributions in flattened theta order.
    grad : Float64[Array, "theta_len"]
        Analytic gradient of the negative objective.
    """
    thetas = thetas.reshape(data.num_dem_vars + 1, num_classes - 1)
    predicted_class_probs = _predict_class_membership_probs(thetas, data)
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
def _demographic_design_matrix(
    data: Data,
) -> Float64[Array, "panels dem_vars_plus_one"]:
    """Return the intercept-augmented demographic design matrix."""
    dems = _require_demographics(data)
    return jnp.concatenate(
        [jnp.ones((dems.shape[0], 1), dtype=dems.dtype), dems],
        axis=1,
    )


@filter_jit
def _predict_class_membership_probs(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"], data: Data
) -> Float64[Array, "panels classes"]:
    """Compute predicted class-membership probabilities from demographics.

    Parameters
    ----------
    thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Baseline-category multinomial-logit coefficients. The first row is the
        intercept and the remaining rows correspond to ``data.dems`` columns.
    data : :class:`~lcl._struct.Data`
        Estimation data containing the panel-level demographic matrix.

    Returns
    -------
    predicted_class_probs : Float64[Array, "panels classes"]
        Class-membership probabilities for each panel, including the baseline class.
    """
    return softmax(_class_membership_logits(thetas, data), axis=1)


@filter_jit
def _class_membership_logits(
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"], data: Data
) -> Float64[Array, "panels classes"]:
    """Return baseline-category logits, including the zero reference class.

    Parameters
    ----------
    thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"]
        Baseline-category multinomial-logit coefficients.
    data : :class:`~lcl._struct.Data`
        Estimation data containing the panel-level demographic matrix.

    Returns
    -------
    Float64[Array, "panels classes"]
        Class logits with a zero-valued baseline column.
    """
    dems = _require_demographics(data)
    V = thetas[None, 0] + dems @ thetas[1:]
    return jnp.concatenate([jnp.zeros((V.shape[0], 1), dtype=V.dtype), V], axis=1)
