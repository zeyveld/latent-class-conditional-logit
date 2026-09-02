"""Expectation-maximization algorithm."""

import math

import jax
import jax.numpy as jnp
import numpy as onp
from equinox import combine, filter_jit, is_array, partition
from jax import lax
from jax.nn import softmax
from jaxtyping import Array, Float64

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    pullback_negative_derivatives,
)
from lcl._case_utils import _loglik_gradient, _loglik_value, _to_structural_betas
from lcl._demographics import _predict_class_membership_probs, _update_thetas
from lcl._jax_compat import Mesh, NamedSharding, P, shard_map
from lcl._kernels import _diff_log_kernels
from lcl._optimize import exact_newton_minimize
from lcl._struct import (
    Data,
    DiffUnchosenChosen,
    EMVars,
    FitOptions,
    OptimizationOptions,
)


def _em_step(
    em_vars: EMVars,
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    optimization_options: OptimizationOptions,
    fit_options: FitOptions,
    numeraire_idx: int | None = None,
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
) -> EMVars:
    """Execute a single step of the Expectation-Maximization (EM) algorithm.

    The step updates posterior class membership, class-specific taste
    coefficients, and then aggregate shares or demographic coefficients.

    Parameters
    ----------
    em_vars : :class:`~lcl._struct.EMVars`
        Container holding the current state of betas, thetas, and class shares.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.
    num_classes : int
        Number of latent classes.
    optimization_options : :class:`~lcl._struct.OptimizationOptions`
        Optimization settings for the exact-Newton MLE solver.
    fit_options : :class:`~lcl._struct.FitOptions`
        EM settings, including device count and reproducible partition seed.
    numeraire_idx : int | None, optional
        Column index of the numeraire variable.
    numeraire_min_abs : float, default=1e-5
        Minimum absolute value imposed on the numeraire coefficient.

    Returns
    -------
    :class:`~lcl._struct.EMVars`
        Updated parameter state following the complete EM recursion.
    """
    if em_vars.structural_betas is None:
        raise ValueError("Structural betas are required before running an EM step.")
    if em_vars.latent_betas is None:
        raise ValueError("Latent betas are required before running an EM step.")
    if em_vars.shares is None:
        raise ValueError("Class shares are required before running an EM step.")

    structural_betas = em_vars.structural_betas
    latent_betas = em_vars.latent_betas
    shares = em_vars.shares

    # Update posterior class-membership probabilities from choices and demographics.
    updated_class_probs_by_panel, updated_class_probs_by_choice = (
        _compute_conditional_class_probs(
            structural_betas,
            em_vars.thetas,
            shares,
            diff_unchosen_chosen,
            data,
        )
    )

    # Update class-specific taste coefficients using posterior case weights.
    updated_latent_betas = _update_betas(
        latent_betas,
        updated_class_probs_by_choice,
        diff_unchosen_chosen,
        optimization_options,
        fit_options,
        numeraire_idx,
        numeraire_min_abs,
    )

    # Without demographics, update the aggregate class-share vector directly.
    if data.dems is None:
        updated_shares = (
            updated_class_probs_by_panel.sum(axis=0)
            / updated_class_probs_by_panel.sum()
        )  # (C,)
        if data.num_panels is None:
            raise ValueError("Panel identifiers are required for latent-class models.")
        unconditional_class_probs_by_panel = jnp.repeat(
            updated_shares[None, :], repeats=data.num_panels, axis=0
        )  # (Np, C)
        updated_thetas = None  # Not applicable

    # With demographics, update the class-membership regression.
    else:
        # Initialize class membership model coefficients if not provided
        if em_vars.thetas is None:
            em_vars = em_vars._replace(
                thetas=jnp.zeros(((data.num_dem_vars + 1), (num_classes - 1)))
            )
        if em_vars.thetas is None:
            raise ValueError("Class-membership parameters could not be initialized.")

        # Update coefficients and recover unconditional class membership probabilities
        updated_thetas, unconditional_class_probs_by_panel = _update_thetas(
            em_vars.thetas,
            updated_class_probs_by_panel,
            data,
            num_classes,
            optimization_options,
        )
        updated_shares = unconditional_class_probs_by_panel.mean(axis=0)

    # Evaluate the observed-data likelihood at the completed EM update.
    updated_structural_betas = _to_structural_betas(
        updated_latent_betas, numeraire_idx, numeraire_min_abs
    )
    unconditional_loglik = _compute_unconditional_loglik(
        updated_structural_betas,
        unconditional_class_probs_by_panel,
        diff_unchosen_chosen,
        data,
    )

    return EMVars(
        latent_betas=updated_latent_betas,
        structural_betas=updated_structural_betas,
        thetas=updated_thetas,
        shares=updated_shares,
        unconditional_loglik=unconditional_loglik,
        class_probs_by_panel=updated_class_probs_by_panel,
    )


def _compute_conditional_class_probs(
    structural_betas: Float64[Array, "alt_vars classes"],
    thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"] | None,
    shares: Float64[Array, "classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> tuple[Float64[Array, "panels classes"], Float64[Array, "cases classes"]]:
    """Compute posterior probabilities of class membership for each decision-maker.

    Uses Bayesian updating to weight the unconditional prior probabilities (either
    fixed shares or demographic predictions) by the likelihood of observing the
    decision-maker's actual choice sequence.

    Parameters
    ----------
    structural_betas : Float64[Array, "alt_vars classes"]
        Taste parameters for each latent class.
    thetas : Float64[Array, "dem_vars_plus_one classes_minus_one"] | None
        Coefficients for the fractional response regression on demographics.
    shares : Float64[Array, "classes"]
        Aggregate unconditional class shares.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.

    Returns
    -------
    updated_class_probs_by_panel : Float64[Array, "panels classes"]
        Matrix of posterior class probabilities assigned to each decision-maker.
    updated_class_probs_by_choice : Float64[Array, "cases classes"]
        Matrix of posterior class probabilities expanded to each choice situation.
    """
    if thetas is None:
        log_class_probs = jnp.log(jnp.maximum(shares, 1e-300))[None, :]

    else:
        class_probs_given_dems = _predict_class_membership_probs(thetas, data)
        log_class_probs = jnp.log(jnp.maximum(class_probs_given_dems, 1e-300))

    log_kernels = _compute_log_kernels(structural_betas, diff_unchosen_chosen, data)
    conditional_class_probs = softmax(log_class_probs + log_kernels, axis=1)

    if data.num_cases_per_panel is None:
        raise ValueError("Panel identifiers are required for latent-class models.")

    return conditional_class_probs, jnp.repeat(
        conditional_class_probs,
        data.num_cases_per_panel,
        axis=0,
        total_repeat_length=data.num_cases,
    )


def _update_betas(
    betas: Float64[Array, "alt_vars classes"],
    class_probs_by_choice: Float64[Array, "cases classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    optimization_options: OptimizationOptions,
    fit_options: FitOptions,
    numeraire_idx: int | None,
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
) -> Float64[Array, "alt_vars classes"]:
    """Optimize taste parameters using strict SPMD multi-GPU parallelism.

    Parameters
    ----------
    betas : Float64[Array, "alt_vars classes"]
        Current unconstrained taste parameters.
    class_probs_by_choice : Float64[Array, "cases classes"]
        Posterior class membership probabilities to act as case weights.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    optimization_options : :class:`~lcl._struct.OptimizationOptions`
        MLE solver configurations.
    fit_options : :class:`~lcl._struct.FitOptions`
        EM settings, including the number of devices.
    numeraire_idx : int | None
        Column index of the numeraire variable.
    numeraire_min_abs : float, default=1e-5
        Minimum absolute value imposed on the numeraire coefficient.

    Returns
    -------
    Float64[Array, "alt_vars classes"]
        Updated taste parameters optimized for the current EM step.
    """
    num_classes = betas.shape[1]
    num_devices = fit_options.num_devices

    # Pad classes so every selected accelerator receives the same workload.
    classes_per_device = math.ceil(num_classes / num_devices)
    padded_num_classes = classes_per_device * num_devices
    pad_size = padded_num_classes - num_classes

    # Pad with zeros if num_classes is not cleanly divisible by num_devices
    if pad_size > 0:
        pad_betas = jnp.zeros((betas.shape[0], pad_size))
        betas_padded = jnp.concatenate([betas, pad_betas], axis=1)

        pad_weights = jnp.zeros((class_probs_by_choice.shape[0], pad_size))
        weights_padded = jnp.concatenate([class_probs_by_choice, pad_weights], axis=1)
    else:
        betas_padded = betas
        weights_padded = class_probs_by_choice

    # Reshape to (devices, classes_per_device, features/cases) for sharding.
    betas_reshaped = betas_padded.T.reshape(num_devices, classes_per_device, -1)
    weights_reshaped = weights_padded.T.reshape(num_devices, classes_per_device, -1)

    devices = onp.asarray(jax.devices()[:num_devices])
    mesh = Mesh(devices, ("class_device",))
    sharding = NamedSharding(mesh, P("class_device", None, None))
    betas_sharded = jax.device_put(betas_reshaped, sharding)
    weights_sharded = jax.device_put(weights_reshaped, sharding)
    dyn_diff, static_diff = partition(diff_unchosen_chosen, is_array)
    diff_specs = jax.tree_util.tree_map(lambda _: P(), dyn_diff)

    with mesh:
        mapped_update = shard_map(
            lambda device_betas, device_weights, dynamic_diff: _distributed_update(
                device_betas,
                device_weights,
                combine(dynamic_diff, static_diff),
                numeraire_idx,
                numeraire_min_abs,
                optimization_options,
            ),
            mesh=mesh,
            in_specs=(
                P("class_device", None, None),
                P("class_device", None, None),
                diff_specs,
            ),
            out_specs=P("class_device", None, None),
            check_vma=False,
        )
        out_betas = mapped_update(betas_sharded, weights_sharded, dyn_diff)

    # Flatten the result back to standard shape and slice off the dummy padding.
    out_betas = out_betas.reshape(padded_num_classes, -1).T
    return out_betas[:, :num_classes]


def _distributed_update(
    device_betas: Float64[Array, "... classes_per_device alt_vars"],
    device_weights: Float64[Array, "... classes_per_device cases"],
    diff: DiffUnchosenChosen,
    numeraire_idx: int | None,
    numeraire_min_abs: float,
    optimization_options: OptimizationOptions,
) -> Float64[Array, "... classes_per_device alt_vars"]:
    """Update class-specific betas on one shard.

    Parameters
    ----------
    device_betas : Float64[Array, "... classes_per_device alt_vars"]
        Current latent beta vectors assigned to this shard. Some JAX execution
        paths include a leading singleton shard axis, which is preserved on return.
    device_weights : Float64[Array, "... classes_per_device cases"]
        Case weights assigned to each class on this shard.
    diff : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix shared by all class updates.
    numeraire_idx : int | None
        Optional column index constrained through the softplus transform.
    numeraire_min_abs : float
        Minimum absolute value imposed on the numeraire coefficient.
    optimization_options : :class:`~lcl._struct.OptimizationOptions`
        Newton optimization settings.

    Returns
    -------
    Float64[Array, "... classes_per_device alt_vars"]
        Optimized latent beta vectors for the shard, with any leading singleton
        shard axis restored.
    """
    has_shard_axis = device_betas.ndim == 3
    if has_shard_axis:
        device_betas = device_betas[0]
        device_weights = device_weights[0]

    dyn_diff, static_diff = partition(diff, is_array)

    def optimize_single_class(
        mapped_inputs: tuple[
            Float64[Array, "alt_vars"],
            Float64[Array, "cases"],
        ],
    ) -> Float64[Array, "alt_vars"]:
        """Optimize the beta vector for one latent class on the current shard."""
        b, w = mapped_inputs

        def _value_fn_closure(
            p: Float64[Array, "alt_vars"],
            d_diff: object,
            w_inner: Float64[Array, "cases"],
        ) -> Float64[Array, ""]:
            """Evaluate the objective using the dynamic/static diff PyTree split."""
            full_diff = combine(d_diff, static_diff)
            p_struct = _to_structural_betas(p, numeraire_idx, numeraire_min_abs)
            scale = jnp.maximum(jnp.sum(w_inner), 1.0)
            return _loglik_value(p_struct, full_diff, w_inner) / scale

        def _loglik_fn_closure(
            p: Float64[Array, "alt_vars"],
            d_diff: object,
            w_inner: Float64[Array, "cases"],
        ) -> tuple[
            Float64[Array, ""],
            Float64[Array, "alt_vars"],
            Float64[Array, "alt_vars alt_vars"],
        ]:
            """Evaluate objective, gradient, and Hessian with numeraire chain rule."""
            full_diff = combine(d_diff, static_diff)
            p_struct = _to_structural_betas(p, numeraire_idx, numeraire_min_abs)

            (val, aux), grad, hessian = _loglik_gradient(p_struct, full_diff, w_inner)

            grad, aux, hessian = pullback_negative_derivatives(
                p, numeraire_idx, grad, aux, hessian, numeraire_min_abs
            )

            scale = jnp.maximum(jnp.sum(w_inner), 1.0)
            return val / scale, grad / scale, hessian / scale

        optim_res = exact_newton_minimize(
            _value_fn_closure,
            _loglik_fn_closure,
            b,
            dyn_diff,
            w,
            maxiter=optimization_options.maxiter,
            tol=optimization_options.gradient_tol,
            damping=optimization_options.hessian_damping,
            max_step_norm=optimization_options.max_step_norm,
            line_search_maxiter=optimization_options.line_search_maxiter,
            accept_any_decrease=optimization_options.accept_any_decrease,
        )
        return optim_res.params

    updated = lax.map(optimize_single_class, (device_betas, device_weights))
    return updated[None, ...] if has_shard_axis else updated


@filter_jit
def _compute_panel_logliks(
    betas: Float64[Array, "alt_vars classes"],
    unconditional_class_probs_by_panel: Float64[Array, "panels classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, "panels"]:
    """Compute the unconditional log-likelihood contribution of each decision-maker.

    Parameters
    ----------
    betas : Float64[Array, "alt_vars classes"]
        Current taste parameters.
    unconditional_class_probs_by_panel : Float64[Array, "panels classes"]
        Prior probabilities of class membership (does not reflect their observed choices).
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.

    Returns
    -------
    Float64[Array, "panels"]
        Vector of log-likelihood contributions per decision-maker.
    """
    log_kernels = _compute_log_kernels(betas, diff_unchosen_chosen, data)
    weighted_log_kernels = (
        jnp.log(jnp.maximum(unconditional_class_probs_by_panel, 1e-300)) + log_kernels
    )
    row_max = jnp.max(weighted_log_kernels, axis=1, keepdims=True)
    return row_max[:, 0] + jnp.log(
        jnp.sum(jnp.exp(weighted_log_kernels - row_max), axis=1)
    )


@filter_jit
def _compute_unconditional_loglik(
    structural_betas: Float64[Array, "alt_vars classes"],
    class_probs_by_panel: Float64[Array, "panels classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, ""]:
    """Aggregate panel log likelihoods to a scalar for convergence checking."""
    panel_logliks = _compute_panel_logliks(
        structural_betas, class_probs_by_panel, diff_unchosen_chosen, data
    )
    return jnp.sum(panel_logliks)


@filter_jit
def _compute_log_kernels(
    betas: Float64[Array, "alt_vars classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, "panels classes"]:
    """Compute panel-level log likelihood kernels by latent class."""
    if data.panels_of_cases is None or data.num_panels is None:
        raise ValueError("Panel identifiers are required for latent-class models.")

    return _diff_log_kernels(
        diff_unchosen_chosen.X,
        betas,
        diff_unchosen_chosen.cases,
        diff_unchosen_chosen.num_cases,
        data.panels_of_cases,
        data.num_panels,
    )
