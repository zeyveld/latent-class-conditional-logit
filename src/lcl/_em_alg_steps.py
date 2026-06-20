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
from lcl._struct import Data, DiffUnchosenChosen, EMAlgConfig, EMVars, MleConfig


def _em_alg(
    em_vars: EMVars,
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    mle_config: MleConfig,
    em_alg_config: EMAlgConfig,
    numeraire_idx: int | None = None,
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
) -> EMVars:
    """Execute a single step of the Expectation-Maximization (EM) algorithm.

    The step proceeds iteratively:
    1. (E-Step) Update conditional class membership probabilities for each decision-maker.
    2. (M-Step 1) Update taste coefficients for each latent class via MLE.
    3. (M-Step 2) Update aggregate class shares or demographic regression coefficients.

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
    mle_config : :class:`~lcl._struct.MleConfig`
        Optimization settings for the MLE solver (L-BFGS).
    em_alg_config : :class:`~lcl._struct.EMAlgConfig`
        Configuration containing the JAX PRNG seed for reproducible partitioning.
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

    # 1. Compute conditional class membership probabilities given choices and demographics
    updated_class_probs_by_panel, updated_class_probs_by_choice = (
        _compute_conditional_class_probs(
            structural_betas,
            em_vars.thetas,
            shares,
            diff_unchosen_chosen,
            data,
        )
    )

    # 2. Update classes' respective taste coefficients based on conditional
    # class membership probabilities
    updated_latent_betas = _update_betas(
        latent_betas,
        updated_class_probs_by_choice,
        diff_unchosen_chosen,
        mle_config,
        em_alg_config,
        numeraire_idx,
        numeraire_min_abs,
    )

    # 3. Update class membership model coefficients or class share vectors

    # 3.1 If demographics omitted, update class share vectors
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

    # 3.2 Otherwise, update class membership model coefficients
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
            em_vars.thetas, updated_class_probs_by_panel, data, num_classes, mle_config
        )
        updated_shares = unconditional_class_probs_by_panel.mean(axis=0)

    # 4. Compute unconditional log likelihood given taste coefficients and class membership
    # coefficients
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
        class_probs_given_dems, *_ = _predict_class_membership_probs(thetas, data)
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
    mle_config: MleConfig,
    em_alg_config: EMAlgConfig,
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
    mle_config : :class:`~lcl._struct.MleConfig`
        MLE solver configurations.
    em_alg_config : :class:`~lcl._struct.EMAlgConfig`
        Configuration containing the JAX PRNG seed for reproducible partitioning.
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
    num_devices = em_alg_config.num_devices

    # 1. Padding to ensure perfectly balanced workload across GPUs
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

    # 2. Reshape for sharded execution: (devices, classes_per_device, features/cases)
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
                mle_config,
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
    mle_config: MleConfig,
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
    mle_config : :class:`~lcl._struct.MleConfig`
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
            return _loglik_value(p_struct, full_diff, w_inner)

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
                p, numeraire_idx, grad, aux, hessian
            )

            return val, grad, hessian

        optim_res = exact_newton_minimize(
            _value_fn_closure,
            _loglik_fn_closure,
            b,
            dyn_diff,
            w,
            maxiter=mle_config.maxiter,
            tol=mle_config.ftol,
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
    """Wrapper to aggregate panel log-likelihoods to a scalar for convergence checking."""

    # Call the array function and sum it to a scalar
    panel_logliks = _compute_panel_logliks(
        structural_betas, class_probs_by_panel, diff_unchosen_chosen, data
    )
    return jnp.sum(panel_logliks)


@filter_jit
def _compute_kernels(
    betas: Float64[Array, "alt_vars classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, "panels classes"]:
    """Compute the probability of observing a decision-maker's entire sequence of choices.

    Parameters
    ----------
    betas : Float64[Array, "alt_vars classes"]
        Taste parameters for each latent class.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.

    Returns
    -------
    Float64[Array, "panels classes"]
        Matrix mapping the joint probability of each decision-maker's sequence
        conditional on membership in each latent class.
    """
    return jnp.exp(_compute_log_kernels(betas, diff_unchosen_chosen, data))


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


def _compute_probs_and_exp_utility(
    latent_betas: Float64[Array, "alt_vars"], data: Data
) -> tuple[Float64[Array, "alts_by_case"], Float64[Array, "alts_by_case"]]:
    """Compute conditional choice probabilities for an individual latent class.

    Parameters
    ----------
    latent_betas : Float64[Array, "alt_vars"]
        Vector of taste parameters for a specific latent class.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.

    Returns
    -------
    probs : Float64[Array, "alts_by_case"]
        Vector of conditional choice probabilities across all alternatives.
    eV : Float64[Array, "alts_by_case"]
        Exponentiated representative utility across all alternatives.
    """
    from lcl._kernels import _choice_probabilities_and_logsum

    probs, logsum = _choice_probabilities_and_logsum(
        data.X, latent_betas[:, None], data.cases, data.num_cases
    )
    eV = jnp.exp(data.X.dot(latent_betas.T) - logsum[data.cases, 0])
    probs = probs[:, 0]
    return probs, eV
