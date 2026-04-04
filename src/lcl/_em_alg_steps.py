"""Expectation-maximization algorithm."""

import jax
import jax.numpy as jnp
import numpy as onp
from equinox import combine, filter_jit, is_array, partition
from jax import Array, lax, shard_map
from jax.nn import sigmoid
from jax.ops import segment_sum
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxopt import BFGS
from jaxtyping import Float64

from lcl._case_utils import _loglik_gradient, _to_structural_betas
from lcl._demographics import _predict_class_membership_probs, _update_thetas
from lcl._struct import Data, DiffUnchosenChosen, EMAlgConfig, EMVars, MleConfig


def _em_alg(
    em_vars: EMVars,
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    mle_config: MleConfig,
    em_alg_config: EMAlgConfig,
    numeraire_idx: int | None = None,
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
        Configuration for the EM algorithm loop (includes hardware sharding settings).
    numeraire_idx : int | None, optional
        Column index of the numeraire variable.

    Returns
    -------
    :class:`~lcl._struct.EMVars`
        Updated parameter state following the complete EM recursion.
    """

    # 1. Compute conditional class membership probabilities given choices and demographics
    updated_class_probs_by_panel, updated_class_probs_by_choice = (
        _compute_conditional_class_probs(
            em_vars.structural_betas,
            em_vars.thetas,
            em_vars.shares,
            diff_unchosen_chosen,
            data,
        )
    )

    # 2. Update classes' respective taste coefficients based on conditional
    # class membership probabilities
    updated_latent_betas = _update_betas(
        em_vars.latent_betas,
        updated_class_probs_by_choice,
        diff_unchosen_chosen,
        mle_config,
        em_alg_config,
        numeraire_idx,
    )

    # 3. Update class membership model coefficients or class share vectors

    # 3.1 If demographics omitted, update class share vectors
    if data.dems is None:
        updated_shares = (
            updated_class_probs_by_panel.sum(axis=0)
            / updated_class_probs_by_panel.sum()
        )  # (C,)
        assert data.num_panels is not None
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
        assert em_vars.thetas is not None

        # Update coefficients and recover unconditional class membership probabilities
        updated_thetas, unconditional_class_probs_by_panel = _update_thetas(
            em_vars.thetas, updated_class_probs_by_panel, data, num_classes
        )
        print(f"updated_thetas : {updated_thetas}")
        updated_shares = unconditional_class_probs_by_panel.mean(axis=0)
        print(f"updated_shares : {updated_shares}")

    # 4. Compute unconditional log likelihood given taste coefficients and class membership
    # coefficients
    updated_structural_betas = _to_structural_betas(updated_latent_betas, numeraire_idx)
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
    kernels = _compute_kernels(structural_betas, diff_unchosen_chosen, data)
    if thetas is None:
        weighted_kernels = kernels * shares[None, :]

    else:
        class_probs_given_dems, *_ = _predict_class_membership_probs(thetas, data)
        weighted_kernels = kernels * class_probs_given_dems  # (Np, C)

    # Remove zero kernels from floating point errors
    assert isinstance(weighted_kernels, Array)
    weighted_kernels_plus_delta = weighted_kernels + 1e-100
    weighted_kernels = (
        weighted_kernels_plus_delta / weighted_kernels_plus_delta.sum(axis=1)[:, None]
    )  # (Np, C)

    conditional_class_probs = (
        weighted_kernels / jnp.sum(weighted_kernels, axis=1)[:, None]
    )  # (Np, C)

    assert data.num_cases_per_panel is not None  # Panels required for this model

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
) -> Float64[Array, "alt_vars classes"]:
    """Optimize the taste parameters for all latent classes simultaneously via `jax.lax.map`.

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
        EM algorithm configurations (supplies device count for hardware sharding).
    numeraire_idx : int | None
        Column index of the numeraire variable.

    Returns
    -------
    Float64[Array, "alt_vars classes"]
        Updated taste parameters optimized for the current EM step.
    """
    # Separate the PyTree into dynamic arrays and static metadata
    dynamic_diff, static_diff = partition(diff_unchosen_chosen, is_array)

    def _loglik_fn_closure(
        p, dyn_diff, w
    ) -> tuple[
        tuple[Float64[Array, "..."], Float64[Array, "..."]], Float64[Array, "..."]
    ]:
        """Impose nonnegativity on numeraire (if provided)"""
        p_struct = _to_structural_betas(p, numeraire_idx)
        full_diff = combine(dyn_diff, static_diff)
        (val, aux), grad = _loglik_gradient(p_struct, full_diff, w)

        if numeraire_idx is not None:
            grad = grad.at[numeraire_idx].multiply(sigmoid(p[numeraire_idx]))

        N_eff = jnp.clip(jnp.sum(w), a_min=1.0)
        return (val / N_eff, aux), grad / N_eff

    solver = BFGS(
        fun=_loglik_fn_closure,
        value_and_grad=True,
        has_aux=True,
        linesearch="zoom",
        max_stepsize=1.0,
        maxiter=mle_config.maxiter,
        tol=mle_config.ftol,
        implicit_diff=False,
        verbose=False,
    )

    def optimize_single_class(mapped_inputs) -> Float64[Array, "..."]:
        beta_vec, weight_vec = mapped_inputs
        params, _ = solver.run(beta_vec, dynamic_diff, weight_vec)
        return params

    num_devices = em_alg_config.num_devices
    num_classes = betas.shape[1]

    # Transpose upfront so the batch dimension (classes) is leading
    betas_T = betas.T
    weights_T = class_probs_by_choice.T

    # Check if we have multiple GPUs AND the classes distribute evenly
    if num_devices > 1 and num_classes % num_devices == 0:
        # 1. Define the hardware mesh
        devices = onp.array(jax.devices())
        mesh = Mesh(devices, ("gpus",))

        # 2. Define the local function that runs on each device's chunk of data
        def optimize_device_chunk(betas_chunk, weights_chunk):
            # lax.map sequentially processes the classes assigned to this specific GPU
            return lax.map(optimize_single_class, (betas_chunk, weights_chunk))

        # 3. Map the chunk function across the mesh
        # P('gpus', None) means: Shard the 0th axis (classes) across 'gpus',
        # and leave the 1st axis (parameters/cases) replicated on device memory.
        sharded_opt = shard_map(
            optimize_device_chunk,
            mesh=mesh,
            in_specs=(P("gpus", None), P("gpus", None)),
            out_specs=P("gpus", None),
        )

        # 4. JIT compile the entire globally sharded operation
        updated_betas_T = jax.jit(sharded_opt)(betas_T, weights_T)

    else:
        # Fallback to standard JIT-compiled loop on a single device
        updated_betas_T = lax.map(optimize_single_class, (betas_T, weights_T))

    return updated_betas_T.T


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
    kernels = _compute_kernels(betas, diff_unchosen_chosen, data)
    weighted_kernels = jnp.einsum(
        "nc,nc->n", unconditional_class_probs_by_panel, kernels
    )
    return jnp.log(jnp.clip(weighted_kernels, a_min=1e-250))


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
    Vd_by_class = jnp.einsum("nk,kc->nc", diff_unchosen_chosen.X, betas)
    eVd_by_class, Vd_by_class = jnp.exp(jnp.clip(Vd_by_class, a_max=700.0)), None

    # Compute chosen alts' conditional choice probabalities by latent class
    probs_by_class = 1 / (
        1
        + segment_sum(
            eVd_by_class,
            diff_unchosen_chosen.cases,
            num_segments=diff_unchosen_chosen.num_cases,
        )
    )  # (N, C)

    assert data.panels_of_cases is not None  # Panels required for this model

    # Use exp(segment_sum(log(probs))) to bypass JAX autodiff limitations
    log_probs_by_class = jnp.log(jnp.clip(probs_by_class, a_min=1e-250))
    sum_log_probs = segment_sum(
        log_probs_by_class, data.panels_of_cases, num_segments=data.num_panels
    )
    return jnp.exp(sum_log_probs)


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
    eV = jnp.exp(jnp.clip((data.X.dot(latent_betas.T)), a_max=700.0))
    probs = eV / segment_sum(eV, data.cases, num_segments=data.num_cases)[data.cases]
    return probs, eV


# EOF
