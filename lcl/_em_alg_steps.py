"""Expectation-maximization algorithm."""

import jax.numpy as jnp
from equinox import combine, filter_jit, is_array, partition
from jax import Array, lax
from jax.nn import sigmoid
from jax.ops import segment_sum
from jaxopt import BFGS
from jaxtyping import Float64

from lcl._case_utils import _loglik_gradient, _to_structural_betas
from lcl._demographics import _predict_class_membership_probs, _update_thetas
from lcl._struct import Data, DiffUnchosenChosen, EMVars, MleConfig


def _em_alg(
    em_vars: EMVars,
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    mle_config: MleConfig,
    numeraire_idx: int | None = None,
) -> EMVars:
    """Run EM algorithm, which proceeds as follows:
        1. Update conditional class membership probabilities associated with each panel
        2. Update class coefficients
        3. Update aggregate class shares

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    shares : array
        (C,) vector of class shares.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    cases : array
        (N,) vector of choice situation cases.
    panels_of_cases : array
        (Nc,) vector of consumer cases, one per choice situation.
    num_alt_vars : int
        Number of explanatory variables
    num_classes: int
        Number of latent classes
    num_cases : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.
    dems : array, optional
        (Np, D) matrix of panel demographic characteristics.
    thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    _grouped_data_loglik_fn : fun, optional
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.
    _class_membership_probs_fn : fun, optional
        Function that computes class membership probabilities given class membership
        coefficients and explanatory variables.

    Returns
    -------
    updated_betas : array
        (K, C) matrix of coefficients associated with each latent class.
    updated_thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes. Only returned if demographic
        variables are provided.
    updated_shares : array, optional
        (C,) vector of class shares.
    _unconditional_loglik : float
        Unconditional log-likelihood of parameters.
    updated_class_probs_by_panel : array
        (Np, C) array of conditional class membership probabilities, given observed
        choices and demographics.

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
    """Update conditional class membership probabilities of all classes.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    cases : array
        (N,) vector of choice situation cases.
    panels_of_cases : array
        (Nc,) vector of consumer cases, one per choice situation.
    num_cases_per_panel : array
        (Np,) vector with each panel's number of observed choices.
    num_cases : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.
    num_classes: int
        Number of latent classes.
    thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    dems : array, optional
        (Np, D) matrix of demographic variables.
    _class_membership_probs_fn : fun, optional
        Function that computes predicted probabilities of class membership
        based on demographics
    shares : array, optional
        (C,) vector of class shares.

    Returns
    -------
    updated_class_probs_by_panel : array
        (Np, C) matrix of conditional class membership probabilities for each panel.
    updated_class_probs_by_choice : array
        (N, C) matrix of conditional class membership probabilities for each choice situation.

    """
    kernels = _compute_kernels(structural_betas, diff_unchosen_chosen, data)
    if thetas is None:
        weighted_kernels = kernels * shares[None, :]

    else:
        class_probs_given_dems, *_ = _predict_class_membership_probs(thetas, data)
        # _class_membership_probs_fn(thetas, dems)  # (Np, C)
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
    numeraire_idx: int | None,
) -> Float64[Array, "alt_vars classes"]:
    """Update the coefficients of each class (contained in a matrix)

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    class_probs_by_choice : array
        (Np, C) matrix of conditional class membership probabilities for each panel.
    cases : array
        (N,) vector of choice situation cases.
    num_alt_vars : int
        Number of explanatory variables.
    num_classes: int
        Number of latent classes.
    num_cases : int
        Number of choices observed.

    Returns
    -------
    updated_betas : array
        (K, C) matrix of coefficients associated with each latent class.

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

        # Combine the partitioned diff struct
        full_diff = combine(dyn_diff, static_diff)

        # Pass combined struct and weights to the gradient function
        (val, aux), grad = _loglik_gradient(p_struct, full_diff, w)

        # Apply chain rule (sigmoid is derivative of softplus)
        if numeraire_idx is not None:
            grad = grad.at[numeraire_idx].multiply(sigmoid(p[numeraire_idx]))

        # Normalize to prevent softplus step explosions
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
        # dynamic_diff feeds into `dyn_diff`, weight_vec feeds into `w`
        params, _ = solver.run(beta_vec, dynamic_diff, weight_vec)
        return params

    mapped_inputs = (betas.T, class_probs_by_choice.T)
    updated_betas_T = lax.map(optimize_single_class, mapped_inputs)

    return updated_betas_T.T


@filter_jit
def _compute_panel_logliks(
    betas: Float64[Array, "alt_vars classes"],
    unconditional_class_probs_by_panel: Float64[Array, "panels classes"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, "panels"]:
    """Compute unconditional log-likelihood for each panel.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    unconditional_class_probs_by_panel : array
        (Np, C) matrix of unconditional probabilities that each panel belongs to each
        class. May depend on panels_of_cases' demographic characteristics, but does NOT directly
        reflect their respective choices.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    cases : array
        (N,) vector of choice situation cases.
    panels_of_cases : array
        (Nc,) vector of consumer cases, one per choice situation.
    num_cases : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.

    Returns
    -------
    panel_logliks : array
        Vector of panels' log-likelihood contributions.

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
    """Wrapper to compute scalar log-likelihood for EM alg convergence check."""

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
    """Compute conditional probabilities of observed choice sequences.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    cases : array
        (N,) vector of choice situation cases.
    panels_of_cases : array
        (Nc,) vector of consumer cases, one per choice situation.
    num_cases : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.

    Returns
    -------
    kernels_by_class : array
        (Np, C) matrix of logit kernels assocaited with each latent class.

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
    """Compute conditional choice probabilities for an individual class.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients (which are transposed from their usual shape!)
    cases : array
        (N,) vector of choice situation cases.
    num_cases : int
        Number of choices observed.

    Returns
    -------
    probs : array
        (N * J,) vector of conditional choice probabilities.
    eV : array
        Exponentiated representative utility.

    """
    eV = jnp.exp(jnp.clip((data.X.dot(latent_betas.T)), a_max=700.0))
    probs = eV / segment_sum(eV, data.cases, num_segments=data.num_cases)[data.cases]
    return probs, eV


# EOF
