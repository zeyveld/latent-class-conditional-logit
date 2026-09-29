"""Expectation-Maximization (EM) algorithm initialization routines."""

import jax.numpy as jnp
import numpy as onp
from equinox import filter_jit
from jax.nn import softmax
from jaxtyping import Array, Float64

from lcl.constraints import NegativeCoefficientBound
from lcl._kernels import _class_membership_log_probs
from lcl._em_alg_steps import (
    _compute_em_log_kernels,
    _update_betas,
    class_mesh_sharding,
    _posterior_and_loglik,
)
from lcl._jax_compat import device_put_array_leaves
from lcl.options import FitOptions, OptimizationOptions
from lcl._struct import Data, DiffUnchosenChosen, EMVars


@filter_jit
def _fit_starting_betas(
    weights: Float64[Array, "panels classes"],
    diff: DiffUnchosenChosen,
    data: Data,
    optimization_options: OptimizationOptions,
    num_devices: int,
    negative_bound: NegativeCoefficientBound,
) -> Float64[Array, "alt_vars classes"]:
    """Fit random panel partitions on one fixed full-data shape.

    Zero weights remove the other partitions from each class's objective,
    gradient, and Hessian. Each solve starts at zero and is normalized by its
    own number of cases, just as a physically sliced conditional-logit fit is.
    """
    initial = jnp.zeros((diff.X.shape[1], weights.shape[1]))
    betas, _ = _update_betas(
        initial,
        weights,
        diff,
        optimization_options,
        num_devices,
        negative_bound,
        panels_of_cases=data.panels_of_cases,
    )
    return betas


def _get_starting_vals(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    fit_options: FitOptions,
    optimization_options: OptimizationOptions,
    negative_bound: NegativeCoefficientBound = NegativeCoefficientBound(),
) -> EMVars:
    """Generate robust initial parameter estimates to seed the EM algorithm.

    Because the EM objective function is highly non-convex for latent class models,
    careful initialization is required to avoid local optima. This function randomly
    partitions decision-makers into `num_classes` subsets and estimates a standard
    conditional logit objective for each subset using full-data 0/1 panel weights.

    Parameters
    ----------
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        The differenced design matrix for the full sample.
    data : :class:`~lcl._struct.Data`
        The core estimation data and metadata.
    num_classes : int
        The number of latent classes to initialize.
    fit_options : :class:`~lcl.options.FitOptions`
        EM settings containing the reproducible partition seed.
    optimization_options : :class:`~lcl.options.OptimizationOptions`
        Optimization settings for the subset-level Newton routines.
    negative_bound : NegativeCoefficientBound
        Resolved negative coefficient constraint, or an unconstrained record.

    Returns
    -------
    :class:`~lcl._struct.EMVars`
        Container holding the initialized taste parameters, starting shares, the
        membership coefficients that reproduce them, the observed-data log
        likelihood at those values, and first-pass posterior class probabilities.
    """
    if data.num_panels is None:
        raise ValueError("Panel identifiers are required for latent-class models.")
    weights = _random_class_weights(data.num_panels, num_classes, fit_options.seed)
    weights = device_put_array_leaves(
        weights, class_mesh_sharding(fit_options.num_devices)
    )
    betas = _fit_starting_betas(
        weights,
        diff_unchosen_chosen,
        data,
        optimization_options,
        fit_options.num_devices,
        negative_bound,
    )

    log_kernels = _compute_em_log_kernels(betas, diff_unchosen_chosen, data)
    starting_class_probs_by_panel = softmax(log_kernels, axis=1)
    starting_shares = jnp.mean(starting_class_probs_by_panel, axis=0)

    # Seed the membership model at the intercepts that reproduce the starting
    # shares exactly.  Starting it at zeros would instead impose a uniform prior,
    # so the first membership M-step would not be warm started at the prior its
    # own E-step used -- the one place the generalized-EM ascent guarantee could
    # otherwise slip.  Initializing here also keeps the EM state's PyTree
    # structure fixed across iterations, so the compiled step is traced once.
    if data.dems is None:
        thetas = None
    else:
        clipped_shares = jnp.clip(starting_shares, 1e-10)
        normalized_shares = clipped_shares / clipped_shares.sum()
        intercepts = jnp.log(normalized_shares[1:] / normalized_shares[0])
        thetas = (
            jnp.zeros((data.num_dem_vars + 1, num_classes - 1)).at[0, :].set(intercepts)
        )

    if thetas is None:
        log_prior = jnp.broadcast_to(
            jnp.log(jnp.maximum(starting_shares, 1e-300)),
            (data.num_panels, num_classes),
        )
    else:
        log_prior = _class_membership_log_probs(thetas, data.dems, data.num_panels)
    starting_class_probs_by_panel, starting_loglik = _posterior_and_loglik(
        log_kernels, log_prior, prior_is_log=True
    )

    return EMVars(
        betas=betas,
        thetas=thetas,
        shares=starting_shares,
        unconditional_loglik=starting_loglik,
        class_probs_by_panel=starting_class_probs_by_panel,
    )


def _random_class_weights(
    num_panels: int, num_classes: int, seed: int
) -> Float64[Array, "panels classes"]:
    """Assign each panel to one class, preserving the seeded partition order."""
    if num_classes > num_panels:
        raise ValueError("num_classes cannot exceed the number of panels.")
    shuffled = onp.random.default_rng(seed).permutation(num_panels)
    weights = onp.zeros((num_panels, num_classes))
    for class_idx, panels in enumerate(onp.array_split(shuffled, num_classes)):
        weights[panels, class_idx] = 1.0
    return jnp.asarray(weights)
