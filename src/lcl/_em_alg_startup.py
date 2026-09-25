"""Expectation-Maximization (EM) algorithm initialization routines."""

from collections.abc import Iterator

import jax.numpy as jnp
import numpy as onp
from equinox import filter_jit
from jax.nn import softmax
from jaxtyping import Array, Float64

from lcl.constraints import NegativeCoefficientBound
from lcl._case_utils import _loglik_gradient, _loglik_value
from lcl._demographics import _predict_class_membership_probs
from lcl._em_alg_steps import (
    _compute_em_log_kernels,
    _posterior_and_loglik,
)
from lcl._optimize import exact_newton_minimize, newton_kwargs, scaled_objective
from lcl.options import FitOptions, OptimizationOptions
from lcl._struct import Data, DiffUnchosenChosen, EMVars


@filter_jit
def _fit_starting_beta(
    diff: DiffUnchosenChosen,
    optimization_options: OptimizationOptions,
    negative_bound: NegativeCoefficientBound,
) -> Float64[Array, "alt_vars"]:
    """Fit one starting subset, reusing the executable across equal shapes.

    Subset arrays are dynamic arguments, never dataset-sized closure constants.
    Unequal subset shapes still specialize normally; changing a seed or array
    values alone does not require another compilation.
    """
    weights = jnp.ones(diff.num_cases)
    scale = max(diff.num_cases, 1)

    value, derivatives = scaled_objective(_loglik_value, _loglik_gradient, scale)

    initial = jnp.zeros(diff.X.shape[1])
    state = exact_newton_minimize(
        value,
        derivatives,
        initial,
        diff,
        weights,
        **newton_kwargs(optimization_options),
        upper_bounds=negative_bound.upper_bounds(initial),
    )
    return state.params


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
    conditional logit model on each subset to derive distinct starting taste parameters.

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
    diff_unchosen_chosen_by_class = _random_class_partition(
        diff_unchosen_chosen, data, num_classes, fit_options
    )

    betas_list = []

    for class_diff_unchosen_chosen in diff_unchosen_chosen_by_class:
        betas_list.append(
            _fit_starting_beta(
                class_diff_unchosen_chosen,
                optimization_options,
                negative_bound,
            )
        )

    # Stack the independently estimated parameter vectors into a (K, C) matrix
    betas = jnp.column_stack(betas_list)

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
        prior_by_panel = jnp.repeat(starting_shares[None, :], data.num_panels, axis=0)
    else:
        prior_by_panel = _predict_class_membership_probs(thetas, data)
    starting_class_probs_by_panel, starting_loglik = _posterior_and_loglik(
        log_kernels, prior_by_panel
    )

    return EMVars(
        betas=betas,
        thetas=thetas,
        shares=starting_shares,
        unconditional_loglik=starting_loglik,
        class_probs_by_panel=starting_class_probs_by_panel,
    )


def _random_class_partition(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    fit_options: FitOptions,
) -> Iterator[DiffUnchosenChosen]:
    """Randomly partition decision-makers to initialize class-specific parameters.

    Ensures that all choice situations belonging to a specific decision-maker (panel)
    are kept together within the same random subset. Natively squashes IDs to remain
    strictly contiguous and zero-indexed to satisfy downstream JAX requirements.

    Parameters
    ----------
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        The complete differenced design matrix.
    data : :class:`~lcl._struct.Data`
        The core estimation data and metadata.
    num_classes : int
        The number of mutually exclusive subsets to generate.
    fit_options : :class:`~lcl.options.FitOptions`
        EM settings containing the reproducible partition seed.

    Yields
    ------
    :class:`~lcl._struct.DiffUnchosenChosen`
        One independent differenced subset at a time, so startup does not retain
        a second copy of the entire differenced design split across classes.
    """
    if diff_unchosen_chosen.panels is None or data.num_panels is None:
        raise ValueError(
            "Panel identifiers are required for latent-class initialization."
        )
    if num_classes > data.num_panels:
        raise ValueError("num_classes cannot exceed the number of panels.")

    # Randomly assign each panel to one initial class.
    rng = onp.random.default_rng(fit_options.seed)
    shuffled_panels = rng.permutation(data.num_panels)
    panels_per_class = onp.array_split(shuffled_panels, num_classes)
    if any(len(panels_in_class) == 0 for panels_in_class in panels_per_class):
        raise ValueError("Initialization produced an empty latent class.")

    panel_to_class = onp.empty(data.num_panels, dtype=onp.int32)
    for class_idx, panels_in_class in enumerate(panels_per_class):
        panel_to_class[panels_in_class] = class_idx

    # Map panel assignments to long-format observations.
    row_classes = panel_to_class[onp.array(diff_unchosen_chosen.panels)]

    for class_idx in range(num_classes):
        # Select the observations assigned to the current class.
        mask = row_classes == class_idx

        # Filter each aligned array with the class mask.
        class_X = diff_unchosen_chosen.X[mask]
        class_alts = diff_unchosen_chosen.alts[mask]
        raw_cases = diff_unchosen_chosen.cases[mask]
        raw_panels = diff_unchosen_chosen.panels[mask]

        # Re-index cases and panels to contiguous, zero-based identifiers.
        # The return_inverse array provides the perfect remapped IDs for JAX segment_sum.
        _, contiguous_cases = onp.unique(raw_cases, return_inverse=True)
        _, contiguous_panels = onp.unique(raw_panels, return_inverse=True)

        num_cases = (
            int(onp.max(contiguous_cases) + 1) if len(contiguous_cases) > 0 else 0
        )

        yield DiffUnchosenChosen(
            X=jnp.array(class_X),
            alts=jnp.array(class_alts),
            cases=jnp.array(contiguous_cases, dtype="uint32"),
            panels=jnp.array(contiguous_panels, dtype="uint32"),
            num_cases=num_cases,
        )
