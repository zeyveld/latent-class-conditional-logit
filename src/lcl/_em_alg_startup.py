"""Expectation-Maximization (EM) algorithm initialization routines."""

import jax.numpy as jnp
import numpy as onp

from lcl._case_utils import _loglik_gradient, _to_structural_betas
from lcl._em_alg_steps import _compute_conditional_class_probs
from lcl._optimize import _minimize
from lcl._struct import Data, DiffUnchosenChosen, EMAlgConfig, EMVars, MleConfig


def _get_starting_vals(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    em_alg_config: EMAlgConfig,
    mle_config: MleConfig,
    numeraire_idx: int | None = None,
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
    em_alg_config : :class:`~lcl._struct.EMAlgConfig`
        Configuration containing the JAX PRNG seed for reproducible partitioning.
    mle_config : :class:`~lcl._struct.MleConfig`
        Optimization settings for the subset-level L-BFGS routines.
    numeraire_idx : int | None, optional
        Column index of the numeraire variable, if applicable.

    Returns
    -------
    :class:`~lcl._struct.EMVars`
        Container holding the initialized taste parameters, uniform starting shares,
        and first-pass posterior class probabilities.
    """
    diff_unchosen_chosen_by_class = _random_class_partition(
        diff_unchosen_chosen, data, num_classes, em_alg_config
    )

    latent_betas_list = []

    for class_diff_unchosen_chosen in diff_unchosen_chosen_by_class:
        optim_res = _minimize(
            loglik_fn=_loglik_gradient,
            params=jnp.zeros(data.num_alt_vars),
            args=(
                class_diff_unchosen_chosen,
                jnp.ones(class_diff_unchosen_chosen.num_cases),
            ),
            mle_config=mle_config,
            numeraire_idx=numeraire_idx,
        )
        latent_betas_list.append(optim_res.params)

    # Stack the independently estimated parameter vectors into a (K, C) matrix
    latent_betas = jnp.column_stack(latent_betas_list)
    structural_betas = _to_structural_betas(latent_betas, numeraire_idx)

    thetas = None
    shares = jnp.repeat(1.0 / num_classes, num_classes)

    starting_class_probs_by_panel, _ = _compute_conditional_class_probs(
        structural_betas, thetas, shares, diff_unchosen_chosen, data
    )

    return EMVars(
        latent_betas=latent_betas,
        structural_betas=structural_betas,
        thetas=thetas,
        shares=jnp.mean(starting_class_probs_by_panel, axis=0),
        unconditional_loglik=jnp.array(1.0),  # Placeholder prior to first EM step
        class_probs_by_panel=starting_class_probs_by_panel,
    )


def _random_class_partition(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    em_alg_config: EMAlgConfig,
) -> list[DiffUnchosenChosen]:
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
    em_alg_config : :class:`~lcl._struct.EMAlgConfig`
        Configuration containing the PRNG seed for reproducibility.

    Returns
    -------
    list[:class:`~lcl._struct.DiffUnchosenChosen`]
        A list of length `num_classes`, where each element is a valid, independent
        differenced design matrix subset ready for conditional logit estimation.
    """
    assert diff_unchosen_chosen.panels is not None
    assert data.num_panels is not None

    # 1. Randomly assign each panel to one of K classes
    onp.random.seed(em_alg_config.jax_prng_seed)
    shuffled_panels = onp.random.permutation(data.num_panels)
    panels_per_class = onp.array_split(shuffled_panels, num_classes)

    panel_to_class = onp.empty(data.num_panels, dtype=onp.int32)
    for class_idx, panels_in_class in enumerate(panels_per_class):
        panel_to_class[panels_in_class] = class_idx

    # 2. Map class assignments down to the observation level
    row_classes = panel_to_class[onp.array(diff_unchosen_chosen.panels)]

    diff_unchosen_chosen_by_class = []

    for class_idx in range(num_classes):
        # 3. Create boolean mask for the current class subset
        mask = row_classes == class_idx

        # 4. Filter the arrays
        class_X = diff_unchosen_chosen.X[mask]
        class_alts = diff_unchosen_chosen.alts[mask]
        raw_cases = diff_unchosen_chosen.cases[mask]
        raw_panels = diff_unchosen_chosen.panels[mask]

        # 5. Crucial: Re-index cases and panels to be strictly contiguous and zero-indexed!
        # The return_inverse array provides the perfect remapped IDs for JAX segment_sum.
        _, contiguous_cases = onp.unique(raw_cases, return_inverse=True)
        _, contiguous_panels = onp.unique(raw_panels, return_inverse=True)

        num_cases = (
            int(onp.max(contiguous_cases) + 1) if len(contiguous_cases) > 0 else 0
        )

        diff_unchosen_chosen_by_class.append(
            DiffUnchosenChosen(
                X=jnp.array(class_X),
                alts=jnp.array(class_alts),
                cases=jnp.array(contiguous_cases),
                panels=jnp.array(contiguous_panels),
                num_cases=num_cases,
            )
        )

    return diff_unchosen_chosen_by_class


# EOF
