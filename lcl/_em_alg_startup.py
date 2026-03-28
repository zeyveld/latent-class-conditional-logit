"""EM algorithm initialization."""

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax.random import key, permutation

from lcl._case_utils import _loglik_gradient, _to_structural_betas
from lcl._em_alg_steps import _compute_conditional_class_probs
from lcl._optimize import _minimize
from lcl._struct import Data, DiffUnchosenChosen, EMAlgConfig, EMVars, MleConfig
from lcl.utils import _ensure_sequential


def _get_starting_vals(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    em_alg_config: EMAlgConfig,
    mle_config: MleConfig,
    numeraire_idx: int | None = None,
) -> EMVars:
    """Obtain starting values for betas and class probs."""
    diff_unchosen_chosen_by_class = _random_class_partition(
        diff_unchosen_chosen, data, num_classes, em_alg_config
    )
    latent_betas = jnp.empty((data.num_alt_vars, num_classes))
    for class_idx, class_diff_unchosen_chosen in enumerate(
        diff_unchosen_chosen_by_class
    ):
        class_starting_coeffs = _minimize(
            _loglik_gradient,
            jnp.repeat(0.0, data.num_alt_vars),
            (
                class_diff_unchosen_chosen,
                jnp.ones(class_diff_unchosen_chosen.num_cases),
            ),
            mle_config,
            numeraire_idx,
        ).params
        latent_betas = latent_betas.at[:, class_idx].set(class_starting_coeffs)

    structural_betas = _to_structural_betas(latent_betas, numeraire_idx)
    thetas, shares = None, jnp.repeat(1 / num_classes, num_classes)

    starting_class_probs_by_panel, _ = _compute_conditional_class_probs(
        structural_betas, thetas, shares, diff_unchosen_chosen, data
    )
    return EMVars(
        latent_betas=latent_betas,
        structural_betas=structural_betas,
        thetas=thetas,
        shares=jnp.mean(starting_class_probs_by_panel, axis=0),
        unconditional_loglik=jnp.array(1.0),  # Placeholder
        class_probs_by_panel=starting_class_probs_by_panel,
    )


def _random_class_partition(
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    num_classes: int,
    em_alg_config: EMAlgConfig,
) -> list[DiffUnchosenChosen]:
    """Randomly divide panels into classes."""
    assert data.panels is not None and data.num_panels is not None
    panels_unique = onp.unique(data.panels)
    num_panels_per_class = -(data.num_panels // -num_classes)  # Ceiling division

    unshuffled_starting_classes = jnp.repeat(
        jnp.arange(num_classes), num_panels_per_class
    )[: data.num_panels]
    starting_classes = permutation(
        key(em_alg_config.jax_prng_seed), unshuffled_starting_classes
    )
    starting_classes_by_panel_df = pl.DataFrame(
        {"panels": panels_unique, "starting_classes": onp.array(starting_classes)}
    )

    est_df = (
        pl.from_numpy(onp.array(diff_unchosen_chosen.X))
        .with_columns(
            pl.Series(name="panels", values=onp.asarray(diff_unchosen_chosen.panels)),
            pl.Series(name="cases", values=onp.array(diff_unchosen_chosen.cases)),
            pl.Series(name="alts", values=onp.array(diff_unchosen_chosen.alts)),
        )
        .join(starting_classes_by_panel_df, on="panels", how="left", coalesce=True)
    )
    est_dfs_by_class = est_df.partition_by("starting_classes", include_key=False)

    diff_unchosen_chosen_by_class = []
    for class_est_df in est_dfs_by_class:
        sorted_class_est_df = class_est_df.sort("panels", "cases", "alts")
        class_Xd = (
            sorted_class_est_df.drop("panels", "cases", "alts")
            .cast(pl.Float64)
            .to_jax()
        )
        alts = _ensure_sequential(sorted_class_est_df["alts"].cast(pl.UInt32).to_jax())
        class_cases = _ensure_sequential(sorted_class_est_df["cases"].to_jax())
        panels = _ensure_sequential(
            sorted_class_est_df["panels"].cast(pl.UInt32).to_jax()
        )
        diff_unchosen_chosen_by_class.append(
            DiffUnchosenChosen(
                X=class_Xd,
                alts=alts,
                cases=class_cases,
                panels=panels,
                num_cases=jnp.unique(class_cases).shape[0],
            )
        )
    return diff_unchosen_chosen_by_class


# EOF
