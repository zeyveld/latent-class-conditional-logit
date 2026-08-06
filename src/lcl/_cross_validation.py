"""Out-of-sample cross-validation for latent-class model selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as onp
import polars as pl

from lcl._struct import (
    FitOptions,
    InferenceOptions,
    OptimizationOptions,
)
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
from lcl.spec import LCLSpec, resolve_lcl_spec

logger = logging.getLogger(__name__)


def cv_optimal_classes(
    data: Any,
    alts_col: str | None = None,
    cases_col: str | None = None,
    panels_col: str | None = None,
    num_classes_list: Sequence[int] | None = None,
    utility_formula: str | None = None,
    membership_formula: str | None = None,
    choice_col: str | None = None,
    case_varnames: Sequence[str] | None = None,
    dem_varnames: Sequence[str] | None = None,
    dems_data: Any | None = None,
    numeraire: str | None = None,
    folds: int = 5,
    seed: int = 42,
    *,
    spec: LCLSpec | None = None,
    fit_options: FitOptions | None = None,
    optimization_options: OptimizationOptions | None = None,
    inference: InferenceOptions | None = None,
    numeraire_min_abs: float | None = None,
) -> pl.DataFrame:
    """Select a latent-class count with blocked panel-level cross-validation.

    Every test fold is transformed by the encoder fitted on its training fold.
    Formula-derived columns therefore retain their training-time categorical
    meaning, and unseen categories produce Formulaic's explicit warning or error.

    ``Avg_OOS_LL`` is the mean held-out log likelihood per panel, pooled across
    all folds. It is set to ``NaN`` when any fold fails so an incomplete class
    sweep cannot silently compete with complete results. The successful-fold-only
    value remains available in ``Avg_Successful_OOS_LL`` for diagnosis.

    Parameters
    ----------
    data : Any
        Long-format choice data.
    alts_col : str | None, optional
        Alternative column.
    cases_col, panels_col, choice_col : str | None, optional
        Choice-situation, panel, and chosen-alternative indicator columns.
    num_classes_list : Sequence[int] | None, optional
        Candidate class counts, each at least two.
    utility_formula, membership_formula : str | None, optional
        Separate Formulaic specifications.
    case_varnames, dem_varnames : Sequence[str] | None, optional
        Explicit utility and class-membership variables.
    dems_data : Any | None, optional
        Separate panel-level demographic data.
    numeraire : str | None, optional
        Utility coefficient constrained to be strictly negative.
    folds : int, default=5
        Number of blocked panel folds.
    seed : int, default=42
        Random seed for panel shuffling.
    spec : LCLSpec | None, optional
        Preferred declarative model specification.
    fit_options : FitOptions | None, optional
        EM settings, including independent starts per training fold.
    optimization_options : OptimizationOptions | None, optional
        M-step optimizer settings.
    inference : InferenceOptions | None, optional
        Inference settings. By default CV skips covariance work.
    numeraire_min_abs : float | None, optional
        Minimum absolute magnitude for a keyword-specified numeraire.

    Returns
    -------
    pl.DataFrame
        One row per valid class count, with aggregate metrics and list-valued
        per-fold diagnostics.
    """
    if num_classes_list is None:
        raise ValueError("num_classes_list is required.")
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    base_spec = resolve_lcl_spec(
        spec=spec,
        alts_col=alts_col,
        cases_col=cases_col,
        panels_col=panels_col,
        choice_col=choice_col,
        case_varnames=case_varnames,
        dem_varnames=dem_varnames,
        utility_formula=utility_formula,
        membership_formula=membership_formula,
        numeraire=numeraire,
        numeraire_min_abs=numeraire_min_abs,
    )
    if fit_options is None:
        fit_options = FitOptions()
    if optimization_options is None:
        optimization_options = OptimizationOptions()
    if inference is None:
        inference = InferenceOptions(skip=True)

    if isinstance(data, pl.DataFrame):
        df = data
    elif hasattr(data, "columns"):
        df = pl.from_pandas(data)
    else:
        df = pl.DataFrame(data)

    panel_col = base_spec.ids.panel
    if panel_col not in df.columns:
        raise ValueError(f"Panel column {panel_col!r} was not found in data.")
    unique_panels = df[panel_col].unique(maintain_order=True).to_numpy()
    if folds > len(unique_panels):
        raise ValueError(
            f"folds={folds} exceeds the number of unique panels ({len(unique_panels)})."
        )

    rng = onp.random.default_rng(seed)
    shuffled_panels = rng.permutation(unique_panels)
    fold_panel_lists = onp.array_split(shuffled_panels, folds)
    rows: list[dict[str, object]] = []

    for num_classes in num_classes_list:
        if num_classes < 2:
            logger.warning(
                "Skipping %s class: latent-class models require at least 2.",
                num_classes,
            )
            continue

        logger.info("Evaluating model with %s latent classes.", num_classes)
        fold_lls: list[float] = []
        fold_mean_lls: list[float] = []
        fold_converged: list[bool | None] = []
        fold_train_panels: list[int] = []
        fold_test_panels: list[int] = []
        fold_errors: list[str | None] = []

        for fold_index, test_panels in enumerate(fold_panel_lists, start=1):
            test_df = df.filter(pl.col(panel_col).is_in(test_panels))
            train_df = df.filter(~pl.col(panel_col).is_in(test_panels))
            train_panel_count = train_df[panel_col].n_unique()
            test_panel_count = test_df[panel_col].n_unique()
            fold_train_panels.append(train_panel_count)
            fold_test_panels.append(test_panel_count)

            model = LatentClassConditionalLogit(
                spec=replace(base_spec, classes=num_classes)
            )
            try:
                result = model.fit(
                    data=train_df,
                    dems_data=dems_data,
                    fit_options=fit_options,
                    optimization_options=optimization_options,
                    inference=inference,
                )
                panel_scores = result.loglik(
                    test_df,
                    dems_data=dems_data,
                    per_panel=True,
                )
                if not isinstance(panel_scores, pl.DataFrame):
                    raise TypeError("Per-panel scoring did not return a DataFrame.")
                if panel_scores.height != test_panel_count:
                    raise ValueError(
                        "Per-panel scoring returned "
                        f"{panel_scores.height} rows for {test_panel_count} "
                        "held-out panels."
                    )
                total_ll = float(panel_scores["log_likelihood"].sum())
                mean_ll = total_ll / test_panel_count
                fold_lls.append(total_ll)
                fold_mean_lls.append(mean_ll)
                fold_converged.append(bool(result.converged))
                fold_errors.append(None)
            except Exception as exc:
                logger.warning(
                    "Model evaluation failed for %s classes, fold %s: %s",
                    num_classes,
                    fold_index,
                    exc,
                )
                fold_lls.append(float("nan"))
                fold_mean_lls.append(float("nan"))
                fold_converged.append(None)
                fold_errors.append(str(exc))

        successful = onp.isfinite(onp.asarray(fold_lls))
        successful_folds = int(successful.sum())
        failed_folds = folds - successful_folds
        converged_folds = sum(value is True for value in fold_converged)
        nonconverged_folds = sum(value is False for value in fold_converged)
        successful_total_ll = float(onp.nansum(onp.asarray(fold_lls, dtype=float)))
        successful_panels = sum(
            count
            for count, succeeded in zip(fold_test_panels, successful.tolist())
            if succeeded
        )
        successful_mean = (
            successful_total_ll / successful_panels
            if successful_panels
            else float("nan")
        )
        complete_mean = successful_mean if failed_folds == 0 else float("nan")

        rows.append(
            {
                "Num_Classes": num_classes,
                "Avg_OOS_LL": complete_mean,
                "Avg_Successful_OOS_LL": successful_mean,
                "Total_OOS_LL": (
                    successful_total_ll if failed_folds == 0 else float("nan")
                ),
                "Successful_Folds": successful_folds,
                "Failed_Folds": failed_folds,
                "Converged_Folds": converged_folds,
                "Nonconverged_Folds": nonconverged_folds,
                "Fold": list(range(1, folds + 1)),
                "Fold_OOS_LL": fold_lls,
                "Fold_Mean_Panel_LL": fold_mean_lls,
                "Fold_Converged": fold_converged,
                "Fold_Train_Panels": fold_train_panels,
                "Fold_Test_Panels": fold_test_panels,
                "Fold_Errors": fold_errors,
            }
        )

    return pl.DataFrame(rows)
