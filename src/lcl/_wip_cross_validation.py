"""Out-of-sample cross validation for model selection."""

from typing import Any, Optional, Sequence

import numpy as onp
import polars as pl

from lcl._case_utils import _diff_unchosen_chosen
from lcl._struct import EMAlgConfig, MleConfig
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def cv_optimal_k(
    data: Any,
    alts_col: str,
    cases_col: str,
    panels_col: str,
    k_max: int,
    formula: Optional[str] = None,
    choice_col: Optional[str] = None,
    case_varnames: Optional[Sequence[str]] = None,
    dem_varnames: Optional[Sequence[str]] = None,
    dems_data: Optional[Any] = None,
    numeraire: Optional[str] = None,
    folds: int = 5,
    seed: int = 42,
    em_alg_config: EMAlgConfig = EMAlgConfig(),
    mle_config: MleConfig = MleConfig(),
) -> pl.DataFrame:
    """Perform blocked K-Fold Cross Validation to determine the optimal number of latent classes.

    Splits the data safely at the panel (decision-maker) level to ensure that
    the same decision-maker does not appear in both the training and test folds.

    Parameters
    ----------
    data : Any
        The main dataset containing choice situations and alternatives.
    alts_col : str
        Name of the column containing alternative identifiers.
    cases_col : str
        Name of the column grouping observations into distinct choice situations.
    panels_col : str
        Name of the column mapping observations to specific decision-makers.
    k_max : int
        The maximum number of latent classes to test (evaluates K = 1 through K = k_max).
    formula : str | None, optional
        R-style formula string (e.g., "choice ~ price + C(brand) | income").
    choice_col : str | None, optional
        Name of the boolean/binary column indicating chosen alternatives.
    case_varnames : Sequence[str] | None, optional
        List of alternative-specific variables.
    dem_varnames : Sequence[str] | None, optional
        List of demographic variables.
    dems_data : Any | None, optional
        A separate dataset containing panel-level demographics.
    numeraire : str | None, optional
        The variable to be constrained as strictly positive (e.g., "price").
    folds : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random seed for replicable panel splitting.
    em_alg_config : :class:`~lcl._struct.EMAlgConfig`, optional
        Configuration for the EM algorithm loop.
    mle_config : :class:`~lcl._struct.MleConfig`, optional
        Configuration for the inner L-BFGS optimization routines.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Average Out-of-Sample Log-Likelihood for each K.
    """
    # Unify input into Polars for easy fold splitting
    # Unify input into Polars for easy fold splitting
    if isinstance(data, pl.DataFrame):
        df = data
    elif hasattr(data, "columns"):
        df = pl.from_pandas(data)
    else:
        df = pl.DataFrame(data)

    unique_panels = df[panels_col].unique().to_numpy()

    onp.random.seed(seed)
    shuffled_panels = onp.random.permutation(unique_panels)
    fold_panel_lists = onp.array_split(shuffled_panels, folds)

    results = []

    for k in range(1, k_max + 1):
        print(f"Evaluating K = {k}...")
        fold_lls = []

        for f in range(folds):
            test_panels = fold_panel_lists[f]

            # Split data at the dataframe level
            test_df = df.filter(pl.col(panels_col).is_in(test_panels))
            train_df = df.filter(~pl.col(panels_col).is_in(test_panels))

            # 1. Instantiate and Fit Model on Training Fold
            model = LatentClassConditionalLogit(num_classes=k, numeraire=numeraire)

            try:
                res = model.fit(
                    data=train_df,
                    alts_col=alts_col,
                    cases_col=cases_col,
                    panels_col=panels_col,
                    formula=formula,
                    choice_col=choice_col,
                    case_varnames=case_varnames,
                    dem_varnames=dem_varnames,
                    dems_data=dems_data,
                    em_alg_config=em_alg_config,
                    mle_config=mle_config,
                )

                # 2. Package Test Data using the ingestion engine
                # This guarantees that the test fold gets properly ranked contiguous IDs
                parsed_test = model._ingest_data(
                    data=test_df,
                    alts_col=alts_col,
                    cases_col=cases_col,
                    panels_col=panels_col,
                    formula=formula,
                    choice_col=choice_col,
                    case_varnames=case_varnames,
                    dem_varnames=dem_varnames,
                    dems_data=dems_data,
                )

                test_data, *_ = model._setup_data(parsed_test)
                test_diff = _diff_unchosen_chosen(test_data)

                # 3. Evaluate Out-of-Sample Log Likelihood
                oos_ll = res._full_loglik_fn(res.flat_params, test_diff, test_data)
                fold_lls.append(float(oos_ll))

            except Exception as e:
                print(f"Warning: Model evaluation failed for K={k}, Fold={f + 1}: {e}")
                fold_lls.append(onp.nan)

        avg_oos_ll = onp.nanmean(fold_lls)
        results.append({"K": k, "Avg_OOS_LL": avg_oos_ll})

    return pl.DataFrame(results)


# EOF
