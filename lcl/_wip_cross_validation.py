"""Out-of-sample cross validation for model selection."""

from typing import Optional, Sequence

import numpy as onp
import polars as pl
from jax.typing import ArrayLike

from lcl._case_utils import _diff_unchosen_chosen
from lcl._struct import EMAlgConfig, MleConfig
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def cv_optimal_k(
    X: ArrayLike,
    y: ArrayLike,
    case_varnames: Sequence[str],
    alts: ArrayLike,
    cases: ArrayLike,
    panels: ArrayLike,
    k_max: int,
    dems: Optional[ArrayLike] = None,
    dem_varnames: Optional[Sequence[str]] = None,
    numeraire: Optional[str] = None,
    folds: int = 5,
    seed: int = 42,
    em_alg_config: EMAlgConfig = EMAlgConfig(),
    mle_config: MleConfig = MleConfig(),
) -> pl.DataFrame:
    """Perform blocked K-Fold Cross Validation to determine the optimal number of latent classes.

    THIS UTILITY IS WIP; USE WITH CAUTION!!!

    Evaluates Out-of-Sample Log-Likelihood (OOS-LL) across a range of K classes. Blocks by
    panel ID to prevent data leakage in panel/repeated choice data.

    Parameters
    ----------
    X : ArrayLike
        (N, K) array of explanatory variables.
    y : ArrayLike
        (N,) boolean array indicating chosen alternatives.
    case_varnames : Sequence[str]
        Names of the alternative-specific variables.
    alts : ArrayLike
        (N,) vector of alternative IDs.
    cases : ArrayLike
        (N,) vector of choice situation IDs.
    panels : ArrayLike
        (N,) vector of decision-maker IDs.
    k_max : int
        The maximum number of latent classes to test (evaluates K = 1 through k_max).
    dems : ArrayLike, optional
        (N, D) array of demographic variables. Must be at the observation level.
    dem_varnames : Sequence[str], optional
        Names of the demographic variables.
    numeraire : str, optional
        The variable to be used as the numeraire (e.g., cost).
    folds : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        PRNG seed for randomly distributing panels across folds.
    em_alg_config : EMAlgConfig, optional
        Algorithm settings for EM.
    mle_config : MleConfig, optional
        Optimization settings for MLE.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing K and the Average Out-of-Sample Log-Likelihood.
    """

    # Convert inputs to numpy for efficient CPU-side masking and slicing before JAX compilation
    X_arr = onp.asarray(X)
    y_arr = onp.asarray(y)
    alts_arr = onp.asarray(alts)
    cases_arr = onp.asarray(cases)
    panels_arr = onp.asarray(panels)

    dems_arr = onp.asarray(dems) if dems is not None else None

    # Identify unique decision makers to create blocked folds
    unique_panels = onp.unique(panels_arr)

    # Shuffle and split panel IDs into roughly equal folds
    onp.random.seed(seed)
    shuffled_panels = onp.random.permutation(unique_panels)
    fold_panel_lists = onp.array_split(shuffled_panels, folds)

    results = []

    for k in range(1, k_max + 1):
        print(f"Evaluating K = {k}...")
        fold_lls = []

        for f in range(folds):
            test_panels = fold_panel_lists[f]

            # Create observation-level masks based on panel IDs
            test_mask = onp.isin(panels_arr, test_panels)
            train_mask = ~test_mask

            X_train, X_test = X_arr[train_mask], X_arr[test_mask]
            y_train, y_test = y_arr[train_mask], y_arr[test_mask]
            alts_train, alts_test = alts_arr[train_mask], alts_arr[test_mask]
            cases_train, cases_test = cases_arr[train_mask], cases_arr[test_mask]
            panels_train, panels_test = panels_arr[train_mask], panels_arr[test_mask]

            dems_train, dems_test = None, None
            if dems_arr is not None:
                if dems_arr.shape[0] == X_arr.shape[0]:
                    dems_train = dems_arr[train_mask]
                    dems_test = dems_arr[test_mask]
                else:
                    raise ValueError(
                        "For cross-validation, `dems` must be provided at the observation "
                        "level (same rows as X) so it can be cleanly split across folds."
                    )

            # 1. Instantiate and Fit Model on Training Fold
            model = LatentClassConditionalLogit(num_classes=k, numeraire=numeraire)

            try:
                res = model.fit(
                    X=X_train,
                    y=y_train,
                    case_varnames=case_varnames,
                    alts=alts_train,
                    cases=cases_train,
                    panels=panels_train,
                    dems=dems_train,
                    dem_varnames=dem_varnames,
                    em_alg_config=em_alg_config,
                    mle_config=mle_config,
                )

                # 2. Package Test Data using internal structs
                # This inherently applies `_ensure_sequential` to test cases/panels
                test_data, *_ = model._setup_data(
                    X=X_test,
                    dems=dems_test,
                    y=y_test,
                    cases=cases_test,
                    alts=alts_test,
                    panels=panels_test,
                )

                # Format differences for conditional logit evaluation
                test_diff = _diff_unchosen_chosen(test_data)

                # 3. Evaluate Out-of-Sample Log Likelihood
                # _full_loglik_fn leverages _panel_logliks, dynamically adapting to test_data
                oos_ll = res._full_loglik_fn(res.flat_params, test_diff, test_data)
                fold_lls.append(float(oos_ll))

            except Exception as e:
                print(f"Warning: Model evaluation failed for K={k}, Fold={f + 1}: {e}")
                fold_lls.append(onp.nan)

        # Average Out-of-Sample Log-Likelihood across all folds
        avg_oos_ll = onp.nanmean(fold_lls)
        results.append({"K": k, "Avg_OOS_LL": avg_oos_ll})

    return pl.DataFrame(results)


# EOF
