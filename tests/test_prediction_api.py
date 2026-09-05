"""Regression tests for shared prediction and rich inference APIs."""

import numpy as np
import polars as pl
import pytest
import jax.numpy as jnp

from lcl import (
    ConditionalLogit,
    FitOptions,
    InferenceOptions,
    LatentClassConditionalLogit,
    OptimizationOptions,
    PartitionType,
    WTPRequest,
)
from lcl.results import CLPrediction, ResultsProtocol


def _conditional_data() -> pl.DataFrame:
    rows = []
    for panel in (10, 20, 30, 40, 50, 60):
        for occasion in (1, 2, 3):
            case = panel * 10 + occasion
            for alt in ("car", "rail", "bus"):
                alt_index = {"car": 0, "rail": 1, "bus": 2}[alt]
                quality = 0.3 * alt_index + 0.02 * panel + 0.1 * occasion
                cost = 1.0 + alt_index + 0.03 * panel + 0.15 * occasion * alt_index
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "quality": quality,
                        "cost": cost,
                        "choice": alt_index == ((panel // 10 + occasion) % 3),
                    }
                )
    return pl.DataFrame(rows)


def test_conditional_prediction_exposes_diagnostics_wtp_and_aggregation() -> None:
    """CL predictions implement the common post-estimation tools."""
    data = _conditional_data()
    result = ConditionalLogit(numeraire="cost").fit(
        data,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["quality", "cost"],
        optimization_options=OptimizationOptions(maxiter=20),
        inference=InferenceOptions(covariance="unadjusted"),
    )
    prediction = result.predict(
        data,
        panel_weights={10: 1.0, 20: 1.0, 30: 2.0, 40: 2.0, 50: 3.0, 60: 3.0},
    )

    assert isinstance(result, ResultsProtocol)
    assert isinstance(prediction, CLPrediction)
    assert np.isclose(prediction.market_shares()["market_share"].sum(), 1.0)
    elasticities = prediction.elasticities("quality")
    aggregate = prediction.aggregate_elasticities("quality")
    assert elasticities.height == data.height * 3
    assert aggregate.height == 9
    assert prediction.denominator_diagnostics().height == 1
    assert prediction.wtp("quality", se="delta").height == 1
    assert result.loglik(data) == pytest.approx(float(result.loglikelihood), rel=1e-7)
    assert {"observed_score_max", "mcfadden_r2"}.issubset(
        set(result.diagnostics().to_frame()["check"])
    )


def test_cl_parametric_bootstrap_is_seed_reproducible() -> None:
    """Ratio bootstrap results are deterministic under an explicit seed."""
    data = _conditional_data()
    result = ConditionalLogit(numeraire="cost").fit(
        data,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["quality", "cost"],
        inference=InferenceOptions(covariance="unadjusted"),
    )
    prediction = result.predict(data)
    first = prediction.wtp(
        "quality", se="bootstrap", bootstrap_draws=50, bootstrap_seed=9
    )
    second = prediction.wtp(
        "quality", se="bootstrap", bootstrap_draws=50, bootstrap_seed=9
    )
    assert first["std_error"].to_list() == second["std_error"].to_list()


def test_lcl_rich_inference_and_weighted_bootstrap_wtp() -> None:
    """LCL exposes class, membership, classification, and ratio inference."""
    data = _conditional_data()
    result = LatentClassConditionalLogit(num_classes=2, numeraire="cost").fit(
        data,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["quality", "cost"],
        fit_options=FitOptions(max_em_iter=8, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=20),
        inference=InferenceOptions(skip=True),
    )
    # Inject a finite covariance to isolate reporting transformations from the
    # deliberately tiny sample's information-rank limitations.  Inference reads
    # the latent-space matrix; the public one is its structural counterpart.
    result.latent_cov_matrix = jnp.eye(result.num_params) * 1e-4
    result.cov_matrix = result._structural_covariance(result.latent_cov_matrix)

    assert isinstance(result, ResultsProtocol)
    assert {"coefficient", "std_error"}.issubset(result.class_coefficients().columns)
    assert {"coefficient", "std_error", "reference_class"}.issubset(
        result.membership_coefficients().columns
    )
    assert "std_error" in result.class_shares().columns
    assert {
        "average_posterior",
        "odds_correct_classification",
        "entropy_r2",
    }.issubset(result.classification_diagnostics().columns)
    assert float(result.aic3) > float(result.aic)

    prediction = result.predict(
        data,
        panel_weights={
            panel: float(index + 1)
            for index, panel in enumerate((10, 20, 30, 40, 50, 60))
        },
    )
    partitions = pl.DataFrame(
        {
            "panel": [10, 20, 30, 40, 50, 60],
            "segment": ["a", "a", "a", "b", "b", "b"],
        }
    )
    tables = prediction.compute_wtp(
        WTPRequest("quality", "segment", PartitionType.CATEGORICAL),
        partition_data=partitions,
        panel_col="panel",
        se="bootstrap",
        bootstrap_draws=25,
        bootstrap_seed=4,
        show=False,
    )
    table = next(iter(tables.values()))
    assert table["Panel_Count"].to_list() == [3, 3]
    assert np.all(np.isfinite(table["Standard_Error"].to_numpy()))
    assert np.isclose(prediction.market_shares()["market_share"].sum(), 1.0)
