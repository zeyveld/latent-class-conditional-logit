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
    # The balanced choices in _conditional_data have a boundary price optimum.
    # Ordinary bootstrap covariance requires an identified interior estimate.
    rng = np.random.default_rng(83)
    x = rng.normal(size=(240, 3, 2))
    chosen = np.argmax(x @ np.array([0.7, -1.2]) + rng.gumbel(size=(240, 3)), axis=1)
    data = pl.DataFrame(
        {
            "panel": np.repeat(np.arange(80), 9),
            "case": np.repeat(np.arange(240), 3),
            "alt": np.tile(["car", "rail", "bus"], 240),
            "quality": x[:, :, 0].ravel(),
            "cost": x[:, :, 1].ravel(),
            "choice": (np.arange(3)[None, :] == chosen[:, None]).ravel(),
        }
    )
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


def test_cl_boundary_fit_does_not_manufacture_bootstrap_covariance() -> None:
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
    assert float(result.coeff_[1]) == pytest.approx(-1e-5, abs=1e-12)
    with pytest.raises(ValueError, match="finite covariance matrix"):
        result.predict(data).wtp("quality", se="bootstrap", bootstrap_draws=50)


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
    # Use known interior coefficients and covariance to isolate reporting from
    # this tiny sample's boundary and information-rank limitations.
    result.em_res = result.em_res._replace(
        betas=result.em_res.betas.at[result.model.numeraire_idx].set(-1.0)
    )
    result.flat_params = result._pack_params()
    result.cov_matrix = jnp.eye(result.num_params) * 1e-4

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


@pytest.mark.parametrize("estimator", ["cl", "lcl"])
def test_gaussian_boundary_screen_only_applies_to_monetary_ratios(estimator):
    from scipy.stats import norm

    data = _conditional_data()
    common = dict(
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["quality", "cost"],
        inference=InferenceOptions(skip=True),
    )
    # Deliberately specified interior coefficients and a known Gaussian law test
    # the inference dispatch independently of the tiny sample's rank/boundary.
    if estimator == "cl":
        result = ConditionalLogit(numeraire="cost").fit(data, **common)
        result.coeff_ = jnp.array([0.7, -0.2])
        indices = [1]
    else:
        result = LatentClassConditionalLogit(num_classes=2, numeraire="cost").fit(
            data,
            fit_options=FitOptions(max_em_iter=2, num_devices=1, polish=False),
            **common,
        )
        result.em_res = result.em_res._replace(
            betas=jnp.array([[0.7, 0.8], [-0.2, -0.2]])
        )
        result.flat_params = result._pack_params()
        indices = [2, 3]
    variance = jnp.full(result.flat_params.size, 0.01).at[jnp.array(indices)].set(1.0)
    result.cov_matrix = jnp.diag(variance)
    original_covariance = np.asarray(result.cov_matrix).copy()
    prediction = result.predict(data)
    counterfactual = result.predict(data.with_columns(pl.col("cost") * 1.1))

    def wtp_errors(**kwargs):
        if estimator == "cl":
            return prediction.wtp("quality", **kwargs)["std_error"].to_numpy()
        partitions = (
            data.select("panel").unique().with_columns(pl.lit("all").alias("group"))
        )
        tables = prediction.compute_wtp(
            WTPRequest("quality", "group", PartitionType.CATEGORICAL),
            partition_data=partitions,
            panel_col="panel",
            show=False,
            **kwargs,
        )
        return next(iter(tables.values()))["Standard_Error"].to_numpy()

    diagnostics = prediction.denominator_diagnostics()
    np.testing.assert_allclose(diagnostics["denominator_std_error"], 1.0)
    np.testing.assert_allclose(
        diagnostics["gaussian_bound_crossing_probability"],
        norm.sf(0.2 - result.model.numeraire_min_abs),
    )
    np.testing.assert_allclose(
        diagnostics["gaussian_zero_crossing_probability"], norm.sf(0.2)
    )
    for seed, draws in [(1, 2), (3, 100)]:
        kwargs = dict(se="bootstrap", bootstrap_seed=seed, bootstrap_draws=draws)
        assert np.isfinite(
            prediction.market_shares(**kwargs)["std_error"].to_numpy()
        ).all()
        assert np.isfinite(
            prediction.aggregate_elasticities("quality", **kwargs)[
                "elasticity_quality_se"
            ].to_numpy()
        ).all()
        for compute in [
            lambda: wtp_errors(**kwargs),
            lambda: prediction.mean_surplus(**kwargs),
            lambda: prediction.mean_surplus_change(counterfactual, **kwargs),
        ]:
            with pytest.raises(ValueError, match="bound-crossing probability"):
                compute()
        if estimator == "cl":
            with pytest.raises(ValueError, match="bound-crossing probability"):
                prediction.compute_wtp("quality", **kwargs)
    # Coefficient and local delta-method inference retain the original covariance.
    assert np.isfinite(wtp_errors(se="delta")).all()
    np.testing.assert_array_equal(result.cov_matrix, original_covariance)
