from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
    cv_optimal_classes,
)
from lcl._kernels import _class_membership_probs
from lcl._params import ParamPacking
from lcl._prediction import _apply_wtp_partition
from lcl._results import _parsed_prediction_arrays
from lcl._struct import PartitionType, WTPRequest
from lcl._struct import ErrorConfig, resolve_inference_options
from lcl.conditional_logit import ConditionalLogit
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
from lcl.spec import resolve_lcl_spec


def _choice_rows(panel_ids: list[int] | None = None) -> pl.DataFrame:
    panels = panel_ids if panel_ids is not None else [30, 10, 20, 40]
    rows = []
    for panel_index, panel in enumerate(panels):
        for case in [2, 1]:
            for alt in [1, 0]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == ((panel_index + case) % 2),
                        "x": float(alt + 0.2 * case),
                        "cost": float(1 + alt + case),
                    }
                )
    return pl.DataFrame(rows)


def test_case_weights_are_realigned_from_input_appearance_order() -> None:
    df = pl.DataFrame(
        {
            "case": ["z", "z", "a", "a"],
            "alt": [0, 1, 0, 1],
            "choice": [True, False, False, True],
            "x": [0.0, 1.0, 0.0, 1.0],
            "weight": [2.0, 2.0, 7.0, 7.0],
        }
    )
    model = ConditionalLogit()
    parsed = model._ingest_data(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="case",
        formula=None,
        utility_formula=None,
        membership_formula=None,
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=None,
        dems_data=None,
    )

    vector_weights = model._resolve_case_weights(
        df,
        parsed,
        [2.0, 7.0],
        cases_col="case",
        panels_col="case",
    )
    column_weights = model._resolve_case_weights(
        df,
        parsed,
        "weight",
        cases_col="case",
        panels_col="case",
    )
    mapping_weights = model._resolve_case_weights(
        df,
        parsed,
        {"z": 2.0, "a": 7.0},
        cases_col="case",
        panels_col="case",
    )

    assert np.array_equal(vector_weights, [7.0, 2.0])
    assert np.array_equal(column_weights, vector_weights)
    assert np.array_equal(mapping_weights, vector_weights)


def test_repeated_case_ids_require_joint_weight_mapping_keys() -> None:
    df = _choice_rows([2, 1])
    model = ConditionalLogit()
    parsed = model._ingest_data(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        formula=None,
        utility_formula=None,
        membership_formula=None,
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=None,
        dems_data=None,
    )

    with pytest.raises(ValueError, match="repeat across panels"):
        model._resolve_case_weights(
            df,
            parsed,
            {1: 1.0, 2: 2.0},
            cases_col="case",
            panels_col="panel",
        )


def test_weighted_cl_fit_is_invariant_to_case_label_sort_order() -> None:
    case_labels = [30, 10, 20, 40, 50, 5]
    relabeled_cases = list(range(len(case_labels)))
    case_weights = [1.0, 4.0, 2.0, 3.0, 1.5, 2.5]
    slopes = [0.2, 1.0, -0.7, 1.8, -1.2, 0.5]
    choices = [1, 1, 0, 1, 0, 0]

    def frame(labels: list[int]) -> pl.DataFrame:
        rows = []
        for label, slope, chosen in zip(labels, slopes, choices):
            for alt in [0, 1]:
                rows.append(
                    {
                        "case": label,
                        "alt": alt,
                        "choice": alt == chosen,
                        "x": float(alt * slope),
                    }
                )
        return pl.DataFrame(rows)

    def fit(data: pl.DataFrame) -> float:
        result = ConditionalLogit().fit(
            data,
            alts_col="alt",
            cases_col="case",
            choice_col="choice",
            case_varnames=["x"],
            weights=case_weights,
            optimization_options=OptimizationOptions(maxiter=30),
            inference=InferenceOptions(skip=True),
        )
        return float(result.coeff_[0])

    assert fit(frame(case_labels)) == pytest.approx(
        fit(frame(relabeled_cases)),
        abs=1e-10,
    )


def test_fitted_encoder_is_immutable_and_public_loglik_reuses_it() -> None:
    df = _choice_rows()
    model = LatentClassConditionalLogit(num_classes=2)
    result = model.fit(
        df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        fit_options=FitOptions(max_em_iter=1, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=2),
        inference=InferenceOptions(skip=True),
    )

    assert result.loglik(df) == pytest.approx(
        float(result.em_res.unconditional_loglik), abs=1e-8
    )
    panel_scores = result.loglik(df, per_panel=True)
    assert isinstance(panel_scores, pl.DataFrame)
    assert panel_scores.height == df["panel"].n_unique()
    assert panel_scores["log_likelihood"].sum() == pytest.approx(result.loglik(df))

    with pytest.raises(RuntimeError, match="already has a fitted encoder"):
        model._ingest_data(
            data=df,
            alts_col="alt",
            cases_col="case",
            panels_col="panel",
            formula=None,
            utility_formula=None,
            membership_formula=None,
            choice_col="choice",
            case_varnames=["x"],
            dem_varnames=None,
            dems_data=None,
        )
    assert model._encoder is not None
    with pytest.raises(AttributeError, match="immutable"):
        model._encoder.case_varnames = ["changed"]


def test_held_out_formula_scoring_keeps_training_categorical_columns() -> None:
    def categorical_frame(panels: range, nonbase_brand: str) -> pl.DataFrame:
        rows = []
        for panel in panels:
            for case in [0, 1]:
                for alt, brand in [(0, "common"), (1, nonbase_brand)]:
                    rows.append(
                        {
                            "panel": panel,
                            "case": case,
                            "alt": alt,
                            "choice": alt == ((panel + case) % 2),
                            "x": float(alt),
                            "brand": brand,
                        }
                    )
        return pl.DataFrame(rows)

    train = categorical_frame(range(4), "train_only")
    test = categorical_frame(range(10, 12), "test_only")
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ x + C(brand)",
        classes=2,
    )
    result = LatentClassConditionalLogit(spec=spec).fit(
        train,
        fit_options=FitOptions(max_em_iter=1, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=2),
        inference=InferenceOptions(skip=True),
    )
    training_columns = list(result.model.case_varnames)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        held_out_ll = result.loglik(test)

    assert np.isfinite(held_out_ll)
    assert any("categories outside" in str(item.message) for item in caught_warnings)
    assert result.model.case_varnames == training_columns
    assert any("train_only" in column for column in training_columns)
    assert all("test_only" not in column for column in training_columns)


def test_cv_reports_fold_failures_and_uses_per_panel_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _choice_rows()
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility=["cost", "x"],
        classes=2,
        constraints={"cost": NegativeCoefficient(min_abs=0.123)},
    )
    seen_options: list[tuple[int, bool, float]] = []

    class FakeResult:
        converged = True

        def __init__(self, model: LatentClassConditionalLogit) -> None:
            self.model = model

        def loglik(
            self,
            test_data: object,
            dems_data: object | None = None,
            *,
            per_panel: bool = False,
        ) -> pl.DataFrame:
            del dems_data
            parsed = self.model._transform_data(test_data, require_choice=True)
            panels = pl.Series(parsed.original_panels).unique(maintain_order=True)
            if 40 in panels:
                raise RuntimeError("deliberate held-out failure")
            assert per_panel
            return pl.DataFrame(
                {"panel": panels, "log_likelihood": [-1.0] * len(panels)}
            )

    def fake_fit(
        self: LatentClassConditionalLogit,
        data: Any,
        **kwargs: Any,
    ) -> FakeResult:
        assert self.spec is not None
        fit_options = kwargs["fit_options"]
        inference = kwargs["inference"]
        seen_options.append(
            (
                fit_options.starts,
                inference.skip_std_errs,
                self.spec.numeraire_min_abs,
            )
        )
        parsed = self._ingest_data(
            data=data,
            alts_col=self.spec.ids.alt,
            cases_col=self.spec.ids.case,
            panels_col=self.spec.ids.panel,
            formula=self.spec.formula,
            utility_formula=self.spec.utility_formula,
            membership_formula=self.spec.membership_formula,
            choice_col=self.spec.ids.choice,
            case_varnames=self.spec.utility,
            dem_varnames=self.spec.membership,
            dems_data=kwargs.get("dems_data"),
        )
        self._pre_fit(
            parsed.case_varnames,
            parsed.dem_varnames,
            self.spec.numeraire,
        )
        return FakeResult(self)

    monkeypatch.setattr(LatentClassConditionalLogit, "fit", fake_fit)

    cv = cv_optimal_classes(
        df,
        spec=spec,
        num_classes_list=[2],
        folds=2,
        fit_options=FitOptions(starts=3, max_em_iter=1, num_devices=1),
    )
    row = cv.row(0, named=True)

    assert row["Successful_Folds"] == 1
    assert row["Failed_Folds"] == 1
    assert np.isnan(row["Avg_OOS_LL"])
    assert row["Avg_Successful_OOS_LL"] == pytest.approx(-1.0)
    assert len(row["Fold_OOS_LL"]) == 2
    assert any(error is not None for error in row["Fold_Errors"])
    assert seen_options == [(3, True, 0.123), (3, True, 0.123)]


def test_quantile_and_custom_wtp_partitions_have_numeric_bin_order() -> None:
    df = pl.DataFrame(
        {
            "panel_idx": [0, 1, 2, 3, 4],
            "score": [30.0, 50.0, 20.0, 40.0, 10.0],
        }
    )
    quintiles = _apply_wtp_partition(
        df,
        WTPRequest("x", "score", PartitionType.QUINTILES),
    ).sort("_partition_order")
    assert quintiles["Partition"].cast(pl.String).to_list() == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]

    custom = _apply_wtp_partition(
        df,
        WTPRequest("x", "score", PartitionType.CUSTOM_BREAKS, bins=[20.0, 40.0]),
    ).sort("_partition_order")
    assert custom["_partition_order"].to_list() == [0, 0, 1, 1, 2]


def test_param_packing_owns_structural_map_and_flat_layout() -> None:
    packing = ParamPacking(
        num_alt_vars=2,
        num_classes=3,
        num_dem_vars=0,
        numeraire_idx=0,
        numeraire_min_abs=0.01,
    )
    latent_betas = jnp.arange(6.0).reshape(2, 3)
    shares = jnp.array([0.2, 0.3, 0.5])
    flat = packing.pack(latent_betas, None, shares)
    unpacked_betas, unpacked_thetas = packing.unpack(flat)

    assert jnp.array_equal(unpacked_betas, latent_betas)
    assert unpacked_thetas.shape == (1, 2)
    assert jnp.allclose(
        packing.class_probs(unpacked_thetas, None, 2)[0],
        shares,
    )
    assert jnp.all(packing.to_structural(latent_betas)[0] <= -0.01)


def test_array_demographics_can_be_validated_and_reordered_by_panel_id() -> None:
    parsed = _parsed_prediction_arrays(
        X=np.array([[0.0], [1.0], [0.0], [1.0]]),
        dems=np.array([[2.0], [1.0]]),
        alts=np.array([0, 1, 0, 1]),
        cases=np.array([1, 1, 1, 1]),
        panels=np.array([10, 10, 20, 20]),
        dem_panel_ids=np.array([20, 10]),
        case_varnames=["x"],
        dem_varnames=["income"],
    )

    assert parsed.dems is not None
    assert np.array_equal(parsed.dems, [[1.0], [2.0]])

    with pytest.raises(ValueError, match="match the unique prediction panels"):
        _parsed_prediction_arrays(
            X=np.array([[0.0], [1.0], [0.0], [1.0]]),
            dems=np.array([[2.0], [1.0]]),
            alts=np.array([0, 1, 0, 1]),
            cases=np.array([1, 1, 1, 1]),
            panels=np.array([10, 10, 20, 20]),
            dem_panel_ids=np.array([20, 30]),
            case_varnames=["x"],
            dem_varnames=["income"],
        )


def test_class_membership_without_required_demographics_fails_clearly() -> None:
    with pytest.raises(ValueError, match="dems is required"):
        _class_membership_probs(jnp.zeros((2, 1)), None, 3)


def test_numeraire_pvalue_is_suppressed_and_panel_bic_is_consistent() -> None:
    df = _choice_rows()
    result = ConditionalLogit(numeraire="cost").fit(
        df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["cost"],
        optimization_options=OptimizationOptions(maxiter=2),
        inference=InferenceOptions(skip=True),
    )

    assert np.isnan(result.coefficient_table()["p_value"][0])
    expected_bic = (
        np.log(df["panel"].n_unique()) * len(result.coeff_) - 2 * result.loglikelihood
    )
    assert result.bic == pytest.approx(float(expected_bic))


def test_inference_options_are_canonical_and_conflicts_are_rejected() -> None:
    assert not InferenceOptions(robust=False).robust
    assert InferenceOptions(covariance="clustered").robust

    with pytest.raises(ValueError, match="inference or error_config"):
        resolve_inference_options(
            InferenceOptions(),
            ErrorConfig(),
        )


def test_spec_resolver_applies_explicit_interface_overrides_once() -> None:
    base = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ x",
        membership_formula="~ income",
        classes=3,
    )

    resolved = resolve_lcl_spec(spec=base, case_varnames=["x", "z"])

    assert resolved.utility == ["x", "z"]
    assert resolved.utility_formula is None
    assert resolved.membership_formula == "~ income"
    assert resolved.classes == 3
