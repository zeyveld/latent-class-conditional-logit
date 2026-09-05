from __future__ import annotations

import importlib
import inspect
import itertools
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

import lcl
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
from lcl._cross_validation import _annotate_cv_selection, _resolve_panel_folds
from lcl._params import ParamPacking
from lcl._prediction import _apply_wtp_partition
from lcl._results import _parsed_prediction_arrays
from lcl.options import PartitionType, WTPRequest
from lcl._struct import EMStepDiagnostics, EMVars
from lcl.conditional_logit import ConditionalLogit
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
from lcl.spec import resolve_lcl_spec

_lcl_model_module = importlib.import_module("lcl.latent_class_conditional_logit")


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
                        "x": float(alt * (1.0 + 0.01 * panel + 0.2 * case)),
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


def _fake_em_state(data: Any, num_classes: int, loglik: float) -> EMVars:
    """Build a minimal EM state for stopping-rule tests."""
    assert data.num_panels is not None
    betas = jnp.zeros((data.num_alt_vars, num_classes))
    return EMVars(
        latent_betas=betas,
        structural_betas=betas,
        thetas=None,
        shares=jnp.full(num_classes, 1.0 / num_classes),
        unconditional_loglik=jnp.array(loglik),
        class_probs_by_panel=jnp.full(
            (data.num_panels, num_classes), 1.0 / num_classes
        ),
    )


@pytest.mark.parametrize("rate", [0.5, 0.9], ids=["fast-rate", "slow-rate"])
def test_em_stops_on_the_aitken_extrapolated_gap(
    monkeypatch: pytest.MonkeyPatch, rate: float
) -> None:
    """The stopping rule targets the remaining ascent, not one step's change.

    A geometric log-likelihood sequence with rate ``r`` still has
    ``d * r / (1 - r)`` to gain after an increment of ``d``.  At ``r = 0.9`` that
    tail is nine times the increment, so a rule that looked at the increment
    alone would stop while a ninth of the ascent was still outstanding.
    """
    em_tol = 1e-3
    max_em_iter = 300
    increments = [rate**step for step in range(max_em_iter + 5)]
    logliks = list(itertools.accumulate(increments, initial=-99.0))[1:]
    remaining = list(logliks)

    def fake_starting_values(
        _diff: Any, data: Any, num_classes: int, *_args: Any
    ) -> EMVars:
        return _fake_em_state(data, num_classes, -99.0)

    def fake_em_step(
        em_vars: EMVars, _diff: Any, _data: Any, *_args: Any
    ) -> tuple[EMVars, EMStepDiagnostics]:
        return (
            em_vars._replace(unconditional_loglik=jnp.array(remaining.pop(0))),
            EMStepDiagnostics(
                beta_newton_error=jnp.zeros(2),
                membership_newton_error=jnp.array(0.0),
            ),
        )

    monkeypatch.setattr(_lcl_model_module, "_get_starting_vals", fake_starting_values)
    monkeypatch.setattr(_lcl_model_module, "_em_step", fake_em_step)

    result = LatentClassConditionalLogit(num_classes=2).fit(
        _choice_rows(),
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        fit_options=FitOptions(
            max_em_iter=max_em_iter, em_tol=em_tol, polish=False, num_devices=1
        ),
        inference=InferenceOptions(skip=True),
    )

    assert result.em_criterion_met
    assert result.total_recursions < max_em_iter
    threshold = em_tol * result.data.num_panels
    tail_factor = rate / (1.0 - rate)
    realized = result.em_history_["loglik"].to_list()
    steps = [after - before for before, after in zip(realized, realized[1:])]

    # It stopped at the first iteration whose extrapolated tail cleared the
    # tolerance, and not before.
    assert steps[-1] * tail_factor <= threshold
    assert steps[-2] * tail_factor > threshold
    naive_stop = next(
        index for index, step in enumerate(steps) if step <= threshold
    )
    if rate > 0.5:
        # A naive log-likelihood-change rule would have stopped strictly earlier,
        # with the geometric tail still outstanding.
        assert naive_stop < len(steps) - 1
    else:
        # At r = 0.5 the tail equals the increment, so the two rules coincide.
        assert naive_stop == len(steps) - 1


def test_em_respects_the_iteration_cap_when_the_criterion_is_never_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crawling sequence runs the full budget and reports the criterion unmet."""
    max_em_iter = 9
    remaining = [-99.0 + 0.5 * step for step in range(1, max_em_iter + 5)]

    def fake_starting_values(
        _diff: Any, data: Any, num_classes: int, *_args: Any
    ) -> EMVars:
        return _fake_em_state(data, num_classes, -99.0)

    def fake_em_step(
        em_vars: EMVars, _diff: Any, _data: Any, *_args: Any
    ) -> tuple[EMVars, EMStepDiagnostics]:
        return (
            em_vars._replace(unconditional_loglik=jnp.array(remaining.pop(0))),
            EMStepDiagnostics(
                beta_newton_error=jnp.zeros(2),
                membership_newton_error=jnp.array(0.0),
            ),
        )

    monkeypatch.setattr(_lcl_model_module, "_get_starting_vals", fake_starting_values)
    monkeypatch.setattr(_lcl_model_module, "_em_step", fake_em_step)

    result = LatentClassConditionalLogit(num_classes=2).fit(
        _choice_rows(),
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        fit_options=FitOptions(
            max_em_iter=max_em_iter, em_tol=1e-8, polish=False, num_devices=1
        ),
        inference=InferenceOptions(skip=True),
    )
    assert result.total_recursions == max_em_iter
    assert result.em_criterion_met is False
    assert result.em_history_.height == max_em_iter


def test_converged_flag_tracks_the_observed_data_score() -> None:
    """A stationary point, not a small log-likelihood change, defines convergence."""
    df = _choice_rows()
    result = LatentClassConditionalLogit(num_classes=2).fit(
        df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        fit_options=FitOptions(max_em_iter=2, polish=False, num_devices=1),
        inference=InferenceOptions(skip=True),
    )
    assert result.converged is (result.observed_score_max <= result.score_tol)
    diagnostics = result.diagnostics().to_frame()
    score_row = diagnostics.filter(pl.col("check") == "observed_score_max")
    assert (score_row["status"][0] == "ok") is result.converged


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
                            "x": float(alt * (1.0 + 0.1 * panel + 0.2 * case)),
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

    with pytest.raises(ValueError, match="categories outside"):
        result.loglik(test)
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
                inference.skip,
                self.spec.numeraire_min_abs,
            )
        )
        parsed = self._ingest_data(
            data=data,
            alts_col=self.spec.ids.alt,
            cases_col=self.spec.ids.case,
            panels_col=self.spec.ids.panel,
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


def test_user_cv_folds_are_validated_as_a_panel_partition() -> None:
    """Explicit fold maps preserve panel identity and reject leakage."""
    panels = np.array([10, 20, 30, 40])
    mapped = _resolve_panel_folds({10: "a", 20: "b", 30: "a", 40: "b"}, panels, seed=9)
    assert [group.tolist() for group in mapped] == [[10, 30], [20, 40]]
    explicit = _resolve_panel_folds([[40, 10], [20, 30]], panels, seed=9)
    assert [group.tolist() for group in explicit] == [[40, 10], [20, 30]]

    with pytest.raises(ValueError, match="more than one"):
        _resolve_panel_folds([[10, 20], [20, 30, 40]], panels, seed=9)
    with pytest.raises(ValueError, match="cover exactly"):
        _resolve_panel_folds([[10, 20], [30]], panels, seed=9)


def test_cv_one_se_rule_selects_the_smallest_eligible_class_count() -> None:
    """One-SE selection uses the best model's uncertainty on the LL scale."""
    selected = _annotate_cv_selection(
        pl.DataFrame(
            {
                "Num_Classes": [2, 3, 4],
                "Avg_OOS_LL": [-1.00, -0.92, -0.90],
                "SE_OOS_LL": [0.03, 0.04, 0.12],
            }
        )
    )
    assert selected.filter(pl.col("Selected_Best"))["Num_Classes"].item() == 4
    assert selected.filter(pl.col("Selected_One_SE"))["Num_Classes"].item() == 2


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

    tied = _apply_wtp_partition(
        pl.DataFrame({"panel_idx": range(8), "score": [1, 1, 1, 2, 2, 3, 3, 3]}),
        WTPRequest("x", "score", PartitionType.QUINTILES),
    )
    assert tied.group_by("score").agg(pl.col("Partition").n_unique())[
        "Partition"
    ].to_list() == [1, 1, 1]
    assert all("of" in label for label in tied["Partition"].unique())


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


def test_inference_options_are_canonical() -> None:
    assert InferenceOptions(covariance="none").covariance == "unadjusted"
    assert InferenceOptions(covariance="robust").covariance == "robust"
    with pytest.raises(ValueError, match="InferenceOptions.covariance"):
        InferenceOptions(covariance="invalid")


def test_legacy_public_api_patterns_are_removed() -> None:
    for name in ("EMAlgConfig", "MleConfig", "ErrorConfig"):
        assert not hasattr(lcl, name)

    lcl_fit_params = inspect.signature(LatentClassConditionalLogit.fit).parameters
    cl_fit_params = inspect.signature(ConditionalLogit.fit).parameters
    cv_params = inspect.signature(cv_optimal_classes).parameters
    for params in (lcl_fit_params, cl_fit_params, cv_params):
        assert "formula" not in params
        assert "mle_config" not in params
        assert "error_config" not in params
    assert "em_alg_config" not in lcl_fit_params
    assert "em_alg_config" not in cv_params


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
