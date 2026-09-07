"""Regression coverage for option routing, input contracts, and package metadata."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import importlib
import warnings

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

import lcl
from lcl._encoding import ChoiceDataEncoder
from lcl.spec import resolve_lcl_spec
from tests.test_orchestration import _choice_rows


@pytest.fixture(scope="module")
def frame():
    return _choice_rows().with_columns(
        (pl.col("panel") / 10).alias("income"),
        (1 + pl.col("panel") / 20).alias("weight"),
    )


@pytest.fixture(scope="module")
def spec():
    return lcl.LCLSpec(
        ids=lcl.ChoiceIds("alt", "case", "panel", "choice"),
        utility=["x", "cost"],
        classes=2,
        constraints={"cost": lcl.NegativeCoefficient(min_abs=0.01, warn_below=100.0)},
    )


@pytest.fixture(scope="module")
def quick_options():
    return lcl.Options(
        fit=lcl.FitOptions(max_em_iter=1, polish=False, num_devices=1),
        optimization=lcl.OptimizationOptions(maxiter=2),
        inference=lcl.InferenceOptions(skip=True, cluster="absent_cluster"),
        diagnostics=lcl.DiagnosticsOptions(check_collinearity=False),
    )


@pytest.fixture(scope="module")
def latent(frame, spec, quick_options):
    return lcl.fit(frame, spec, options=quick_options)


@pytest.fixture(scope="module")
def conditional(frame, quick_options):
    return lcl.ConditionalLogit(numeraire="cost").fit(
        frame,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x", "cost"],
        weights="weight",
        options=quick_options,
    )


def test_optimizer_alias_does_not_restore_an_old_tolerance_on_replace():
    original = lcl.OptimizationOptions(newton_decrement_tol=0.2)
    with warnings.catch_warnings(record=True) as caught:
        reset = replace(original, newton_decrement_tol=1e-5)
        copied = replace(original, maxiter=4)
    assert not caught
    assert reset.newton_decrement_tol == reset.gradient_tol == 1e-5
    assert copied.newton_decrement_tol == 0.2
    with pytest.warns(DeprecationWarning):
        alias = lcl.OptimizationOptions(gradient_tol=0.2)
    assert alias == original and hash(alias) == hash(original)
    with pytest.warns(DeprecationWarning):
        explicit_default = lcl.OptimizationOptions(
            newton_decrement_tol=1e-5, gradient_tol=0.2
        )
    assert explicit_default.newton_decrement_tol == 1e-5


def test_constructor_overrides_survive_spec_resolution(frame, spec, quick_options):
    model = lcl.LatentClassConditionalLogit(
        spec=spec, num_classes=3, numeraire_min_abs=0.4
    )
    result = model.fit(frame, options=quick_options)
    assert result.model.spec.classes == 3
    assert result._param_packing.numeraire_min_abs == 0.4
    assert np.all(-np.asarray(result.em_res.structural_betas)[1] >= 0.4)
    assert result.model.spec.negative_constraint.warn_below == 100.0
    inherited = lcl.LatentClassConditionalLogit(spec=spec)
    assert inherited.num_classes == 2 and inherited.numeraire_min_abs == 0.01
    assert lcl.LatentClassConditionalLogit().num_classes == 5


def test_numeraire_is_added_to_an_explicitly_empty_constraint_collection(spec):
    empty = replace(spec, constraints=[])
    resolved = resolve_lcl_spec(spec=empty, numeraire="cost", numeraire_min_abs=0.3)
    assert resolved.numeraire == "cost" and resolved.numeraire_min_abs == 0.3


def test_refit_rejection_preserves_original_result_spec(latent, frame):
    original = latent.model.spec
    with pytest.raises(RuntimeError, match="already has a fitted encoder"):
        latent.model.fit(frame, case_varnames=["cost"])
    assert latent.model.spec is original


def test_skipped_inference_does_not_require_a_cluster_column(latent, conditional):
    assert not latent.covariance_available
    assert not conditional.covariance_available


def test_fitted_results_copy_mutable_option_sections(
    latent, conditional, quick_options
):
    for result in (latent, conditional):
        assert result.inference == quick_options.inference
        assert result.inference is not quick_options.inference
        assert result.diagnostics_config is not quick_options.diagnostics
    resolved = importlib.import_module("lcl.options")._resolve_options(quick_options)
    quick_options.inference.skip = False
    quick_options.diagnostics.check_collinearity = True
    try:
        assert resolved.inference.skip
        assert not resolved.diagnostics.check_collinearity
        for result in (latent, conditional):
            assert (
                result.inference.skip
                and not result.diagnostics_config.check_collinearity
            )
    finally:
        quick_options.inference.skip = True
        quick_options.diagnostics.check_collinearity = False


def test_constraint_specific_warning_threshold_is_honored(latent):
    row = latent.diagnostics().to_frame().filter(pl.col("check") == "min_abs_numeraire")
    assert row.item(0, "status") == "warning"


@pytest.mark.parametrize("aggregate", [True, False])
def test_cv_forwards_all_option_sections(
    monkeypatch, frame, spec, quick_options, aggregate
):
    module = importlib.import_module("lcl._cross_validation")
    seen = []

    def fake_fit(self, data, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            converged=True,
            loglik=lambda data, **kw: pl.DataFrame(
                {
                    "panel": data["panel"].unique(),
                    "log_likelihood": [-2.0] * data["panel"].n_unique(),
                }
            ),
        )

    monkeypatch.setattr(module.LatentClassConditionalLogit, "fit", fake_fit)
    kwargs = (
        {"options": quick_options}
        if aggregate
        else {
            "fit_options": quick_options.fit,
            "optimization_options": quick_options.optimization,
            "inference": quick_options.inference,
            "diagnostics": quick_options.diagnostics,
        }
    )
    result = lcl.cv_optimal_classes(
        frame, spec=spec, num_classes_list=[2], folds=2, **kwargs
    )
    assert result.item(0, "Failed_Folds") == 0
    assert len(seen) == 2
    for call in seen:
        assert call["diagnostics"] == quick_options.diagnostics
        assert call["fit_options"] == quick_options.fit
        assert call["optimization_options"] == quick_options.optimization
        assert call["inference"] == quick_options.inference


def test_cv_treats_nonfinite_scores_as_reported_fold_failures(monkeypatch, frame, spec):
    module = importlib.import_module("lcl._cross_validation")
    monkeypatch.setattr(
        module.LatentClassConditionalLogit,
        "fit",
        lambda *args, **kw: SimpleNamespace(
            converged=True,
            loglik=lambda data, **kw: pl.DataFrame(
                {"log_likelihood": [float("inf")] * data["panel"].n_unique()}
            ),
        ),
    )
    row = lcl.cv_optimal_classes(frame, spec=spec, num_classes_list=[2], folds=2).row(
        0, named=True
    )
    assert row["Failed_Folds"] == 2
    assert all("finite" in error for error in row["Fold_Errors"])
    assert not row["Selected_Best"] and np.isnan(row["Avg_OOS_LL"])


@pytest.mark.parametrize("formula", [False, True])
@pytest.mark.parametrize("invalid", [-1.0, 2.0, 0.5, float("nan"), None])
def test_invalid_choices_are_rejected_before_boolean_conversion(
    frame, formula, invalid
):
    bad = frame.with_columns(
        pl.when(pl.col("choice"))
        .then(pl.lit(invalid, dtype=pl.Float64))
        .otherwise(0.0)
        .alias("choice")
    )
    encoder = ChoiceDataEncoder(
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        utility_formula="choice ~ x + cost" if formula else None,
        explicit_case_varnames=None if formula else ["x", "cost"],
    )
    with pytest.raises(ValueError, match="0/1|produced.*rows|Unable to evaluate"):
        encoder.fit_transform(bad)


def test_formula_and_explicit_outcomes_cannot_disagree(frame):
    encoder = ChoiceDataEncoder(
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="other_choice",
        utility_formula="choice ~ x + cost",
    )
    with pytest.raises(ValueError, match="conflicts"):
        encoder.fit_transform(
            frame.with_columns((~pl.col("choice")).alias("other_choice"))
        )


@pytest.mark.parametrize("membership", [None, "explicit", "formula"])
def test_external_demographics_reach_utility_with_every_membership_interface(
    frame, membership
):
    def encoder():
        return ChoiceDataEncoder(
            alts_col="alt",
            cases_col="case",
            panels_col="panel",
            choice_col="choice",
            utility_formula="choice ~ cost + x:income",
            membership_formula="~ income" if membership == "formula" else None,
            explicit_dem_varnames=["income"] if membership == "explicit" else None,
        )

    separate = frame.select("panel", "income").unique().reverse()
    expected = encoder().fit_transform(frame)
    actual_encoder = encoder()
    actual = actual_encoder.fit_transform(frame.drop("income"), dems_data=separate)
    prediction = actual_encoder.transform(frame.drop("income"), dems_data=separate)
    np.testing.assert_array_equal(expected.X, actual.X)
    np.testing.assert_array_equal(expected.X, prediction.X)
    if expected.dems is not None:
        np.testing.assert_array_equal(expected.dems, actual.dems)


def test_conflicting_design_sources_are_rejected(spec):
    with pytest.raises(ValueError, match="either utility"):
        replace(spec, utility_formula="choice ~ cost")
    with pytest.raises(ValueError, match="either utility"):
        resolve_lcl_spec(
            spec=spec, case_varnames=["cost"], utility_formula="choice ~ x"
        )
    with pytest.raises(ValueError, match="either membership"):
        replace(spec, membership=["income"], membership_formula="~ income")
    # A single override still intentionally replaces the corresponding base design.
    changed = resolve_lcl_spec(spec=spec, utility_formula="choice ~ x")
    assert changed.utility is None and changed.utility_formula == "choice ~ x"


@pytest.mark.parametrize(
    "kwargs",
    [{"X": np.ones((8, 2))}, {"dems": np.ones((4, 1))}, {"alts": np.arange(8)}],
)
def test_tabular_prediction_rejects_competing_arrays(latent, frame, kwargs):
    with pytest.raises(ValueError, match="either tabular"):
        latent.predict(frame, **kwargs)


def test_array_prediction_rejects_tabular_demographics(latent, frame):
    with pytest.raises(ValueError, match="dems_data requires tabular"):
        latent.predict(dems_data=frame)


def test_prediction_identifiers_are_validated_and_deprecated(conditional, frame):
    with pytest.raises(ValueError, match="must match"):
        conditional.predict(frame, panels_col="different")
    with pytest.warns(DeprecationWarning):
        actual = conditional.predict(frame, panels_col="panel")
    assert actual.predicted_probs.equals(conditional.predict(frame).predicted_probs)


def test_weighted_cl_scoring_uses_joint_ids_and_fit_alignment(conditional, frame):
    total = conditional.loglik(frame.reverse(), weights="weight")
    by_case = conditional.loglik(frame.reverse(), per_case=True, weights="weight")
    assert total == pytest.approx(float(conditional.loglikelihood))
    assert by_case["log_likelihood"].sum() == pytest.approx(total)
    assert by_case.select("panel", "case").unique().height == by_case.height
    assert conditional.loglik(frame) != pytest.approx(total)


def test_cl_tradeoff_uses_the_same_target_contract_as_wtp(conditional, frame):
    prediction = conditional.predict(frame)
    assert prediction.tradeoff("x", se="none").equals(prediction.wtp("x", se="none"))


def test_numpy_jax_and_sequence_prediction_inputs_are_equivalent(latent, frame):
    expected = latent.predict(frame).predicted_probs
    args = {
        "X": frame.select("x", "cost").to_numpy(),
        "alts": frame["alt"].to_numpy(),
        "cases": frame["case"].to_numpy(),
        "panels": frame["panel"].to_numpy(),
    }
    for convert in (np.asarray, jnp.asarray, lambda value: value.tolist()):
        actual = latent.predict(**{key: convert(value) for key, value in args.items()})
        assert actual.predicted_probs.equals(expected)
    for result in (latent,):
        actual = result.predict(frame, panel_weights=jnp.ones(4)).market_shares(
            se="none"
        )
        assert actual.equals(result.predict(frame).market_shares(se="none"))


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (lcl.OptimizationOptions, {"newton_decrement_tol": float("nan")}),
        (lcl.OptimizationOptions, {"max_step_norm": float("inf")}),
        (lcl.FitOptions, {"em_tol": float("nan")}),
        (lcl.DiagnosticsOptions, {"near_zero_numeraire_threshold": float("nan")}),
        (lcl.NegativeCoefficient, {"min_abs": float("inf")}),
    ],
)
def test_nonfinite_configuration_is_rejected(factory, kwargs):
    with pytest.raises(ValueError, match="finite"):
        factory(**kwargs)


def test_wtp_request_rejects_unused_or_ambiguous_partition_settings():
    with pytest.raises(ValueError, match="bins is only used"):
        lcl.WTPRequest("x", "income", "quintiles", bins=[1.0])
    with pytest.raises(ValueError, match="dummy_labels requires"):
        lcl.WTPRequest("x", "income", "categorical", dummy_labels=["A"])
    with pytest.raises(ValueError, match="must be distinct"):
        lcl.WTPRequest(
            "x", "income", "categorical", dummy_vars=["a"], dummy_labels=["base"]
        )
    with pytest.raises(ValueError, match="finite"):
        lcl.WTPRequest("x", "income", "custom_breaks", bins=[float("nan")])
    assert str(lcl.PartitionType.QUINTILES) == "quintiles"


def test_array_annotations_do_not_regress_to_unshaped_array_types():
    root = Path(lcl.__file__).parent
    bare_types = {
        "Array",
        "ArrayLike",
        "jax.Array",
        "jnp.ndarray",
        "onp.ndarray",
        "np.ndarray",
    }
    errors = []

    def inspect_annotation(node, path):
        if node is None:
            return
        if ast.unparse(node) in bare_types:
            errors.append(f"{path.name}:{node.lineno}: {ast.unparse(node)}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # jaxtyping dtype[backend, shape] is the explicit array contract.
            if node.value.id in {
                "Float64",
                "Float",
                "Real",
                "Int",
                "Integer",
                "UInt",
                "Bool",
                "Shaped",
            }:
                return
        for child in ast.iter_child_nodes(node):
            inspect_annotation(child, path)

    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.arg):
                inspect_annotation(node.annotation, path)
            elif isinstance(node, ast.AnnAssign):
                inspect_annotation(node.annotation, path)
            elif isinstance(node, ast.FunctionDef):
                inspect_annotation(node.returns, path)
    assert not errors, "\n".join(errors)


def test_model_convergence_is_synchronized_with_results(frame):
    model = lcl.ConditionalLogit()
    assert not model.convergence
    result = model.fit(
        frame,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        inference=lcl.InferenceOptions(skip=True),
    )
    assert result.converged
    assert model.convergence == result.converged


def test_completion_event_requires_successful_result_construction(
    monkeypatch, frame, spec, quick_options
):
    module = importlib.import_module("lcl.latent_class_conditional_logit")
    events = []

    def failed_results(**kwargs):
        raise RuntimeError("results construction failed")

    monkeypatch.setattr(module, "LCLResults", failed_results)
    with pytest.raises(RuntimeError, match="results construction failed"):
        lcl.fit(frame, spec, options=quick_options, progress_callback=events.append)
    assert events and all(event["event"] != "complete" for event in events)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (lcl.FitOptions, {"max_em_iter": 2.5}),
        (lcl.FitOptions, {"starts": True}),
        (lcl.OptimizationOptions, {"maxiter": 2.5}),
        (lcl.LatentClassConditionalLogit, {"num_classes": True}),
    ],
)
def test_iteration_and_class_counts_are_integers(factory, kwargs):
    with pytest.raises((ValueError, TypeError)):
        factory(**kwargs)


def test_numpy_string_ids_survive_array_prediction(latent, frame):
    renamed = frame.with_columns(
        pl.col("panel").cast(pl.String),
        pl.col("case").cast(pl.String),
        pl.col("alt").cast(pl.String),
    )
    actual = latent.predict(
        X=renamed.select("x", "cost").to_numpy(),
        alts=renamed["alt"].to_numpy(),
        cases=renamed["case"].to_numpy(),
        panels=renamed["panel"].to_numpy(),
    )
    assert actual.predicted_probs.equals(latent.predict(renamed).predicted_probs)


def test_boolean_array_weights_keep_their_existing_conversion_contract(
    latent, conditional, frame
):
    # Boolean masks are valid nonnegative weights after the normal float conversion.
    for result in (latent, conditional):
        actual = result.predict(frame, panel_weights=np.ones(4, dtype=bool))
        expected = result.predict(frame)
        assert actual.market_shares(se="none").equals(expected.market_shares(se="none"))
    assert conditional.loglik(
        frame, weights=np.ones(8, dtype=bool)
    ) == conditional.loglik(frame)
