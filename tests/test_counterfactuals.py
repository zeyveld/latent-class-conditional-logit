import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from lcl import (
    FutureSimulationConfig,
    InferenceOptions,
    LatentClassConditionalLogit,
    OptimizationOptions,
    simulate_future_choice_sets,
)
from lcl._struct import FitOptions
from lcl._welfare import _mix_welfare_components, _train_welfare_by_class


def _panel_choice_data() -> pl.DataFrame:
    rows = []
    for panel in [10, 20]:
        for case in range(4):
            for alt in [0, 1]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "day": case * 10,
                        "intervention": case == 2,
                        "alt": alt,
                        "choice": alt == case % 2,
                        "price": float(1 + case + alt),
                    }
                )
    return pl.DataFrame(rows)


def _welfare_fit_data() -> pl.DataFrame:
    rows = []
    for panel_idx, panel in enumerate([101, 205, 309, 415]):
        for case in [1, 2, 3]:
            for alt in [0, 1]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == ((panel_idx + case) % 2),
                        "cost": float(1.0 + alt + 0.1 * case),
                        "quality": float(alt + 0.25 * case),
                    }
                )
    return pl.DataFrame(rows)


def test_future_worlds_use_separate_information_sets_and_whole_choice_sets() -> None:
    data = _panel_choice_data()
    config = FutureSimulationConfig(
        num_draws=12, horizon_days=20, max_trips_per_panel=4, seed=7
    )

    worlds = simulate_future_choice_sets(
        data,
        panel_col="panel",
        case_col="case",
        time_col="day",
        intervention_col="intervention",
        choice_col="choice",
        config=config,
    )

    assert set(worlds.anticipated["source_case"].to_list()) <= {0, 1, 2}
    assert 3 in worlds.realized["source_case"].to_list()
    assert "choice" not in worlds.anticipated.columns
    assert not worlds.anticipated["intervention"].any()
    assert set(worlds.anticipated.group_by("case").len()["len"].to_list()) == {2}
    assert worlds.anticipated.select("panel", "simulation_round").unique().height == 24
    assert worlds.anticipated["day"].min() >= 20
    assert worlds.anticipated["day"].max() <= 40
    assert worlds.trip_summary["simulated_trips_per_draw"].to_list() == [2, 2, 2, 2]

    repeated = simulate_future_choice_sets(
        data,
        panel_col="panel",
        case_col="case",
        time_col="day",
        intervention_col="intervention",
        choice_col="choice",
        config=config,
    )
    assert worlds.anticipated.equals(repeated.anticipated)
    assert worlds.realized.equals(repeated.realized)


def test_future_worlds_require_one_intervention_choice_set_per_panel() -> None:
    data = _panel_choice_data().with_columns(intervention=pl.lit(False))

    with pytest.raises(ValueError, match="exactly one choice set per panel"):
        simulate_future_choice_sets(
            data,
            panel_col="panel",
            case_col="case",
            time_col="day",
            intervention_col="intervention",
        )


def test_train_welfare_matches_closed_form_logit_expression() -> None:
    anticipated_X = jnp.array([[1.0, 2.0], [1.0, 1.0]])
    experienced_X = jnp.array([[1.0, 0.0], [1.0, 1.0]])
    betas = jnp.array([[-1.0], [1.0]])

    welfare = _train_welfare_by_class(
        anticipated_X,
        experienced_X,
        betas,
        jnp.array([0, 0], dtype=jnp.uint32),
        1,
    )

    probability_alt_zero = np.exp(1.0) / (np.exp(1.0) + 1.0)
    anticipated = np.log(np.exp(1.0) + 1.0)
    experienced = anticipated - 2.0 * probability_alt_zero
    perfect_foresight = np.log(np.exp(-1.0) + 1.0)
    assert np.allclose(welfare["anticipated_surplus"], anticipated)
    assert np.allclose(welfare["experience_effect"], -2.0 * probability_alt_zero)
    assert np.allclose(welfare["experienced_surplus"], experienced)
    assert np.allclose(welfare["perfect_foresight_surplus"], perfect_foresight)
    assert np.allclose(
        welfare["foreknowledge_loss"], perfect_foresight - experienced
    )


def test_dollar_welfare_divides_within_class_before_marginalizing() -> None:
    welfare_by_class = {
        "experienced_surplus": jnp.array([[2.0, 2.0]]),
    }
    mixed = _mix_welfare_components(
        welfare_by_class,
        class_probs_by_case=jnp.array([[0.5, 0.5]]),
        marginal_utility_income=jnp.array([1.0, 2.0]),
    )

    assert np.allclose(mixed["experienced_surplus_utils"], 2.0)
    assert np.allclose(mixed["experienced_surplus_dollars"], 1.5)


def test_prediction_reports_train_welfare_and_acceptance_probability() -> None:
    data = _welfare_fit_data()
    model = LatentClassConditionalLogit(num_classes=2, numeraire="cost")
    results = model.fit(
        data=data,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["cost", "quality"],
        fit_options=FitOptions(max_em_iter=1, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=2),
        inference=InferenceOptions(skip=True),
    )
    experienced = data.with_columns(
        pl.when(pl.col("alt") == 1)
        .then(pl.col("quality") - 0.5)
        .otherwise(pl.col("quality"))
        .alias("quality")
    )

    prediction = results.predict(data=data, experienced_data=experienced)
    surplus = prediction.surplus

    expected_columns = {
        "anticipated_surplus_utils",
        "experience_effect_utils",
        "experienced_surplus_utils",
        "perfect_foresight_surplus_utils",
        "foreknowledge_loss_utils",
        "anticipated_surplus_dollars",
        "experience_effect_dollars",
        "experienced_surplus_dollars",
        "perfect_foresight_surplus_dollars",
        "foreknowledge_loss_dollars",
        "surplus",
    }
    assert expected_columns <= set(surplus.columns)
    assert np.allclose(
        surplus["experienced_surplus_utils"],
        surplus["anticipated_surplus_utils"] + surplus["experience_effect_utils"],
    )
    assert np.allclose(
        surplus["foreknowledge_loss_utils"],
        surplus["perfect_foresight_surplus_utils"]
        - surplus["experienced_surplus_utils"],
    )
    assert np.allclose(surplus["surplus"], surplus["experienced_surplus_dollars"])
    assert np.all(surplus["foreknowledge_loss_utils"].to_numpy() >= -1e-10)

    acceptance = prediction.acceptance_probability(1)
    manual = (
        prediction.predicted_probs.filter(pl.col("alts") == 1)
        .sort(["panels", "cases"])["choice_probs"]
        .to_numpy()
    )
    assert np.allclose(
        acceptance.sort(["panels", "cases"])["acceptance_probability"], manual
    )


def test_prediction_rejects_misaligned_experienced_choice_sets() -> None:
    data = _welfare_fit_data()
    model = LatentClassConditionalLogit(num_classes=2, numeraire="cost")
    results = model.fit(
        data=data,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["cost", "quality"],
        fit_options=FitOptions(max_em_iter=1, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=1),
        inference=InferenceOptions(skip=True),
    )

    with pytest.raises(ValueError, match="exactly the same panels, cases"):
        results.predict(data=data, experienced_data=data.filter(pl.col("alt") == 0))
