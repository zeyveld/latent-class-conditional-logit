"""Independent economic identities and adversarial counterfactual designs."""

from copy import copy

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp, softmax

import lcl
from lcl import ChoiceIds, FitOptions, InferenceOptions, LCLSpec, NegativeCoefficient
from lcl import OptimizationOptions, PartitionType, WTPRequest, PastChoicesData
from lcl._presentation import format_class_coefficients, format_lcl_beta_summary
from lcl._presentation import format_membership_coefficients


@pytest.fixture(scope="module")
def known():
    """Install known parameters after encoding; isolate prediction from estimation."""
    rng = np.random.default_rng(17)
    rows = []
    for panel in range(12):
        income = panel / 6 - 1
        for case in range(3):
            for alt in range(3):
                rows.append(
                    dict(
                        panel=panel,
                        case=case,
                        alt=alt,
                        choice=alt == case,
                        price=float(rng.uniform(0.5, 3)),
                        quality=float(rng.uniform(0.2, 4)),
                        income=income,
                        segment="low" if income < 0 else "high",
                    )
                )
    data = pl.DataFrame(rows)
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality + quality:income",
        membership_formula="~ income",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    result = lcl.fit(
        data,
        spec,
        fit_options=FitOptions(max_em_iter=1, num_devices=1, polish=False),
        optimization_options=OptimizationOptions(maxiter=2),
        inference=InferenceOptions(skip=True),
    )
    beta = np.array([[-1.8, -0.5], [0.4, 1.6], [0.2, 0.8]])
    latent = beta.copy()
    latent[0] = np.log(np.expm1(-beta[0] - result.model.numeraire_min_abs))
    theta = np.array([[-0.2], [1.1]])
    prior = softmax(
        np.c_[np.zeros(12), theta[0, 0] + np.arange(12) / 6 * 1.1 - 1.1], axis=1
    )
    result.em_res = result.em_res._replace(
        structural_betas=jnp.asarray(beta),
        latent_betas=jnp.asarray(latent),
        thetas=jnp.asarray(theta),
        shares=jnp.asarray(prior.mean(axis=0)),
    )
    result.flat_params = result._pack_params()
    root = rng.normal(size=(result.num_params, result.num_params)) * 0.002
    result.latent_cov_matrix = jnp.asarray(root @ root.T)
    result.cov_matrix = result._structural_covariance(result.latent_cov_matrix)
    return data, result


def _independent_prior(result, income):
    theta = np.asarray(result.em_res.thetas)
    return softmax(
        np.c_[np.zeros(len(income)), theta[0, 0] + income * theta[1, 0]], axis=1
    )


def test_short_partial_history_matches_bayes_and_keeps_new_consumer_priors(known):
    data, result = known
    history = data.filter((pl.col("panel") == 7) & (pl.col("case") == 0))
    prediction = result.predict(data=data, past_choices=history.reverse())
    prior = _independent_prior(result, np.arange(12) / 6 - 1)
    x = history.select(
        "price", "quality", (pl.col("quality") * pl.col("income")).alias("interaction")
    ).to_numpy()
    utility = x @ np.asarray(result.em_res.structural_betas)
    log_likelihood = utility[0] - logsumexp(utility, axis=0)
    expected = prior.copy()
    expected[7] = softmax(np.log(prior[7]) + log_likelihood)
    np.testing.assert_allclose(prediction.class_probs_by_panel, expected, atol=1e-12)
    assert np.max(abs(expected[7] - prior[7])) > 0.01
    membership = prediction.class_membership()
    assert membership.filter(pl.col("panels") == 7)["history_cases"].to_list() == [1, 1]
    assert membership.filter(pl.col("panels") != 7)[
        "probability_source"
    ].unique().to_list() == ["prior"]
    # A short history need not identify the three fitted coefficients anew.
    assert np.isfinite(result.loglik(history))
    arrays = PastChoicesData(
        X=x,
        y=history["choice"].to_numpy(),
        alts=history["alt"].to_numpy(),
        cases=history["case"].to_numpy(),
        panels=history["panel"].to_numpy(),
    )
    from_arrays = result.predict(data=data, past_choices=arrays)
    np.testing.assert_allclose(from_arrays.class_probs_by_panel, expected, atol=1e-12)
    aggregate = prediction.market_shares(se="none").sort("alts")
    manual = (
        prediction.predicted_probs.group_by("alts")
        .agg(pl.col("choice_probs").sum())
        .sort("alts")
    )
    np.testing.assert_allclose(
        aggregate["market_share"], manual["choice_probs"].to_numpy() / 36
    )


def test_ragged_formula_elasticities_are_derivatives_of_market_demand(known):
    data, result = known
    data = data.filter(~((pl.col("panel") < 5) & (pl.col("alt") == 0)))
    prediction = result.predict(data=data)
    aggregate = prediction.aggregate_elasticities("quality", se="none")
    base = dict(
        prediction.market_shares(se="none").select("alts", "market_share").iter_rows()
    )
    step = 1e-5
    for target in range(3):
        scenarios = []
        for sign in [1, -1]:
            changed = data.with_columns(
                pl.when(pl.col("alt") == target)
                .then(pl.col("quality") * (1 + sign * step))
                .otherwise(pl.col("quality"))
                .alias("quality")
            )
            scenarios.append(
                dict(
                    result.predict(data=changed)
                    .market_shares(se="none")
                    .select("alts", "market_share")
                    .iter_rows()
                )
            )
        for affected in range(3):
            actual = aggregate.filter(
                (pl.col("alts") == affected) & (pl.col("target_alts") == target)
            )["elasticity_quality"][0]
            expected = (scenarios[0][affected] - scenarios[1][affected]) / (
                2 * step * base[affected]
            )
            assert actual == pytest.approx(expected, rel=2e-8, abs=1e-10)
    # The raw input frame follows encoded order even when the first case lacks alt 0.
    assert (
        prediction.raw_prediction_data["alt"].to_list()
        == prediction.predicted_probs["alts"].to_list()
    )


def test_uniform_price_change_is_an_exact_money_loss_with_zero_se(known):
    data, result = known
    history = data.filter((pl.col("panel") > 5) & (pl.col("case") == 0))
    base = result.predict(data=data, past_choices=history)
    changed = result.predict(
        data=data.with_columns((pl.col("price") + 0.75).alias("price")),
        past_choices=history,
    )
    np.testing.assert_allclose(
        base.surplus_change(changed)["surplus_change"], -0.75, atol=1e-12
    )
    summary = base.mean_surplus_change(changed)
    assert summary["mean_surplus_change"][0] == pytest.approx(-0.75, abs=1e-12)
    assert summary["std_error"][0] < 1e-12
    assert summary["change_identified"][0]
    removed = result.predict(data=data.filter(pl.col("alt") != 0), past_choices=history)
    assert (base.surplus_change(removed)["surplus_change"] < 0).all()


def test_welfare_checks_case_identity_model_and_weights(known):
    data, result = known
    baseline = result.predict(data=data)
    wrong_cases = result.predict(
        data=data.with_columns((pl.col("case") + 100).alias("case"))
    )
    other_model = copy(baseline)
    other_model.results = copy(result)
    other_weights = result.predict(data=data, panel_weights=[2.0] * 12)
    for method in [baseline.surplus_change, baseline.mean_surplus_change]:
        for other in [wrong_cases, other_model, other_weights]:
            with pytest.raises(ValueError):
                method(other)


def test_demographic_changes_do_not_identify_welfare_when_money_scales_coincide(known):
    data, result = known
    equal = copy(result)
    beta = result.em_res.structural_betas.at[0].set(-1.0)
    latent = result.em_res.latent_betas.at[0].set(
        np.log(np.expm1(1 - result.model.numeraire_min_abs))
    )
    equal.em_res = result.em_res._replace(structural_betas=beta, latent_betas=latent)
    equal.flat_params = equal._pack_params()
    base = equal.predict(data=data)
    changed = equal.predict(
        data=data.with_columns((pl.col("income") + 1).alias("income"))
    )
    summary = base.mean_surplus_change(changed, se="none")
    assert abs(summary["normalisation_sensitivity"][0]) < 1e-12
    assert not summary["change_identified"][0]
    assert not base.surplus_change(changed)["change_identified"].any()


def test_wtp_averages_class_ratios_and_includes_demographic_interactions(known):
    data, result = known
    history = data.filter((pl.col("panel") % 2 == 0) & (pl.col("case") == 0))
    weights = np.arange(1, 13, dtype=float)
    prediction = result.predict(data=data, past_choices=history, panel_weights=weights)
    income = np.arange(12) / 6 - 1
    betas = np.asarray(result.em_res.structural_betas)
    ratios = (betas[1] + income[:, None] * betas[2]) / -betas[0]
    expected = np.sum(np.asarray(prediction.class_probs_by_panel) * ratios, axis=1)
    rows = prediction.marginal_wtp("quality")
    np.testing.assert_allclose(rows["marginal_wtp"], np.repeat(expected, 9), rtol=1e-9)
    request = WTPRequest("quality", "segment", PartitionType.CATEGORICAL)
    delta = next(iter(prediction.compute_wtp(request, show=False).values()))
    point = next(iter(prediction.compute_wtp(request, se="none", show=False).values()))
    np.testing.assert_allclose(
        delta["Mean_Marginal_WTP"], point["Mean_Marginal_WTP"], rtol=1e-12
    )
    for row in point.iter_rows(named=True):
        mask = income < 0 if row["segment"] == "low" else income >= 0
        assert row["Mean_Marginal_WTP"] == pytest.approx(
            np.average(expected[mask], weights=weights[mask]), rel=1e-9
        )
    assert (delta["Standard_Error"] > 0).all()
    assert (rows["class_sd"] > 0).all()


def test_wtp_zero_weight_groups_and_missing_partitions_are_rejected(known):
    data, result = known
    prediction = result.predict(data=data, panel_weights=[0.0] * 6 + [1.0] * 6)
    request = WTPRequest("quality", "segment", PartitionType.CATEGORICAL)
    with pytest.raises(ValueError, match="positive total"):
        prediction.compute_wtp(request, se="none", show=False)
    with pytest.raises(ValueError, match="missing"):
        result.predict(data=data).compute_wtp(
            request,
            partition_data=pl.DataFrame({"panels": [0], "segment": ["x"]}),
            se="none",
            show=False,
        )


def test_underflow_does_not_contaminate_positive_demand_elasticities(known):
    data, result = known
    data = data.with_columns(
        pl.when(pl.col("alt") == 0).then(1e5).otherwise(pl.col("price")).alias("price")
    )
    values = result.predict(data=data).aggregate_elasticities("price", se="none")
    assert np.isfinite(
        values.filter(pl.col("alts") != 0)["elasticity_price"].to_numpy()
    ).all()
    assert values.filter(pl.col("alts") == 0)["elasticity_price"].is_nan().all()


def test_64_class_tables_preserve_math_and_aggregate_style():
    table = pl.DataFrame(
        [
            dict(
                variable="travel_time",
                label=r"Travel $t_n$ & time",
                **{"class": c},
                coefficient=float(c),
                std_error=0.1,
            )
            for c in range(64)
        ]
    )
    for renderer in [format_class_coefficients, format_membership_coefficients]:
        output = renderer(table, 64, 3)
        latex, terminal = output.split("--- Table preview ---")
        assert r"Travel $t_n$ \& time" in latex
        assert r"\textbackslash" not in latex
        assert "$" not in terminal and r"\&" not in terminal
        assert "64 &" in latex
        assert "Class 64" not in latex.split(r"\midrule")[0]
    moments = pl.DataFrame(
        [dict(label=r"$\beta_t$", mean=2.0, mean_se=0.1, sd=0.5, sd_se=0.2)]
    )
    aggregate = format_lcl_beta_summary(moments, ("Variable", "Mean", "SD"), 3)
    assert "$\\beta_t$ & 2.000 & 0.500 \\\\\n & (0.100) & (0.200) \\\\" in aggregate


def test_price_interactions_disable_money_metrics_but_keep_predictions(known):
    data, result = known
    altered = copy(result)
    altered.model = copy(result.model)
    altered.model.case_varnames = ["price", "quality", "price:income"]
    # Array designs expose the same specification guard as formula designs.
    prediction = altered.predict(
        X=result.data.X,
        dems=result.data.dems,
        alts=data["alt"].to_numpy(),
        cases=data["case"].to_numpy(),
        panels=data["panel"].to_numpy(),
    )
    assert prediction.surplus_units == "undefined"
    assert prediction.surplus["surplus"].is_nan().all()
    assert np.isfinite(prediction.predicted_probs["choice_probs"].to_numpy()).all()
    for method in [prediction.mean_surplus, prediction.wtp_by_class]:
        with pytest.raises(ValueError, match="without interactions"):
            method()


def test_posterior_wtp_se_matches_independent_parameter_differences(known):
    data, result = known
    history = data.filter((pl.col("panel") % 2 == 0) & (pl.col("case") == 0))
    income = np.arange(12) / 6 - 1
    weights = np.arange(1, 13, dtype=float)
    prediction = result.predict(data=data, past_choices=history, panel_weights=weights)
    table = next(
        iter(
            prediction.compute_wtp(
                WTPRequest("quality", "segment", PartitionType.CATEGORICAL), show=False
            ).values()
        )
    )

    def evaluate(params):
        beta = params[:6].reshape(3, 2).copy()
        beta[0] = -np.logaddexp(0.0, beta[0]) - result.model.numeraire_min_abs
        theta = params[6:]
        logprior = np.c_[np.zeros(12), theta[0] + theta[1] * income]
        for panel in range(0, 12, 2):
            case = history.filter(pl.col("panel") == panel).sort("alt")
            quality = case["quality"].to_numpy()
            x = np.c_[case["price"].to_numpy(), quality, quality * income[panel]]
            utility = x @ beta
            logprior[panel] += utility[0] - logsumexp(utility, axis=0)
        posterior = softmax(logprior, axis=1)
        ratios = (beta[1] + income[:, None] * beta[2]) / -beta[0]
        values = (posterior * ratios).sum(axis=1)
        return np.average(values[income >= 0], weights=weights[income >= 0])

    params = np.asarray(result.flat_params)
    steps = np.eye(params.size) * 1e-5
    gradient = np.array(
        [(evaluate(params + s) - evaluate(params - s)) / 2e-5 for s in steps]
    )
    expected_se = np.sqrt(gradient @ np.asarray(result.latent_cov_matrix) @ gradient)
    row = table.filter(pl.col("segment") == "high").row(0, named=True)
    assert row["Mean_Marginal_WTP"] == pytest.approx(evaluate(params), rel=1e-9)
    assert row["Standard_Error"] == pytest.approx(expected_se, rel=1e-8)


def test_extreme_prior_odds_are_not_replaced_by_a_probability_floor(known):
    data, result = known
    extreme = copy(result)
    beta = jnp.array([[-1.0, -1.0], [900.0, 0.0], [0.0, 0.0]])
    latent = beta.at[0].set(np.log(np.expm1(1 - result.model.numeraire_min_abs)))
    extreme.em_res = result.em_res._replace(
        structural_betas=beta,
        latent_betas=latent,
        thetas=jnp.array([[-1000.0], [0.0]]),
        shares=jnp.array([1.0, 0.0]),
    )
    extreme.flat_params = extreme._pack_params()
    history = PastChoicesData(
        X=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        y=np.array([False, True]),
        alts=np.array([0, 1]),
        cases=np.array([0, 0]),
        panels=np.array([6, 6]),
    )
    prediction = extreme.predict(data=data, past_choices=history)
    expected = softmax(np.array([-900.0, -1000.0 - np.log(2)]))[1]
    assert float(prediction.class_probs_by_panel[6, 1]) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )
