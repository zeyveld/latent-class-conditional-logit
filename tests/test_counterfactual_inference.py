"""Standard errors for counterfactual, welfare, and elasticity aggregates.

The delta-method Jacobians are checked against central finite differences, and
the parametric bootstrap is checked against the delta method in the limit where
the covariance shrinks and the linearization becomes exact.  Together those pin
both the derivative and the draw mechanics without asserting hard-coded numbers.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
import pytest
from jax.tree_util import Partial

from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
)
from lcl import fit as lcl_fit
from lcl._delta import apply_delta_method, parametric_bootstrap_se
from lcl._prediction_inference import build_within_case_pairs


def _two_class_panel(seed: int = 4, num_panels: int = 120) -> pl.DataFrame:
    """Panel choice data with two well-separated taste classes."""
    rng = onp.random.default_rng(seed)
    latent = rng.choice(2, size=num_panels, p=[0.55, 0.45])
    price_by_class = onp.array([-1.8, -0.6])
    quality_by_class = onp.array([0.4, 1.6])
    rows = []
    for panel in range(num_panels):
        income = float(rng.normal())
        for case in range(4):
            price = rng.uniform(0.5, 3.0, size=3)
            quality = rng.uniform(0.0, 5.0, size=3)
            utility = (
                price_by_class[latent[panel]] * price
                + quality_by_class[latent[panel]] * quality
                + rng.gumbel(size=3)
            )
            chosen = int(onp.argmax(utility))
            for alt in range(3):
                rows.append(
                    {
                        "panel": panel,
                        "case": panel * 4 + case,
                        "alt": alt,
                        "choice": alt == chosen,
                        "price": float(price[alt]),
                        "quality": float(quality[alt]),
                        "income": income,
                        "region": panel // 12,
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    """Return a converged two-class fit and its in-sample prediction."""
    df = _two_class_panel()
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        membership_formula="~ income",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    results = lcl_fit(
        df,
        spec,
        fit_options=FitOptions(seed=2, num_devices=1),
        inference=InferenceOptions(covariance="clustered"),
    )
    return df, results, results.predict(df)


def _finite_difference_se(func, flat_params, cov_matrix, step=1e-6, **kwargs):
    """Delta-method standard errors from a central-difference Jacobian."""
    target = Partial(func, **kwargs)
    params = jnp.asarray(flat_params)
    columns = [
        (
            onp.asarray(target(params.at[index].add(step)))
            - onp.asarray(target(params.at[index].add(-step)))
        )
        / (2.0 * step)
        for index in range(params.size)
    ]
    jacobian = onp.column_stack([column.reshape(-1) for column in columns])
    covariance = onp.asarray(cov_matrix)
    return onp.sqrt(onp.einsum("ip,pq,iq->i", jacobian, covariance, jacobian))


def test_market_share_standard_errors_match_finite_differences(fitted) -> None:
    """The market-share Jacobian is the numerical one."""
    _, results, prediction = fitted
    from lcl._prediction_inference import market_shares

    data = prediction.predict_data
    row_weights = prediction._row_panel_weights()
    first_case_rows = onp.asarray(data.cases) != onp.roll(onp.asarray(data.cases), 1)
    first_case_rows[0] = True
    kwargs = dict(
        alt_codes=data.alts,
        num_alts=int(onp.asarray(data.alts).max()) + 1,
        row_weights=row_weights,
        weight_total=float(onp.asarray(row_weights)[first_case_rows].sum()),
        **prediction._design_kwargs(),
    )
    reported = prediction.market_shares()
    expected = _finite_difference_se(
        market_shares, results.flat_params, results.latent_cov_matrix, **kwargs
    )
    onp.testing.assert_allclose(
        reported["std_error"].to_numpy(), expected, rtol=1e-5, atol=1e-12
    )
    assert reported["market_share"].sum() == pytest.approx(1.0)


def test_aggregate_elasticities_match_the_row_level_aggregation(fitted) -> None:
    """The differentiable aggregate reproduces the per-row demand-weighted mean."""
    _, _, prediction = fitted
    aggregate = prediction.aggregate_elasticities("price").sort(
        ["alts", "target_alts"]
    )
    rows = prediction.elasticities("price").join(
        prediction.predicted_probs.select(["panels", "cases", "alts", "choice_probs"]),
        on=["panels", "cases", "alts"],
    )
    manual = (
        rows.group_by(["alts", "target_alts"], maintain_order=True)
        .agg(
            (pl.col("choice_probs") * pl.col("elasticity_price")).sum().alias("total"),
            pl.col("choice_probs").sum().alias("demand"),
        )
        .with_columns((pl.col("total") / pl.col("demand")).alias("manual"))
        .select("alts", "target_alts", "manual")
        .sort(["alts", "target_alts"])
    )
    joined = aggregate.join(manual, on=["alts", "target_alts"])
    onp.testing.assert_allclose(
        joined["elasticity_price"].to_numpy(),
        joined["manual"].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
    own = joined.filter(pl.col("alts") == pl.col("target_alts"))
    assert onp.all(own["elasticity_price"].to_numpy() < 0.0)
    cross = joined.filter(pl.col("alts") != pl.col("target_alts"))
    assert onp.all(cross["elasticity_price"].to_numpy() > 0.0)


def test_bootstrap_converges_to_the_delta_method_as_curvature_vanishes(
    fitted,
) -> None:
    """Shrinking the covariance removes the nonlinearity the two methods differ on."""
    _, results, prediction = fitted
    from lcl._prediction_inference import market_shares

    data = prediction.predict_data
    row_weights = prediction._row_panel_weights()
    first_case_rows = onp.asarray(data.cases) != onp.roll(onp.asarray(data.cases), 1)
    first_case_rows[0] = True
    kwargs = dict(
        alt_codes=data.alts,
        num_alts=int(onp.asarray(data.alts).max()) + 1,
        row_weights=row_weights,
        weight_total=float(onp.asarray(row_weights)[first_case_rows].sum()),
        **prediction._design_kwargs(),
    )
    shrunk = jnp.asarray(results.latent_cov_matrix) * 1e-4
    _, delta = apply_delta_method(
        market_shares, results.flat_params, shrunk, **kwargs
    )
    bootstrap = parametric_bootstrap_se(
        market_shares,
        results.flat_params,
        shrunk,
        draws=4000,
        seed=11,
        **kwargs,
    )
    onp.testing.assert_allclose(
        onp.asarray(bootstrap), onp.asarray(delta), rtol=0.05
    )


def test_surplus_change_is_signed_and_shares_parameter_uncertainty(fitted) -> None:
    """A price rise lowers surplus, and the paired difference is far tighter."""
    df, results, baseline = fitted
    counterfactual = results.predict(df.with_columns(pl.col("price") * 1.25))
    # The receiver is the baseline; the argument is the scenario compared to it,
    # matching surplus_change.
    change = baseline.mean_surplus_change(counterfactual)
    reverse = counterfactual.mean_surplus_change(baseline)

    assert float(change["mean_surplus_change"][0]) < 0.0
    assert float(reverse["mean_surplus_change"][0]) == pytest.approx(
        -float(change["mean_surplus_change"][0])
    )
    assert float(reverse["std_error"][0]) == pytest.approx(
        float(change["std_error"][0])
    )
    # Evaluating both scenarios at one parameter vector keeps their correlation,
    # so the change is estimated far more precisely than either level.
    level = baseline.mean_surplus()
    assert float(change["std_error"][0]) < float(level["std_error"][0])


@pytest.mark.parametrize("method", ["delta", "bootstrap", "none"])
def test_point_estimates_do_not_depend_on_the_uncertainty_method(
    fitted, method
) -> None:
    """Switching the standard-error method never moves the estimate itself."""
    df, results, prediction = fitted
    counterfactual = results.predict(df.with_columns(pl.col("price") * 1.1))
    kwargs = {"se": method}
    if method == "bootstrap":
        kwargs |= {"bootstrap_draws": 25, "bootstrap_seed": 3}

    reference = prediction.market_shares(se="none")["market_share"].to_numpy()
    onp.testing.assert_allclose(
        prediction.market_shares(**kwargs)["market_share"].to_numpy(), reference
    )
    onp.testing.assert_allclose(
        prediction.mean_surplus(**kwargs)["mean_surplus"][0],
        prediction.mean_surplus(se="none")["mean_surplus"][0],
    )
    onp.testing.assert_allclose(
        counterfactual.mean_surplus_change(prediction, **kwargs)[
            "mean_surplus_change"
        ][0],
        counterfactual.mean_surplus_change(prediction, se="none")[
            "mean_surplus_change"
        ][0],
    )
    onp.testing.assert_allclose(
        prediction.aggregate_elasticities("price", **kwargs)[
            "elasticity_price"
        ].to_numpy(),
        prediction.aggregate_elasticities("price", se="none")[
            "elasticity_price"
        ].to_numpy(),
    )


def test_within_case_pairs_enumerate_every_ordered_pair() -> None:
    """The elasticity pair index covers each case's ordered pairs exactly once."""
    cases = onp.array([0, 0, 1, 1, 1, 2])
    affected, target = build_within_case_pairs(cases)
    assert affected.size == 4 + 9 + 1
    assert onp.all(cases[affected] == cases[target])
    seen = sorted(zip(affected.tolist(), target.tolist()))
    expected = sorted(
        (j, k)
        for case in onp.unique(cases)
        for j in onp.flatnonzero(cases == case)
        for k in onp.flatnonzero(cases == case)
    )
    assert seen == expected


def test_empty_pair_index_is_handled() -> None:
    """No rows means no pairs, not an exception."""
    affected, target = build_within_case_pairs(onp.empty(0, dtype=onp.int64))
    assert affected.size == 0 and target.size == 0


def test_coarser_clustering_changes_only_the_covariance() -> None:
    """A region cluster reuses the estimate and re-sums the scores."""
    df = _two_class_panel()
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        membership_formula="~ income",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    common = dict(fit_options=FitOptions(seed=2, num_devices=1))
    by_panel = lcl_fit(
        df, spec, inference=InferenceOptions(covariance="clustered"), **common
    )
    by_region = lcl_fit(
        df,
        spec,
        inference=InferenceOptions(covariance="clustered", cluster="region"),
        **common,
    )
    onp.testing.assert_allclose(
        onp.asarray(by_panel.flat_params), onp.asarray(by_region.flat_params)
    )
    assert not onp.allclose(
        onp.asarray(by_panel.cov_matrix), onp.asarray(by_region.cov_matrix)
    )

    # Reproduce the region sandwich directly from the panel-level scores.
    from lcl._analytic_derivatives import _panel_scores_and_hessian
    from lcl._case_utils import _diff_unchosen_chosen
    from lcl._inference import _aggregate_scores, _invert_information, _symmetrize

    diff = _diff_unchosen_chosen(by_region.data)
    scores, hessian = _panel_scores_and_hessian(
        by_region.flat_params, diff, by_region.data, by_region._param_packing
    )
    bread, _ = _invert_information(-hessian)
    cluster_ids = onp.asarray(df.sort("panel")["region"].unique(maintain_order=True))
    grouped = _aggregate_scores(scores, by_region._cluster_ids, len(cluster_ids))
    groups = len(cluster_ids)
    expected_latent = _symmetrize(
        (bread @ (grouped.T @ grouped) @ bread) * (groups / (groups - 1))
    )
    jacobian = jax.jacfwd(by_region._structural_from_latent)(by_region.flat_params)
    expected = _symmetrize(jacobian @ expected_latent @ jacobian.T)
    onp.testing.assert_allclose(
        onp.asarray(by_region.cov_matrix), onp.asarray(expected), rtol=1e-9, atol=1e-12
    )


def test_a_nonconstant_cluster_column_is_rejected() -> None:
    """Clustering coarser than the panel requires nesting."""
    df = _two_class_panel(num_panels=30)
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        classes=2,
    )
    with pytest.raises(ValueError, match="constant within each panel"):
        lcl_fit(
            df,
            spec,
            fit_options=FitOptions(seed=1, max_em_iter=2, num_devices=1),
            inference=InferenceOptions(covariance="clustered", cluster="price"),
        )
    with pytest.raises(ValueError, match="was not found"):
        lcl_fit(
            df,
            spec,
            fit_options=FitOptions(seed=1, max_em_iter=2, num_devices=1),
            inference=InferenceOptions(covariance="clustered", cluster="absent"),
        )


def test_aggregates_use_the_bayesian_posterior_when_past_choices_are_supplied(
    fitted,
) -> None:
    """Aggregates with standard errors must weight classes as the per-case tables do.

    ``predict(..., past_choices=...)`` sharpens class membership from the
    demographics-only prior to the posterior implied by the observed history.
    The per-case probability and surplus tables always used that posterior; the
    aggregates that carry standard errors are differentiable functions of the
    parameters and have to reach the same weights through the Bayes update rather
    than falling back to the prior.
    """
    df, results, _ = fitted
    history = df.filter(pl.col("case") % 4 < 2)
    future = df.filter(pl.col("case") % 4 >= 2)
    prior = results.predict(data=future)
    posterior = results.predict(data=future, past_choices=history)

    assert prior.class_probabilities_source == "prior"
    assert posterior.class_probabilities_source == "posterior"
    prior_weights = onp.asarray(prior.class_probs_by_panel)
    posterior_weights = onp.asarray(posterior.class_probs_by_panel)
    assert onp.abs(prior_weights - posterior_weights).max() > 0.1

    for prediction in (prior, posterior):
        cases = prediction.predicted_probs["cases"].n_unique()
        by_alternative = (
            prediction.predicted_probs.group_by("alts")
            .agg(pl.col("choice_probs").sum())
            .sort("alts")
        )
        expected_shares = by_alternative["choice_probs"].to_numpy() / cases
        shares = prediction.market_shares(se="none")["market_share"].to_numpy()
        assert onp.allclose(shares, expected_shares, atol=1e-12)

        mean_surplus = prediction.mean_surplus(se="none")["mean_surplus"][0]
        assert mean_surplus == pytest.approx(
            prediction.surplus["surplus"].mean(), abs=1e-12
        )

    # The two weightings must actually disagree, or the assertions above would
    # pass even if the posterior branch were never taken.
    assert prior.mean_surplus(se="none")["mean_surplus"][0] != pytest.approx(
        posterior.mean_surplus(se="none")["mean_surplus"][0], abs=1e-6
    )


def test_posterior_weighted_standard_errors_respond_to_membership_parameters(
    fitted,
) -> None:
    """A posterior-weighted aggregate must depend on the membership coefficients.

    The posterior is a function of both the taste and the membership parameters,
    so a delta-method Jacobian that treats it as a fixed constant would leave the
    membership block empty and understate the reported uncertainty.
    """
    df, results, _ = fitted
    history = df.filter(pl.col("case") % 4 < 2)
    future = df.filter(pl.col("case") % 4 >= 2)
    posterior = results.predict(data=future, past_choices=history)

    kwargs = posterior._design_kwargs()
    kwargs.pop("panels")
    kwargs["panels_of_cases"] = posterior.predict_data.panels_of_cases
    kwargs["case_weights"] = posterior._case_panel_weights()
    target = Partial(_mean_surplus_target, **kwargs)
    jacobian = onp.asarray(jax.jacrev(target)(jnp.asarray(results.flat_params)))

    num_taste_params = results.model.num_classes * len(results.model.case_varnames)
    assert onp.abs(jacobian[:num_taste_params]).max() > 0.0
    assert onp.abs(jacobian[num_taste_params:]).max() > 0.0
    assert onp.isfinite(
        posterior.mean_surplus()["std_error"][0]
    )


def test_surplus_change_is_normalisation_free_only_at_fixed_class_weights(
    fitted,
) -> None:
    """The reported sensitivity must be zero exactly when the constants cancel.

    Shifting the unidentified Gumbel location by ``c`` adds ``c / alpha_s`` to
    class ``s``'s money-metric surplus.  Those shifts cancel from a difference
    when both scenarios weight the classes identically -- an attribute
    counterfactual -- and do not when the counterfactual moves a demographic that
    enters the class-membership model.
    """
    df, results, baseline = fitted

    attribute_scenario = results.predict(
        data=df.with_columns((pl.col("price") * 1.1).alias("price"))
    )
    attribute_change = baseline.mean_surplus_change(attribute_scenario, se="none")
    assert attribute_change["normalisation_sensitivity"][0] == pytest.approx(
        0.0, abs=1e-12
    )

    demographic_scenario = results.predict(
        data=df.with_columns((pl.col("income") + 1.0).alias("income"))
    )
    demographic_change = baseline.mean_surplus_change(demographic_scenario, se="none")
    sensitivity = demographic_change["normalisation_sensitivity"][0]
    assert abs(sensitivity) > 1e-3

    # The sensitivity is the exact multiplier on the normalising constant, so it
    # predicts how far a given choice of constant moves the reported change.
    constant = 0.5772156649015329
    shifted = _mean_surplus_change_with_constant(
        results, baseline, demographic_scenario, constant
    )
    unshifted = _mean_surplus_change_with_constant(
        results, baseline, demographic_scenario, 0.0
    )
    assert shifted - unshifted == pytest.approx(constant * sensitivity, abs=1e-9)


def _mean_surplus_target(flat_params, **kwargs):
    """Adapt the module-level mean-surplus function for ``jax.jacrev``."""
    from lcl._prediction_inference import mean_surplus

    return mean_surplus(flat_params, **kwargs)


def _mean_surplus_change_with_constant(results, baseline, counterfactual, constant):
    """Recompute the mean surplus change with an explicit utility constant."""
    betas = onp.asarray(results.em_res.structural_betas)
    alpha = -betas[results.model.numeraire_idx, :]

    def mean_surplus(prediction):
        """Average the per-case surplus after shifting every log-sum by ``constant``."""
        data = prediction.predict_data
        utilities = onp.asarray(data.X) @ betas
        case_ids = onp.asarray(data.cases)
        log_sums = onp.stack(
            [
                _log_sum_exp(utilities[case_ids == case])
                for case in range(data.num_cases)
            ]
        )
        weights = onp.asarray(prediction.class_probs_by_panel)[
            onp.asarray(data.panels_of_cases)
        ]
        return float((weights * ((log_sums + constant) / alpha)).sum(axis=1).mean())

    return mean_surplus(counterfactual) - mean_surplus(baseline)


def _log_sum_exp(values):
    """Stable column-wise log-sum-exp over the alternatives in one choice situation."""
    shift = values.max(axis=0)
    return shift + onp.log(onp.exp(values - shift).sum(axis=0))
