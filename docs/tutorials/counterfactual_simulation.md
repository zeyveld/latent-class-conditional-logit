# Counterfactual future worlds and present-trip welfare

Two distinct calculations commonly arise in substitution counterfactuals:

1. The store must evaluate future profits without knowing future trip times, choice
   sets, prices, or product availability.
2. Present-trip consumer welfare must distinguish the attributes used to choose from
   the attributes actually experienced.

LCL keeps these calculations separate. Future-state resampling does not enter the
present-trip welfare formula.

## Future states without perfect store foresight

`simulate_future_choice_sets` samples complete historical choice sets with
replacement. Sampling a choice set as a unit preserves the observed joint
distribution of availability, price, wholesale cost, and other trip attributes.

```python
import polars as pl

from lcl import FutureSimulationConfig, simulate_future_choice_sets

worlds = simulate_future_choice_sets(
    panel_data,
    panel_col="loyalty_id",
    case_col="transaction_id",
    time_col="date",
    intervention_col="first_substitution",
    choice_col="choice",
    config=FutureSimulationConfig(
        num_draws=20,
        horizon_days=365,
        max_trips_per_panel=104,
        seed=7,
        trip_timing="poisson",
    ),
)
```

The two outputs represent different information sets:

- `worlds.anticipated` samples each customer's choice sets only through the
  original-order/intervention date. Use it when the store selects a substitute.
- `worlds.realized` samples the customer's full observed panel, both before and after
  the intervention. Use it only to evaluate realized outcomes.

Each frame includes `simulation_round`, `simulation_trip`, `source_case`,
`source_time`, and `days_after_intervention`. Synthetic case IDs are unique, and an
observed choice column is removed when `choice_col` is supplied. Compute future
choice probabilities from the fitted model rather than copying the historical choice.

`worlds.trip_summary` makes the trip-rate assumptions auditable. For each customer
and information set, it reports the historical pool size, estimated mean interval,
expected trips per draw, and the rounded fixed-schedule count. A customer's rate is
the elapsed time divided by the number of observed trip intervals. The pooled median
interval is used when a customer has only one observed date, and
`max_trips_per_panel` provides a transparent cap.

By default, the simulator treats shopping as a zero-truncated, capped homogeneous
Poisson process: each world draws at least one trip at the estimated rate and random
trip dates within the horizon. This integrates over uncertainty in future trip timing
while preserving the original implementation's one-trip minimum. Set
`trip_timing="fixed"` to use the simpler evenly spaced schedule from the expected trip
rate.

Generate the worlds once and reuse them for every candidate substitution policy. This
provides common simulated future states across policies and avoids unnecessary data
expansion and Monte Carlo noise. The default of 20 draws is deliberately modest; the
sampler first creates a compact case-level draw map and only then joins product rows.

All rounds can be scored in one batched prediction call. For example, after joining
the returned generic prediction keys to the original column names, expected discounted
future profit can be integrated with a single group-by:

```python
future_prediction = results.predict(
    data=worlds.anticipated,
    past_choices=history_through_original_order,
)

scored_worlds = worlds.anticipated.join(
    future_prediction.predicted_probs,
    left_on=["loyalty_id", "transaction_id", "product_id"],
    right_on=["panels", "cases", "alts"],
)
profit_by_world = scored_worlds.group_by("simulation_round").agg(
    (
        pl.col("retail_margin")
        * pl.col("choice_probs")
        * (0.9998 ** pl.col("days_after_intervention"))
    ).sum().alias("discounted_future_profit")
)
expected_future_profit = profit_by_world["discounted_future_profit"].mean()
```

Use the appropriate acceptance- or rejection-conditioned choice history in
`past_choices` when comparing those two branches. Reuse the same scored worlds for
all candidate substitutes; only the policy-dependent history and profit calculation
should vary.

!!! warning "Do not leak future information into the policy"
    `worlds.realized` contains post-intervention information. It is an ex post
    evaluation device and must never be supplied to the store's policy-selection
    rule.

## Present-trip welfare with anticipated and experienced attributes

Let $W_j$ denote utility anticipated when the consumer chooses, let $U_j$ denote
experienced utility, and define $d_j = U_j-W_j$. Train (2015) shows that expected
experienced surplus under logit is

\[
CS^{\mathrm{experienced}}
= \log\!\left(\sum_j e^{W_j}\right) + \sum_j P_j(W)d_j.
\]

The consumer still chooses using $P_j(W)$; replacing the first term with
`logsum(U)` would incorrectly give the consumer perfect foresight. The corresponding
positive loss from imperfect foreknowledge is

\[
L^{\mathrm{foreknowledge}}
= \log\!\left(\sum_j e^{U_j}\right) - CS^{\mathrm{experienced}}.
\]

Pass the row-aligned anticipated and experienced choice sets separately:

```python
prediction = results.predict(
    data=attributes_at_choice,
    experienced_data=attributes_received,
    past_choices=observed_choice_history,
)

welfare = prediction.surplus
acceptance = prediction.acceptance_probability(accepted_alternatives=[substitute_id])
```

`prediction.predicted_probs` and `acceptance_probability` use anticipated attributes.
The welfare frame contains:

- `anticipated_surplus_utils`
- `experience_effect_utils`, equal to \(\sum_j P_j(W)d_j\)
- `experienced_surplus_utils`
- `perfect_foresight_surplus_utils`
- `foreknowledge_loss_utils`

When the model has a price/cost numeraire, the same five columns are also returned
with a `_dollars` suffix. Conversion to dollars occurs within latent class using that
class's marginal utility of income, before class-specific welfare is averaged. This
is the appropriate mixed-logit integration, although acceptance probability often
remains the most interpretable headline outcome because it does not place large
weight on classes with low price sensitivity.

The legacy `surplus` column is retained for compatibility. It equals experienced
surplus in dollars when a numeraire exists and experienced surplus in utils otherwise.
When `experienced_data` is omitted, experienced and anticipated attributes coincide,
the experience effect and foreknowledge loss are zero, and the formula reduces to the
standard inclusive value.
