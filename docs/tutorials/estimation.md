# Estimation & counterfactuals

This tutorial fits a three-class latent-class conditional logit model to Apollo's
`modeChoice` data. It then evaluates a counterfactual fare increase and summarizes
the value of travel time across income groups.

## 1. Reshape the data

LCL expects long-format data: one row for each decision-maker, choice situation,
and available alternative. Apollo provides these data in wide format, so we first
reshape them.

```python
import polars as pl

df_wide = (
    pl.read_csv("https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv")
      .with_row_index("qID")
)

alts_map = {1: "car", 2: "bus", 3: "air", 4: "rail"}

dfs = []
for num, name in alts_map.items():
    dfs.append(
        df_wide.select([
            pl.col("ID"),
            pl.col("qID"),
            pl.col("income"),
            pl.col("female"),
            pl.col(f"time_{name}").alias("time"),
            pl.col(f"cost_{name}").alias("cost"),
            pl.col(f"av_{name}").alias("av"),
            (pl.col("choice") == num).alias("choice"),
        ]).with_columns(pl.lit(name).alias("alt"))
    )

df_long = (
    pl.concat(dfs)
      .filter(pl.col("av") == 1)        # Drop unavailable alternatives
      .sort(["ID", "qID", "alt"])       # Required: panel, then case, then alt
)
print(df_long.head(8))
```

```text
shape: (8, 9)
┌─────┬─────┬────────┬────────┬───┬──────┬─────┬────────┬──────┐
│ ID  ┆ qID ┆ income ┆ female ┆ … ┆ cost ┆ av  ┆ choice ┆ alt  │
│ --- ┆ --- ┆ ---    ┆ ---    ┆   ┆ ---  ┆ --- ┆ ---    ┆ ---  │
│ i64 ┆ u32 ┆ i64    ┆ i64    ┆   ┆ i64  ┆ i64 ┆ bool   ┆ str  │
╞═════╪═════╪════════╪════════╪═══╪══════╪═════╪════════╪══════╡
│ 1   ┆ 0   ┆ 46705  ┆ 0      ┆ … ┆ 80   ┆ 1   ┆ false  ┆ air  │
│ 1   ┆ 0   ┆ 46705  ┆ 0      ┆ … ┆ 55   ┆ 1   ┆ true   ┆ rail │
│ 1   ┆ 1   ┆ 46705  ┆ 0      ┆ … ┆ 80   ┆ 1   ┆ false  ┆ air  │
│ 1   ┆ 1   ┆ 46705  ┆ 0      ┆ … ┆ 45   ┆ 1   ┆ true   ┆ rail │
│ 1   ┆ 2   ┆ 46705  ┆ 0      ┆ … ┆ 50   ┆ 1   ┆ false  ┆ air  │
│ 1   ┆ 2   ┆ 46705  ┆ 0      ┆ … ┆ 35   ┆ 1   ┆ true   ┆ rail │
│ 1   ┆ 3   ┆ 46705  ┆ 0      ┆ … ┆ 65   ┆ 1   ┆ false  ┆ air  │
│ 1   ┆ 3   ┆ 46705  ┆ 0      ┆ … ┆ 75   ┆ 1   ┆ true   ┆ rail │
└─────┴─────┴────────┴────────┴───┴──────┴─────┴────────┴──────┘
```

!!! note "Why the availability filter matters"
    Unless removed, an unavailable alternative still enters the denominator of
    the logit probability. That changes the choice set and biases the estimates.

## 2. Estimate the model

We estimate three latent classes and treat fare as the numeraire, constraining its
coefficient to be strictly negative through a softplus reparameterization. Formulaic
uses familiar Wilkinson-style formulas to construct the utility and membership
designs:

- `C(alt)` creates alternative-specific constants for bus, car, and rail, with air
  as the reference alternative. The fitted encoder preserves that coding during
  prediction.
- `C(income_band)` adds a categorical panel characteristic to the class-membership
  model without manual indicator columns.

We also attach presentation labels. They appear in printed tables and dedicated
`label` columns, while formulas and programmatic lookups continue to use the raw
variable names.

```python
import lcl
from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
)

# A string categorical demographic; income is constant within a decision-maker.
df_long = df_long.with_columns(
    pl.col("income")
      .qcut(3, labels=["low", "mid", "high"])
      .cast(pl.String)
      .alias("income_band")
)

spec = LCLSpec(
    ids=ChoiceIds(alt="alt", case="qID", panel="ID", choice="choice"),
    utility_formula="choice ~ cost + time + C(alt)",
    membership_formula="~ C(income_band) + female",
    classes=3,
    constraints={"cost": NegativeCoefficient(units="dollars")},
    variable_labels={
        "cost": "Fare",
        "time": "Travel time",
        "alt": "Travel mode",
        "income_band": "Household income band",
        "female": "Female traveler",
    },
)

results = lcl.fit(
    df_long,
    spec,
    fit_options=FitOptions(max_em_iter=60, num_devices=1),
    optimization_options=OptimizationOptions(maxiter=40),
    inference=InferenceOptions(covariance="clustered"),
)

results.summarize_betas()
print(results)
```

```text
--- LaTeX Output ---

\toprule
Variable & Means (\beta's) & Standard deviations (\sigma's) \\
\midrule
%
Fare & -0.061 & 0.024 \\
 & (0.002) & (0.002) \\
Travel time & -0.011 & 0.001 \\
 & (0.001) & (0.001) \\
Travel mode: bus & -1.750 & 1.371 \\
 & (0.230) & (0.313) \\
Travel mode: car & 1.085 & 0.742 \\
 & (0.130) & (0.150) \\
Travel mode: rail & 0.418 & 0.321 \\
 & (0.057) & (0.066) \\
%
\bottomrule

--- Table preview ---

┌───────────────────┬───────────────┬─────────────────────────────┐
│ Variable          │ Means (β's)   │ Standard deviations (σ's)   │
├───────────────────┼───────────────┼─────────────────────────────┤
│ Fare              │ -0.061        │ 0.024                       │
│                   │ (0.002)       │ (0.002)                     │
│ Travel time       │ -0.011        │ 0.001                       │
│                   │ (0.001)       │ (0.001)                     │
│ Travel mode: bus  │ -1.750        │ 1.371                       │
│                   │ (0.230)       │ (0.313)                     │
│ Travel mode: car  │ 1.085         │ 0.742                       │
│                   │ (0.130)       │ (0.150)                     │
│ Travel mode: rail │ 0.418         │ 0.321                       │
│                   │ (0.057)       │ (0.066)                     │
└───────────────────┴───────────────┴─────────────────────────────┘

<LCLResults: 3 Classes | Converged | Log likelihood: -6413.8 | CAIC: 12993.6 | BIC: 12970.6 | Adj. BIC: 12897.6>
```

!!! note "Formulas or explicit lists?"
    `LCLSpec` also accepts `utility=[...]` and `membership=[...]` when the input
    columns are already model-ready. Use formulas for categorical expansion,
    interactions, or transformations. A combined legacy `formula=` remains
    available, but separate utility and membership formulas are easier to audit.

!!! tip "Watching the EM iterations"
    LCL sends iterative progress to Python's `logging` module. Enable it with
    `logging.basicConfig(level=logging.INFO)` when you need the full optimization
    trace.

`summarize_betas()` reports the share-weighted mean and standard deviation of each
structural coefficient, with delta-method standard errors in parentheses. The table
uses labels such as **Travel mode: bus**; the returned frame also retains Formulaic's
raw name, `C(alt)[T.bus]`. Use `results.class_coefficients()` for class-specific
coefficients and `results.class_shares()` for the estimated class composition.

## 3. Inspect the fit with the diagnostic tools

Before interpreting the estimates, inspect the built-in diagnostics.
`results.diagnostics()` returns structured fit, data, class, and coefficient checks.
Use `.print()` for a readable table or `.to_frame()` for programmatic validation.

```python
results.diagnostics().print()
```

```text
section       check                   status           value  message
------------  ----------------------  --------  ------------  ----------------------------------------------------------
fit           converged               ok            1         EM convergence flag.
fit           log_likelihood          ok        -6413.84      Final unconditional log likelihood.
data          panels                  ok          500         Number of decision-maker panels.
data          cases                   ok         8000         Number of choice situations.
latent_class  posterior_entropy_mean  ok            0.278257  Mean entropy of posterior class membership.
latent_class  min_class_share         ok            0.241837  Small classes can indicate weakly identified local optima.
latent_class  min_effective_panels    ok          120.919     Smallest posterior panel mass across classes.
coefficients  max_abs_beta            ok            3.74922   Largest absolute structural coefficient.
coefficients  min_abs_numeraire       ok            0.035543  Small numeraires can dominate WTP/tradeoff ratios.
```

Configure warning thresholds with `DiagnosticsOptions` at fit time. For example,
`DiagnosticsOptions(large_coefficient_threshold=10.0)` changes the large-coefficient
check. `convergence_report()` provides a compact optimization summary:

```python
print(results.convergence_report())
```

```text
Converged: True
EM recursions: 30
Final log likelihood: -6413.84
Warnings: 0
Last EM history row: {'em_iter': 30, 'loglik': -6413.840790186069, 'class_0_share': 0.24183736641517078, 'class_1_share': 0.3050601879815024, 'class_2_share': 0.45310244560332685}
```

`class_shares()` reports each class's aggregate share and posterior panel mass.
`class_coefficients()` returns the structural coefficients underlying the population
moments.

```python
print(results.class_shares())
print(results.class_coefficients())
```

```text
shape: (3, 3)
┌───────┬──────────┬──────────────────┐
│ class ┆ share    ┆ effective_panels │
│ ---   ┆ ---      ┆ ---              │
│ i64   ┆ f64      ┆ f64              │
╞═══════╪══════════╪══════════════════╡
│ 0     ┆ 0.241837 ┆ 120.918674       │
│ 1     ┆ 0.30506  ┆ 152.530099       │
│ 2     ┆ 0.453102 ┆ 226.551228       │
└───────┴──────────┴──────────────────┘
shape: (15, 5)
┌────────────────┬───────────────────┬───────┬─────────────┬─────────────┐
│ variable       ┆ label             ┆ class ┆ coefficient ┆ constrained │
│ ---            ┆ ---               ┆ ---   ┆ ---         ┆ ---         │
│ str            ┆ str               ┆ i64   ┆ f64         ┆ bool        │
╞════════════════╪═══════════════════╪═══════╪═════════════╪═════════════╡
│ cost           ┆ Fare              ┆ 0     ┆ -0.100253   ┆ true        │
│ cost           ┆ Fare              ┆ 1     ┆ -0.035543   ┆ true        │
│ cost           ┆ Fare              ┆ 2     ┆ -0.056122   ┆ true        │
│ time           ┆ Travel time       ┆ 0     ┆ -0.01306    ┆ false       │
│ time           ┆ Travel time       ┆ 1     ┆ -0.010947   ┆ false       │
│ …              ┆ …                 ┆ …     ┆ …           ┆ …           │
│ C(alt)[T.car]  ┆ Travel mode: car  ┆ 1     ┆ 0.078436    ┆ false       │
│ C(alt)[T.car]  ┆ Travel mode: car  ┆ 2     ┆ 1.240218    ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 0     ┆ 0.904493    ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 1     ┆ 0.031566    ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 2     ┆ 0.418674    ┆ false       │
└────────────────┴───────────────────┴───────┴─────────────┴─────────────┘
```

The class-specific estimates reveal heterogeneity that population averages conceal.
`results.audit_report()` combines the specification, fit statistics, class shares,
and diagnostics in a text report. `results.em_history_` and
`results.optimization_history_` expose iteration histories as Polars frames.

## 4. A counterfactual fare increase, conditioned on observed choices

Suppose bus and rail fares rise by 25%. `predict` reuses the fitted encoder, so the
counterfactual requires only a modified DataFrame. Passing `past_choices` updates
each decision-maker's demographic class prior with an observed choice history. We
use the estimation choices here; an applied analysis might supply a separate
pre-policy history.

```python
cf_df = df_long.with_columns(
    pl.when(pl.col("alt").is_in(["bus", "rail"]))
      .then(pl.col("cost") * 1.25)
      .otherwise(pl.col("cost"))
      .alias("cost")
)

prediction = results.predict(data=cf_df, past_choices=df_long)
print(prediction.predicted_probs.head(8))
```

```text
shape: (8, 4)
┌────────┬───────┬──────┬──────────────┐
│ panels ┆ cases ┆ alts ┆ choice_probs │
│ ---    ┆ ---   ┆ ---  ┆ ---          │
│ i64    ┆ u32   ┆ str  ┆ f64          │
╞════════╪═══════╪══════╪══════════════╡
│ 1      ┆ 0     ┆ air  ┆ 0.369361     │
│ 1      ┆ 0     ┆ rail ┆ 0.630639     │
│ 1      ┆ 1     ┆ air  ┆ 0.206002     │
│ 1      ┆ 1     ┆ rail ┆ 0.793998     │
│ 1      ┆ 2     ┆ air  ┆ 0.556804     │
│ 1      ┆ 2     ┆ rail ┆ 0.443196     │
│ 1      ┆ 3     ┆ air  ┆ 0.879005     │
│ 1      ┆ 3     ┆ rail ┆ 0.120995     │
└────────┴───────┴──────┴──────────────┘
```

The resulting `prediction.class_probs_by_panel` contains the updated class
probabilities used for choice probabilities, elasticities, and welfare measures.
Callers who already manage encoded arrays may pass `PastChoicesData` instead of a
DataFrame.

`LCLPrediction` also reports expected consumer surplus by choice situation and a
panel-level willingness-to-pay frame for downstream welfare analysis.

## 5. Elasticities

LCL computes own- and cross-elasticities for every pair of alternatives in each
choice situation.

```python
elast_df = prediction.elasticities(["cost", "time"])
print(elast_df.head(8))
```

```text
shape: (8, 6)
┌────────┬───────┬──────┬─────────────┬─────────────────┬─────────────────┐
│ panels ┆ cases ┆ alts ┆ target_alts ┆ elasticity_cost ┆ elasticity_time │
│ ---    ┆ ---   ┆ ---  ┆ ---         ┆ ---             ┆ ---             │
│ u32    ┆ u32   ┆ u32  ┆ u32         ┆ f64             ┆ f64             │
╞════════╪═══════╪══════╪═════════════╪═════════════════╪═════════════════╡
│ 0      ┆ 0     ┆ 0    ┆ 0           ┆ -3.966249       ┆ -0.370317       │
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆ 3.408495        ┆ 1.036888        │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆ 2.323005        ┆ 0.216892        │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -1.996333       ┆ -0.607298       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -4.431003       ┆ -0.612663       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆ 3.115549        ┆ 1.487897        │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆ 1.149616        ┆ 0.158954        │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -0.808324       ┆ -0.386032       │
└────────┴───────┴──────┴─────────────┴─────────────────┴─────────────────┘
```

`alts` identifies the alternative whose probability changes; `target_alts` identifies
the alternative whose attribute changes. Rows where the two are equal contain own
elasticities, and the remaining rows contain cross-elasticities. The calculation uses
the stored posterior class probabilities.

## 6. Marginal willingness-to-pay

Because `cost` is the numeraire, LCL computes the value of travel time as
$-\beta_{\text{time}}/\beta_{\text{cost}}$. We summarize it by income band and
gender. The raw `income_band` column remains available for grouping even though
Formulaic expands it internally for estimation.

Because `prediction` stores choice-updated probabilities, we request
`class_probabilities="prior"` for this population summary. Delta-method standard
errors currently propagate through the demographic prior, not through the posterior
update. Use `se="none"` to weight point estimates by stored posteriors.

```python
from lcl import PartitionType, WTPRequest

prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income_band",
               partition_type=PartitionType.CATEGORICAL),
    WTPRequest(alt_var="time", demographic_var="female",
               partition_type=PartitionType.CATEGORICAL),
    class_probabilities="prior",
)
```

```text
Marginal WTP for Travel time by Household income band (categorical)

--- LaTeX Output ---

\toprule
Household income band & Mean marginal WTP \\
\midrule
%
mid & -0.2173 \\
 & (0.0124) \\
...
%
\bottomrule

--- Table preview ---

┌─────────────────────────┬─────────────────────┐
│ Household income band   │ Mean marginal WTP   │
├─────────────────────────┼─────────────────────┤
│ mid                     │ -0.2173             │
│                         │ (0.0124)            │
│ high                    │ -0.2421             │
│                         │ (0.0158)            │
│ low                     │ -0.1818             │
│                         │ (0.0114)            │
└─────────────────────────┴─────────────────────┘

Marginal WTP for Travel time by Female traveler (categorical)

--- LaTeX Output ---

\toprule
Female traveler & Mean marginal WTP \\
\midrule
%
0.0 & -0.2120 \\
 & (0.0118) \\
1.0 & -0.2165 \\
 & (0.0122) \\
%
\bottomrule

--- Table preview ---

┌───────────────────┬─────────────────────┐
│   Female traveler │ Mean marginal WTP   │
├───────────────────┼─────────────────────┤
│            0.0000 │ -0.2120             │
│                   │ (0.0118)            │
│            1.0000 │ -0.2165             │
│                   │ (0.0122)            │
└───────────────────┴─────────────────────┘
```

Groups retain their first-observed order. Sort the returned frame when a prescribed
ordering is required.

The estimate varies more across income bands than across gender. Its sign follows the
coefficient convention: travel time enters utility as a disamenity.

!!! note "No manual one-hot encoding"
    `C(income_band)` handles estimation coding while `compute_wtp` groups on the
    original string column. Use `dummy_vars=` only when the grouping variable was
    encoded outside the fitted formula.

`wtp_by_class` returns each class-specific ratio.
`denominator_diagnostics` reports the corresponding numeraire coefficients and flags
the scale information needed to diagnose unstable ratios.

```python
print(prediction.wtp_by_class("time"))
print(prediction.denominator_diagnostics())
```

```text
shape: (3, 7)
┌──────────┬─────────────┬─────────────┬───────────────────┬───────┬───────────┬───────────────────┐
│ variable ┆ label       ┆ denominator ┆ denominator_label ┆ class ┆ tradeoff  ┆ denominator_value │
│ str      ┆ str         ┆ str         ┆ str               ┆ i64   ┆ f64       ┆ f64               │
╞══════════╪═════════════╪═════════════╪═══════════════════╪═══════╪═══════════╪═══════════════════╡
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 0     ┆ -0.130275 ┆ 0.100253          │
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 1     ┆ -0.308002 ┆ 0.035543          │
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 2     ┆ -0.195745 ┆ 0.056122          │
└──────────┴─────────────┴─────────────┴───────────────────┴───────┴───────────┴───────────────────┘
shape: (3, 6)
┌───────┬─────────────┬───────────────────┬───────────────────┬─────────────────┬───────────────┐
│ class ┆ denominator ┆ denominator_label ┆ denominator_value ┆ abs_denominator ┆ min_abs_floor │
│ i64   ┆ str         ┆ str               ┆ f64               ┆ f64             ┆ f64           │
╞═══════╪═════════════╪═══════════════════╪═══════════════════╪═════════════════╪═══════════════╡
│ 0     ┆ cost        ┆ Fare              ┆ 0.100253          ┆ 0.100253        ┆ 0.00001       │
│ 1     ┆ cost        ┆ Fare              ┆ 0.035543          ┆ 0.035543        ┆ 0.00001       │
│ 2     ┆ cost        ┆ Fare              ┆ 0.056122          ┆ 0.056122        ┆ 0.00001       │
└───────┴─────────────┴───────────────────┴───────────────────┴─────────────────┴───────────────┘
```

The smallest fare coefficient produces the largest-magnitude tradeoff. All reported
denominators remain comfortably above the configured floor.

WTP groups need not appear in the membership model. Any panel-constant prediction
column can define a partition; use `partition_data=...` when the grouping variable
lives in a separate table.

```python
prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income",
               partition_type=PartitionType.QUINTILES),
    class_probabilities="prior",
)
```

The fitted `LCLResults` object remains available for additional counterfactuals;
`predict` does not mutate the model.
