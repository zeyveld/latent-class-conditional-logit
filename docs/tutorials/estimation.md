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
    fit_options=FitOptions(
        seed=42,
        starts=3,
        max_em_iter=60,
        num_devices=1,
    ),
    optimization_options=OptimizationOptions(
        maxiter=40,
        gradient_tol=1e-5,
    ),
    inference=InferenceOptions(covariance="clustered"),
)

summary = results.summarize_betas()
print(results)
```

`max_em_iter` is a hard cap on all complete EM recursions, including the strict
final-refit phase. The estimator reserves one recursion for that phase. If the
strict recursion moves the relative log likelihood by more than `em_tol`, it keeps
running EM with the stricter M-step settings until a consecutive strict recursion
moves by at most `em_tol` or the remaining recursion budget is exhausted. This
usually adds only a few recursions near the optimum while preventing a loose
M-step convergence check from being reported as final convergence.

Representative output (the exact optimum can vary with the JAX version and
available hardware):

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
Travel mode: bus & -1.749 & 1.371 \\
 & (0.229) & (0.313) \\
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
│ Travel mode: bus  │ -1.749        │ 1.371                       │
│                   │ (0.229)       │ (0.313)                     │
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
    interactions, or transformations. Keep utility and class-membership designs
    in their separate fields so each design remains easy to audit and reuse.

!!! tip "Use several starts for reported mixture models"
    `FitOptions(starts=3, seed=42)` runs three independent panel-partition starts
    with seeds 42, 43, and 44, then refits the best start with the requested
    inference settings. Increase `starts` for consequential analyses.

!!! tip "Watching the EM iterations"
    LCL sends iterative progress to Python's `logging` module. Enable it with
    `logging.basicConfig(level=logging.INFO)` when you need the full optimization
    trace. Fitting does not write timing messages directly to standard output.

`summarize_betas()` reports the share-weighted mean and standard deviation of each
structural coefficient, with delta-method standard errors in parentheses. The table
uses labels such as **Travel mode: bus**; the returned frame also retains Formulaic's
raw name, `C(alt)[T.bus]`. Use `results.class_coefficients()` for class-specific
coefficients and `results.class_shares()` for the estimated class composition.
In a pipeline that provides its own display, use
`results.summarize_betas(show=False)` (or the `results.summarize(show=False)`
alias) to obtain the same frame without printing LaTeX and terminal tables.

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
latent_class  posterior_entropy_mean  ok            0.27833   Mean entropy of posterior class membership.
latent_class  min_class_share         ok            0.241938  Small classes can indicate weakly identified local optima.
latent_class  min_effective_panels    ok          120.969     Smallest posterior panel mass across classes.
coefficients  max_abs_beta            ok            3.74783   Largest absolute structural coefficient.
coefficients  min_abs_numeraire       ok            0.035545  Small numeraires can dominate WTP/tradeoff ratios.
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
Last EM history row: {'em_iter': 30, 'loglik': -6413.840782946274, 'class_0_share': 0.45298103347438456, 'class_1_share': 0.24193776632753392, 'class_2_share': 0.3050812001980814}
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
│ 0     ┆ 0.452981 ┆ 226.490521       │
│ 1     ┆ 0.241938 ┆ 120.968878       │
│ 2     ┆ 0.305081 ┆ 152.5406         │
└───────┴──────────┴──────────────────┘
shape: (15, 5)
┌────────────────┬───────────────────┬───────┬─────────────┬─────────────┐
│ variable       ┆ label             ┆ class ┆ coefficient ┆ constrained │
│ ---            ┆ ---               ┆ ---   ┆ ---         ┆ ---         │
│ str            ┆ str               ┆ i64   ┆ f64         ┆ bool        │
╞════════════════╪═══════════════════╪═══════╪═════════════╪═════════════╡
│ cost           ┆ Fare              ┆ 0     ┆ -0.056118   ┆ true        │
│ cost           ┆ Fare              ┆ 1     ┆ -0.10024    ┆ true        │
│ cost           ┆ Fare              ┆ 2     ┆ -0.035545   ┆ true        │
│ time           ┆ Travel time       ┆ 0     ┆ -0.010986   ┆ false       │
│ time           ┆ Travel time       ┆ 1     ┆ -0.013059   ┆ false       │
│ …              ┆ …                 ┆ …     ┆ …           ┆ …           │
│ C(alt)[T.car]  ┆ Travel mode: car  ┆ 1     ┆ 2.061444    ┆ false       │
│ C(alt)[T.car]  ┆ Travel mode: car  ┆ 2     ┆ 0.07853     ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 0     ┆ 0.418737    ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 1     ┆ 0.904207    ┆ false       │
│ C(alt)[T.rail] ┆ Travel mode: rail ┆ 2     ┆ 0.031596    ┆ false       │
└────────────────┴───────────────────┴───────┴─────────────┴─────────────┘
```

The class-specific estimates reveal heterogeneity that population averages conceal.
`results.audit_report()` combines the specification, fit statistics, class shares,
and diagnostics in a text report. `results.em_history_` and
`results.optimization_history_` expose iteration histories as Polars frames.

## Score observed choices without refitting

Use `loglik` to evaluate a validation or test sample with the fitted model. The
method reuses the training encoder, including Formulaic's categorical levels and
expanded column order.

```python
validation_ll = results.loglik(validation_long)
validation_by_panel = results.loglik(validation_long, per_panel=True)

print(validation_ll)
print(validation_by_panel.head(3))
```

```text
-1927.846
shape: (3, 2)
┌───────┬────────────────┐
│ panel ┆ log_likelihood │
│ ---   ┆ ---            │
│ i64   ┆ f64            │
╞═══════╪════════════════╡
│ 12    ┆ -11.4902       │
│ 27    ┆ -13.0018       │
│ 31    ┆ -10.7725       │
└───────┴────────────────┘
```

The numbers above are illustrative; the returned `panel` column always contains
the original panel IDs. An unseen categorical level produces Formulaic's warning
instead of being silently assigned a different code.

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
│ 1      ┆ 0     ┆ air  ┆ 0.369339     │
│ 1      ┆ 0     ┆ rail ┆ 0.630661     │
│ 1      ┆ 1     ┆ air  ┆ 0.205966     │
│ 1      ┆ 1     ┆ rail ┆ 0.794034     │
│ 1      ┆ 2     ┆ air  ┆ 0.556787     │
│ 1      ┆ 2     ┆ rail ┆ 0.443213     │
│ 1      ┆ 3     ┆ air  ┆ 0.879016     │
│ 1      ┆ 3     ┆ rail ┆ 0.120984     │
└────────┴───────┴──────┴──────────────┘
```

The resulting `prediction.class_probs_by_panel` contains the updated class
probabilities used for choice probabilities, elasticities, and welfare measures.
Callers who already manage encoded arrays may pass `PastChoicesData` instead of a
DataFrame.

`past_choices` must contain exactly the same panel IDs as the counterfactual data.
LCL validates that set before matching posterior class probabilities to prediction
panels, preventing silent reassignment when one input contains missing or extra
decision-makers.

`LCLPrediction` also reports expected consumer surplus by choice situation and a
panel-level willingness-to-pay frame for downstream welfare analysis.

!!! tip "Prefer tabular prediction when possible"
    `results.predict(data=...)` is the safest interface because the fitted encoder
    aligns formulas and panel characteristics. For array-style prediction, pass
    `dem_panel_ids` alongside `dems`; LCL validates and reorders the demographic
    rows. Without IDs, demographic rows must follow sorted unique panel-ID order.
    A demographic membership model cannot be predicted without `dems`. The same
    alignment rule applies to `PastChoicesData(dems=..., dem_panel_ids=...)`.

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
│ 0      ┆ 0     ┆ 0    ┆ 0           ┆ -3.966785       ┆ -0.370332       │
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆ 3.408956        ┆ 1.036931        │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆ 2.323102        ┆ 0.216881        │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -1.996415       ┆ -0.607266       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -4.431907       ┆ -0.612719       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆ 3.116184        ┆ 1.488032        │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆ 1.149604        ┆ 0.158935        │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -0.808315       ┆ -0.385984       │
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

wtp_tables = prediction.compute_wtp(
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

Categorical groups retain their first-observed order. Numeric quintiles and custom
break partitions are returned in numeric bin order.

The estimate varies more across income bands than across gender. Its sign follows the
coefficient convention: travel time enters utility as a disamenity.

`wtp_tables` is a dictionary whose keys are the displayed titles and whose values
are the corresponding Polars frames. Use `show=False` when a notebook, test, or
application should receive those frames without terminal and LaTeX output.

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
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 0     ┆ -0.195762 ┆ 0.056118          │
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 1     ┆ -0.130273 ┆ 0.10024           │
│ time     ┆ Travel time ┆ cost        ┆ Fare              ┆ 2     ┆ -0.308005 ┆ 0.035545          │
└──────────┴─────────────┴─────────────┴───────────────────┴───────┴───────────┴───────────────────┘
shape: (3, 6)
┌───────┬─────────────┬───────────────────┬───────────────────┬─────────────────┬───────────────┐
│ class ┆ denominator ┆ denominator_label ┆ denominator_value ┆ abs_denominator ┆ min_abs_floor │
│ i64   ┆ str         ┆ str               ┆ f64               ┆ f64             ┆ f64           │
╞═══════╪═════════════╪═══════════════════╪═══════════════════╪═════════════════╪═══════════════╡
│ 0     ┆ cost        ┆ Fare              ┆ 0.056118          ┆ 0.056118        ┆ 0.00001       │
│ 1     ┆ cost        ┆ Fare              ┆ 0.10024           ┆ 0.10024         ┆ 0.00001       │
│ 2     ┆ cost        ┆ Fare              ┆ 0.035545          ┆ 0.035545        ┆ 0.00001       │
└───────┴─────────────┴───────────────────┴───────────────────┴─────────────────┴───────────────┘
```

The smallest fare coefficient produces the largest-magnitude tradeoff. All reported
denominators remain comfortably above the configured floor.

WTP groups need not appear in the membership model. Any panel-constant prediction
column can define a partition; use `partition_data=...` when the grouping variable
lives in a separate table.

```python
income_quintiles = prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income",
               partition_type=PartitionType.QUINTILES),
    class_probabilities="prior",
    show=False,
)
```

The fitted `LCLResults` object remains available for additional counterfactuals;
`predict` does not mutate the model.
