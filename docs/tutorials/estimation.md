# Estimation & counterfactuals

This tutorial fits a three-class latent-class conditional logit on the Apollo `modeChoice` dataset using patsy-style formulas (with `C(...)` categoricals), then uses the estimated model to evaluate a counterfactual fare increase and compute the value of time across income bands. The goal is to illustrate a standard end-to-end workflow.

## 1. Reshape the data

LCL expects a long-format DataFrame: one row per `(decision-maker, choice situation, alternative)` triple. The Apollo data ship in wide format, so the first step is a wide-to-long melt.

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
    An alternative that an individual could not have chosen still contributes to the denominator of every conditional choice probability unless it is dropped. Leaving unavailable alternatives in place silently biases the estimates.

## 2. Estimate the model

Let's estimate three latent classes, treating cost as the numeraire (so its coefficient is constrained strictly negative through a softplus reparameterization). Rather than enumerate the design columns by hand, we describe the utility and class-membership equations with **Formulaic (patsy-style) formula strings**, which buys us two things at once:

- `C(alt)` expands the string mode column (`car`, `bus`, `air`, `rail`) into a set of **alternative-specific constants**. Formulaic builds the dummies—and reuses the same base level at prediction time—so we never one-hot encode the categorical by hand.
- the class-membership equation is just a right-hand-side formula, and `C(income_band)` brings a categorical demographic into the membership design the same way.

So we first bin household income into a string `income_band` (a panel-level categorical), then declare the model. The numeraire is a `NegativeCoefficient` constraint, and the grouped options objects replace the scattered `em_alg_config`/`mle_config`/`error_config` keywords.

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
Estimation time: 14.165 seconds

--- LaTeX Output ---

\toprule
Variable & Means (\beta's) & Standard deviations (\sigma's) \\
\midrule
%
cost & -0.061 & 0.024 \\
 & (0.002) & (0.002) \\
time & -0.011 & 0.001 \\
 & (0.001) & (0.001) \\
C(alt)[T.bus] & -1.750 & 1.371 \\
 & (0.230) & (0.313) \\
C(alt)[T.car] & 1.085 & 0.742 \\
 & (0.130) & (0.150) \\
C(alt)[T.rail] & 0.418 & 0.321 \\
 & (0.057) & (0.066) \\
%
\bottomrule

--- Table preview ---

┌────────────────┬───────────────┬─────────────────────────────┐
│ Variable       │ Means (β's)   │ Standard deviations (σ's)   │
├────────────────┼───────────────┼─────────────────────────────┤
│ cost           │ -0.061        │ 0.024                       │
│                │ (0.002)       │ (0.002)                     │
│ time           │ -0.011        │ 0.001                       │
│                │ (0.001)       │ (0.001)                     │
│ C(alt)[T.bus]  │ -1.750        │ 1.371                       │
│                │ (0.230)       │ (0.313)                     │
│ C(alt)[T.car]  │ 1.085         │ 0.742                       │
│                │ (0.130)       │ (0.150)                     │
│ C(alt)[T.rail] │ 0.418         │ 0.321                       │
│                │ (0.057)       │ (0.066)                     │
└────────────────┴───────────────┴─────────────────────────────┘

<LCLResults: 3 Classes | Converged | Log likelihood: -6413.8 | CAIC: 12993.6 | BIC: 12970.6 | Adj. BIC: 12834.2>
```

!!! note "Formulas or explicit lists?"
    `LCLSpec` also accepts plain `utility=[...]` / `membership=[...]` column lists in place of the formula strings; the two interfaces are interchangeable. Reach for formulas when you want `C(...)` categoricals, interactions, or transformations; reach for the lists when your columns are already model-ready. (A combined `formula="choice ~ cost + time + C(alt) | C(income_band) + female"` is also accepted, but the split `utility_formula`/`membership_formula` pair reads more clearly.)

!!! tip "Watching the EM iterations"
    By default LCL prints only the final summary. It routes per-iteration progress (`EM recursion: …`, `Computing LCL covariance matrix`, the information criteria) through the standard library `logging` module, so a one-liner—`import logging; logging.basicConfig(level=logging.INFO)`—surfaces the full trace when you want it.

`summarize_betas()` reports population-level moments of the structural β's—the share-weighted mean and standard deviation across latent classes—with Delta-method standard errors in parentheses. Formulaic named the expanded columns for us: `C(alt)[T.bus]`, `C(alt)[T.car]`, and `C(alt)[T.rail]` are the alternative-specific constants relative to the omitted base (air), so on average travellers favour car and rail over air and shun the bus, holding cost and time fixed. The class-specific coefficients—including how each class weights those constants—are available with `results.class_coefficients()` and the latent-class composition with `results.class_shares()`; both feature in the diagnostics below.

## 3. Inspect the fit with the diagnostic tools

Before trusting the estimates, run the built-in diagnostics. `results.diagnostics()` collects fit, data, latent-class, and coefficient checks into one structured object; its `.print()` method renders them as a table with each check flagged `ok` or `warning`. (`repr` reports the count—`LCLDiagnostics(checks=9, warnings=0)`—and `.to_frame()` hands back a Polars frame for programmatic gating.)

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
latent_class  posterior_entropy_mean  ok            0.278251  Mean entropy of posterior class membership.
latent_class  min_class_share         ok            0.241837  Small classes can indicate weakly identified local optima.
latent_class  min_effective_panels    ok          120.919     Smallest posterior panel mass across classes.
coefficients  max_abs_beta            ok            3.74922   Largest absolute structural coefficient.
coefficients  min_abs_numeraire       ok            0.035543  Small numeraires can dominate WTP/tradeoff ratios.
```

The thresholds behind the `warning` flags are tunable through `DiagnosticsOptions` at fit time (for example, `DiagnosticsOptions(large_coefficient_threshold=10.0)`). For a one-glance convergence summary, call `convergence_report()`:

```python
print(results.convergence_report())
```

```text
Converged: True
EM recursions: 30
Final log likelihood: -6413.84
Warnings: 0
Last EM history row: {'em_iter': 30, 'loglik': -6413.840653814358, 'class_0_share': 0.24183714729667438, 'class_1_share': 0.305060238815295, 'class_2_share': 0.45310261388803075}
```

The latent-class composition deserves a look as well. `class_shares()` reports each class's aggregate share alongside its posterior ("effective") panel mass, and `class_coefficients()` returns the class-specific structural β's that the population moments above average over.

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
│ 0     ┆ 0.241837 ┆ 120.918571       │
│ 1     ┆ 0.30506  ┆ 152.530121       │
│ 2     ┆ 0.453103 ┆ 226.551308       │
└───────┴──────────┴──────────────────┘
shape: (15, 4)
┌────────────────┬───────┬─────────────┬─────────────┐
│ variable       ┆ class ┆ coefficient ┆ constrained │
│ ---            ┆ ---   ┆ ---         ┆ ---         │
│ str            ┆ i64   ┆ f64         ┆ bool        │
╞════════════════╪═══════╪═════════════╪═════════════╡
│ cost           ┆ 0     ┆ -0.100253   ┆ true        │
│ cost           ┆ 1     ┆ -0.035543   ┆ true        │
│ cost           ┆ 2     ┆ -0.056122   ┆ true        │
│ time           ┆ 0     ┆ -0.01306    ┆ false       │
│ time           ┆ 1     ┆ -0.010947   ┆ false       │
│ time           ┆ 2     ┆ -0.010986   ┆ false       │
│ C(alt)[T.bus]  ┆ 0     ┆ -0.290531   ┆ false       │
│ C(alt)[T.bus]  ┆ 1     ┆ -3.749216   ┆ false       │
│ C(alt)[T.bus]  ┆ 2     ┆ -1.182399   ┆ false       │
│ C(alt)[T.car]  ┆ 0     ┆ 2.062043    ┆ false       │
│ C(alt)[T.car]  ┆ 1     ┆ 0.078436    ┆ false       │
│ C(alt)[T.car]  ┆ 2     ┆ 1.240218    ┆ false       │
│ C(alt)[T.rail] ┆ 0     ┆ 0.904492    ┆ false       │
│ C(alt)[T.rail] ┆ 1     ┆ 0.031566    ┆ false       │
│ C(alt)[T.rail] ┆ 2     ┆ 0.418674    ┆ false       │
└────────────────┴───────┴─────────────┴─────────────┘
```

Class 0 is the most cost-sensitive ($\beta_{\text{cost}} = -0.100$) and the smallest, carrying about 121 panels of effective mass; it also shows the sharpest car preference (`C(alt)[T.car] = 2.06`). Class 1, by contrast, has a near-flat cost coefficient but a strong bus aversion (`C(alt)[T.bus] = -3.75`). Because `class_coefficients()` returns the expanded `C(alt)` constants per class, you can read the taste heterogeneity in the mode constants directly. For a replication appendix, `results.audit_report()` bundles the specification, fit statistics, class shares, and the diagnostics table into a single text block, while `results.em_history_` and `results.optimization_history_` expose the per-iteration log-likelihood path and the final M-step gradient norms as Polars frames.

## 4. A counterfactual fare increase, conditioned on observed choices

Suppose the regulator raises bus and rail fares by 25%. `predict` reuses the fitted encoder, so you only need to pass the modified DataFrame. The optional `past_choices` argument lets you condition the latent-class membership posterior on each decision-maker's observed choices. The intuition is that combining panels' revealed preferences with the (estimated) demographic prior provides sharper class assignments for counterfactual predictions. Here, we reuse `df_long`, which contains the very sequences on which we fitted the model, as the historical record. In practice, you might pass a separate frame of observed choices for each decision-maker prior to the policy change.

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
│ 1      ┆ 1     ┆ air  ┆ 0.206001     │
│ 1      ┆ 1     ┆ rail ┆ 0.793999     │
│ 1      ┆ 2     ┆ air  ┆ 0.556804     │
│ 1      ┆ 2     ┆ rail ┆ 0.443196     │
│ 1      ┆ 3     ┆ air  ┆ 0.879005     │
│ 1      ┆ 3     ┆ rail ┆ 0.120995     │
└────────┴───────┴──────┴──────────────┘
```

Panel 1 chose rail throughout the observed sample, so conditioning on those choices tilts its class posterior toward rail-friendly classes—lifting its rail probabilities above what the demographic prior alone would imply (cases 0 and 1), even though the 25% fare hike still hands air the more attractive option where the rail-cost penalty bites hardest (cases 2 and 3). The same posterior `class_probs_by_panel` is stored on `prediction` and informs the elasticity and welfare calculations below. Pass `past_choices` as a `PastChoicesData` instance instead of a DataFrame when you already manage design matrices and ID arrays directly.

The `LCLPrediction` object also reports expected consumer surplus by choice situation (the log-sum-exp inclusive value rescaled by marginal utility of income) and a per-panel willingness-to-pay frame. Both are useful as inputs to welfare analysis.

## 5. Elasticities

LCL computes the full table of own- and cross-price elasticities—that is, the percentage change in the probability of choosing alternative $j$ given a one-percent change in attribute $k$ of alternative $j'$—in one pass.

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
│ 0      ┆ 0     ┆ 0    ┆ 0           ┆ -3.966251       ┆ -0.370317       │
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆ 3.408497        ┆ 1.036888        │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆ 2.323006        ┆ 0.216892        │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -1.996333       ┆ -0.607298       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -4.431006       ┆ -0.612664       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆ 3.115551        ┆ 1.487898        │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆ 1.149616        ┆ 0.158954        │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -0.808324       ┆ -0.386032       │
└────────┴───────┴──────┴─────────────┴─────────────────┴─────────────────┘
```

`alts` indexes the alternative whose probability changes, while `target_alts` indexes the alternative whose attribute changes. Diagonal entries (`alts == target_alts`) represent own-price elasticities; the rest are cross-price elasticities. Because the elasticities are evaluated at the posterior class probabilities, each panel's table reflects what we have learned about that decision-maker's class membership. So, panels whose observed choices suggest they belong to cost-sensitive classes will show correspondingly larger own-price responses.

## 6. Marginal willingness-to-pay

Because `cost` is the declared numeraire, LCL computes the value of time analytically as the ratio $-\beta_{\text{time}}/\beta_{\text{cost}}$, quantifying uncertainty using the Delta method. We'll break it down by the very `income_band` we fed to `C(...)` in the membership formula, and by gender. Since `income_band` is a plain string column riding along in the prediction data (constant within panel), we partition on it directly—the `C(income_band)` expansion in the model and the raw column in the partition are two views of the same variable, and no `income_q2`/`income_q3`/… dummy bundle is built anywhere.

We built `prediction` with `past_choices`, so the probabilities it stores are Bayesian *posteriors*. Marginal WTP, by contrast, is a population summary whose Delta-method standard errors are propagated through the demographic *prior*—so we ask for `class_probabilities="prior"` explicitly. (To weight by the stored posteriors instead, pass `se="none"`; differentiating the standard errors through the posterior update is not supported.)

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
Marginal WTP for time by income_band (categorical)

--- LaTeX Output ---

\toprule
income\_band & Mean marginal WTP \\
\midrule
%
mid & -0.2173 \\
 & (0.0124) \\
...
%
\bottomrule

--- Table preview ---

┌───────────────┬─────────────────────┐
│ income_band   │ Mean marginal WTP   │
├───────────────┼─────────────────────┤
│ mid           │ -0.2173             │
│               │ (0.0124)            │
│ high          │ -0.2421             │
│               │ (0.0158)            │
│ low           │ -0.1818             │
│               │ (0.0114)            │
└───────────────┴─────────────────────┘

Marginal WTP for time by female (categorical)

--- LaTeX Output ---

\toprule
female & Mean marginal WTP \\
\midrule
%
0.0 & -0.2120 \\
 & (0.0118) \\
1.0 & -0.2165 \\
 & (0.0122) \\
%
\bottomrule

--- Table preview ---

┌──────────┬─────────────────────┐
│   female │ Mean marginal WTP   │
├──────────┼─────────────────────┤
│   0.0000 │ -0.2120             │
│          │ (0.0118)            │
│   1.0000 │ -0.2165             │
│          │ (0.0122)            │
└──────────┴─────────────────────┘
```

(Bands appear in the order they first occur in the panel-sorted data; sort the returned frame if you want a strict low–mid–high ordering.)

The value of time rises with income: the low band will pay roughly 0.18 cost units to shave a minute off the trip, the high band about 0.24. It proves essentially flat across gender once the income band is accounted for. The signs are negative because `time` enters utility as a disamenity—flip the convention if you prefer a marginal-cost framing.

!!! note "No manual one-hot encoding"
    We never built `income_q2`/`income_q3`/… indicators. `C(income_band)` handled the encoding inside the membership regression, and the partition above reused the raw string column. The `dummy_vars=` argument of `WTPRequest` is still there for partitions you one-hot encoded *outside* the model, but with the formula API you rarely need it.

To inspect the class-level building blocks behind those averages, `wtp_by_class` returns the ratio for each latent class, and `denominator_diagnostics` reports the numeraire coefficient sitting in every denominator—an easy way to catch a near-zero $\beta_{\text{cost}}$ that would blow up a tradeoff ratio.

```python
print(prediction.wtp_by_class("time"))
print(prediction.denominator_diagnostics())
```

```text
shape: (3, 5)
┌──────────┬─────────────┬───────┬───────────┬───────────────────┐
│ variable ┆ denominator ┆ class ┆ tradeoff  ┆ denominator_value │
│ ---      ┆ ---         ┆ ---   ┆ ---       ┆ ---               │
│ str      ┆ str         ┆ i64   ┆ f64       ┆ f64               │
╞══════════╪═════════════╪═══════╪═══════════╪═══════════════════╡
│ time     ┆ cost        ┆ 0     ┆ -0.130275 ┆ 0.100253          │
│ time     ┆ cost        ┆ 1     ┆ -0.308002 ┆ 0.035543          │
│ time     ┆ cost        ┆ 2     ┆ -0.195745 ┆ 0.056122          │
└──────────┴─────────────┴───────┴───────────┴───────────────────┘
shape: (3, 5)
┌───────┬─────────────┬───────────────────┬─────────────────┬───────────────┐
│ class ┆ denominator ┆ denominator_value ┆ abs_denominator ┆ min_abs_floor │
│ ---   ┆ ---         ┆ ---               ┆ ---             ┆ ---           │
│ i64   ┆ str         ┆ f64               ┆ f64             ┆ f64           │
╞═══════╪═════════════╪═══════════════════╪═════════════════╪═══════════════╡
│ 0     ┆ cost        ┆ 0.100253          ┆ 0.100253        ┆ 0.00001       │
│ 1     ┆ cost        ┆ 0.035543          ┆ 0.035543        ┆ 0.00001       │
│ 2     ┆ cost        ┆ 0.056122          ┆ 0.056122        ┆ 0.00001       │
└───────┴─────────────┴───────────────────┴─────────────────┴───────────────┘
```

Class 1 has the smallest cost coefficient ($\beta_{\text{cost}} = -0.036$), so it shows the largest value of time (−0.31) and the smallest denominator—well above the `min_abs_floor`, so no tradeoff blows up here.

You are not limited to the partitions used during estimation. Any panel-constant column carried through `predict(data=...)`—say a finer raw-`income` quintile cut that never entered the membership formula—can be summarized on the fly; pass `partition_data=...` when the grouping lives in a separate table.

```python
prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income",
               partition_type=PartitionType.QUINTILES),
    class_probabilities="prior",
)
```

That concludes a complete pass: ingest, estimate, predict, decompose. The same `LCLResults` object remains usable for further counterfactuals; nothing about `predict` mutates the fitted model.
