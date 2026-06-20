# Estimation & counterfactuals

This tutorial fits a three-class latent-class conditional logit on the Apollo `modeChoice` dataset, then uses the estimated model to evaluate a counterfactual fare increase and compute the value of time across income quintiles. The goal is to illustrate a standard end-to-end workflow.

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

Let's estimate three latent classes, treating cost as the numeraire (so its coefficient is constrained strictly negative through a softplus reparameterization). We'll model class membership as a function of two demographic variables: income and an indicator for being female.

The whole model lives in one `LCLSpec`. The numeraire is declared as a `NegativeCoefficient` constraint—optionally annotated with its units and a warning threshold—and the grouped options objects replace the scattered `em_alg_config`/`mle_config`/`error_config` keywords.

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

spec = LCLSpec(
    ids=ChoiceIds(alt="alt", case="qID", panel="ID", choice="choice"),
    utility=["cost", "time"],
    membership=["income", "female"],
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
Estimation time: 16.586 seconds

--- LaTeX Output ---

\toprule
Variable & Means (\beta's) & Standard deviations (\sigma's) \\
\midrule
%
cost & -0.046 & 0.010 \\
 & (0.001) & (0.002) \\
time & -0.011 & 0.003 \\
 & (0.000) & (0.000) \\
%
\bottomrule

--- Table preview ---

┌────────────┬───────────────┬─────────────────────────────┐
│ Variable   │ Means (β's)   │ Standard deviations (σ's)   │
├────────────┼───────────────┼─────────────────────────────┤
│ cost       │ -0.046        │ 0.010                       │
│            │ (0.001)       │ (0.002)                     │
│ time       │ -0.011        │ 0.003                       │
│            │ (0.000)       │ (0.000)                     │
└────────────┴───────────────┴─────────────────────────────┘

<LCLResults: 3 Classes | Converged | Log likelihood: -7618.6 | CAIC: 15323.8 | BIC: 15311.8 | Adj. BIC: 15240.9>
```

!!! tip "Watching the EM iterations"
    By default LCL prints only the final summary. It routes per-iteration progress (`EM recursion: …`, `Computing LCL covariance matrix`, the information criteria) through the standard library `logging` module, so a one-liner—`import logging; logging.basicConfig(level=logging.INFO)`—surfaces the full trace when you want it.

The table reports population-level moments of the structural β's—that is, the share-weighted mean and standard deviation across latent classes—with Delta-method standard errors in parentheses. The class-specific coefficients are available with `results.class_coefficients()` and the latent-class composition with `results.class_shares()`; both feature in the diagnostics below.

## 3. Inspect the fit with the diagnostic tools

Before trusting the estimates, run the built-in diagnostics. `results.diagnostics()` collects fit, data, latent-class, and coefficient checks into one structured object; its `.print()` method renders them as a table with each check flagged `ok` or `warning`. (`repr` reports the count—`LCLDiagnostics(checks=9, warnings=0)`—and `.to_frame()` hands back a Polars frame for programmatic gating.)

```python
results.diagnostics().print()
```

```text
section       check                   status            value  message
------------  ----------------------  --------  -------------  ----------------------------------------------------------
fit           converged               ok            1          EM convergence flag.
fit           log_likelihood          ok        -7618.63       Final unconditional log likelihood.
data          panels                  ok          500          Number of decision-maker panels.
data          cases                   ok         8000          Number of choice situations.
latent_class  posterior_entropy_mean  ok            0.370441   Mean entropy of posterior class membership.
latent_class  min_class_share         ok            0.193648   Small classes can indicate weakly identified local optima.
latent_class  min_effective_panels    ok           96.8239     Smallest posterior panel mass across classes.
coefficients  max_abs_beta            ok            0.0637565  Largest absolute structural coefficient.
coefficients  min_abs_numeraire       ok            0.03708    Small numeraires can dominate WTP/tradeoff ratios.
```

The thresholds behind the `warning` flags are tunable through `DiagnosticsOptions` at fit time (for example, `DiagnosticsOptions(large_coefficient_threshold=10.0)`). For a one-glance convergence summary, call `convergence_report()`:

```python
print(results.convergence_report())
```

```text
Converged: True
EM recursions: 40
Final log likelihood: -7618.63
Warnings: 0
Last EM history row: {'em_iter': 40, 'loglik': -7618.634492495526, 'class_0_share': 0.4367277421868584, 'class_1_share': 0.19364779816101987, 'class_2_share': 0.36962445965212176}
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
│ 0     ┆ 0.436728 ┆ 218.363871       │
│ 1     ┆ 0.193648 ┆ 96.823899        │
│ 2     ┆ 0.369624 ┆ 184.81223        │
└───────┴──────────┴──────────────────┘
shape: (6, 4)
┌──────────┬───────┬─────────────┬─────────────┐
│ variable ┆ class ┆ coefficient ┆ constrained │
│ ---      ┆ ---   ┆ ---         ┆ ---         │
│ str      ┆ i64   ┆ f64         ┆ bool        │
╞══════════╪═══════╪═════════════╪═════════════╡
│ cost     ┆ 0     ┆ -0.046454   ┆ true        │
│ cost     ┆ 1     ┆ -0.063756   ┆ true        │
│ cost     ┆ 2     ┆ -0.03708    ┆ true        │
│ time     ┆ 0     ┆ -0.009385   ┆ false       │
│ time     ┆ 1     ┆ -0.007646   ┆ false       │
│ time     ┆ 2     ┆ -0.014458   ┆ false       │
└──────────┴───────┴─────────────┴─────────────┘
```

Class 1 is the most cost-sensitive ($\beta_{\text{cost}} = -0.064$) but the smallest, carrying just under 97 panels of effective mass; class 2 is the most time-sensitive. For a replication appendix, `results.audit_report()` bundles the specification, fit statistics, class shares, and the diagnostics table into a single text block, while `results.em_history_` and `results.optimization_history_` expose the per-iteration log-likelihood path and the final M-step gradient norms as Polars frames.

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
│ 1      ┆ 0     ┆ air  ┆ 0.541747     │
│ 1      ┆ 0     ┆ rail ┆ 0.458253     │
│ 1      ┆ 1     ┆ air  ┆ 0.39825      │
│ 1      ┆ 1     ┆ rail ┆ 0.60175      │
│ 1      ┆ 2     ┆ air  ┆ 0.666931     │
│ 1      ┆ 2     ┆ rail ┆ 0.333069     │
│ 1      ┆ 3     ┆ air  ┆ 0.858308     │
│ 1      ┆ 3     ┆ rail ┆ 0.141692     │
└────────┴───────┴──────┴──────────────┘
```

Panel 1 chose rail in every observed situation, so the posterior tends towards rail-leaning classes. So, after the 25% fare increase, they're more likely to stick with rail travel than would be suggested by the demographic prior alone. The same posterior `class_probs_by_panel` is stored on `prediction` and informs the elasticity and welfare calculations below. Pass `past_choices` as a `PastChoicesData` instance instead of a DataFrame when you already manage design matrices and ID arrays directly.

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
│ 0      ┆ 0     ┆ 0    ┆ 0           ┆ -1.975733       ┆ -0.196076       │
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆ 1.697896        ┆ 0.549013        │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆ 2.335712        ┆ 0.231801        │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -2.007252       ┆ -0.649043       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -2.525647       ┆ -0.358665       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆ 1.775845        ┆ 0.871044        │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆ 1.67152         ┆ 0.237371        │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -1.175287       ┆ -0.576473       │
└────────┴───────┴──────┴─────────────┴─────────────────┴─────────────────┘
```

`alts` indexes the alternative whose probability changes, while `target_alts` indexes the alternative whose attribute changes. Diagonal entries (`alts == target_alts`) represent own-price elasticities; the rest are cross-price elasticities. Because the elasticities are evaluated at the posterior class probabilities, each panel's table reflects what we have learned about that decision-maker's class membership. So, panels whose observed choices suggest they belong to cost-sensitive classes will show correspondingly larger own-price responses.

## 6. Marginal willingness-to-pay

Because `cost` is the declared numeraire, LCL computes the value of time analytically as the ratio $-\beta_{\text{time}}/\beta_{\text{cost}}$, quantifying uncertainty using the Delta method. Partitions are evaluated lazily; when you ask for income quintiles, the encoder bins the panel-level demographics, and you receive one row per bin.

We built `prediction` with `past_choices`, so the probabilities it stores are Bayesian *posteriors*. Marginal WTP, by contrast, is a population summary whose Delta-method standard errors are propagated through the demographic *prior*—so we ask for `class_probabilities="prior"` explicitly. (To weight by the stored posteriors instead, pass `se="none"`; differentiating the standard errors through the posterior update is not supported.)

```python
from lcl import PartitionType, WTPRequest

prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income",
               partition_type=PartitionType.QUINTILES),
    WTPRequest(alt_var="time", demographic_var="female",
               partition_type=PartitionType.CATEGORICAL),
    class_probabilities="prior",
)
```

```text
Marginal WTP for time by income (quintiles)

--- LaTeX Output ---

\toprule
income & Mean marginal WTP \\
\midrule
%
Q3 & -0.2706 \\
 & (0.0086) \\
...
%
\bottomrule

--- Table preview ---

┌──────────┬─────────────────────┐
│ income   │ Mean marginal WTP   │
├──────────┼─────────────────────┤
│ Q3       │ -0.2706             │
│          │ (0.0086)            │
│ Q5       │ -0.2946             │
│          │ (0.0154)            │
│ Q1       │ -0.1891             │
│          │ (0.0095)            │
│ Q2       │ -0.2377             │
│          │ (0.0083)            │
│ Q4       │ -0.2859             │
│          │ (0.0109)            │
└──────────┴─────────────────────┘

Marginal WTP for time by female (categorical)

--- LaTeX Output ---

\toprule
female & Mean marginal WTP \\
\midrule
%
0.0 & -0.2554 \\
 & (0.0089) \\
1.0 & -0.2558 \\
 & (0.0097) \\
%
\bottomrule

--- Table preview ---

┌──────────┬─────────────────────┐
│   female │ Mean marginal WTP   │
├──────────┼─────────────────────┤
│   0.0000 │ -0.2554             │
│          │ (0.0089)            │
│   1.0000 │ -0.2558             │
│          │ (0.0097)            │
└──────────┴─────────────────────┘
```

(Quintile rows appear in the order each bin first occurs in the panel-sorted data; sort the returned frame if you want a strict Q1–Q5 ordering.)

The value of time rises with income: the lowest quintile will pay roughly 0.19 cost units to shave a minute off the trip, the highest about 0.29. It proves essentially flat across gender once income is accounted for. The signs are negative because `time` enters utility as a disamenity—flip the convention if you prefer a marginal-cost framing.

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
│ time     ┆ cost        ┆ 0     ┆ -0.202024 ┆ 0.046454          │
│ time     ┆ cost        ┆ 1     ┆ -0.119925 ┆ 0.063756          │
│ time     ┆ cost        ┆ 2     ┆ -0.3899   ┆ 0.03708           │
└──────────┴─────────────┴───────┴───────────┴───────────────────┘
shape: (3, 5)
┌───────┬─────────────┬───────────────────┬─────────────────┬───────────────┐
│ class ┆ denominator ┆ denominator_value ┆ abs_denominator ┆ min_abs_floor │
│ ---   ┆ ---         ┆ ---               ┆ ---             ┆ ---           │
│ i64   ┆ str         ┆ f64               ┆ f64             ┆ f64           │
╞═══════╪═════════════╪═══════════════════╪═════════════════╪═══════════════╡
│ 0     ┆ cost        ┆ 0.046454          ┆ 0.046454        ┆ 0.00001       │
│ 1     ┆ cost        ┆ 0.063756          ┆ 0.063756        ┆ 0.00001       │
│ 2     ┆ cost        ┆ 0.03708           ┆ 0.03708         ┆ 0.00001       │
└───────┴─────────────┴───────────────────┴─────────────────┴───────────────┘
```

If you discretized a continuous demographic variable before estimation, pass the dummy bundle as a single categorical factor. For example, a five-level income quintile factor with `income_q1` as the omitted base could be summarized in one request:

```python
wtp_tables = prediction.compute_wtp(
    WTPRequest(
        alt_var="time",
        demographic_var="income_quintile",
        partition_type=PartitionType.CATEGORICAL,
        dummy_vars=["income_q2", "income_q3", "income_q4", "income_q5"],
        dummy_labels=["Q2", "Q3", "Q4", "Q5"],
        base_category="Q1",
    ),
    class_probabilities="prior",
)
```

You can also summarize WTP using a panel-level partition that was not included in the class-membership regression. Raw categorical or binned columns passed through `predict(data=...)` are available automatically when they are constant within panel. If the partition lives in a separate table, provide it through `partition_data`.

```python
income_partitions = (
    df_long
    .select(["ID", "income_quintile"])
    .unique(subset=["ID"], maintain_order=True)
)

wtp_tables = prediction.compute_wtp(
    WTPRequest(
        alt_var="time",
        demographic_var="income_quintile",
        partition_type=PartitionType.CATEGORICAL,
    ),
    partition_data=income_partitions,
    panel_col="ID",
    class_probabilities="prior",
)
```

That concludes a complete pass: ingest, estimate, predict, decompose. The same `LCLResults` object remains usable for further counterfactuals; nothing about `predict` mutates the fitted model.
