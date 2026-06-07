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

```python
import lcl
from lcl import EMAlgConfig, ErrorConfig, MleConfig

model = lcl.LatentClassConditionalLogit(num_classes=3, numeraire="cost")

results = model.fit(
    data=df_long,
    alts_col="alt",
    cases_col="qID",
    panels_col="ID",
    choice_col="choice",
    case_varnames=["cost", "time"],
    dem_varnames=["income", "female"],
    em_alg_config=EMAlgConfig(maxiter=60, num_devices=1),
    mle_config=MleConfig(maxiter=40),
    error_config=ErrorConfig(robust=True),
)

results.summarize_betas()
print(results)
```

```text
Running beta updates on a single device.
EM recursion: 0
EM recursion: 1
...
EM recursion: 39
Estimation time: 16.956 seconds
Computing LCL covariance matrix.
Information criteria: CAIC=15323.8, BIC=15311.8, adjusted BIC=15240.9

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

┌──────────┬─────────────┬───────────────────────────┐
│ Variable │ Means (β's) │ Standard deviations (σ's) │
├──────────┼─────────────┼───────────────────────────┤
│ cost     │ -0.046      │ 0.010                     │
│          │ (0.001)     │ (0.002)                   │
│ time     │ -0.011      │ 0.003                     │
│          │ (0.000)     │ (0.000)                   │
└──────────┴─────────────┴───────────────────────────┘

<LCLResults: 3 Classes | Converged | Log likelihood: -7618.6 |
 CAIC: 15323.8 | BIC: 15311.8 | Adj. BIC: 15240.9>
```

The table reports population-level moments of the structural β's—that is, the share-weighted mean and standard deviation across latent classes—with Delta-method standard errors in parentheses. Class-specific coefficients can be found in `results.em_res.structural_betas`, while posterior class-membership probabilities by panel are located in `results.em_res.class_probs_by_panel`.

## 3. A counterfactual fare increase, conditioned on observed choices

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
│ 1      ┆ 1     ┆ air  ┆ 0.398250     │
│ 1      ┆ 1     ┆ rail ┆ 0.601750     │
│ 1      ┆ 2     ┆ air  ┆ 0.666931     │
│ 1      ┆ 2     ┆ rail ┆ 0.333069     │
│ 1      ┆ 3     ┆ air  ┆ 0.858308     │
│ 1      ┆ 3     ┆ rail ┆ 0.141692     │
└────────┴───────┴──────┴──────────────┘
```

Panel 1 chose rail in every observed situation, so the posterior tends towards rail-leaning classes. So, after the 25% fare increase, they're more likely to stick with rail travel than would be suggested by the demographic prior alone. The same posterior `class_probs_by_panel` is stored on `prediction` and informs the elasticity and welfare calculations below. Pass `past_choices` as a `PastChoicesData` instance instead of a DataFrame when you already manage design matrices and ID arrays directly.

The `LCLPrediction` object also reports expected consumer surplus by choice situation (the log-sum-exp inclusive value rescaled by marginal utility of income) and a per-panel willingness-to-pay frame. Both are useful as inputs to welfare analysis.

## 4. Elasticities

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
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆  1.697896       ┆  0.549013       │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆  2.335712       ┆  0.231801       │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -2.007252       ┆ -0.649043       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -2.525647       ┆ -0.358665       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆  1.775845       ┆  0.871044       │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆  1.671520       ┆  0.237371       │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -1.175287       ┆ -0.576473       │
└────────┴───────┴──────┴─────────────┴─────────────────┴─────────────────┘
```

`alts` indexes the alternative whose probability changes, while `target_alts` indexes the alternative whose attribute changes. Diagonal entries (`alts == target_alts`) represent own-price elasticities; the rest are cross-price elasticities. Because the elasticities are evaluated at the posterior class probabilities, each panel's table reflects what we have learned about that decision-maker's class membership. So, panels whose observed choices suggest they belong to cost-sensitive classes will show correspondingly larger own-price responses.

## 5. Marginal willingness-to-pay

Because we declared `numeraire="cost"`, LCL computes the value of time analytically as the ratio $-\beta_{\text{time}}/\beta_{\text{cost}}$, quantifying uncertainty using the Delta method. Partitions are evaluated lazily; when you ask for income quintiles, the encoder bins the panel-level demographics, and you receive one row per bin.

```python
from lcl import PartitionType, WTPRequest

prediction.compute_wtp(
    WTPRequest(alt_var="time", demographic_var="income",
               partition_type=PartitionType.QUINTILES),
    WTPRequest(alt_var="time", demographic_var="female",
               partition_type=PartitionType.CATEGORICAL),
)
```

```text
Marginal WTP for time by income (quintiles)
shape: (5, 3)
| income | Mean_Marginal_WTP | Standard_Error |
|--------|-------------------|----------------|
| Q1     | -0.1891           | 0.0095         |
| Q2     | -0.2377           | 0.0083         |
| Q3     | -0.2705           | 0.0086         |
| Q4     | -0.2859           | 0.0109         |
| Q5     | -0.2946           | 0.0154         |

Marginal WTP for time by female (categorical)
shape: (2, 3)
| female | Mean_Marginal_WTP | Standard_Error |
|--------|-------------------|----------------|
| 0.0    | -0.2553           | 0.0089         |
| 1.0    | -0.2558           | 0.0097         |
```

The value of time tends to rise with income—wealthier households are willing to pay more to save a minute on the journey—and proves essentially flat across gender after accounting for income. Notice that the signs are negative: this is because `time` enters utility as a disamenity. (Flip the sign convention if you prefer a marginal cost framing.)

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
    )
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
)
```

That concludes a complete pass: ingest, estimate, predict, decompose. The same `LCLResults` object remains usable for further counterfactuals; nothing about `predict` mutates the fitted model.
