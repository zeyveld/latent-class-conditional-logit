# Estimation & counterfactuals

This walkthrough fits a three-class latent-class conditional logit on the Apollo `modeChoice` dataset, then uses the estimated model to evaluate a counterfactual fare increase and compute the value of time across income quintiles. The goal is to illustrate a standard end-to-end workflow.

## 1. Reshape the data

LCL expects a long-format dataframe: one row per `(decision-maker, choice situation, alternative)` triple. The Apollo data ships in wide format, so the first step is a melt. Polars handles this in milliseconds even for millions of rows.

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
      .filter(pl.col("av") == 1)        # drop unavailable alternatives
      .sort(["ID", "qID", "alt"])       # required: panel, then case, then alt
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
    A row that an individual could not have chosen still contributes to the denominator of every conditional choice probability unless it is dropped. Leaving unavailable rows in place silently biases the estimates.

## 2. Estimate the model

Three latent classes, cost as the numeraire (so its coefficient is constrained strictly negative through a softplus reparametrization), and demographics — income and a female indicator — driving class membership.

```python
import lcl
from lcl._struct import EMAlgConfig, ErrorConfig, MleConfig

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

The table reports population-level moments of the structural β's — i.e., the share-weighted mean and standard deviation across latent classes, with Delta-method standard errors in parentheses. Class-specific coefficients are in `results.em_res.structural_betas`, while posterior class-membership probabilities by panel are in `results.em_res.class_probs_by_panel`.

## 3. A counterfactual fare increase

Suppose a regulator raises bus and rail fares by 25 percent. `predict` re-uses the fitted encoder, so you only need to pass the modified DataFrame.

```python
cf_df = df_long.with_columns(
    pl.when(pl.col("alt").is_in(["bus", "rail"]))
      .then(pl.col("cost") * 1.25)
      .otherwise(pl.col("cost"))
      .alias("cost")
)

prediction = results.predict(data=cf_df)
print(prediction.predicted_probs.head(8))
```

```text
shape: (8, 4)
┌────────┬───────┬──────┬──────────────┐
│ panels ┆ cases ┆ alts ┆ choice_probs │
│ ---    ┆ ---   ┆ ---  ┆ ---          │
│ i64    ┆ u32   ┆ str  ┆ f64          │
╞════════╪═══════╪══════╪══════════════╡
│ 1      ┆ 0     ┆ air  ┆ 0.629067     │
│ 1      ┆ 0     ┆ rail ┆ 0.370933     │
│ 1      ┆ 1     ┆ air  ┆ 0.526480     │
│ 1      ┆ 1     ┆ rail ┆ 0.473520     │
│ 1      ┆ 2     ┆ air  ┆ 0.744888     │
│ 1      ┆ 2     ┆ rail ┆ 0.255112     │
│ 1      ┆ 3     ┆ air  ┆ 0.831142     │
│ 1      ┆ 3     ┆ rail ┆ 0.168858     │
└────────┴───────┴──────┴──────────────┘
```

The `LCLPrediction` object also reports expected consumer surplus by choice situation (the log-sum-exp inclusive value rescaled by marginal utility of income) and a per-panel willingness-to-pay frame. Both are useful as inputs to welfare analysis.

## 4. Elasticities

LCL computes the full table of own- and cross-elasticities — the percentage change in the probability of choosing alternative $j$ given a one-percent change in attribute $v$ of alternative $k$ — in one pass.

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
│ 0      ┆ 0     ┆ 0    ┆ 0           ┆ -1.281914       ┆ -0.203956       │
│ 0      ┆ 0     ┆ 0    ┆ 1           ┆  1.101645       ┆  0.571078       │
│ 0      ┆ 0     ┆ 1    ┆ 0           ┆  2.174003       ┆  0.345890       │
│ 0      ┆ 0     ┆ 1    ┆ 1           ┆ -1.868284       ┆ -0.968492       │
│ 0      ┆ 1     ┆ 0    ┆ 0           ┆ -1.583079       ┆ -0.361833       │
│ 0      ┆ 1     ┆ 0    ┆ 1           ┆  1.113102       ┆  0.878737       │
│ 0      ┆ 1     ┆ 1    ┆ 0           ┆  1.760136       ┆  0.402301       │
│ 0      ┆ 1     ┆ 1    ┆ 1           ┆ -1.237595       ┆ -0.977018       │
└────────┴───────┴──────┴─────────────┴─────────────────┴─────────────────┘
```

`alts` indexes the alternative whose probability changes, and `target_alts` indexes the alternative whose attribute changes. Diagonal entries (`alts == target_alts`) represent own-elasticities; the rest are cross-elasticities.

## 5. Marginal willingness-to-pay

Because we declared `numeraire="cost"`, LCL computes the value of time analytically as the ratio $-\beta_{\text{time}}/\beta_{\text{cost}}$ and propagates uncertainty using the Delta method. Partitions are evaluated lazily — you ask for income quintiles, the encoder bins the panel-level demographics, and you receive one row per bin.

```python
from lcl._struct import PartitionType, WTPRequest

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

The value-of-time rises monotonically with income — wealthier households are willing to pay more to save a minute on the journey — and proves essentially flat across gender after accounting for income. The signs are negative because `time` enters utility as a disamenity; flip the sign convention if you prefer the marginal cost framing.

That covers a complete pass: ingest, estimate, predict, decompose. The same `LCLResults` object remains usable for further counterfactuals; nothing about `predict` mutates the fitted model.
