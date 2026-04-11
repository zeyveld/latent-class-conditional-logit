# Estimation & Counterfactuals

This tutorial demonstrates an end-to-end workflow using LCL, from data formatting to out-of-sample counterfactual inference. We will use the classic **Apollo modeChoice** dataset, estimating a model where individuals choose between car, bus, air, and rail travel.

## 1. Data Preparation

Choice models require data in a strictly sorted, "long" format. The Apollo dataset is provided in a "wide" format (one row per choice situation). We use **Polars** to rapidly melt the dataset, filter out unavailable alternatives, and sort the panel data so it is ready for JAX.

```python
import polars as pl
import lcl

# Fetch the dataset directly from the Apollo servers
df_wide = pl.read_csv("[https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv](https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv)")

# Create a unique choice situation ID
df_wide = df_wide.with_row_index("qID")

alts_map = {1: "car", 2: "bus", 3: "air", 4: "rail"}
dfs = []

# Melt to long format
for num, name in alts_map.items():
    alt_df = df_wide.select([
        pl.col("ID"),
        pl.col("qID"),
        pl.col("income"),
        pl.col("female"),
        pl.col(f"time_{name}").alias("time"),
        pl.col(f"cost_{name}").alias("cost"),
        pl.col(f"av_{name}").alias("av"), # Availability flag
        (pl.col("choice") == num).alias("choice")
    ]).with_columns(pl.lit(name).alias("alt"))
    dfs.append(alt_df)

df_long = pl.concat(dfs)

# CRITICAL: Filter out alternatives that were not in the choice set
df_long = df_long.filter(pl.col("av") == 1).sort(["ID", "qID", "alt"])
```

**Expected Output:**
```text
df_wide = pl.read_csv("https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv")

# Create a unique choice situation ID
df_wide = df_wide.with_row_index("qID")

alts_map = {1: "car", 2: "bus", 3: "air", 4: "rail"}
dfs = []

# Melt to long format
for num, name in alts_map.items():
...     alt_df = df_wide.select([
...         pl.col("ID"),
...         pl.col("qID"),
...         pl.col("income"),
...         pl.col("female"),
...         pl.col(f"time_{name}").alias("time"),
...         pl.col(f"cost_{name}").alias("cost"),
...         pl.col(f"av_{name}").alias("av"), # Availability flag
...         (pl.col("choice") == num).alias("choice")
...     ]).with_columns(pl.lit(name).alias("alt"))
...     dfs.append(alt_df)
...
df_long = pl.concat(dfs)

# CRITICAL: Filter out alternatives that were not in the choice set, and sort
df_long = df_long.filter(pl.col("av") == 1).sort(["ID", "qID", "alt"])
df_long = pl.read_parquet("df_long.parquet")
```

## 2. Model Estimation

We will estimate a model with 4 latent consumer classes. By specifying `numeraire="cost"`, LCL applies a softplus constraint to the cost parameter, ensuring marginal utility of income is strictly negative across all classes.

```python
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
from lcl._struct import EMAlgConfig

model = LatentClassConditionalLogit(num_classes=4, numeraire="cost")

results = model.fit(
    data=df_long,
    alts_col="alt",
    cases_col="qID",
    panels_col="ID",
    choice_col="choice",
    case_varnames=["cost", "time"],
    dem_varnames=["income", "female"],
    em_alg_config=EMAlgConfig(maxiter=500)
)

# Output population-level moments with robust standard errors
results.summarize_betas()
print(results)
```

**Expected Output:**
```text
Hardware Status: Distributing 4 classes across 4 GPUs.
EM recursion: 0
EM recursion: 1
EM recursion: 2
...
EM recursion: 48
EM recursion: 49
Estimation time: 350.06757950782776

Computing clustered covariance matrix per steps in Stata's reference manual...
(StataCorp. 2025. Stata 19 Base Reference Manual. College Station, TX: Stata Press.)

Information criteria:
  Consistent Aikake information criterion (CAIC; see Bozdogan [1987]): 15346.4
  Bayesian information criterion (BIC; see Schwartz [1978]): 15329.4
  Adjusted BIC (see Sclove [1987]): 15228.7

--- LaTeX Output ---

\toprule
Variable & Means (\beta's) & Standard deviations (\sigma's) \\
\midrule
%
cost & -0.047 & 0.012 \\
 & (0.001) & (0.002) \\
time & -0.011 & 0.003 \\
 & (0.000) & (0.000) \\
%
\bottomrule

--- Table preview ---

┌────────────┬───────────────┬─────────────────────────────┐
│ Variable   │ Means (β's)   │ Standard deviations (σ's)   │
├────────────┼───────────────┼─────────────────────────────┤
│ cost       │ -0.047        │ 0.012                       │
│            │ (0.001)       │ (0.002)                     │
│ time       │ -0.011        │ 0.003                       │
│            │ (0.000)       │ (0.000)                     │
└────────────┴───────────────┴─────────────────────────────┘

<LCLResults: 4 Classes | Converged | Log likelihood: -7611.9> | CAIC: 15346.4 | BIC: 15329.4 | Adj. BIC: 15228.7>
```

## 3. Counterfactual Inference & Elasticities

Suppose we want to evaluate the impact of a 25% fare increase on transit alternatives (bus and rail). We can modify our dataset and generate counterfactual choice probabilities.

```python
# Create the counterfactual scenario
cf_df = df_long.with_columns(
    pl.when(pl.col("alt").is_in(["bus", "rail"]))
    .then(pl.col("cost") * 1.25)
    .otherwise(pl.col("cost"))
    .alias("cost")
)

# Ingest the counterfactual data using the same specification
parsed_cf = model._ingest_data(
    data=cf_df,
    alts_col="alt",
    cases_col="qID",
    panels_col="ID",
    choice_col="choice",
    case_varnames=["cost", "time"],
    dem_varnames=["income", "female"],
    formula=None,
    dems_data=None,
)

# Generate out-of-sample predictions and consumer surplus
predictions = results.predict(
    X=parsed_cf.X,
    alts=parsed_cf.alts,
    cases=parsed_cf.cases,
    panels=parsed_cf.panels,
    dems=parsed_cf.dems,
)

print(predictions.predicted_probs.head())
```

**Expected Output:**
```text
shape: (5, 4)
┌────────┬───────┬──────┬──────────────┐
│ panels ┆ cases ┆ alts ┆ choice_probs │
│ ---    ┆ ---   ┆ ---  ┆ ---          │
│ u32    ┆ u32   ┆ u32  ┆ f64          │
╞════════╪═══════╪══════╪══════════════╡
│ 0      ┆ 0     ┆ 0    ┆ 0.610393     │
│ 0      ┆ 0     ┆ 3    ┆ 0.389607     │
│ 0      ┆ 1     ┆ 0    ┆ 0.497739     │
│ 0      ┆ 1     ┆ 3    ┆ 0.502261     │
│ 0      ┆ 2     ┆ 0    ┆ 0.72851      │
└────────┴───────┴──────┴──────────────┘
```

We can also extract the full N x N x K matrix of own- and cross-elasticities directly from the prediction container:

```python
elasticities_df = predictions.elasticities(vars=["cost", "time"])
print(elasticities_df.head())
```

## 4. Marginal Willingness-to-Pay (WTP)

Because we specified a numeraire, LCL can analytically compute the marginal Willingness-to-Pay (the ratio of the time coefficient to the cost coefficient). Using the Delta Method, LCL derives analytical standard errors for this non-linear ratio, automatically partitioned by demographic bins.

```python
from lcl._struct import WTPRequest, PartitionType

# Calculate Value of Time by income quintile
wtp_req_income = WTPRequest(
    alt_var="time", 
    demographic_var="income", 
    partition_type=PartitionType.QUINTILES
)

# Calculate Value of Time by gender
wtp_req_gender = WTPRequest(
    alt_var="time",
    demographic_var="female",
    partition_type=PartitionType.CATEGORICAL,
)

predictions.compute_wtp(wtp_req_income, wtp_req_gender)
```

**Expected Output:**
```text
===========================================
Marginal WTP for time by income (quintiles)
===========================================
shape: (5, 3)
| income | Mean_Marginal_WTP | Standard_Error |
| ---    | ---               | ---            |
| str    | f64               | f64            |
|--------|-------------------|----------------|
| Q3     | -0.2758           | 0.0094         |
| Q5     | -0.2969           | 0.0155         |
| Q1     | -0.1837           | 0.0092         |
| Q2     | -0.2347           | 0.0088         |
| Q4     | -0.2936           | 0.0121         |

=============================================
Marginal WTP for time by female (categorical)
=============================================
shape: (2, 3)
| female | Mean_Marginal_WTP | Standard_Error |
| ---    | ---               | ---            |
| str    | f64               | f64            |
|--------|-------------------|----------------|
| 0.0    | -0.2576           | 0.0091         |
| 1.0    | -0.2563           | 0.0104         |
```