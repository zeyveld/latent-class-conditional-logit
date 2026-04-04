# Estimation & Counterfactuals

This tutorial demonstrates an end-to-end workflow using LCL, from data formatting to out-of-sample counterfactual inference. We will use the classic **Apollo modeChoice** dataset, estimating a model where individuals choose between car, bus, air, and rail travel.

## 1. Data Preparation

Choice models require data in a strictly sorted, "long" format. The Apollo dataset is provided in a "wide" format (one row per choice situation). We use **Polars** to rapidly melt the dataset, filter out unavailable alternatives, and sort the panel data so it is ready for JAX.

```python
import urllib.request
import polars as pl
import lcl

# Fetch the dataset directly from the Apollo servers
url = "[https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv](https://www.apollochoicemodelling.com/files/examples/data/apollo_modeChoiceData.csv)"
df_wide = pl.read_csv(urllib.request.urlopen(url))

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

# CRITICAL: Filter out alternatives that were not in the choice set, and sort
df_long = df_long.filter(pl.col("av") == 1).sort(["ID", "qID", "alt"])
```

## 2. Model Estimation

We will estimate a model with 3 latent consumer classes. By specifying `numeraire="cost"`, LCL applies a softplus constraint to the cost parameter, ensuring marginal utility of income is strictly negative across all classes.

```python
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
from lcl._struct import EMAlgConfig

model = LatentClassConditionalLogit(num_classes=3, numeraire="cost")

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