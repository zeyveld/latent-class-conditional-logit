# Cross-Validation & Model Selection

Selecting the optimal number of latent classes is a common challenge in finite mixture modeling. While information criteria (BIC, CAIC) are useful in-sample metrics, out-of-sample predictive performance is often the most rigorous benchmark.

LCL includes an experimental utility, `cv_optimal_classes`, which performs blocked K-Fold Cross Validation. 

!!! note "Panel-Safe Splitting"
    LCL automatically splits the cross-validation folds at the **panel level** (decision-maker). This ensures that an individual's choices do not accidentally leak across both the training and testing sets, preserving the integrity of the out-of-sample log-likelihood.

## Running the Cross-Validation

Using the formatted long dataset from the [Estimation Tutorial](estimation.md), we can evaluate the model with 2 through 6 latent classes.

```python
from lcl._wip_cross_validation import cv_optimal_classes
from lcl._struct import EMAlgConfig

# Evaluate K from 2 through 6
cv_results = cv_optimal_classes(
    data=df_long,
    alts_col="alt",
    cases_col="qID",
    panels_col="ID",
    choice_col="choice",
    case_varnames=["cost", "time"],
    dem_varnames=["income", "female"],
    numeraire="cost",
    num_classes_list=[2, 3, 4, 5, 6],
    folds=5,
    seed=42,
    em_alg_config=EMAlgConfig(maxiter=100) # Lower maxiter can speed up CV sweeps
)

print(cv_results)
```

## Visualizing the Optimal Classes

Reading a table of log-likelihoods can be unintuitive. Using a visualization library like **Vega-Altair**, you can quickly plot the CV curve to visually identify where predictive performance maximizes before the model begins overfitting.

```python
import altair as alt
import polars as pl

def plot_cv_results(cv_df: pl.DataFrame) -> alt.LayerChart:
    # Identify the optimal K
    optimal_df = cv_df.filter(pl.col("Avg_OOS_LL") == pl.col("Avg_OOS_LL").max())

    line = alt.Chart(cv_df).mark_line(color="#1f77b4", size=3).encode(
        x=alt.X("K:O", title="Number of Latent Classes (K)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Avg_OOS_LL:Q", title="Average Out-of-Sample Log-Likelihood", scale=alt.Scale(zero=False)),
    )

    points = line.mark_circle(size=80, color="#1f77b4", opacity=1)

    optimal_point = alt.Chart(optimal_df).mark_circle(size=150, color="#d62728").encode(
        x="K:O",
        y="Avg_OOS_LL:Q",
        tooltip=[
            alt.Tooltip("K:O", title="Classes (K)"),
            alt.Tooltip("Avg_OOS_LL:Q", title="OOS-LL", format=".2f"),
        ],
    )

    text = optimal_point.mark_text(
        align="left", baseline="middle", dx=12, fontSize=12, fontWeight="bold", color="#d62728"
    ).encode(text=alt.value("Optimal K"))

    chart = (line + points + optimal_point + text).properties(
        title="Cross-Validation: Optimal Number of Latent Classes", width=600, height=400
    ).configure_title(fontSize=16, anchor="start", offset=20)
    
    return chart

# Render the interactive HTML plot
chart = plot_cv_results(cv_results)
chart.save("cv_plot.html")
```
When viewing `cv_plot.html` in a browser, you will see a clean, interactive line plot highlighting the exact point where out-of-sample likelihood is maximized.