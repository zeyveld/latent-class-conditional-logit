# Cross-validation & model selection

The class count controls both fit and interpretability. LCL reports BIC, CAIC, and
adjusted BIC; panel-blocked cross-validation supplies a complementary measure of
out-of-sample performance.

`cv_optimal_classes` assigns each decision-maker's complete choice history to one
fold, preventing leakage between training and validation data.

!!! warning "Experimental"
    The cross-validation API remains experimental and may change between minor
    releases.

## Running the sweep

We reuse the long-format Apollo data and labeled `LCLSpec` from the
[estimation tutorial](estimation.md). `num_classes_list` overrides `spec.classes` for
each candidate. Every fold fits its formula encoder on the training data. The shorter
shorter budget below keeps the example practical, but it is not suitable for a final
model-selection decision.

```python
import lcl
from lcl import (
    ChoiceIds,
    FitOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
)

spec = LCLSpec(
    ids=ChoiceIds(alt="alt", case="qID", panel="ID", choice="choice"),
    utility_formula="choice ~ cost + time + C(alt)",
    membership_formula="~ C(income_band) + female",
    constraints={"cost": NegativeCoefficient()},
    variable_labels={
        "cost": "Fare",
        "time": "Travel time",
        "alt": "Travel mode",
        "income_band": "Household income band",
        "female": "Female traveler",
    },
)

cv_results = lcl.cv_optimal_classes(
    df_long,
    spec,
    num_classes_list=[2, 3, 4, 5],
    folds=3,
    seed=42,
    fit_options=FitOptions(max_em_iter=25, num_devices=1),
    optimization_options=OptimizationOptions(maxiter=30),
)
print(cv_results)
```

```text
shape: (4, 2)
┌─────────────┬──────────────┐
│ Num_Classes ┆ Avg_OOS_LL   │
│ ---         ┆ ---          │
│ i64         ┆ f64          │
╞═════════════╪══════════════╡
│ 2           ┆ -2178.832838 │
│ 3           ┆ -2148.768534 │
│ 4           ┆ -2147.732898 │
│ 5           ┆ -2148.065152 │
└─────────────┴──────────────┘
```

!!! warning "Illustrative iteration budget"
    These transcript values use 25 EM iterations, and the fold fits reached that
    cap. They verify the documented workflow but are not publication-ready model
    comparisons. Increase `max_em_iter` until every candidate fit satisfies the
    convergence checks before selecting a class count.

The held-out log likelihood improves substantially from two to three classes. The
differences among three, four, and five classes are much smaller, so the result should
be read alongside convergence diagnostics and information criteria rather than as a
decisive ranking.

## Plotting the curve

An interactive curve is easier to scan when the candidate set grows. Altair can
render the result directly:

```python
import altair as alt
import polars as pl

def plot_cv(cv_df: pl.DataFrame) -> alt.LayerChart:
    optimal = cv_df.filter(pl.col("Avg_OOS_LL") == pl.col("Avg_OOS_LL").max())

    line = (
        alt.Chart(cv_df)
           .mark_line(color="#3F2B47", size=3)
           .encode(
               x=alt.X("Num_Classes:O",
                       title="Number of latent classes",
                       axis=alt.Axis(labelAngle=0)),
               y=alt.Y("Avg_OOS_LL:Q",
                       title="Average out-of-sample log-likelihood",
                       scale=alt.Scale(zero=False)),
           )
    )
    points = line.mark_circle(size=80, color="#3F2B47", opacity=1)
    peak = (
        alt.Chart(optimal)
           .mark_circle(size=160, color="#E37449")
           .encode(x="Num_Classes:O", y="Avg_OOS_LL:Q",
                   tooltip=[
                       alt.Tooltip("Num_Classes:O", title="Classes"),
                       alt.Tooltip("Avg_OOS_LL:Q", title="OOS-LL", format=".2f"),
                   ])
    )
    label = peak.mark_text(align="left", baseline="middle",
                           dx=12, fontSize=12, fontWeight="bold",
                           color="#E37449").encode(text=alt.value("Best K"))

    return (line + points + peak + label).properties(
        title="Held-out log-likelihood by class count",
        width=600, height=380,
    ).configure_title(fontSize=16, anchor="start", offset=20)

plot_cv(cv_results).save("cv_plot.html")
```

Open `cv_plot.html` for the interactive chart. Save as SVG for a static figure.

## Practical notes

- **Use a small fold count for initial screening.** Increase it after narrowing the candidate range.
- **Use the same `FitOptions` across folds.** Differing iteration budgets across folds confound the comparison.
- **Investigate failed folds.** A `NaN` indicates a failed fold. Refit it directly and inspect the diagnostics.
- **Compare several criteria.** Prefer a simpler model when held-out performance is effectively tied and its classes are better identified.
