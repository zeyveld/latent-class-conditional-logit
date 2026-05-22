# Cross-validation & model selection

Choosing the number of latent classes is an important modelling decision for any finite mixture. Information criteria (BIC, CAIC, adjusted BIC) are useful, and LCL reports all three on every fit. But these criteria can disagree, and they reward in-sample fit by construction. Out-of-sample log-likelihood under a panel-respecting split is the cleaner benchmark when you can spare the time (and shoulder the GPU expense).

`cv_optimal_classes` automates that exercise. Folds are drawn at the decision-maker level: an individual's entire choice history sits in exactly one fold so that a panel cannot leak between training and held-out data.

!!! warning "Experimental"
    The cross-validation utility is functional but still labelled experimental. Expect occasional refinements as I use this functionality in my own work.

## Running the sweep

Let's re-use the long-format Apollo frame from the [estimation tutorial](estimation.md). The search below evaluates two, three, four, and five classes with three-fold CV. To keep the example speedy on a single device, we trim the inner EM loop to twenty-five iterations; in practice you'd loosen this when the optimum is already obvious from a coarse sweep.

```python
import lcl
from lcl import EMAlgConfig, MleConfig

cv_results = lcl.cv_optimal_classes(
    data=df_long,
    alts_col="alt",
    cases_col="qID",
    panels_col="ID",
    choice_col="choice",
    case_varnames=["cost", "time"],
    dem_varnames=["income", "female"],
    numeraire="cost",
    num_classes_list=[2, 3, 4, 5],
    folds=3,
    seed=42,
    em_alg_config=EMAlgConfig(maxiter=25, num_devices=1),
    mle_config=MleConfig(maxiter=30),
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
│ 2           ┆ -2554.779595 │
│ 3           ┆ -2544.394819 │
│ 4           ┆ -2543.206710 │
│ 5           ┆ -2543.117616 │
└─────────────┴──────────────┘
```

The out-of-sample log-likelihood rises sharply from two to three classes, increases marginally at four, and remains essentially flat at five. On Apollo, three or four classes are defensible choices; the diminishing returns past four suggest that a fifth component is fitting noise.

## Plotting the curve

Eye-balling a table of likelihoods is easy when there are four rows, but tougher when you're comparing a dozen candidate Ks. Vega-Altair generates a tidy interactive curve:

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

Open `cv_plot.html` and you have an interactive plot of the curve with the peak highlighted. For a paper figure, swap `.save("cv_plot.html")` for `.save("cv_plot.svg")`.

## Practical notes

- **Stick with a small number of folds for initial screening.** Three folds is plenty to see the qualitative shape of the curve. Bump to five or ten once you've narrowed the range.
- **Use the same `EMAlgConfig` across folds.** Differing iteration budgets across folds confound the comparison.
- **Inspect a fold that fails to converge.** `cv_optimal_classes` swallows individual fold failures and records `NaN`; if you see one, refit that fold standalone with `LatentClassConditionalLogit.fit` to see the diagnostic logs.
- **Pair CV with the information criteria.** When CV picks $K^*$ and BIC picks $K^* - 1$, the latter is often the right call for inference; the smaller model trades a small amount of fit for tighter standard errors.
