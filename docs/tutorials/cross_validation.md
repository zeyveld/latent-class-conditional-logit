# Cross-validation & model selection

The class count controls both fit and interpretability. LCL reports BIC, CAIC, and
adjusted BIC; panel-blocked cross-validation supplies a complementary measure of
out-of-sample performance.

[`cv_optimal_classes`][lcl._cross_validation.cv_optimal_classes] assigns each
decision-maker's complete choice history to one fold. No panel contributes choices
to both training and validation data.

!!! warning "Experimental"
    The cross-validation API remains experimental and may change between minor
    releases.

## Run a reproducible sweep

Reuse the labeled `LCLSpec` from the
[estimation tutorial](estimation.md). `num_classes_list` replaces `spec.classes`
for each candidate while leaving the rest of the empirical specification unchanged.
Pass the specification with the explicit `spec=` keyword.

Each fold fits its formula encoder on training data and uses that fitted encoder to
score the holdout. This preserves training-time categorical coding and prevents a
validation fold from influencing its own feature construction.

```python
import lcl
import polars as pl
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
    spec=spec,
    num_classes_list=[2, 3, 4, 5],
    folds=3,
    seed=42,
    fit_options=FitOptions(
        seed=42,
        starts=3,
        max_em_iter=60,
        num_devices=1,
    ),
    optimization_options=OptimizationOptions(
        maxiter=30,
        gradient_tol=1e-5,
    ),
)

print(
    cv_results.select(
        "Num_Classes",
        "Avg_OOS_LL",
        "Successful_Folds",
        "Failed_Folds",
        "Converged_Folds",
    )
)
```

Representative output:

```text
shape: (4, 5)
┌─────────────┬────────────┬──────────────────┬──────────────┬─────────────────┐
│ Num_Classes ┆ Avg_OOS_LL ┆ Successful_Folds ┆ Failed_Folds ┆ Converged_Folds │
│ ---         ┆ ---        ┆ ---              ┆ ---          ┆ ---             │
│ i64         ┆ f64        ┆ i64              ┆ i64          ┆ i64             │
╞═════════════╪════════════╪══════════════════╪══════════════╪═════════════════╡
│ 2           ┆ -13.0736   ┆ 3                ┆ 0            ┆ 3               │
│ 3           ┆ -12.8926   ┆ 3                ┆ 0            ┆ 3               │
│ 4           ┆ -12.8788   ┆ 3                ┆ 0            ┆ 1               │
│ 5           ┆ -12.8800   ┆ 3                ┆ 0            ┆ 0               │
└─────────────┴────────────┴──────────────────┴──────────────┴─────────────────┘
```

Numerical estimates can vary with JAX version, device, iteration budget, and
starting values. The important change from older LCL transcripts is the scale:
`Avg_OOS_LL` is the pooled mean held-out log likelihood **per panel**, not an
average of fold totals. Unequal fold sizes therefore receive the correct weight.

!!! warning "Illustrative iteration budget"
    Some four- and five-class fold fits reached the 60-iteration cap. The sweep
    demonstrates the workflow but is not a publication-ready model comparison.
    Increase `max_em_iter` until every successful fold converges, and use several
    starts before selecting a class count.

The held-out score improves substantially from two to three classes. The
differences among three, four, and five classes are much smaller, so the result
should be read alongside convergence diagnostics, class identification, and
information criteria.

Cross-validation skips covariance estimation by default because covariance does
not affect held-out likelihood. Pass an explicit `InferenceOptions` only when fold
level inference is genuinely needed.

## Inspect fold-level diagnostics

The result contains aggregate metrics and equal-length list columns for the folds:

- `Fold_OOS_LL` and `Fold_Mean_Panel_LL`
- `Fold_Converged`
- `Fold_Train_Panels` and `Fold_Test_Panels`
- `Fold_Errors`

Explode those columns to inspect one candidate:

```python
fold_details = (
    cv_results
    .filter(pl.col("Num_Classes") == 4)
    .select(
        "Fold",
        "Fold_OOS_LL",
        "Fold_Mean_Panel_LL",
        "Fold_Converged",
        "Fold_Train_Panels",
        "Fold_Test_Panels",
        "Fold_Errors",
    )
    .explode(
        "Fold",
        "Fold_OOS_LL",
        "Fold_Mean_Panel_LL",
        "Fold_Converged",
        "Fold_Train_Panels",
        "Fold_Test_Panels",
        "Fold_Errors",
    )
)
print(fold_details)
```

If any fold fails, the candidate's `Avg_OOS_LL` and `Total_OOS_LL` are `NaN`.
This prevents an incomplete candidate from silently competing with candidates
evaluated on every panel. `Avg_Successful_OOS_LL` is retained only as a debugging
aid, and `Fold_Errors` records the error text.

Nonconvergence is reported separately from failure. A fold can produce a finite
held-out score while `Fold_Converged` is false; do not use that candidate as a
final estimate until its optimization settings are adequate.

## Plot the comparison

Plot only complete candidates. The filter below prevents a partially evaluated
candidate from being marked as best:

```python
import altair as alt


def plot_cv(cv_df: pl.DataFrame) -> alt.LayerChart:
    """Plot complete cross-validation candidates and mark the best score."""
    complete = cv_df.filter(pl.col("Failed_Folds") == 0)
    optimal = complete.filter(
        pl.col("Avg_OOS_LL") == pl.col("Avg_OOS_LL").max()
    )

    line = (
        alt.Chart(complete)
        .mark_line(color="#3F2B47", size=3)
        .encode(
            x=alt.X(
                "Num_Classes:O",
                title="Number of latent classes",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Avg_OOS_LL:Q",
                title="Mean held-out log likelihood per panel",
                scale=alt.Scale(zero=False),
            ),
        )
    )
    points = line.mark_circle(size=80, color="#3F2B47", opacity=1)
    peak = (
        alt.Chart(optimal)
        .mark_circle(size=160, color="#E37449")
        .encode(
            x="Num_Classes:O",
            y="Avg_OOS_LL:Q",
            tooltip=[
                alt.Tooltip("Num_Classes:O", title="Classes"),
                alt.Tooltip(
                    "Avg_OOS_LL:Q",
                    title="Mean panel OOS-LL",
                    format=".3f",
                ),
            ],
        )
    )
    label = peak.mark_text(
        align="left",
        baseline="middle",
        dx=12,
        fontSize=12,
        fontWeight="bold",
        color="#E37449",
    ).encode(text=alt.value("Best complete candidate"))

    return (line + points + peak + label).properties(
        title="Held-out log likelihood by class count",
        width=600,
        height=380,
    )


plot_cv(cv_results).save("cv_plot.html")
```

## Practical checklist

- Use a small fold count and candidate range for initial screening, then rerun
  the short list with a publication-quality iteration budget.
- Use the same `FitOptions` and `OptimizationOptions` across candidates.
- Use several starts for final comparisons; mixture models can settle at local
  optima.
- Require `Failed_Folds == 0` and investigate every nonconverged fold.
- Prefer a simpler model when held-out performance is effectively tied and its
  classes are easier to identify and interpret.
