# Conditional logit

The McFadden conditional logit estimates one taste vector for the full sample. It
provides a useful homogeneous benchmark for latent-class specifications.

Use `utility_formula` for new formula-based specifications, and use the current
optimization and inference option objects:

```python
from lcl import ConditionalLogit, InferenceOptions, OptimizationOptions, Options

results = ConditionalLogit(numeraire="price").fit(
    data,
    alts_col="alternative",
    cases_col="choice_situation",
    panels_col="respondent",
    utility_formula="chosen ~ price + time + C(mode)",
    weights="survey_weight",
    options=Options(
        optimization=OptimizationOptions(gradient_tol=1e-6),
        inference=InferenceOptions(covariance="clustered"),
    ),
)

coefficient_table = results.summarize_betas(show=False)
```

Weights are case-level. Prefer a column name or case-keyed mapping because those
forms preserve identity when rows are reordered. If case IDs repeat across panels,
key a mapping by `(panel_id, case_id)`. A sequence is interpreted in
first-case-appearance order and realigned after encoding.

With `panels_col`, BIC, CAIC, and adjusted BIC use the number of panels as their
sample size; otherwise they use the number of choice situations. A
softplus-constrained numeraire does not have an ordinary zero-null p-value, so its
reported p-value is `NaN`.

`covariance="clustered"` clusters at the panel level when `panels_col` is
provided. `covariance="robust"` always requests case-level Huber–White inference;
the two labels are not aliases. The result also reports the null log likelihood,
McFadden rho-squared, final score, and information diagnostics.

Prediction returns a [`CLPrediction`][lcl.results.CLPrediction] rather than a bare
frame. Probabilities remain in `prediction.predicted_probs`, with WTP,
elasticities, market shares, aggregate elasticities, denominator diagnostics,
and surplus available through the same methods as latent-class prediction.

```python
prediction = results.predict(counterfactual_data, panel_weights="survey_weight")
wtp = prediction.wtp("time", se="bootstrap", bootstrap_draws=1_000)
elasticities = prediction.elasticities(["price", "time"])
market_shares = prediction.market_shares()
```

## Model

::: lcl.ConditionalLogit

## Results

::: lcl.results.CLResults

::: lcl.results.CLPrediction
