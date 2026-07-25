# Conditional logit

The McFadden conditional logit estimates one taste vector for the full sample. It
provides a useful homogeneous benchmark for latent-class specifications.

Use `utility_formula` for new formula-based specifications, and use the current
optimization and inference option objects:

```python
from lcl import ConditionalLogit, InferenceOptions, OptimizationOptions

results = ConditionalLogit(numeraire="price").fit(
    data,
    alts_col="alternative",
    cases_col="choice_situation",
    panels_col="respondent",
    utility_formula="chosen ~ price + time + C(mode)",
    weights="survey_weight",
    optimization_options=OptimizationOptions(gradient_tol=1e-6),
    inference=InferenceOptions(covariance="clustered"),
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

## Model

::: lcl.conditional_logit.ConditionalLogit

## Results

::: lcl.conditional_logit.CLResults
