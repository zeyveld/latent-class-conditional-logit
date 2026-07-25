# Best practices & migration

LCL's high-level API keeps model specification, estimation settings, and
post-estimation work separate. New projects should follow the patterns on this
page. The older interfaces remain available temporarily so existing analyses can
be migrated deliberately.

## Specify the model once

Put identifiers, formulas, class count, constraints, and labels in an
[`LCLSpec`][lcl.spec.LCLSpec], then reuse that immutable specification for fitting
and cross-validation.

```python
import lcl
from lcl import ChoiceIds, LCLSpec, NegativeCoefficient

spec = LCLSpec(
    ids=ChoiceIds(
        alt="alternative",
        case="choice_situation",
        panel="respondent",
        choice="chosen",
    ),
    utility_formula="chosen ~ price + time + C(mode)",
    membership_formula="~ income + C(region)",
    classes=3,
    constraints={"price": NegativeCoefficient()},
)

results = lcl.fit(data, spec)
```

Use separate `utility_formula` and `membership_formula` fields. The combined
`formula="choice ~ ... | ..."` form is deprecated because a single string makes
the two designs harder to inspect and reuse.

For lower-level orchestration, pass the specification by keyword:

```python
model = lcl.LatentClassConditionalLogit(spec=spec)
results = model.fit(data)
```

Passing an `LCLSpec` as the first positional argument to
`LatentClassConditionalLogit` or `cv_optimal_classes` is deprecated.

## Group settings by responsibility

Use the three current option objects:

```python
from lcl import FitOptions, InferenceOptions, OptimizationOptions

results = lcl.fit(
    data,
    spec,
    fit_options=FitOptions(
        seed=42,
        starts=5,
        max_em_iter=500,
        em_tol=1e-6,
    ),
    optimization_options=OptimizationOptions(
        maxiter=75,
        gradient_tol=1e-5,
    ),
    inference=InferenceOptions(covariance="clustered"),
)
```

- `FitOptions` controls the EM loop and independent starting values.
- `OptimizationOptions` controls the exact-Newton M-step. `gradient_tol` is the
  tolerance on the mean negative-log-likelihood gradient.
- `InferenceOptions` controls covariance work and standard errors.

Latent-class likelihoods can have local optima, so use more than one start for
reported estimates and final model comparisons. Start `i` uses
`FitOptions.seed + i`; setting the seed therefore makes the sweep reproducible.

`EMAlgConfig`, `MleConfig`, and `ErrorConfig` are deprecated compatibility
objects. Do not pass a legacy object together with its replacement; LCL rejects
the conflict instead of guessing which setting should win.

## Score held-out choices with the fitted encoder

Use [`LCLResults.loglik`][lcl._results.LCLResults.loglik] rather than rebuilding
formula matrices. This keeps training-time categorical coding and column order
intact.

```python
total_log_likelihood = results.loglik(test_data)
panel_scores = results.loglik(test_data, per_panel=True)
```

`panel_scores` has one row per original panel ID:

```text
shape: (3, 2)
┌────────────┬────────────────┐
│ panel      ┆ log_likelihood │
│ ---        ┆ ---            │
│ str        ┆ f64            │
╞════════════╪════════════════╡
│ person-101 ┆ -10.2841       │
│ person-205 ┆ -12.9176       │
│ person-410 ┆ -9.6632        │
└────────────┴────────────────┘
```

Formulaic warns when held-out data contains a category that was absent during
training. Treat that warning as a data/specification issue; do not silently
remap the category.

## Keep panel alignment explicit

Tabular prediction is the safest default because the fitted encoder aligns IDs,
formula columns, and demographics:

```python
prediction = results.predict(data=counterfactual_data)
```

When arrays are already prepared, pass demographic panel IDs so LCL can validate
and reorder the demographic rows:

```python
prediction = results.predict(
    X=X,
    alts=alternative_ids,
    cases=case_ids,
    panels=panel_ids,
    dems=demographics,
    dem_panel_ids=demographic_panel_ids,
)
```

Without `dem_panel_ids`, rows of `dems` must follow sorted unique panel-ID order.
If the fitted membership model uses demographics, omitting `dems` is an error.
The same rule applies to [`PastChoicesData`][lcl._struct.PastChoicesData].

## Separate computation from display

Summary and WTP methods return structured Polars frames. Use `show=False` in
notebooks, tests, pipelines, and web applications that provide their own display:

```python
coefficient_table = results.summarize_betas(show=False)

wtp_tables = prediction.compute_wtp(
    request,
    class_probabilities="prior",
    show=False,
)
```

Keep the default `show=True` for an immediate LaTeX and terminal preview.
`compute_wtp` returns a dictionary from the displayed title to its corresponding
data frame. Numeric quintile and custom-break partitions are returned in numeric
bin order.

## Prefer keyed case weights

Conditional-logit weights are defined per choice situation. A column name or
mapping makes their identity explicit and remains safe if the input rows are
reordered:

```python
cl_results = lcl.ConditionalLogit().fit(
    data,
    alts_col="alternative",
    cases_col="choice_situation",
    panels_col="respondent",
    utility_formula="chosen ~ price + time",
    weights="survey_weight",
    optimization_options=OptimizationOptions(gradient_tol=1e-6),
    inference=InferenceOptions(covariance="clustered"),
)
```

A mapping may be keyed by case ID. If case IDs repeat across panels, key it by
`(panel_id, case_id)`. A sequence is also accepted in first-case-appearance order,
but keyed forms are more robust for durable data pipelines.

## Migration reference

| Older pattern | Current pattern |
| --- | --- |
| `LCLSpec(formula="choice ~ x \| z")` | `LCLSpec(utility_formula="choice ~ x", membership_formula="~ z")` |
| `LatentClassConditionalLogit(spec)` | `LatentClassConditionalLogit(spec=spec)` |
| `cv_optimal_classes(data, spec, ...)` | `cv_optimal_classes(data, spec=spec, ...)` |
| `em_alg_config=EMAlgConfig(...)` | `fit_options=FitOptions(...)` |
| `mle_config=MleConfig(ftol=...)` | `optimization_options=OptimizationOptions(gradient_tol=...)` |
| `error_config=ErrorConfig(...)` | `inference=InferenceOptions(...)` |
| Capture printed summaries | Use the returned frame and `show=False` |
| Manually rebuild held-out formula matrices | `results.loglik(held_out_data)` |

Deprecation warnings are intentional migration aids. Update the call site rather
than filtering them globally.
