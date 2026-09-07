# Latent-class conditional logit

The latent-class estimator fits a finite mixture of conditional logits by
expectation-maximization. Class membership may be represented by aggregate shares or
modeled as a function of panel characteristics with fractional multinomial logit.

Most users should start with [`LCLSpec` and `lcl.fit`](specification.md). The lower-level
class remains available for direct use.

```python
import lcl
from lcl import FitOptions, Options

options = Options(fit=FitOptions(starts=3))
results = lcl.fit(data, spec, options=options)

# Lower-level orchestration when a persistent model object is useful:
model = lcl.LatentClassConditionalLogit(spec=spec)
results = model.fit(data, options=options)
```

Pass `spec` by keyword to the lower-level constructor. Explicit `num_classes`
and `numeraire_min_abs` override their base-specification values; omitted values
inherit them. The historical direct-constructor default without a spec is five
classes, while `LCLSpec` defaults to two. Create a new model for each fit.

## Model

::: lcl.LatentClassConditionalLogit

## Results

::: lcl.results.LCLResults

### Held-out scoring

`LCLResults.loglik` transforms observed choices with the fitted encoder:

```python
total_ll = results.loglik(test_data)
panel_ll = results.loglik(test_data, per_panel=True)
```

The panel-level form returns original panel IDs and their log-likelihood
contributions. It is also the scoring path used by cross-validation.

Summary methods return Polars frames. Pass `show=False` to suppress their LaTeX
and terminal renderings:

```python
summary = results.summarize_betas(show=False)
class_coefficients = results.class_coefficients()  # includes std_error
membership = results.membership_coefficients()    # class 0 is the reference
classification = results.classification_diagnostics()
```

`parameter_names()` labels covariance rows and columns exactly. `converged`,
`cov_matrix`, and `adjusted_bic` are the canonical names shared with conditional
logit; `convergence`, `covariance`, and `abic` are deprecated aliases.

## Diagnostics

::: lcl.results.LCLDiagnostics

## Prediction and counterfactuals

Tabular prediction is preferred because it reuses the fitted encoder:

```python
prediction = results.predict(data=counterfactual_data)
shares = prediction.market_shares()
aggregate = prediction.aggregate_elasticities(["cost", "time"])
```

Pass `panel_weights=` to `predict` as a panel-keyed mapping, a prediction-data
column name, or a vector in sorted prediction-panel order. WTP supports
`se="delta"`, `se="bootstrap"` (an asymptotic parametric bootstrap), and
`se="none"`. Both inference methods propagate uncertainty through the Bayesian
update when `past_choices` is supplied. History can cover a subset of prediction
consumers; `prediction.class_membership()` reports the probability source and
history count for each consumer. Prediction demographics supply the prior.

Surplus frames include `surplus_units` (`money` with a numeraire, otherwise
`utils`). Use `baseline_prediction.surplus_change(counterfactual_prediction)`
for the identified welfare change rather than comparing unnormalised levels.
Changes check model, consumer/occasion identity, and weights, and include
`change_identified`. Monetary summaries require a linear numeraire without extra
price transforms or interactions. `marginal_wtp("quality")` evaluates the full
raw-attribute derivative at each offered profile; `compute_wtp` aggregates it by
consumer and demographic group. See the
[economic definitions and worked examples](../tutorials/prediction_welfare.md).

For array-style prediction, supply `dem_panel_ids` with `dems` so demographic
rows can be validated and reordered. Tabular and array prediction inputs cannot
be combined in one call. Without those IDs, demographic rows must
follow sorted unique panel-ID order.

::: lcl.results.LCLPrediction

::: lcl.options.WTPRequest

::: lcl.options.PartitionType

::: lcl.options.PastChoicesData
