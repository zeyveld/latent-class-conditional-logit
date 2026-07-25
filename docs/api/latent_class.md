# Latent-class conditional logit

The latent-class estimator fits a finite mixture of conditional logits by
expectation-maximization. Class membership may be represented by aggregate shares or
modeled as a function of panel characteristics with fractional multinomial logit.

Most users should start with [`LCLSpec` and `lcl.fit`](specification.md). The lower-level
class remains available for direct use.

```python
import lcl
from lcl import FitOptions

results = lcl.fit(data, spec, fit_options=FitOptions(starts=3))

# Lower-level orchestration when a persistent model object is useful:
model = lcl.LatentClassConditionalLogit(spec=spec)
results = model.fit(data, fit_options=FitOptions(starts=3))
```

Pass `spec` by keyword to the lower-level constructor. The positional
`LatentClassConditionalLogit(spec)` form is deprecated.

## Model

::: lcl.latent_class_conditional_logit.LatentClassConditionalLogit

## Results

::: lcl._results.LCLResults

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
```

## Diagnostics

::: lcl._diagnostics.LCLDiagnostics

## Prediction and counterfactuals

Tabular prediction is preferred because it reuses the fitted encoder:

```python
prediction = results.predict(data=counterfactual_data)
```

For array-style prediction, supply `dem_panel_ids` with `dems` so demographic
rows can be validated and reordered. Without those IDs, demographic rows must
follow sorted unique panel-ID order.

::: lcl._prediction.LCLPrediction

::: lcl._struct.WTPRequest

::: lcl._struct.PartitionType

::: lcl._struct.PastChoicesData
