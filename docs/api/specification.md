# Specification & options

Define a model with [`LCLSpec`][lcl.LCLSpec], then estimate it with
[`lcl.fit`][lcl.fit]. The four focused option objects can be collected in one
[`Options`][lcl.options.Options] bundle shared by every fitting entry point.

```python
import lcl
from lcl import FitOptions, InferenceOptions, OptimizationOptions, Options

results = lcl.fit(
    data,
    spec,
    options=Options(
        fit=FitOptions(seed=42, starts=3, max_em_iter=500),
        optimization=OptimizationOptions(maxiter=75, gradient_tol=1e-5),
        inference=InferenceOptions(covariance="clustered"),
    ),
)
```

The legacy `fit_options=`, `optimization_options=`, `inference=`, and
`diagnostics=` keywords remain available, but do not mix them with `options=`;
ambiguous partial merges raise an error.

Use separate `utility_formula` and `membership_formula` fields for formula-based
designs. `LCLSpec` is immutable, so it can be reused safely across fitting and
cross-validation.

## Fitting

::: lcl.fit

## Model specification

::: lcl.LCLSpec

::: lcl.ChoiceIds

::: lcl.NegativeCoefficient

## Options

::: lcl.options.FitOptions

::: lcl.options.OptimizationOptions

::: lcl.options.InferenceOptions

::: lcl.options.DiagnosticsOptions

::: lcl.options.Options
