# Specification & options

Define a model with [`LCLSpec`][lcl.spec.LCLSpec], then estimate it with
[`lcl.fit`][lcl.fit]. Three option objects configure the EM loop, Newton
optimizer, and inference without expanding the fitting signature.
`DiagnosticsOptions` separately controls diagnostic thresholds.

```python
import lcl
from lcl import FitOptions, InferenceOptions, OptimizationOptions

results = lcl.fit(
    data,
    spec,
    fit_options=FitOptions(seed=42, starts=3, max_em_iter=500),
    optimization_options=OptimizationOptions(
        maxiter=75,
        gradient_tol=1e-5,
    ),
    inference=InferenceOptions(covariance="clustered"),
)
```

Use separate `utility_formula` and `membership_formula` fields for formula-based
designs. `LCLSpec` is immutable, so it can be reused safely across fitting and
cross-validation.

## Fitting

::: lcl.fit

## Model specification

::: lcl.spec.LCLSpec

::: lcl.spec.ChoiceIds

::: lcl.constraints.NegativeCoefficient

## Options

::: lcl._struct.FitOptions

::: lcl._struct.OptimizationOptions

::: lcl._struct.InferenceOptions

::: lcl._struct.DiagnosticsOptions
