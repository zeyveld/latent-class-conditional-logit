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

Use `utility_formula` and `membership_formula` for formula-based designs. The
combined `formula` field is deprecated. `LCLSpec` is immutable, so it can be
reused safely across fitting and cross-validation.

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

## Compatibility objects

`EMAlgConfig`, `MleConfig`, and `ErrorConfig` remain available while existing
analyses migrate. Their corresponding keyword arguments emit deprecation warnings.
Passing an old and new object for the same responsibility raises `ValueError`.

| Deprecated | Preferred |
| --- | --- |
| `em_alg_config=EMAlgConfig(...)` | `fit_options=FitOptions(...)` |
| `mle_config=MleConfig(ftol=...)` | `optimization_options=OptimizationOptions(gradient_tol=...)` |
| `error_config=ErrorConfig(...)` | `inference=InferenceOptions(...)` |

See [Best practices & migration](../guides/best_practices.md) for complete
before-and-after examples.
