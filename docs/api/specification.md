# Specification & options

The high-level entry point. Describe the model declaratively with an [`LCLSpec`][lcl.spec.LCLSpec], then estimate it with [`lcl.fit`][lcl.fit]. Behaviour is tuned through four grouped options objects rather than a long keyword list: `FitOptions` (the EM loop), `OptimizationOptions` (the M-step optimizer), `InferenceOptions` (covariance and standard errors), and `DiagnosticsOptions` (the post-fit health checks).

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
