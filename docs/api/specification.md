# Specification & options

Define a model with [`LCLSpec`][lcl.spec.LCLSpec], then estimate it with
[`lcl.fit`][lcl.fit]. Four option objects configure the EM loop, Newton optimizer,
inference, and diagnostics without expanding the fitting signature.

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
