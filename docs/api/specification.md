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
        optimization=OptimizationOptions(maxiter=75, newton_decrement_tol=1e-5),
        inference=InferenceOptions(covariance="clustered"),
    ),
)
```

The legacy `fit_options=`, `optimization_options=`, `inference=`, and
`diagnostics=` keywords remain available, but do not mix them with `options=`;
ambiguous partial merges raise an error. See [API contracts and compatibility](contracts.md)
for configuration precedence, which sections each estimator uses, and array ordering.

Use separate `utility_formula` and `membership_formula` fields for formula-based
designs. `LCLSpec` has frozen top-level fields and can be reused across fitting
and cross-validation. Treat its nested variable lists and mappings as read-only;
use `dataclasses.replace` to derive another specification.

## Boundary inference options (0.1.42)

```python
inference = InferenceOptions(
    covariance="clustered", boundary="projected",
    boundary_draws=2048, boundary_seed=0,
)
results = lcl.fit(data, spec, inference=inference)
summary = results.beta_summary()
```

| Option | Default | Contract |
| --- | --- | --- |
| `boundary` | `"strict"` | Require an interior estimate and full positive-definite information; `"conditional"` instead fixes numerically binding prices for covariance; `"projected"` additionally estimates boundary-aware mean/SD uncertainty in `beta_summary()`. |
| `boundary_draws` | `2048` | Integer, at least 100; Gaussian simulation draws for summaries, with no model refits. |
| `boundary_seed` | `0` | Nonnegative integer; repeated summary calls use a cached result. |

These modes are LCL-only. `ConditionalLogit` rejects `"conditional"` and
`"projected"`. With `skip=True`, covariance remains unavailable; structural KKT
checking of binding prices still runs. The price constraint and optimizer settings
are unchanged. `boundary="projected"` does not authorize normal Wald intervals or
extend projected inference to prediction/WTP methods.

See the [result-field contracts](latent_class.md#boundary-results-and-diagnostics),
[runnable tutorial](../tutorials/boundary_prices.md), and
[literature and assumptions](../boundary_inference.md#literature-and-implemented-approximation).

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
