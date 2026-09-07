# API contracts and compatibility

This page explains how configuration, data identity, and return values carry
through fitting, scoring, and prediction. The
[package mechanics audit](../development/package-mechanics-audit.md) records the
corrections made in the source checkout. These audit corrections have not been
published to PyPI as part of this work.

## Configuration has one source

Every fitting entry point accepts `options=Options(...)`. Pass either that bundle
or the individual option arguments; mixing them raises `ValueError`.

| Section | Conditional logit | Latent-class logit and CV |
| --- | --- | --- |
| `fit` | No EM stage; this section does not apply | Initialization, starts, EM stopping, devices, and polishing |
| `optimization` | Maximum-likelihood solver | Starting fits and both M-steps; also the documented polishing controls |
| `inference` | Covariance method, grouping, finite-sample correction, and skipping | Same settings; case-level `robust` is unsupported for a panel mixture |
| `diagnostics` | `check_collinearity` controls information reporting | All diagnostic switches and thresholds |

Mutable inference and diagnostic objects are copied when fitting. Editing an
option object later does not relabel an already-computed covariance matrix or
change a result's diagnostic configuration.

`InferenceOptions(skip=True)` skips covariance work and does not require a custom
cluster column to exist. For CV, covariance is skipped when both `options` and
`inference` are omitted. An explicit bundle is respected in full: use
`Options(inference=InferenceOptions(skip=True))` to request the same behavior.

Use `newton_decrement_tol` for solver tolerance. Polishing retains its separate
`FitOptions.polish_maxiter` budget and fixed `1e-10` Newton tolerance; the remaining
solver controls also apply to polishing. `FitOptions.score_tol` determines LCL's
final convergence flag after EM and polishing. The EM stopping flag is separately
available as `em_criterion_met`.

For numeraire warnings, `NegativeCoefficient.warn_below`, when present, overrides
`DiagnosticsOptions.near_zero_numeraire_threshold`. The
`warn_near_zero_numeraire` switch still controls the warning status.

## Specifications and overrides

`LCLSpec.classes` defaults to two. The direct constructor retains its historical
five-class default when neither `spec` nor `num_classes` is supplied. Specify the
class count explicitly when moving between these interfaces.

```python
model = lcl.LatentClassConditionalLogit(
    spec=spec,
    num_classes=3,
    numeraire_min_abs=0.01,
)
```

Explicit constructor class counts and coefficient floors override the base
specification; omitted values inherit it. A conflicting numeraire name raises an
error. A single explicit design override in `fit` or CV replaces the corresponding
base-specification design. Supplying both a formula and an explicit variable list
for the same design at the same configuration level raises an error.

If a formula supplies the choice outcome and `choice_col` is also supplied, they
must agree. Choices must be binary before conversion to boolean, with exactly one
chosen alternative per `(panel, case)`.

`LCLSpec` freezes top-level fields; its nested variable lists and mappings should
be treated as read-only. Derive changes with `dataclasses.replace`.

Create a new model instance for each fit. An attempted second fit raises an error
before changing the original result's model specification.

## Data identity and weights

Separate `dems_data` is joined by panel ID before evaluating utility or membership
features. It may supply utility interactions even without a membership formula.
External columns must not overlap the choice frame except for the panel ID, and
membership variables must be constant within panel.

Tabular prediction reuses fitted formula transformations and categorical levels.
Supply raw columns, including categorical columns, rather than constructing new
dummies. Unseen categories raise an explicit error. Prediction identifier columns
come from the fitted encoder.

LCL array prediction accepts NumPy arrays, JAX arrays, and ordinary sequences:

| Argument | Shape and ordering |
| --- | --- |
| `X` | `(rows, alt_vars)`, in fitted expanded-column order |
| `alts`, `cases`, `panels` | `(rows,)`, aligned with `X`; original IDs are preserved |
| `dems` | `(panels, dem_vars)`, in fitted membership-column order |
| `dem_panel_ids` | `(panels,)`, identifies demographic rows for alignment |

Without `dem_panel_ids`, demographic rows follow sorted unique panel-ID order.
Do not combine tabular `data` with array arguments. `dems_data` requires tabular
prediction; array prediction uses `dems`.

Historical choices may cover a subset of prediction consumers. Prediction
demographics supply membership priors. `PastChoicesData.dems` remains accepted
for compatibility, but does not override those priors; historical utility
interactions must already be represented in its `X`.

| Weight argument | Applies to | Vector ordering |
| --- | --- | --- |
| CL `fit(..., weights=...)` | Estimation | First appearance of each choice situation in the input |
| CL `loglik(..., weights=...)` | Scoring | Same rule as fitting |
| Both `predict(..., panel_weights=...)` methods | Aggregate summaries | Sorted unique prediction-panel order |

Prefer a column or ID-keyed mapping when row order may change. When case IDs
repeat across panels, CL weight mappings use `(panel_id, case_id)` keys. Scoring
weights are explicit: omitting them scores cases equally and does not reuse
training weights. Prediction weights affect summaries, while individual choice
probabilities remain unchanged.

## Return values

Both `fit` methods return result objects, and both `predict` methods return
prediction objects. Tables are Polars DataFrames.

| Operation | Conditional logit | Latent-class logit |
| --- | --- | --- |
| Canonical result attributes | `converged`, `cov_matrix`, `adjusted_bic` | Same |
| Covariance labels | `parameter_names()` | Same |
| Probabilities | `prediction.predicted_probs` | Same |
| Aggregate predictions | `market_shares`, `aggregate_elasticities`, `mean_surplus`, `mean_surplus_change` | Same |
| Per-profile WTP | `marginal_wtp(target)` | Same |
| WTP summary | `wtp(target)`, with `compute_wtp` and `tradeoff` aliases | `compute_wtp(*WTPRequest)`, with `tradeoff` alias, for demographic partitions |
| Contribution scoring | `loglik(data, per_case=True)` | `loglik(data, per_panel=True)` |

CL scoring tables include both `panel` and `case` identifiers, so repeated case
IDs remain distinguishable. Their weighted contributions sum to the total from
the same `loglik` call.

Aggregate uncertainty methods accept `se="delta"`, `se="bootstrap"`, and
`se="none"`; bootstrap draws and seeds are forwarded to the shared inference
implementation. Summary printing can be disabled with `show=False` where offered.

## Compatibility surface

| Older argument or name | Current usage |
| --- | --- |
| `OptimizationOptions.gradient_tol` | Use `newton_decrement_tol`. Passing the alias warns; reading returns the resolved value. `dataclasses.replace` operates on the canonical field. |
| Individual fit option arguments | Still supported. Do not mix with `options=`. |
| `results.convergence`, `results.covariance`, `results.abic` | Deprecated aliases for `converged`, `cov_matrix`, and `adjusted_bic`. |
| CL prediction identifier keywords | Redundant and deprecated. Supplied names must match the fitted encoder. |
| `FitOptions.start_method` | Only `"panel_partition"` is implemented; other values raise an error. |
| `WTPRequest.bins` | A finite, increasing list for `"custom_breaks"` only. Integer bin counts and bins for other partition types are unsupported. |

The package uses jaxtyping annotations for JAX and NumPy arrays, including
internal solver state and identifier shapes. Public input aliases preserve the
accepted array and sequence forms; ingestion validates values and converts
numerical designs to 64-bit arrays. Scalar arrays have shape `""`; general
nonlinear targets retain variable output shapes. Runtime type checking and
explicit value checks complement each other.
