# Package mechanics audit

The September 2026 audit covered configuration routing, input encoding, fit and
prediction APIs, result state, docstrings, array annotations, and package metadata.
The core mathematics, econometrics, and economics were outside its scope.
This report describes the audited source checkout; publishing a PyPI release is
a separate step.

## Findings and corrections

| Finding | Correction and result impact |
| --- | --- |
| CV discarded `Options.diagnostics` after resolving the bundle | Forward diagnostics to each training fit and accept the individual `diagnostics` argument. Diagnostic reporting now respects the caller. |
| The direct LCL constructor overwrote explicit class counts and coefficient floors whenever a spec was supplied | Distinguish omission from an explicit value; inherit only omitted values. Fits can change when honoring a previously ignored override. |
| An explicitly empty constraints collection prevented adding a keyword numeraire | Resolve the requested constraint even when the base collection is empty. The requested bound is now enforced. |
| The deprecated tolerance alias stored a duplicate resolved value | Keep one canonical tolerance. `dataclasses.replace(..., newton_decrement_tol=1e-5)` can reset a nondefault tolerance without restoring the old value or emitting a spurious warning. |
| Skipping covariance still looked up custom cluster columns | Bypass those lookups when inference is skipped. Invalid LCL case-level covariance requests fail before estimation when inference is enabled. |
| `NegativeCoefficient.warn_below` was accepted but unused | Apply it as the constraint-specific diagnostic threshold; retain the global warning switch. Estimates are unchanged. |
| Fitted results retained callers' mutable inference and diagnostic objects | Snapshot both sections so later caller mutation cannot misdescribe an existing result. |
| Choice values were converted to boolean before the binary check | Check original values and formula output width first. Reject conflicting explicit and formula outcomes. Previously accepted malformed data now raises an error. |
| Separate demographics reached utility formulas only when a membership formula was also present | Join external data once before feature evaluation; reuse the aligned frame for demographics. Valid existing designs produce the same encoded arrays. |
| Formula/list conflicts, and tabular/array prediction conflicts, silently chose one source | Reject competing inputs at the same configuration level; retain documented single-override precedence. |
| CL prediction ignored redundant identifier arguments, including `panels_col` without warning | Validate names against the fitted encoder and warn consistently on redundant arguments. |
| CL held-out scoring could not accept weights and returned ambiguous repeated case IDs | Add explicit scoring weights with fit-time alignment rules and include panel IDs in case contributions. Unweighted scoring is unchanged. |
| CL inherited a partition-request `tradeoff` signature inconsistent with its own WTP method | Give the alias the same target contract as CL `wtp`. |
| A rejected second LCL fit could first mutate the spec referenced by the original result | Reject refits before mutation. Synchronize the model's existing convergence field with the completed result. Emit the completion callback after result construction succeeds. |
| CV could record nonfinite scores without a useful fold error | Treat nonfinite contributions as failed folds; exclude incomplete sweeps from selection. |
| Nonfinite tolerances, floors, thresholds, initial coefficients, and fractional iteration counts were not consistently rejected | Validate these controls before numerical work. Remove unused WTP bin-count typing and reject ignored or colliding partition settings. |
| Many internal arrays used bare `Array`, `ArrayLike`, or `ndarray` annotations | Add informative jaxtyping shapes and shared boundary aliases. Correct the distributed weight axis to reflect both case- and panel-weight paths. |
| The declared Python 3.10 support conflicted with `enum.StrEnum` usage | Preserve enum string behavior using the Python 3.10-compatible `str, Enum` combination. |
| Declared dependency floors admitted incompatible versions; NumPy was used directly but only depended on transitively | Require Polars 1.24.0 and jaxtyping 0.3.7, declare NumPy 1.25 directly, and align SciPy with JAX's 1.11.1 minimum. Preserve locked dependency versions. |
| Website examples used a deprecated tolerance spelling; public docstrings omitted options or described old return types | Update examples, signatures, shape/order documentation, option precedence, and the [API contracts guide](../api/contracts.md). |

## Compatibility decisions

Working public aliases remain available rather than being abruptly removed.
`start_method` remains a validated, single-choice compatibility field.
`PastChoicesData.dems` remains accepted, with its compatibility role documented:
prediction demographics determine membership priors. CL's WTP summary and LCL's
partitioned WTP summary remain distinct, documented APIs; no new aggregation
formula was introduced to make their signatures look identical.

`LCLSpec` remains a shallowly frozen dataclass; documentation now explicitly
requires treating nested lists and mappings as read-only. Deep freezing would
change their public container behavior and is not introduced here.

The historical five-class direct-constructor default and the two-class `LCLSpec`
default are preserved. Conditional logit has no EM stage, so `Options.fit` is
explicitly documented as inapplicable. Its information-reporting switch is
honored, while membership and class-warning controls apply to LCL.

Removed internal redundancy includes an unused feature-encoding argument, the
membership-formula-only external-data gate, and a second demographic join path.

## Validation

- Baseline suite before changes: **226 passed, 8 skipped**.
- Full suite with the mechanics fixes: **266 passed, 8 skipped**.
- Final targeted mechanics checks, including additional state/callback and input-form regressions: **48 passed**.
- Isolated Python **3.10.19** checks: **79 passed** with JAX **0.5.3**, Equinox **0.13.6**, jaxtyping **0.3.7**, Polars **1.24.0**, NumPy **1.25.0**, pandas **2.0.0**, SciPy **1.11.1**, beartype **0.17.0**, and Formulaic **1.2.1**. Ten warnings come from pandas 2.0's use of a deprecated NumPy API.
- A separate two-device CPU run passes **32 sharding checks**, including the eight checks skipped by the ordinary single-device suite. Final prediction/inference checks pass **73 tests**.
- Ruff passes on package source and tests; mypy passes on **37 source modules**.
- MkDocs builds in strict mode. A wheel and source distribution build successfully; the built wheel imports, fits, and predicts independently of the source checkout. The existing `py.typed` marker and new input aliases are included.
- A GitHub Actions workflow now runs package tests on Python 3.10 and 3.14, plus lint and static typing on 3.14.

The normal run exposes one CPU device; the separate sharding run exposes two
virtual CPU devices through JAX. Physical GPU execution was not tested.
The Python 3.10 floor check initially reproduced an
Equinox import failure with jaxtyping 0.2.25; the corrected 0.3.7 floor passes.
Polars' [1.24.0 implementation](https://github.com/pola-rs/polars/blob/py-1.24.0/py-polars/polars/dataframe/frame.py)
provides the ordered-join interface used by prediction alignment.

A syntax-tree comparison, ignoring annotations, imports, and docstrings, verified
that the computational bodies in `_analytic_derivatives`, `_case_utils`,
`_demographics`, `_delta`, `_em_alg_startup`, `_em_alg_steps`, `_inference`,
`_kernels`, `_optimize`, `_params`, `_polish`, and `_prediction_inference` are
unchanged. Existing numerical regression tests remain in place to check that
annotation changes work through JIT, differentiation, and supported array layouts.
