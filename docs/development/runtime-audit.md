# Compilation, placement, and numerical audit

This audit checks the reported runtime issues against the source on September 29,
2026. Some reports describe behavior that had already been corrected. The tests
and measurements below distinguish those cases from new fixes.

## Assessment and changes

| Report | Finding and action |
| --- | --- |
| EM compiles twice for even class splits | Already fixed at the EM return boundary. All 11 existing placement/count cases passed before editing. The new full-data initializer also needs replicated beta-update outputs, including partial class splits. The constraint now lives at that shared boundary as well. |
| Startup compiles for each subset shape | Confirmed. Startup now passes full data and seeded 0/1 panel weights to one compiled, class-sharded beta update. It preserves the original partitions, zero starts, per-class case normalization, and coefficient bounds. |
| Every fold and class count recompiles | Different shapes and parameter layouts still specialize. This is JAX's expected static-shape behavior. General fold/class bucketing is not implemented: it would require valid-panel, valid-case, alternative, and class masks throughout estimation, stopping, diagnostics, and inference. Simply padding the current `Data` changes the likelihood and sample-size calculations. Persistent caching is documented for repeated runs. |
| Cached closures can retain arrays | The factories capture only immutable settings and packing metadata. Added weak-reference checks for data and derivative-chunk arrays while both factories remain cached, and again after clearing JAX/Equinox caches. Equinox's default does not donate user arrays, although its internal JAX wrapper still specifies a donated argument slot; array-bearing static closures should still be avoided. |
| CPU lookup fails when inference is skipped | Confirmed. Skipped covariance now returns before requesting a CPU backend. Other inference paths use the first local CPU, with a logged warning and a local default-device fallback if the CPU backend is disabled. Boundary score validation still runs when needed, even if covariance is skipped. |
| Deprecated `with mesh:` | Confirmed for JAX 0.10.1. Removed the context; `shard_map` already receives its mesh explicitly. No `jax.set_mesh` dependency was introduced. |
| Global device count includes other hosts | Confirmed. Defaults, validation, and class meshes now use local device counts/devices. This provides local class parallelism; it does not implement distributed fitting across hosts. |
| Data is broadcast on every EM call | The loop formerly passed uncommitted input arrays. It now explicitly places array leaves on the replicated class mesh before startup and the recursion. Python count metadata stays static. Transfer-guard tests cover repeated warm calls and seeds. |
| Polish and score checks can exhaust GPU memory before CPU inference | Confirmed by the call path. Fitting now moves its final state and data to the inference device before polish, score checks, and result construction. Derivatives also accumulate complete-panel chunks, so they do not retain all panel scores. |
| Panel score cross-product is computed twice | Confirmed. Each derivative chunk computes its cross-product once, subtracts it from the Hessian, and adds it to the sandwich accumulator. Coarser clusters aggregate scores across chunks before their outer product. Boundary inference preserves centering by the common panel mean, including unequal cluster sizes. |
| Bootstrap vmaps all draws on GPU | Confirmed. Target evaluation now uses the same CPU/fallback policy as the delta method and batches of 32 draws. Online centered second moments replace the full draw-output array. Random draws, seed behavior, denominator checks, and sample-SD normalization are preserved. |
| Results retain GPU data and replicated posteriors | Confirmed for the fitting path. Completed LCL fits now retain these arrays on the inference device. If the CPU backend is disabled, the documented fallback necessarily retains them on the default device. Explicitly constructing an `LCLResults` from user-placed arrays preserves that placement. |
| Only the beta M-step is explicitly parallel | Confirmed as the current sharding design. E-step and membership calculations remain replicated within EM. Replication is required by the supported older compiler paths and the fixed state layout; this audit does not claim whole-pipeline multi-device speedup. Post-EM work now runs on one inference device. |
| Information rank depends on coefficient units | Confirmed. Rank, definiteness, and conditioning now use diagonal-equilibrated information; the Cholesky solve is transformed back to the original units. Singular and indefinite matrices still yield unavailable covariance. Diagnostic eigenvalues/condition numbers now describe the equilibrated matrix. |
| Import changes process-wide x64 | Confirmed. Import no longer changes JAX precision. Public fitting, prediction, and numerical reporting methods enter a float64 context and restore the caller's setting, including on exceptions. Tests and audit scripts that call private kernels explicitly enable x64 themselves. |
| GPU reductions need not be bitwise reproducible | A backend limitation, not a seed bug. Retained the 64-ULP likelihood monotonicity allowance and documented that near ties and stopping boundaries can differ. No deterministic GPU guarantee is made. |
| Likelihood clamps membership priors but derivatives use log-softmax | Confirmed. Production likelihood, startup, EM, polish, and held-out scoring now use log priors directly when membership logits are available. A regression with logits separated by 800 checks the objective, gradient, and Hessian together. |

The JAX floor was already **0.5.3**. Compatibility tests cover it rather than
raising the requirement unnecessarily. The compatibility layer continues to
support both `check_rep` and `check_vma` and both spellings of the x64 context.

## Compilation counts and numerical comparisons

The before/after startup comparison uses 101 unbalanced panels, 250 cases, 609
unchosen rows, two utility variables, one demographic, eight classes, three
seeds, and three EM recursions per seed. Counts are actual JAX compilation log
events, with synchronization before inspecting results.

| Seeds completed | Old startup compilations | New startup compilations | Old/new EM compilations |
| --- | ---: | ---: | ---: |
| 1 | 8 | 1 | 1 / 1 |
| 2 | 16 | 1 | 1 / 1 |
| 3 | 23 | 1 | 1 / 1 |

Across those runs, the largest starting-coefficient difference was `6.94e-17`;
after three EM steps it was `1.56e-16`. Starting log likelihoods were identical
at displayed float64 precision; subsequent likelihoods differed by at most
`5.69e-14`. These are rounding differences from summation order.
The [measurement record](runtime-audit-results.json) includes the baseline commit,
per-seed counts, likelihoods, coefficient differences, and executable buffer sizes.

Regression tests repeat compilation counts after all the changes, across one,
two, and four devices; odd and even class counts; constraints; demographics;
changed array values; changed seeds; and coefficients leaving a binding bound.
The bootstrap test counts one target compilation even for an incomplete final
batch. Its cache receives closure-converted targets: captured arrays are explicit
dynamic inputs, so a bound prediction method cannot keep a dataset alive through
a static JIT argument. Lifetime regressions cover both closures and bound methods,
including captured arrays placed on a different device. Different shapes, dtypes,
solver settings, devices, and class counts can still require separate executables.

## Derivative storage

Chunks contain at most 256 complete panels. Each posterior is computed using
that panel's full choice history. Invalid padded rows and cases are discarded
by segment reductions; invalid panels contribute exactly zero score and
curvature. Only the tail of the ordered design is padded, rather than storing a
separate padded design for every chunk. The largest panel block determines the
row workspace, so exceptionally long individual panels still require memory.

For 5,000 ragged panels, six utility variables, five demographics, eight classes,
and 90 parameters, JAX 0.9.2 CPU executable analysis reported:

| Buffer category | Full score-matrix kernel | Chunked summary kernel |
| --- | ---: | ---: |
| Temporary bytes | 30,163,840 | 1,744,768 |
| Output bytes | 3,664,832 | 130,360 |
| Argument bytes | 1,870,420 | 1,972,760 |

The score, Hessian, and sandwich-middle differences were respectively at most
`2.28e-12`, `4.67e-12`, and `1.14e-11`. The chunked and full kernels compute the
same mathematical quantities. These are executable buffer measurements, not
process peak RSS or measured GPU allocation. The parameter-squared information
and covariance matrices remain necessary; coarser clustering also keeps one
score vector per cluster.

## Verification

Final checks after the bootstrap lifetime correction:

- Full JAX 0.9.2 suite: **382 passed, 32 skipped**. The skips require extra devices.
- Five-device CPU regressions on JAX 0.5.3: **94 passed**.
- The same regressions on JAX 0.10.1 with `-W error`: **94 passed**.
- Ruff passes; mypy reports no issues in **41 source files**.
- Documentation builds with MkDocs strict mode.

The runtime regressions compare chunk sizes 1, 7, and 256 against the original
full score/Hessian calculation, including ragged cases, multiple class counts,
demographics, incomplete tails, coarser clusters, and centered boundary meat.
The existing analytic-versus-autodiff, parameter recovery, prediction, and
boundary tests remain in the suite.

Reproduction commands:

```sh
.venv/bin/python -m pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=5 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -W error \
  tests/test_em_efficiency.py tests/test_runtime_regressions.py \
  tests/test_results_device_placement.py
.venv/bin/ruff check src/lcl tests
.venv/bin/mypy src/lcl
```

The five devices are virtual CPU devices. No physical CUDA device was available;
the missing-CPU-backend path is tested by injecting the backend lookup failure.

## References

- [JAX compilation and shape specialization](https://docs.jax.dev/en/latest/201/jit.html).
- [JAX persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html).
- [JAX 0.10.1 mesh-context deprecation](https://docs.jax.dev/en/latest/changelog.html#jax-0-10-1-may-20-2026).
- [Equinox filtered JIT and donation](https://docs.kidger.site/equinox/api/transformations/).
- [JAX scoped precision and operations on returned arrays](https://docs.jax.dev/en/latest/101/default_dtypes.html).
- [JAX maintainers on GPU reduction nondeterminism](https://github.com/jax-ml/jax/discussions/10674).
