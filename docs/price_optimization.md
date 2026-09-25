# Direct coefficient bounds and projected Newton

LCL stores economic coefficients directly and enforces `beta_price <= -min_abs`
in optimization. The same values are used by the likelihood, EM state,
parameter packing, covariance, prediction, and reported coefficient tables.
There is no separate transformed coefficient or covariance representation.

## Why direct bounds

A smooth monotone transformation approaching a finite bound cannot have its
slope uniformly bounded away from zero throughout that tail. A tiny transformed
gradient can therefore conceal substantial feasible improvement in the economic
coefficient. Changing the stopping tolerance does not recover a derivative
that has underflowed. Squaring the parameter introduces a stationary point at
the bound; restarting or clipping an auxiliary parameter introduces additional
heuristics. CERN's [Minuit2 guide, section 6.3.1](https://root.cern/root/htmldoc/guides/minuit2/Minuit2.html#631-getting-the-right-minimum-with-limits)
documents the analogous failure of transformed derivatives at parameter limits.

Direct bounds avoid this geometry and permit exact finite storage of a boundary
estimate. Keeping a transformation solely for storage would add conversions,
chain rules, and rounding without helping the constrained solver.

## Algorithm and literature

The implementation follows the two-metric projected Newton structure of
[Bertsekas (1982), SIAM J. Control Optim. 20:221–246](https://doi.org/10.1137/0320018)
([author's paper](https://www.mit.edu/~dimitrib/ProjectedNewton.pdf)). Near-bound
coordinates whose objective gradient points outward use a positive diagonal
metric. The remaining coordinates use the dense Newton metric. Projection and
the projected Armijo rule account for the actual feasible displacement; merely
clipping an arbitrary dense Newton direction would not suffice.

The active neighborhood shrinks with the scaled projected-gradient residual.
An inward slope releases a binding coordinate immediately. Existing Hessian
regularization, curvature scaling, and trust-radius safeguards remain in use.
The stopping statistic combines the free-block Newton decrement and feasible
active displacement. Exactly zero-weight classes stop without iteration.
Directions, active sets, local scales, regularization shifts, and decrements
are cached in the iterate. Trust-radius updates measure the accepted step in
the same positive curvature metric as the solve (the Hessian in the interior,
with active blocks and regularization included when needed).

[Wu and Xie (2024), section 4](https://arxiv.org/html/2409.05321v1#S4) provides a
modern description of these rules. LCL does not implement that paper's later
scaled variant or claim its complexity bounds for the complete safeguard policy.
The method is established; this particular JAX implementation is new and tested.
Conditional logit is convex in coefficients. The mixture likelihood remains
nonconvex, so multiple starts and convergence diagnostics still matter.

The solver covers standalone conditional logit, subset initialization, EM taste
updates, and observed-data polishing. In the binary regression example, starting
at the price bound recovers `-log(4) = -1.3862943611` rather than stopping there.

## Parameters and inference

- `NegativeCoefficient(min_abs=m)` specifies the closed bound `beta <= -m`.
- CL `init_beta` contains economic coefficients in expanded design-column order.
  The solver projects infeasible starts. The default is zero before projection.
- EM stores one `betas` matrix, with variables in rows and classes in columns.
- LCL `flat_params` contains that matrix in row-major order, followed by
  non-baseline membership logits. CL `flat_params` is its coefficient vector.
- `cov_matrix` has the same coordinates and ordering as `flat_params`.
  Analytic derivatives and delta-method targets use these coordinates directly.

At an upper bound, a nonnegative likelihood score satisfies the one-sided KKT
condition; a negative score still indicates feasible ascent. Observed-score
and class diagnostics account for this distinction.

A bound can invalidate ordinary Gaussian/Wald inference even when the coefficient
Hessian is nonsingular. The default strict mode explicitly withholds covariance
at a binding estimate. Existing LCL conditional and projected boundary modes use
the direct coefficient information; see [boundary inference](boundary_inference.md)
and [Geyer (1994)](https://doi.org/10.1214/aos/1176325768).

`se="bootstrap"` remains Gaussian parameter simulation, not data resampling and
refitting. It draws directly from `flat_params` and `cov_matrix`. For WTP and
monetary surplus, simulation is refused when any numeraire coordinate has more
than 0.1% Gaussian probability above its configured bound. This deterministic
screen uses the fitted mean and marginal variance, not the realized draws;
`denominator_diagnostics()` reports its inputs and probabilities. Shares and
elasticities do not divide by a numeraire and do not use this screen. All calls
still require a finite, positive-semidefinite covariance and finite quantities.

The 0.1% cutoff is a conservative diagnostic policy, not a coverage or
finite-moment guarantee. A nondegenerate Gaussian denominator generally makes
a ratio's simulation variance undefined even if its density near zero is tiny.
Passing the screen does not remove that problem: the reported bootstrap SE is
still a finite Monte Carlo sample SD. The default delta-method SE is the local
asymptotic uncertainty calculation and is unchanged. Neither method resolves
weak identification or an invalid ordinary covariance at a binding estimate.

No draws are clipped or rejected, including occasional bound crossings after a
passing screen. Truncation would change the distribution and the meaning of the
reported uncertainty. Conditional boundary covariance keeps its designated
binding coefficients fixed. See the
[refinement assessment](development/projected-newton-refinement.md) for the
statistical distinctions and numerical changes.

## JAX execution

Bounds, active masks, Hessians, and loop state have fixed shapes. Newton,
regularization, and line-search iterations use `lax.while_loop`; the solver
contains no host optimizer, callback, or dynamically sized free-coordinate
extraction. Existing `lax.map` and `shard_map` class scheduling is retained.
Tests also run different active sets under `jit(vmap(...))`.

This follows JAX's requirements for
[fixed-shape loop state](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html)
and [stable JIT cache keys](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching).
Tests count actual XLA solver compilation announcements using
[`jax.log_compiles`](https://docs.jax.dev/en/latest/_autosummary/jax.log_compiles.html).
Each fixed configuration compiles once across changed starts, data values,
weights, seeds, or active bounds. Shapes, dtypes, shardings, and static options
can legitimately require another compilation.

## Reproduction and performance

Run `.venv/bin/python benchmarks/price_constraint.py` in a separate process,
without other compute-heavy jobs. It synchronizes each result, measures 30 warm
calls, counts actual solver compilations, and reports XLA buffer estimates and
process peak RSS separately. Use `--maxiter 1` for per-iteration cost and
`--price-effect 0.8` for a binding optimum. `--start` is a coefficient value.

CPU measurements recorded before these refinements and the original 0.1.43 baseline
(`24efba17a85d3ee0365864f69a356bf5bdb20499`) are in
[the benchmark data](https://github.com/zeyveld/latent-class-conditional-logit/blob/main/benchmarks/results/price_constraint_cpu.json).
They use JAX 0.9.2 on macOS ARM64, 10,000 cases, four classes, eight variables,
and four alternatives. These measure class M-steps rather than complete model
fits. GPU performance has not been measured. There is no universal claim that
a constrained fit takes the same number of Newton iterations as a transformed fit.

| Workload | Original 0.1.43 median ms | Direct coefficients median ms |
| --- | ---: | ---: |
| Interior optimum, same economic start | 10.003 | 12.735 |
| One Newton iteration, same economic start | 4.552 | 4.518 |
| Binding optimum, same economic start | 75.721 | 12.730 |

The direct solver's default projected-zero start took 15.540 ms on the
interior workload. Starting values affect iteration counts: the same-start
interior comparison is about 27% slower than the original solver, while the
binding case is about 83% faster. A universal unchanged-runtime condition is
therefore **not** satisfied. Per-iteration cost is essentially unchanged.

Removing the storage transformation itself changed the same-start interior
median from 12.688 to 12.735 ms, within measurement noise.
XLA's total argument/output/temporary buffer estimate is identical before and
after this cleanup: 7,723,880 bytes. Compared with original 0.1.43 it is only
16 bytes larger (about 7.7 MB total). Current process peak RSS is 439–450 MB;
it includes imports, host data, JIT compilation, and allocator overhead, so it
is not a precise device-buffer estimate. These observations do not establish
hardware-independent performance or peak memory guarantees.

All four current benchmark configurations counted **one solver compilation**,
including extra calls with changed starting values and weights. Regression tests
also count one compilation per tested static configuration for standalone CL,
subset initialization, EM recursion, class updates, and observed-data polish.

Validation before these refinements: **322 passed, 8 skipped** on one CPU device. All **16** EM
sharding tests passed separately with two simulated CPU devices, covering the
eight skipped cases. Ruff, mypy, and the strict documentation build passed.
Coverage includes exact boundary storage, inward release, scaled price units,
zero-mass classes, correlated bounds compared with SciPy L-BFGS-B, analytic
versus autodiff derivatives, covariance and prediction propagation, and
the former realized-draw bound check (now replaced by deterministic screening).
