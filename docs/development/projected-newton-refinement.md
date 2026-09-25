# Projected Newton refinement assessment

The numerical refactor preserves structural coefficients, the likelihood,
parameter ordering, covariance formulas, delta-method derivatives, and boundary
inference modes. Gaussian parameter simulation keeps its original distribution
and sample-SD output; only the screening policy and diagnostics change.

## Gaussian simulation and welfare inference

The reported cliff was real. All affected calls share the parameter-simulation
helper; WTP has separate dispatch paths rather than going through `_quantity_se`. For one coordinate with crossing probability `p`,
independent Gaussian draws give probability `1 - (1 - p)^B` of at least one
crossing among `B` draws. At two standard deviations from the bound, `p` is about
0.02275 and a 500-draw call fails with probability about 0.99999. Correlation
between coefficients matters for joint crossing, but not this single-coordinate
calculation or independence across draws.

For a coefficient with estimate `beta`, standard error `sigma`, and upper bound
`u`, the crossing probability is `Phi((beta - u) / sigma)`. The proposed
`Phi((u - beta) / sigma)` instead gives the probability of satisfying the bound.
Zero variance is a point mass: an estimate exactly at its bound has zero strict
crossing probability, which matters for conditional boundary covariance.

| Suggestion | Assessment and implementation |
| --- | --- |
| Deterministic Gaussian gate | Adopted as a screening policy. The cutoff is 0.001 per denominator coordinate, checked before drawing. It corresponds to roughly 3.09 marginal SEs below the bound. This is an explicit conservative heuristic, not a theorem about coverage or moments. |
| Scope the gate to ratios | Adopted for WTP, partitioned WTP, and monetary surplus levels/changes in CL and LCL. Shares and elasticities have no division by the numeraire. Their ordinary covariance can still be invalid at a boundary, so the existing finite-covariance and boundary-inference restrictions remain. |
| Labelled truncated draws | Not adopted for the existing SE API. Conditioning a joint Gaussian on feasibility changes means, variances, correlations, and downstream uncertainty. An acceptance fraction and a label disclose this change but do not preserve the original inference. |
| Percentile intervals | Not substituted for SEs. Quantiles need no finite variance and can be useful additional summaries, but they do not supply the existing SE, z-statistic, or Wald interval. Near a weak denominator, a finite percentile interval also cannot express the unbounded or disconnected sets possible with Fieller/test-inversion inference. Such an interval API would require its own coverage and boundary treatment. |

`denominator_diagnostics()` now reports `denominator_std_error`,
`gaussian_bound_crossing_probability`, `gaussian_zero_crossing_probability`,
and `bootstrap_bound_probability_limit`, in addition to the existing levels and
floor. These are marginal diagnostics. A per-coordinate cutoff does not control
the probability of at least one crossing across many correlated classes.
Unavailable covariance gives NaNs rather than an assertion of zero risk.

The screen does not enforce feasibility on the simulated sample. All draws are
retained if it passes; neither rejection sampling nor clipping is performed.
Consequently an otherwise accepted call can contain rare infeasible draws, and
the sample SD still varies with the seed and draw count. Nonfinite simulated
quantities still raise an error.

A ratio with a nondegenerate Gaussian denominator generally has no finite second
moment, however small the density near zero; special exact cancellations are
exceptions. The bootstrap SD remains a finite Monte Carlo approximation rather
than a consistent estimator of that nonexistent Gaussian-ratio moment. A sign
truncation at zero alone does not fix this; truncating away from zero at the
positive `min_abs` floor does control that denominator, but changes the
simulation law and can make uncertainty sensitive to the chosen floor.
This concerns the Gaussian approximation to estimation error, not the fitted
finite distribution of tastes across latent classes. Also, a truncated Gaussian
is generally not the sampling limit of a constrained estimator: projection can
produce boundary mass, as discussed in [boundary inference](../boundary_inference.md).

These facts do not invalidate the **local asymptotic delta method at an
identified interior parameter with a nonzero denominator**. Under its regularity
conditions it propagates coefficient uncertainty through a smooth map. Nor does
that local result establish uniform validity as the true denominator approaches
zero or the estimator approaches a constraint. The coefficient covariance and
existing default delta SEs are unchanged here.

The distinctions between asymptotic SEs, simulated ratio moments, and confidence
sets are discussed in [Daly, Hess and de Jong (2012)](https://eprints.whiterose.ac.uk/84335/1/Daly_Hess_deJong_revised_July_30_2011.pdf)
and [Daly, Hess and Ortuzar (2023)](https://eprints.whiterose.ac.uk/204222/1/Daly_Hess_Ortuzar_TRA_2003.pdf).
Neither truncation nor percentile intervals should be presented as a universal
repair of standard errors near the bound.

## Numerical and implementation changes

| Concern | Assessment and implementation |
| --- | --- |
| Two direction solves per iteration | Valid. `NewtonState` now retains a `NewtonDirection` containing the direction, decrement, scale, active mask, and regularization shift. Initialization solves once; each accepted new iterate solves once; a failed line search retains the old cache. |
| Inconsistent trust-radius bookkeeping | Valid. Both branches now measure the actual accepted step as `sqrt(s' M s)`, using the positive metric `M` of the solve. In the unregularized interior `M=H`, so bounded and unbounded paths agree. Raw `H` alone is not a norm when mixture curvature is indefinite. The metric includes the active diagonal block and standardized regularization shift. The stopping statistic remains the projected decrement. |
| Large direction closure | Valid. Curvature scaling, active-set identification, regularized solve, projected decrement, and accepted-step norm are module-level pure functions with direct numerical tests. The active-set rule still releases an attained bound immediately when the gradient points inward. |
| Polish bypasses `newton_kwargs` | Valid. The cached polish takes the frozen `OptimizationOptions`; its caller uses `replace` for the polish iteration budget and tolerance. Every solver path now goes through the same option translation. |
| Repeated weight normalization | Valid. `scaled_objective` wraps the scalar and derivative kernels with one common mass divisor, floored at one as before. It handles analytic score auxiliaries and the polish's plain derivative tuple. The wrappers are constructed inside traced kernels so their identities do not create repeated JIT compilation. |
| Index/floor threaded separately | Valid. A frozen `NegativeCoefficientBound` resolves the design index and floor once per numerical entry point. Its sole bounds constructor serves standalone/class vectors and packed class blocks. Public `NegativeCoefficient` remains a named specification; existing model and packing metadata remain compatible. |

The positive two-metric structure and projected Armijo rule follow
[Bertsekas (1982)](https://www.mit.edu/~dimitrib/ProjectedNewton.pdf).
Caching changes work performed, not the search direction. Trust-radius
bookkeeping can change a path and therefore which stationary point a nonconvex
mixture reaches; it does not imply a global-optimum guarantee.

New regression coverage includes direct KKT neighborhood/decrement checks,
regularized and active curvature metrics, interior path equality at several
iteration budgets, runtime direction-evaluation counts, Gaussian tail orientation,
zero-variance bounds, seed/count-independent screening, retained rare crossing
draws, and every monetary prediction route for both estimators. Existing tests
continue to cover independent-solver agreement, covariance propagation,
compilation reuse, and multi-device class scheduling. Historical performance
numbers in `price_optimization.md` predate this refinement; no new wall-clock
speedup is claimed from the factorization-count reduction alone.

Validation after refinement: **345 passed, 8 skipped** on one CPU device. All
**16** EM sharding cases passed separately with two simulated CPU devices,
including the eight skipped cases. Ruff, mypy, the strict documentation build,
and whitespace checks passed.
