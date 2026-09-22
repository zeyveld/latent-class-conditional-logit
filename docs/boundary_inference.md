# Boundary prices, uncertainty, and identification

**Available from 0.1.42.** Start with the [runnable tutorial](tutorials/boundary_prices.md),
then consult the [options](api/specification.md) and [result contracts](api/latent_class.md#boundary-results-and-diagnostics).

A negative coefficient is `-(softplus(raw) + min_abs)`. At its upper bound,
the derivative with respect to `raw` vanishes. Consequently, a perfectly useful
constrained predictive fit can have a singular *latent-coordinate* information
matrix. A small raw score alone can also conceal a feasible improving direction.
Neither fact establishes collinearity among product attributes.

```python
from lcl import InferenceOptions, fit

result = fit(data, spec, inference=InferenceOptions(
    covariance="clustered", boundary="projected",
    boundary_draws=2048, boundary_seed=0,
))
result.beta_summary()
result.boundary_summary_diagnostics
result.diagnostics().print()
```

The default `boundary="strict"` preserves the existing requirement for a full
positive-definite information matrix. `"conditional"` holds numerically binding
prices fixed. `"projected"` also computes boundary-aware uncertainty for the
population coefficient means and between-class standard deviations. These modes
are for LCL; ordinary conditional-logit fits reject the two new modes explicitly.

## What is reported

- `class_coefficients()` suppresses the SE of each numerically binding price.
  Other class-specific SEs condition on those binding prices being fixed.
- `cov_matrix`, membership, prediction, and other delta-method inference use that
  same conditional covariance when prices bind. Zero covariance rows encode the
  conditioning assumption, not knowledge of the unconstrained price coefficient.
- `beta_summary()` uses joint taste and estimated membership uncertainty,
  including cross-covariances. Its `inference_status` distinguishes regular,
  projected, conditional, and conditional-fallback results.
- Summary weights are average *prior* class probabilities over the observed
  demographics, as before. Demographics, the chosen class count, and generated
  regressors such as a previously estimated control function are held fixed.
  These are not SEs for model selection or first-stage estimation uncertainty.
- `boundary_summary_diagnostics` records selected constraints, multiplier tests,
  zero-spread selections, draw count/seed, information diagnostics, fallback reason,
  and time. `beta_summary()` caches its draws' summary, so repeated exports agree.

## Literature and implemented approximation

Constrained estimators generally have projected-Gaussian, nonnormal limits;
ordinary inverse-Hessian Wald inference need not apply at a boundary.
[Geyer (1994)](https://doi.org/10.1214/aos/1176325768) develops this projection
characterization, and [Andrews (1999)](https://doi.org/10.1111/1468-0262.00082)
treats estimation with parameters on the boundary. A concrete simulation procedure
for SEs using Gaussian draws and small quadratic programs is given by
[Kim, Stone, and White (2000), section 4](https://www.nottingham.ac.uk/economics/documents/discussion-papers/00-18.pdf).
The implementation applies this general theory to the structural LCL parameters;
these papers do not themselves validate a particular fitted LCL mixture.

Let `I` be total structural observed information and `B` the centered, optionally
clustered score cross-product. For a maximized likelihood with `beta <= upper`,
a positive population score at the boundary is a strictly binding multiplier.
Its first-order perturbation is zero. A zero-multiplier boundary instead permits
perturbations `h_beta <= 0`. The distinction matters when an apparently positive
price response is constrained negative. See the explicit critical-cone statement
and strict-complementarity discussion in
[Liao and Kroer (2023), appendix F and section 3](https://proceedings.mlr.press/v202/liao23a/liao23a.pdf).

After fixing strictly binding coordinates, draw
`z ~ N(0, inverse(I) @ B @ inverse(I))` and solve
`min_h (h-z)' I (h-z)` subject to the remaining boundary inequalities.
For `covariance="unadjusted"`, the Gaussian covariance is `inverse(I)` instead;
this option requires the information equality and should not be preferred for a
misspecified predictive model. The dual quadratic program has only as many
coordinates as selected price constraints. Correlations with non-price tastes
and membership coefficients propagate through the information metric.

The active set is unknown. This implementation uses a transparent **pointwise**
selection rule with `c_G = sqrt(log(max(G,3)))`, where `G` counts independent
clusters. A numerically binding price (distance at most `1e-8` from its bound)
is strictly binding if its total structural score divided by its estimated
score standard deviation exceeds `c_G`. Other prices within `c_G` unconstrained
SEs of the bound receive inequality constraints, including interior estimates
near the bound. The threshold diverges but is smaller than `sqrt(G)`, separating
fixed interior, zero-multiplier boundary, and strictly binding cases under the
usual regularity conditions. It is a tuning choice, not a uniform-coverage
result for local alternatives. Inspect the recorded multiplier statistics when
classification is marginal.

Analytic Jacobians propagate these joint draws to coefficient means and variances.
For a positive, separated between-class SD, use its ordinary derivative. For a
spread within `c_G` times the estimated scale of class-coefficient differences,
use the weighted-norm directional derivative at zero. This pointwise selection
avoids dividing by an SD that is sampling noise around zero. The delta method
for directional derivatives is developed by
[Fang and Santos (2019)](https://doi.org/10.1093/restud/rdy049).
This implementation's cutoff is an explicit engineering choice within that
framework; it is not a claim of uniformly reliable inference for barely
separated classes or tiny heterogeneous effects.

Reported SEs are the standard deviations of the resulting first-order changes,
not their root mean squares. The distribution can be biased, asymmetric, or have
point mass. **Do not construct normal Wald confidence intervals simply by
multiplying these SEs by 1.96.** When all price coefficients are strictly binding,
their population mean and SD can have zero first-order SE. That does not imply
finite-sample certainty or identify unconstrained positive price effects.

## Assumptions, failure, and fallback

This approximation assumes a fixed, locally identified number of distinct
classes with nonvanishing shares, a consistent local optimum, an appropriate
independent-panel/cluster CLT, smooth likelihood derivatives, and positive
definite information on the subspace being used. Label permutations do not alter
the reported population moments. Vanishing classes, duplicated class profiles,
separation, or utility aliases can violate these assumptions. Boundary handling
cannot cure them.

A negative structural price score at a binding upper bound means a feasible
move into the interior improves the likelihood. The package now checks that KKT
condition even when inference is skipped and marks the fit nonconverged if its
violation exceeds `score_tol`. This check does not change optimizer estimates.

If the free conditional information is not positive definite, covariance stays
unavailable. If it is valid but the larger subspace needed for projected summary
uncertainty fails, the least-bad fallback is explicitly labelled
`conditional_on_boundary_fallback`, with its reason. It holds boundary prices
fixed and can understate summary uncertainty. It is **not** a substitute for
unconditional boundary inference. No ridge, pseudo-inverse, or naive normal SE
silently manufactures identification.

## Cost and diagnosis

The structural score/Hessian pass replaces the existing inference derivative
pass. Centered score products are computed algebraically without retaining a
second panels-by-parameters array. Projection needs only parameter-size matrices
and `draws x parameters` arrays; it never creates `draws x households x classes`
or refits a model. Increasing draws affects this small kernel alone. Repeated
summary calls use the cache. Skipped-inference fits with saturated prices need
one additional structural-score check; JAX discards unused Hessian outputs.

Before optimization, exact utility aliases and membership aliases are rejected
with named weak directions. Utility checks use chosen-alternative differences,
so common utility shifts are accounted for. A normalized design condition above
`1e4` warns without dropping terms. After fitting, problematic information matrices
report named, diagonally scaled parameter directions in the audit, alongside
class mass and membership-separation diagnostics. Loadings identify combinations
to investigate; they are not a rule to delete whichever variable appears first.


## Implementation checks against the formulas

The dual projection solves only for the inequality multipliers; tests compare
correlated, multiple-constraint solutions against an independent constrained
optimizer. The one-dimensional test checks the mean and variance of
`min(Z, 0)` for a standard normal `Z`. A separate test checks the folded-normal
law of the between-class SD's directional derivative when two tastes coincide.
Analytic mean/variance Jacobians, including membership terms, agree with JAX
automatic differentiation. Centered coarser-cluster score products agree with
explicit panel centering and aggregation. Interior structural covariance agrees
with the ordinary latent-coordinate calculation at a stationary fit.

These checks verify implementation identities and synthetic behavior, not
uniform confidence-interval coverage for arbitrary finite mixtures. In
particular, the `sqrt(log G)` active-set threshold is our documented tuning
choice within the asymptotic framework, not a cutoff prescribed by these papers.
