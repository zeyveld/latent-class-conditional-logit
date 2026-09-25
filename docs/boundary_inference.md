# Boundary prices, uncertainty, and identification

**Available from 0.1.42.** Start with the [runnable tutorial](tutorials/boundary_prices.md),
then consult the [options](api/specification.md) and [result contracts](api/latent_class.md#boundary-results-and-diagnostics).

A negative coefficient is stored directly and constrained to
`beta_price <= -min_abs`. At a binding upper bound, a nonnegative log-likelihood
score is consistent with optimality: the improving direction is infeasible.
A negative score still requires moving into the feasible interior. Convergence
uses this KKT condition, not a transformed or artificially small gradient.

A binding constraint can make ordinary Wald inference invalid even when the
coefficient information matrix is positive definite. This does not establish
collinearity among product attributes. See the
[price optimization investigation](price_optimization.md) for the algorithm,
literature, JIT checks, and performance measurements.

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

The default `boundary="strict"` requires an interior estimate and a full
positive-definite information matrix; otherwise covariance is unavailable. `"conditional"` holds numerically binding
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
[Kim, Stone, and White (2000), section 4](https://www.nottingham.ac.uk/economics/documents/discussion-papers/00-18.pdf),
published as [Kim, White, and Stone (2005)](https://doi.org/10.1093/jjfinec/nbi015).
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
misspecified predictive model. A strictly binding price is itself evidence of
such misspecification: under a correctly specified model whose true parameter
satisfies the constraint, the population score at the truth is zero. A sample
multiplier does not prove population misspecification, and misspecification
does not rule out an accidental information equality. The package warns when
its selection rule classifies a price as strictly binding, because the usual
justification for `"unadjusted"` is then doubtful. The dual
quadratic program has only as many coordinates as selected price constraints.
Correlations with non-price tastes and membership coefficients propagate through
the information metric.

The active set is unknown. This implementation uses a transparent **pointwise**
selection rule with `c_G = sqrt(log(max(G,3)))`, where `G` counts independent
clusters. This is the BIC-type constant that
[Andrews and Soares (2010)](https://doi.org/10.3982/ECTA7502) recommend for
generalized moment selection; hard-threshold "shrinking" of nuisance parameters
toward the boundary plays the same role in
[Cavaliere, Nielsen, Pedersen, and Rahbek (2022)](https://doi.org/10.1016/j.jeconom.2020.05.006).
A numerically binding price (distance at most `1e-8` from its bound)
is strictly binding if its estimated KKT multiplier, divided by that
multiplier's own sampling SD, exceeds `c_G`. Holding the binding set `A` fixed
and re-optimizing the free coordinates `F`, the multiplier is to first order the
nuisance-adjusted score `S_A - I_AF inverse(I_FF) S_F`, so its SD comes from the
sandwich of that linear combination, as in a robust Lagrange-multiplier test.
Under information equality, the raw variance `B_AA` exceeds the adjusted
variance by the positive-semidefinite term `I_AF inverse(I_FF) I_FA`.
Under misspecification or clustering, the adjustment can increase or decrease
the variance; correlation alone does not determine the direction. With weak
boundaries, this fixed-face reference scale is not the unconditional sampling
SD of the constrained multiplier, nor does it give a conventional normal-test
p-value. Other prices within `c_G`
unconstrained SEs of the bound receive inequality constraints, including
interior estimates near the bound. The threshold diverges but is smaller than
`sqrt(G)`, separating fixed interior, zero-multiplier boundary, and strictly
binding cases under the usual regularity conditions. It is a tuning choice, not
a uniform-coverage result for local alternatives. In the scalar case with one
weak constraint, the error has a known direction: if the true price lies a small distance
`delta / sqrt(G)` inside the bound, its limit is `min(Z, delta)`, whose variance
rises with `delta`. Projecting at `delta = 0` therefore understates the SE of
such a near-boundary interior price; it is not a conservative bound. This
one-dimensional comparison does not order arbitrary correlated summary SEs. Inspect the
recorded multiplier statistics and distances when classification is marginal.

Analytic Jacobians propagate the projected limit to coefficient means and variances.
Means and separated SDs are linear in `h = z - inverse(I)[:, A] mu(z_A)`, where
`mu` holds the dual multipliers of the weak constraints and depends on `z_A`
alone. Write `V = Cov(z)`, `D = L inverse(I)[:, A]`, and
`a = L V[:, A] V[A,A]^+`. Gaussian regression gives `Lz = a z_A + e`, with
`e` independent of `z_A`. Consequently,

```text
Var(Lh) = L V L' - a V[A,A] a' + Var(a z_A - D mu(z_A)).
```

This is an exact variance identity for the chosen Gaussian approximation,
including its cross-covariances; it does not add an independent boundary
variance to a delta-method variance. The residual variance is computed exactly,
and only the low-dimensional final term is simulated. The regression is solved
after standardizing the weak coordinates, so different units do not cause a
small but relevant variance direction to be discarded. When no
weak constraint is selected, these SEs therefore equal the delta method on the
reduced subspace and do not depend on `boundary_seed`. A functional with `D=0`
also keeps its exact delta-method variance, even if its score correlates with
weak prices. Quality summaries need not satisfy `D=0`: correlations in the
information matrix and uncertain membership can transmit the price constraint.
Here **exact means free of Monte Carlo error**, not exact finite-sample inference.
For a positive, separated between-class SD, use its ordinary derivative. For a
spread within `c_G` times the estimated scale of class-coefficient differences,
use the weighted-norm directional derivative at zero. This pointwise selection
avoids dividing by an SD that is sampling noise around zero. The delta method
for directional derivatives is developed by
[Fang and Santos (2019)](https://doi.org/10.1093/restud/rdy049).
This implementation's cutoff is an explicit engineering choice within that
framework; it is not a claim of uniformly reliable inference for barely
separated classes or tiny heterogeneous effects.

The remaining simulation uses independent Gaussian draws. Antithetic pairs
are useful for some expectations, but their benefit does not carry over to
variance estimation. With no projection, a zero-spread SD's directional norm
has the same value at `z` and `-z`: pairing duplicates observations and roughly
doubles the Monte Carlo variance at the same total draw count. A scalar
`min(Z, 0)` SE benefits only slightly from pairing. Also, for `N` paired draws
of a functional `f`, the expected ordinary sample variance is
`Var(f(Z)) - Cov(f(Z), f(-Z))/(N-1)`; `ddof=1` is unbiased for the variance
with independent draws, but generally not with pairs. Independent draws avoid
this extra bias, although the square root used for an SE still has the usual
small Monte Carlo bias.

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
and draws over the simulated coordinates: the weak prices plus the class
coefficients of any zero-spread SD (`simulated_dimension` in the diagnostics).
The simulated dimension can equal the number of free parameters when every
coordinate is needed. It never creates `draws x households x classes` arrays,
and it never refits a model. Draws with every weak coordinate nonpositive need
no quadratic program, and a single weak price has a closed-form multiplier.
Increasing draws affects this small kernel alone. Repeated summary calls use the
cache. Skipped-inference fits with saturated prices need
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
The low-dimensional simulation agrees with brute-force projection of full
parameter draws. Without weak constraints it reproduces the delta method to
rounding error, for every seed. Multiplier statistics agree with explicit
per-panel nuisance-adjusted score influence functions. Tests also cover an
unaffected summary with correlated weak-price noise, weak coordinates with
very different variances, and a robust multiplier adjustment that increases
the sampling variance.
Analytic mean/variance Jacobians, including membership terms, agree with JAX
automatic differentiation. Centered coarser-cluster score products agree with
explicit panel centering and aggregation. Interior structural covariance agrees
with the ordinary coefficient-coordinate calculation at a stationary fit.

These checks verify implementation identities and synthetic behavior, not
uniform confidence-interval coverage for arbitrary finite mixtures. In
particular, the `sqrt(log G)` active-set threshold follows the moment-selection
literature's recommended rate. It is not a cutoff validated for LCL mixtures.
