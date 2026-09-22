# Boundary inference audit

The September 22, 2026 review covers the four proposed inference changes and
antithetic simulation. The SE simplification is mathematically valid for the
selected first-order distribution. Two implementation gaps were corrected;
the distinction between simulation accuracy and sampling validity remains
essential.

## Why the SE simplification is valid

After removing strictly binding coordinates, let `I` denote information,
`V = inverse(I) B inverse(I)` the sandwich covariance, and `A` the weak price
coordinates. The projected Gaussian perturbation is

```text
z ~ N(0, V)
h = z - inverse(I)[:, A] mu(z_A).
```

For a scalar smooth summary with derivative `L`, define
`a = L V[:, A] V[A,A]^+` and `D = L inverse(I)[:, A]`. Joint Gaussianity gives
`Lz = a z_A + e`, where `e` is independent of `z_A` and hence of `mu(z_A)`.
Therefore

```text
Var(Lh) = [L V L' - a V[A,A] a'] + Var(a z_A - D mu(z_A)).
```

Integrating the first term exactly changes computational precision, not the
target variance. All taste/membership covariance terms remain included. If
there are no weak constraints, the smooth summary has variance `L V L'`.
The same conclusion holds for an individual summary with `D=0` even when
other summaries are affected. A zero between-class SD is not differentiable
in the ordinary sense; its directional norm still needs simulation under the
current implementation.

This calculation inherits the assumptions behind the projected limit:
fixed and locally identified classes, nonvanishing shares, an appropriate
cluster CLT, stationarity, and positive definite information on the selected
subspace. The active-set selection is pointwise, not uniform near transitions.
Nonnormal SEs alone do not justify normal Wald confidence intervals. These
restrictions are explained in the [method guide](../boundary_inference.md),
including the critical-cone distinction used in
[Liao and Kroer, appendix F](https://proceedings.mlr.press/v202/liao23a/liao23a.pdf).

## Findings and corrections

| Change | Assessment |
| --- | --- |
| Exact Gaussian residual variance | Valid. The previous implementation nevertheless simulated an unaffected summary when its Gaussian score correlated with weak prices. Such summaries now retain their exact delta-method variance. |
| Regression on the weak covariance | A raw pseudoinverse could drop a small but relevant variance direction. Standardizing the weak coordinates before the pseudoinverse fixes this scale dependence while allowing singular score covariance. |
| Multiplier standardization | Correct on the fixed binding face: use the variance of `S_A - I_AF inverse(I_FF) S_F`. Under robust covariance the adjustment can either increase or decrease the variance. The previous claim that it always decreases the SD was incorrect. With weak constraints, this is a selection scale rather than the unconditional SD of the constrained multiplier. |
| Unadjusted warning | Appropriate. The warning now says that sample classification suggests population misspecification. It does not claim to establish it or to exclude an accidental information equality. Both conditional and projected modes are tested. |
| Cholesky solve | Correct. It solves against the same symmetric positive definite matrix using its Cholesky factor, avoiding the two general linear solves. Existing rank and curvature checks remain intact. |
| Antithetic draws | Mixed effects on SE precision, and ordinary `ddof=1` loses its unbiased-variance interpretation with paired draws. Independent draws are now used. |

The scale regression test originally reported an SE of approximately
`1.1591e-10` where the clipped-normal target is `5.8382e-11`. The corrected
calculation agrees with the target within its simulation tolerance. Additional
tests cover identical Gaussian coordinates and zero score covariance.

Economically, a binding price in the constrained fit describes the best fit
within the imposed sign restriction. It does not establish that consumers
actually have zero causal price sensitivity. A positive *population* multiplier
indicates that the smooth likelihood would improve beyond the permitted price
bound. Robust inference then concerns the constrained pseudo-true parameter.
The bound is `-min_abs`, so even exact zero sensitivity lies outside a model
with a positive `min_abs`. Price endogeneity or omitted attributes can also
produce an apparent positive price response. A nearly zero denominator remains
problematic for willingness-to-pay ratios.

## Fitted examples

The two-class fixtures use 240 panels, six occasions, four alternatives,
estimated demographic membership, and 2,048 draws.

The strict-boundary example selects multiplier statistic **11.2988259** against
threshold **2.3410764**. Its simulated dimension is zero, and all four summary
SEs are identical across seeds 0–9:

| Variable | Mean SE | Between-class SD SE |
| --- | ---: | ---: |
| Price | 0.04803810 | 0.03202479 |
| Quality | 0.13627654 | 0.08272736 |

The zero-price-effect example selects one weak price. With independent draws,
the seed 0–9 range divided by the average SE is approximately **0.283%** for
the price mean and **0.815%** for the price SD. Quality is extremely stable,
but **not exact**: corresponding ranges are **0.000548%** and **0.002525%**.
Its derivative has a small, nonzero projection loading. Independent variation
of price and quality in the data generator does not eliminate covariance in
the fitted mixture.

## Antithetic precision

At a fixed total of `N=2m` draws, an even functional satisfies `f(z)=f(-z)`.
Antithetic pairing therefore gives only `m` distinct observations for that
functional. The zero-spread SD norm without projection is an example: the
Monte Carlo variance is asymptotically doubled and its Monte Carlo SD increases
by `sqrt(2)`. For the SE of `min(Z,0)`, direct Gaussian moment calculation gives
an antithetic-to-independent Monte Carlo SD ratio of **0.96723**, a modest gain.

A 1,000-seed comparison on the fitted weak-boundary example gives:

| Reported SE | Antithetic / independent Monte Carlo SD |
| --- | ---: |
| Price mean | 0.909 |
| Price SD | 0.933 |
| Quality mean | 1.114 |
| Quality SD | 1.349 |

Thus pairing helps these price summaries but hurts the quality summaries. The
choice of independent draws is a general default, not a claim that they
minimize simulation noise for every summary. If `c = Cov(f(Z), f(-Z))`, the
ordinary sample variance from pairs has expectation `Var(f(Z)) - c/(N-1)`.
Independent draws remove this pairing bias. Taking a square root still gives
the usual small Monte Carlo bias in an estimated SE.

## Performance and verification

A synthetic 64-class kernel benchmark used eight utility coefficients, three
membership design columns, 701 free parameters, and 2,048 draws. Medians over
five runs, with peak allocations measured by `tracemalloc`, were:

| Weak prices | Directional SD variables | Simulated dimension | Seconds | Peak MB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.00040 | 0.34 |
| 1 | 0 | 1 | 0.00090 | 1.06 |
| 4 | 0 | 4 | 0.01976 | 1.23 |
| 4 | 1 | 68 | 0.02133 | 5.87 |

These measurements exclude fitting, the structural derivative pass, and the
already allocated information/covariance matrices. They confirm the reduced
simulation cost; they are not a reproduction of an unspecified benchmark.
If all free coordinates are needed by directional summaries, the simulated
dimension can still reach the full free dimension.

For positive definite matrices of dimensions 64, 256, and 701, seven-run median
Cholesky-solve speedups over the previous two-general-solve implementation were
**2.16x, 3.77x, and 2.66x**. Maximum absolute differences were below `2e-18`
for these well-conditioned, scaled matrices. This is a numerical equivalence
check, not a bound for every ill-conditioned information matrix.

Regression checks compare projections with an independent constrained
optimizer, full-dimensional Monte Carlo, clipped/folded-normal laws, explicit
score influence functions, automatic differentiation, and conditional delta
methods. Package source retains jaxtyping shape annotations; mypy and Ruff
pass. The full suite reports **302 passed, 8 skipped**; the strict MkDocs build
also passes. These checks establish implementation identities and selected examples,
not finite-sample coverage for arbitrary latent-class mixtures.
