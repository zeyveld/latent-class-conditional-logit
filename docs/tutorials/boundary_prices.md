# Predictive fits with a boundary price coefficient

New in **0.1.42**: LCL can retain a valid constrained predictive fit when one or
more classes have essentially no price sensitivity, and report uncertainty in
coefficient means and standard deviations across classes. This is opt-in;
`InferenceOptions(boundary="strict")` remains the default.

## A reproducible two-class example

This synthetic panel has one class with an apparently positive price response.
The fitted model constrains both price coefficients to be negative, so a boundary
solution is appropriate for that class. Quality and price vary independently.
The example requires only the package's existing dependencies.

```python
import numpy as np
import polars as pl
from lcl import ChoiceIds, FitOptions, InferenceOptions, LCLSpec, NegativeCoefficient, fit


def synthetic_choices(first_price_effect: float = 0.8) -> pl.DataFrame:
    rng = np.random.default_rng(321)
    rows = []
    for household in range(240):
        income = rng.normal()
        group = int(rng.random() < 1 / (1 + np.exp(-income)))
        for occasion in range(6):
            price = rng.uniform(1, 5, 4)
            quality = np.array([0., 0., 1., 1.])
            price_effect = first_price_effect if group == 0 else -1.2
            quality_effect = -2.0 if group == 0 else 2.0
            chosen = np.argmax(price_effect * price + quality_effect * quality
                               + rng.gumbel(size=4))
            for alternative in range(4):
                rows.append({
                    "household": household, "case": household * 6 + occasion,
                    "alternative": alternative, "choice": alternative == chosen,
                    "price": price[alternative], "quality": quality[alternative],
                    "income": income,
                })
    return pl.DataFrame(rows)


spec = LCLSpec(
    ids=ChoiceIds(alt="alternative", case="case", panel="household", choice="choice"),
    utility=("price", "quality"), membership=("income",), classes=2,
    constraints={"price": NegativeCoefficient(min_abs=1e-5)},
)
fit_options = FitOptions(seed=82, max_em_iter=300, num_devices=1)
inference = InferenceOptions(
    covariance="clustered", boundary="projected",
    boundary_draws=2048, boundary_seed=0,
)
result = fit(synthetic_choices(), spec, fit_options=fit_options, inference=inference)

print(result.convergence_report())
print(result.class_coefficients())
print(result.beta_summary())
print(result.boundary_summary_diagnostics)
```

The fit should retain one numerically binding price. Its class-specific
`std_error` is `NaN`; other class-specific SEs condition on that price being
fixed. The coefficient summary still reports positive SEs for the population
price mean and between-class price SD: the other price and estimated membership
weights remain uncertain. Class numbering can vary across fits.

Here the binding price is strictly binding, so no inequality is left to simulate
(`simulated_dimension` is `0`). The summary SEs are then the exact delta method
on the remaining parameters and do not change with `boundary_seed`.

Read both inference labels: `result.inference_status` describes the covariance
matrix, while `result.beta_summary()["inference_status"]` describes the moment
SEs. A conditional covariance matrix and projected summary SEs can coexist.

## Zero sensitivity differs from a strictly binding price

A zero population score at the boundary permits inward movement in repeated
samples. A strictly positive score at the constrained optimum fixes that
coordinate to first order. The implementation distinguishes these cases using
the multiplier statistics recorded in the diagnostics.

```python
zero_effect = fit(
    synthetic_choices(first_price_effect=0.0), spec,
    fit_options=fit_options, inference=inference,
)
print(zero_effect.beta_summary())
print(zero_effect.boundary_summary_diagnostics["strict_parameters"])
print(zero_effect.boundary_summary_diagnostics["weak_parameters"])
```

This second example selects a weak constraint. A statistically near-boundary
interior estimate can also enter `weak_parameters`; exact numerical boundary
hits alone are insufficient. These indices align with `parameter_names()`.

## Interpret the output before using it

| Output | Meaning |
| --- | --- |
| `boundary=True` in `class_coefficients()` | Coefficient is within the numerical tolerance of its upper bound. Its individual SE is suppressed. |
| Summary `critical_cone_projection` | Mean/SD SEs propagate joint projected taste and membership uncertainty. |
| Summary `conditional_on_boundary_fallback` | Projection was unavailable; binding prices are held fixed. Inspect `fallback_reason`; these SEs can understate uncertainty. |
| `covariance_available=False` | Even the covariance required by the selected mode is unavailable. Inspect stationarity, aliases, class mass, and information curvature. |

The method assumes a fixed, locally identified mixture and independent panels or
valid coarser clusters. It does not repair duplicate classes or collinear utility
columns, and it does not include class-count selection or first-stage
control-function uncertainty. The active-set cutoff is a pointwise approximation,
not a uniform coverage guarantee near transitions. SEs of a nonnormal boundary
distribution do **not** justify ordinary `estimate ± 1.96 × SE` intervals.

Projected inference currently applies to `beta_summary()` and its presentation
aliases. Prediction, elasticity, WTP, and membership SEs use the stored covariance
and are conditional when prices bind. In particular, accepting a boundary price
for prediction does not make a WTP ratio with a nearly zero denominator stable.

The [method and identification guide](../boundary_inference.md) gives formulas,
computational costs, and limitations. The projection framework follows
[Geyer (1994)](https://doi.org/10.1214/aos/1176325768) and
[Andrews (1999)](https://doi.org/10.1111/1468-0262.00082), with an explicit
Gaussian/QP simulation example in
[Kim, Stone, and White (2000), section 4](https://www.nottingham.ac.uk/economics/documents/discussion-papers/00-18.pdf).
Strict and weak boundary directions are distinguished as in
[Liao and Kroer (2023), appendix F](https://proceedings.mlr.press/v202/liao23a/liao23a.pdf).
The treatment of a zero between-class SD uses directional differentiation;
see [Fang and Santos (2019)](https://doi.org/10.1093/restud/rdy049).
