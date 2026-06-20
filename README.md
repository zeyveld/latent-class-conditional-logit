# LCL

[![PyPI version](https://badge.fury.io/py/lcl-choice.svg)](https://badge.fury.io/py/lcl-choice)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LCL is a Python package for estimating latent-class conditional logit models. It runs an expectation-maximization (EM) algorithm on JAX, sharding the per-class M-steps across available accelerators. After estimation, the results object supports counterfactual predictions and consumer welfare analysis.

Although I'm an economist by training, this package is intended for all social scientists who study household-level panel data: marketers, transportation researchers, operations researchers, political scientists, and public policy researchers, among others. 

## Key features

- **A declarative, high-level API**: describe the model once with an `LCLSpec` and fit it with `lcl.fit`. Estimation, optimizer, inference, and diagnostic behaviour are each tuned through a single grouped options object (`FitOptions`, `OptimizationOptions`, `InferenceOptions`, `DiagnosticsOptions`).
- **`LatentClassConditionalLogit`**: finite-mixture conditional logit with a fractional-response multinomial logit regression of class membership on demographics.
- **`ConditionalLogit`**: standard conditional logit, useful both as a baseline and as the inner kernel of the M-step.
- **`cv_optimal_classes`**: blocked K-fold cross-validation for choosing the number of latent classes. Folds are split at the decision-maker level, so no individuals' choices appear in both training and hold-out data.
- **Counterfactual prediction**: out-of-sample choice probabilities, expected consumer surplus, own- and cross-elasticities, and marginal willingness-to-pay broken out by demographic partitions.
- **Inference & diagnostics**: clustered sandwich covariance at the panel level, the Delta method for non-linear functions of the parameters (such as the value of time), and one-call diagnostic reports (`results.diagnostics()`, `convergence_report()`, `audit_report()`).

Types are enforced at runtime by `jaxtyping` and `beartype`. A wrongly shaped design matrix should raise a readable error at the call site rather than a cryptic XLA trace.

## Documentation

Full documentation—worked tutorials, an API reference, and a model-selection guide—is hosted at [zeyveld.github.io/latent-class-conditional-logit](https://zeyveld.github.io/latent-class-conditional-logit/).

## Installation

The wheel is published on PyPI as `lcl-choice` (it imports as `lcl`):

```bash
pip install lcl-choice
```

If you plan to use a GPU, install the CUDA-matched JAX build first; see the [JAX installation notes](https://github.com/jax-ml/jax#installation).

## Quickstart

A two-class model on a small synthetic panel. The [estimation tutorial](https://zeyveld.github.io/latent-class-conditional-logit/tutorials/estimation/) provides a full example, including counterfactual fares and value-of-time partitions.

```python
import numpy as onp
import polars as pl
import lcl
from lcl import ChoiceIds, FitOptions, LCLSpec, NegativeCoefficient, OptimizationOptions

rng = onp.random.default_rng(7)

# Two latent classes: one is price-sensitive, the other prefers quality.
n_panels, n_choices, n_alts = 200, 4, 3
true_class = rng.choice(2, size=n_panels, p=[0.55, 0.45])
beta_price   = onp.array([-1.8, -0.3])
beta_quality = onp.array([ 0.4,  1.6])

rows = []
for panel in range(n_panels):
    income = rng.normal()
    for case in range(n_choices):
        prices  = rng.uniform(0.5, 3.0, size=n_alts)
        quality = rng.uniform(0.0, 5.0, size=n_alts)
        u = (beta_price[true_class[panel]]   * prices
           + beta_quality[true_class[panel]] * quality
           + rng.gumbel(size=n_alts))
        chosen = int(onp.argmax(u))
        for alt in range(n_alts):
            rows.append({
                "panel": panel,
                "case":  panel * n_choices + case,
                "alt":   alt,
                "choice":  alt == chosen,
                "price":   float(prices[alt]),
                "quality": float(quality[alt]),
                "income":  float(income),
            })

df = pl.DataFrame(rows)

# Describe the model once, then fit it. The numeraire (price) is declared as a
# strictly-negative coefficient; options are grouped, not scattered keywords.
spec = LCLSpec(
    ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
    utility=["price", "quality"],
    membership=["income"],
    classes=2,
    constraints={"price": NegativeCoefficient()},
)

results = lcl.fit(
    df,
    spec,
    fit_options=FitOptions(max_em_iter=50, num_devices=1),
    optimization_options=OptimizationOptions(maxiter=40),
)

results.summarize_betas()
print(results)
```

A representative end-of-run printout (`summarize_betas()` also emits a LaTeX version of the table, elided here):

```text
Estimation time: 15.344 seconds

--- Table preview ---

┌────────────┬───────────────┬─────────────────────────────┐
│ Variable   │ Means (β's)   │ Standard deviations (σ's)   │
├────────────┼───────────────┼─────────────────────────────┤
│ price      │ -1.124        │ 0.723                       │
│            │ (0.114)       │ (0.128)                     │
│ quality    │ 0.905         │ 0.611                       │
│            │ (0.097)       │ (0.130)                     │
└────────────┴───────────────┴─────────────────────────────┘

<LCLResults: 2 Classes | Converged | Log likelihood: -597.8 | CAIC: 1233.4 | BIC: 1227.4 | Adj. BIC: 1197.4>
```

The parentheses enclose Delta-method standard errors of the population moments. `summarize_betas()` also returns those moments as a tidy Polars frame; the class-specific β's are available with `results.class_coefficients()`.

## Roadmap

The estimator is fairly stable and the results object covers the cases I routinely encounter in my own work. I'm hoping to make at least two extensions:

- **Model selection.** Blocked K-fold cross-validation is included but still marked experimental; expect refinements as I use this utility in my research.
- **Documentation.** A mathematical appendix and additional worked examples beyond Apollo's mode-choice data.

If there is a constraint, optimization routine, or post-estimation tool you'd like to see, please [open an issue](https://github.com/zeyveld/latent-class-conditional-logit/issues).

## Contributing

The project uses `uv` for dependency management:

```bash
git clone https://github.com/zeyveld/latent-class-conditional-logit.git
cd latent-class-conditional-logit
uv sync --all-extras --dev
uv run pytest tests/
```

## Acknowledgments

LCL is built on JAX, Polars, equinox, jaxopt, jaxtyping, beartype, and formulaic. The differenced-design-matrix kernel at the heart of the conditional logit likelihood evaluation owes a particular debt to the [xlogit](https://github.com/arteagac/xlogit/) package by Cristian Arteaga, JeeWoong Park, Prithvi Bhat Beeramoole, and Alexander Paz.

The documentation site is set in [Luciole](https://luciole-vision.com/), a typeface designed for visually impaired readers by Laurent Bourcellier and Jonathan Perez in collaboration with the Centre Technique Régional pour la Déficience Visuelle and typographies.fr, released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@software{lcl_2026,
  author = {Zeyveld, Andrew},
  title  = {LCL: Latent-Class Conditional Logit Estimation in Python},
  year   = {2026},
  url    = {https://github.com/zeyveld/latent-class-conditional-logit}
}
```
