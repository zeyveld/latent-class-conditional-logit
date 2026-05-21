# LCL: Latent-Class Conditional Logit Estimation in Python

[![PyPI version](https://badge.fury.io/py/lcl-choice.svg)](https://badge.fury.io/py/lcl-choice)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LCL** is a Python package for fitting latent-class conditional logit models. It runs the expectation-maximization algorithm on JAX, sharding the per-class M-steps across whatever accelerators it finds, and returns a results object with clustered robust standard errors, counterfactual predictions, and Delta-method willingness-to-pay distributions.

It is written for econometricians who routinely outgrow `mlogit`, `gmnl`, or `Apollo` on large panel datasets. Eighty thousand households with twenty choice occasions apiece is a comfortable working size on a single H200.

## What is in the package

* **`LatentClassConditionalLogit`** — finite-mixture conditional logit with a fractional-response multinomial regression for class membership on demographics.
* **`ConditionalLogit`** — a standard conditional logit, useful as both a baseline and the inner kernel of the M-step.
* **`cv_optimal_classes`** — blocked K-fold cross-validation for choosing the number of latent classes; folds are split at the decision-maker level.
* **Counterfactual prediction** — out-of-sample choice probabilities, expected consumer surplus, own- and cross-elasticities, and marginal willingness-to-pay broken out by demographic partitions.
* **Inference** — clustered sandwich covariance at the panel level and the Delta method for non-linear parameter combinations.

Type contracts are enforced at runtime through `jaxtyping` and `beartype`: a wrongly shaped design matrix raises a readable error at the call site rather than a trace through XLA.

## Installation

The wheel is published on PyPI as `lcl-choice` (it imports as `lcl`):

```bash
pip install lcl-choice
```

If you intend to run on a GPU, install the CUDA-matched JAX build first; see the [JAX installation notes](https://github.com/jax-ml/jax#installation).

## Quickstart

A two-class model on a small synthetic panel — one class price-sensitive, the other quality-loving.

```python
import numpy as onp
import polars as pl
import lcl
from lcl._struct import EMAlgConfig, MleConfig

rng = onp.random.default_rng(7)

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

model = lcl.LatentClassConditionalLogit(num_classes=2, numeraire="price")
results = model.fit(
    data=df,
    alts_col="alt",
    cases_col="case",
    panels_col="panel",
    choice_col="choice",
    case_varnames=["price", "quality"],
    dem_varnames=["income"],
    em_alg_config=EMAlgConfig(maxiter=50, num_devices=1),
    mle_config=MleConfig(maxiter=40),
)

results.summarize_betas()
print(results)
```

A representative end-of-run printout:

```text
Estimation time: 15.705 seconds
Information criteria: CAIC=1233.4, BIC=1227.4, adjusted BIC=1197.4

--- Table preview ---

┌──────────┬─────────────┬───────────────────────────┐
│ Variable │ Means (β's) │ Standard deviations (σ's) │
├──────────┼─────────────┼───────────────────────────┤
│ price    │ -1.124      │ 0.723                     │
│          │ (0.114)     │ (0.128)                   │
│ quality  │  0.905      │ 0.611                     │
│          │ (0.097)     │ (0.130)                   │
└──────────┴─────────────┴───────────────────────────┘

<LCLResults: 2 Classes | Converged | Log likelihood: -597.8 |
 CAIC: 1233.4 | BIC: 1227.4 | Adj. BIC: 1197.4>
```

A complete walkthrough using the Apollo `modeChoice` data — counterfactual fares, value of time by income quintile, own- and cross-elasticities — is in the [estimation tutorial on the docs site](https://zeyveld.github.io/latent-class-conditional-logit/tutorials/estimation/).

## Roadmap

LCL is under active development. The estimator is stable and the results object covers the cases we encounter in our own work. Active work is on:

* **Model selection.** Blocked K-fold cross-validation is included but still labelled experimental; expect refinements on highly unbalanced panels.
* **Documentation.** A mathematical appendix and worked examples beyond Apollo's mode-choice data.
* **Companion paper.** A working paper covering the econometric framework, hardware benchmarks, and Monte Carlo coverage tests.

Feature requests are welcome on the [issue tracker](https://github.com/zeyveld/latent-class-conditional-logit/issues).

## Development

The project uses `uv` for dependency management.

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
  author = {Jeffries, Anna and Zeyveld, Andrew},
  title  = {LCL: Latent-Class Conditional Logit Estimation in Python},
  year   = {2026},
  url    = {https://github.com/zeyveld/latent-class-conditional-logit}
}
```
