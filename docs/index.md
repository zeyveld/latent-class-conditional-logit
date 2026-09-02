---
hide:
  - navigation
  - toc
---

<div class="lcl-hero" markdown>

<span class="lcl-kicker">Discrete choice models in Python</span>

# Latent classes, clear inference

LCL estimates conditional-logit and latent-class conditional-logit models with
JAX. It combines a declarative model specification, safeguarded Newton updates,
panel-clustered inference, and practical tools for counterfactual analysis.

[Start the tutorial](tutorials/estimation.md){ .md-button .md-button--primary }
[Browse the API](api/specification.md){ .md-button }

</div>

<div class="lcl-feature-grid" markdown>

<div class="lcl-feature-card" markdown>

### Specify once

Define identifiers, utility and membership formulas, constraints, and
publication labels in one `LCLSpec`.

</div>

<div class="lcl-feature-card" markdown>

### Estimate efficiently

JAX compiles the numerical kernels and shards independent class updates across
available accelerators.

</div>

<div class="lcl-feature-card" markdown>

### Analyze behavior

Compute choice probabilities, consumer surplus, elasticities, class shares, and
marginal willingness-to-pay from the fitted result.

</div>

</div>

## Install

The distribution is named `lcl-choice`; the import is `lcl`.

```bash
pip install lcl-choice
```

For GPU use, install the JAX build that matches your CUDA environment before
installing LCL. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## A compact model

This example estimates two latent classes and assigns readable labels to raw
variables. Labels affect presentation only: formulas, constraints, prediction
inputs, and returned `variable` columns continue to use the original names.

```python
import lcl
from lcl import (
    ChoiceIds,
    FitOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
)

spec = LCLSpec(
    ids=ChoiceIds(
        alt="alt",
        case="case",
        panel="panel",
        choice="choice",
    ),
    utility_formula="choice ~ price + quality",
    membership_formula="~ income",
    classes=2,
    constraints={"price": NegativeCoefficient()},
    variable_labels={
        "price": "Price",
        "quality": "Product quality",
        "income": "Household income",
    },
)

results = lcl.fit(
    data,
    spec,
    fit_options=FitOptions(seed=7, starts=3, max_em_iter=50, num_devices=1),
    optimization_options=OptimizationOptions(
        maxiter=40,
        gradient_tol=1e-5,
    ),
)

coefficient_table = results.summarize_betas()
print(coefficient_table.select("variable", "label", "mean", "mean_se"))
```

The printed coefficient table uses the labels:

```text
┌─────────────────┬───────────────┬─────────────────────────────┐
│ Variable        │ Means (β's)   │ Standard deviations (σ's)  │
├─────────────────┼───────────────┼─────────────────────────────┤
│ Price           │ -1.124        │ 0.723                       │
│                 │ (0.114)       │ (0.128)                     │
│ Product quality │ 0.906         │ 0.612                       │
│                 │ (0.097)       │ (0.131)                     │
└─────────────────┴───────────────┴─────────────────────────────┘
```

The returned frame keeps both identities:

```text
┌──────────┬─────────────────┬────────┬─────────┐
│ variable ┆ label           ┆ mean   ┆ mean_se │
╞══════════╪═════════════════╪════════╪═════════╡
│ price    ┆ Price           ┆ -1.124 ┆ 0.114   │
│ quality  ┆ Product quality ┆ 0.906  ┆ 0.097   │
└──────────┴─────────────────┴────────┴─────────┘
```

Use `results.summarize_betas(show=False)` when you want the returned frame
without the LaTeX and terminal preview.

The same fitted encoder scores held-out choices and powers panel-blocked model
selection:

```python
test_log_likelihood = results.loglik(test_data)
panel_scores = results.loglik(test_data, per_panel=True)

cv_results = lcl.cv_optimal_classes(
    data,
    spec=spec,
    num_classes_list=[2, 3, 4],
    fit_options=FitOptions(seed=7, starts=3),
)
```

Cross-validation reports pooled mean held-out log likelihood per panel, fold
standard errors, best and one-SE selections, convergence, panel counts, and
explicit failure diagnostics. User-defined panel folds are supported.

## What the package covers

- **Latent-class conditional logit.** Estimate a finite mixture of conditional
  logits, with optional demographic predictors of class membership.
- **Standard conditional logit.** Fit a homogeneous benchmark with the same
  data-ingestion and inference conventions.
- **Inference and diagnostics.** Request panel-clustered sandwich covariance,
  class/membership and nonlinear ratio standard errors, classification metrics,
  convergence summaries, and structured audit output.
- **Counterfactuals.** Reuse the fitted encoder for new choice sets, weighted
  market shares, aggregate elasticities, and explicitly unit-labeled welfare
  changes; optionally update class probabilities with observed choice histories.
- **Model selection.** Compare class counts using blocked cross-validation that
  keeps each decision-maker entirely within one fold.

## Who it is for

LCL is intended for researchers who work with repeated-choice data, including
transportation, marketing, operations research, political science, economics,
and public policy. Runtime shape checking catches many specification errors
before they become opaque compiled-kernel failures.

## Next steps

- Follow the [estimation and counterfactual tutorial](tutorials/estimation.md).
- Compare class counts with [panel-blocked cross-validation](tutorials/cross_validation.md).
- Review the [`LCLSpec` and options API](api/specification.md).

The project is open source under the MIT license. Bug reports, focused feature
requests, and reproducible examples are welcome on
[GitHub](https://github.com/zeyveld/latent-class-conditional-logit/issues).

<div class="lcl-font-note" markdown>

This site uses [Luciole](https://luciole-vision.com/), designed by Laurent
Bourcellier and Jonathan Perez for readers with low vision and released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

</div>
