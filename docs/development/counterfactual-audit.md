# Prediction, welfare, and reporting audit

This September 2026 audit reviews economic definitions, parameter uncertainty,
and implementation against independent calculations and held-out choices. The
results below describe the development source in this repository. They do not
establish performance for every specification or dataset.

## Economic requirements retained from the literature

* **Personal prediction:** use the full mixture, with
  `h[n,c] ∝ pi[c | demographics[n]] × product_t P(observed choice[n,t] | c)`.
  Multiply historical probabilities within class before averaging over classes;
  use log probabilities for long or improbable histories. With no history,
  retain the demographic prior. Do not substitute a modal class or a posterior
  mean coefficient into a logit probability. These nonlinear operations do not
  commute. History must precede the prediction target when evaluating accuracy.
  [Train (2009), chapter 11](https://eml.berkeley.edu/books/choice2nd/Ch11_p259-281.pdf)
  and [Sarrias and Daziano (2018)](https://doi.org/10.1016/j.jocm.2017.10.004).
* **Consumer welfare:** average class-specific logsum **differences**, each divided
  by its own positive marginal utility of income. The standard formula assumes
  utility linear in expenditure, a constant marginal utility of income over the
  policy change, fixed preferences and the same population and information in
  both scenarios. Absolute utility locations are unidentified. An outside option
  must be explicitly included to model nonpurchase. Removing an alternative
  cannot improve expected welfare at fixed tastes.
  [Train (2009), chapter 3](https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf),
  [PyBLP consumer-surplus reference](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_consumer_surpluses.html).
* **WTP heterogeneity:** average individual/class marginal rates of substitution,
  not ratios of average utility coefficients. Scale cancels within a coefficient
  ratio. Demographic membership coefficients are log odds, not WTP slopes or
  causal demographic effects. Utilities containing attribute interactions or
  transformations require derivatives or explicitly defined finite changes.
  Posterior means compress the underlying taste distribution; dispersion of
  those means is not the population taste dispersion.
  [Train and Weeks (2005)](https://eml.berkeley.edu/~train/papers/trainweeks.pdf),
  [Scarpa, Thiene and Train (2008)](https://eml.berkeley.edu/~train/papers/ScarpaThieneTrain.pdf),
  [Boxall and Adamowicz (2002)](https://doi.org/10.1023/A:1021351721619).
* **Elasticities:** differentiate the mixture probability, retaining the within-class
  joint substitution term. A market elasticity is the derivative of total demand
  divided by that demand. Consumers who cannot access the changing product
  contribute zero derivatives, but still contribute to demand for the affected
  product. This denominator requirement follows directly from differentiating
  the aggregate; averaging only across joint availability changes the estimand.
  [Train (2009), chapter 6](https://eml.berkeley.edu/books/choice2nd/Ch06_p134-150.pdf).
* **Uncertainty:** propagate the complete joint covariance through utility
  constraints, demographic membership, posterior updating, and aggregation.
  Difference two scenarios within a single parameter function to retain their
  covariance. Asymptotic normal parameter simulation (Krinsky–Robb style) is not
  a bootstrap that resamples consumers and refits. Neither approach makes a
  weakly identified price denominator reliable. Report estimation SEs separately
  from uncertainty about an individual's class and future random choices.
  [Daly, Hess and de Jong (2012)](https://doi.org/10.1016/j.trb.2011.10.008),
  [Hole (2007)](https://doi.org/10.1002/hec.1197),
  [Daly, Hess and Train (2012)](https://eml.berkeley.edu/~train/papers/DHT_WTP.pdf).

The repository's papers by Greene and Hensher (2003), Hynes, Hanley and Scarpa
(2008), Zheng et al. (2016), and Wu and Daziano (2023) were also consulted for
panel mixtures, welfare distributions, applied segmentation, and the limitations
of hard class assignments. The requirements above, rather than copied literature
passages, are the working specification for the audit.

## Scope and interpretation

Counterfactuals hold the fitted model fixed. They do not by themselves identify
causal price responses in observational data, solve a supply-side equilibrium,
or value policies with nonlinear income effects. Forecast validation needs both
new-consumer holdouts and future-choice holdouts for existing consumers. Better
historical information can improve average predictive scores without making
every individual's realized prediction better.

## Verification results

### Findings and corrections

| Area | Assessment and resulting behavior |
| --- | --- |
| Bayesian prediction | The mixture and posterior logic were economically correct for matching, fully observed panels. Prediction now also accepts short and partially observed histories, uses prediction demographics for the prior, and matches consumers by ID. Extreme log prior odds survive Bayesian updating. `class_membership()` exposes prior/posterior probabilities and history counts. |
| Held-out scoring | Prediction and scoring no longer require the new choice data to identify coefficients anew. Rank checks remain enforced for estimation. |
| Welfare | Class-specific money conversion was correct. Comparisons now validate model, case identity, and weights; arbitrary class-specific normalization changes are flagged even when the common-shift sensitivity is zero. Extra price interactions/transforms disable the standard monetary summaries while leaving demand prediction available. |
| WTP heterogeneity | Class-weighted coefficient ratios were correct for simple linear attributes. Raw-attribute WTP now incorporates utility interactions and transforms. Group summaries average profiles within consumers before applying consumer weights. Profile-level WTP distinguishes class dispersion from estimation SEs. |
| Aggregate elasticities | The old denominator included only consumers with joint availability of both alternatives. It now includes all demand for the affected alternative, matching an actual market-demand perturbation. The implementation avoids division by tiny individual probabilities before aggregation. |
| Formula alignment | Raw prediction rows now follow encoded row identity. Previously differing choice sets could produce a different alternative order, misaligning raw values and formula derivatives. External demographic inputs are retained for derivative evaluation. |
| Grouping and weights | WTP grouping uses demographic values rather than similarly named WTP columns. Missing partition values and groups with zero total weight are rejected. NumPy weight vectors pass the runtime API checks. |
| Inference | Delta propagation already used the complete joint latent covariance, including the price transform and posterior update. Independent numerical gradients confirm the calculations. Parameter simulation now rejects non-finite quantities. Extremely small class shares are no longer inflated to `1e-10` when packing inference parameters. |
| Presentation | The aggregate utility mean/SD and SE renderer is unchanged. Class tables retain the same estimate/parenthesized-SE style, switch to class rows at nine classes, and were checked at 64 classes. Mathematical labels survive LaTeX export and read cleanly in the terminal. Notes distinguish skipped inference from unidentified covariance and explain encoded demographic derivatives. |

The changes do **not** estimate tastes by regressing posterior means on
demographics, assign consumers to a modal class for prediction, average utility
coefficients before forming WTP, or interpret demographic class logits as causal
WTP slopes. Those shortcuts would change the economic objects being estimated.

### Genuine holdouts

Both experiments estimate on one group of consumers and evaluate a separate
group's later choices. The history used for personalization contains only that
test group's earlier choices. The synthetic experiment uses 500 training and 300
test consumers (1,200 scored choices). The Apollo experiment uses 350 training and
150 test consumers (900 scored choices). Apollo's **stated-preference observations
only** are used, avoiding an unmodelled equality of RP and SP utility scales.

| Predictive information/model | Synthetic log loss | Apollo SP log loss |
| --- | ---: | ---: |
| Homogeneous conditional logit | 0.7458 | 0.8568 |
| LCL tastes with population class shares | 0.7163 | 0.8479 |
| LCL with demographic priors | 0.6889 | 0.8315 |
| LCL with demographics and eight earlier choices | **0.5685** | **0.7676** |

Lower log loss is better. Adding history to demographic priors reduced it by
17.5% and 7.7%, respectively. The two demographic-prior forecasts also outperform
their pooled-share counterparts on this score. On synthetic data, choice accuracy
increased from 69.8% to 76.4% with history; on Apollo it increased from 61.9% to
65.6%. The corresponding Brier scores improved from 0.4126 to 0.3346 and from
0.4889 to 0.4538. Accuracy need not rank models identically to probability scores:
the homogeneous Apollo benchmark had 62.4% accuracy despite worse log loss.

All primary fits converged with finite covariance. Independent NumPy/SciPy
calculations reproduced choice probabilities within `5.6e-16` and surplus levels
within `2.9e-14`. A 10% increase in every offered price produced mean welfare
changes −0.17478 (SE 0.000681) in synthetic money units and −4.99793 (SE 0.01836)
in Apollo's cost units, per choice occasion. These are model-based illustrations,
not policy estimates validated against an actual price intervention.

Independent central differences of the entire posterior-conditioned welfare
function reproduced its reported SE within `3.8e-13` and `7.2e-10`. Thus the SE
tests cover both the utility and membership blocks and preserve the covariance
between baseline and counterfactual.

### Monte Carlo uncertainty diagnostic

The 100-replication experiment generated 220 consumers with six choices each from
a known two-class model. The target was mean quality WTP among consumers with
nonnegative income, conditional on each replication's realized demographic
design. Every fit converged and produced finite SEs.

| Quantity | Result |
| --- | ---: |
| Mean estimation error | 0.0458 |
| Empirical RMSE | 0.4019 |
| Mean reported delta SE | 0.3884 |
| Coverage of estimate ± 1.96 SE | 92 / 100 |

The SE scale is close to the repeated-sample error scale. The finite-sample
coverage is below nominal; 100 replications do not establish exact calibration.
Price-denominator curvature, weak class separation, small classes, and the choice
of class count remain substantive inference concerns. A negative-price floor
protects the sign, not the information content of the data.

### Economic identities and adversarial cases

Automated checks verify the following independently of an optimizer's success:

* exact Bayes updates from a single historical case and partial panel coverage;
* extreme prior log odds (−1,000) updated without replacing them by a probability floor;
* posterior WTP values and SEs against NumPy calculations and numerical parameter derivatives;
* quality × income WTP heterogeneity, weighted demographic means, and zero-weight groups;
* elasticity agreement with direct market-demand perturbations under unequal availability;
* no contamination of positive-demand elasticities from zero-probability alternatives;
* a common monetary price increment causes exactly the same welfare loss, with zero parameter SE;
* removal of an available alternative cannot improve welfare at fixed tastes;
* refusal of mismatched model/population/weight comparisons and invalid monetary specifications;
* class-specific normalization dependence even with identical class price coefficients;
* readable 64-class tables and unchanged aggregate utility LaTeX output.

The final regression suite passed **226 tests**, with eight existing skips.
Ruff and mypy checks passed, as did the strict MkDocs build. The new guide and
audit tables were also inspected in a browser preview.

### Reproduction

The public dataset is available from
[Apollo's examples page](https://www.apollochoicemodelling.com/examples.html).
The exact input file hash and complete results are saved in
[counterfactual-audit-results.json](counterfactual-audit-results.json).
No private data were used.

```bash
.venv/bin/python tools/audit_predictions.py \
  --apollo /path/to/apollo_modeChoiceData.csv \
  --output /tmp/lcl-prediction-audit-results.json \
  --replications 100
.venv/bin/python -m pytest -q
```

The [experiment script](https://github.com/zeyveld/latent-class-conditional-logit/blob/main/tools/audit_predictions.py)
stores seeds, sample sizes, scores, numerical error checks, and all Monte Carlo
estimates and standard errors. Tests inject known parameters and a specified
covariance where needed to isolate algebra from estimation; the held-out and
Monte Carlo experiments instead estimate their parameters and covariance from
the simulated or real training observations.
