# Prediction, elasticities, and consumer welfare

A fitted LCL model can forecast new consumers from demographics, update forecasts
for returning consumers using earlier decisions, and evaluate attribute or price
changes. The calculations retain the full class distribution. A consumer is not
assigned permanently to their most likely class.

## Choose the information available at prediction time

```python
# New consumers: membership probabilities depend on their demographics.
prior = results.predict(data=future)

# Returning consumers: update that prior using observed earlier decisions.
personal = results.predict(data=future, past_choices=history)
personal.class_membership()
personal.market_shares()  # estimate and delta-method standard error
```

`class_membership()` returns consumer IDs, class IDs, prior probabilities,
probabilities used in the forecast, and the number of historical choice occasions.
History may cover **only some** prediction consumers. The others retain their
demographic priors. History containing unrelated consumer IDs is rejected; filter
it to the prediction population first. A single choice is a valid history even
when it could not identify the fitted model on its own.

The membership prior uses the **prediction demographics**. History supplies the
likelihood of past choices and need not repeat variables used only in membership.
If demographics enter historical *utility*, supply their historical values in
`history` or `past_choices_dems_data`. The model assumes class tastes remain stable
across the history and forecast periods; it is not a model of evolving preferences.

For class $c$, consumer $n$, and observed history $H_n$, the update is

\[
h_{nc}=\frac{\pi_c(z_n)\prod_{t\in H_n}P_c(y_{nt}\mid X_{nt})}
{\sum_r\pi_r(z_n)\prod_{t\in H_n}P_r(y_{nt}\mid X_{nt})},
\qquad
P_{nj}=\sum_c h_{nc}P_c(j\mid X_n).
\]

LCL evaluates the history in log space. No historical choices means $h=\pi$.
Neither applying logit to a mean coefficient nor selecting the largest class
probability reproduces this mixture forecast. This follows the individual
prediction approach in [Train, chapter 11](https://eml.berkeley.edu/books/choice2nd/Ch11_p259-281.pdf).

For validation, hold out consumers to evaluate population forecasts, and hold out
later choices to evaluate personalization. Never put the scored choice itself in
its history. A policy simulation conditioned on observed baseline choices is
different from a held-out predictive accuracy exercise. More information can
improve average scores without improving every realized individual prediction.

## Compute a welfare change for the same consumers

```python
import polars as pl

changed_data = future.with_columns(
    pl.when(pl.col("alt") == "rail")
      .then(pl.col("price") * 1.10)
      .otherwise(pl.col("price"))
      .alias("price")
)
baseline = results.predict(data=future, past_choices=history)
changed = results.predict(data=changed_data, past_choices=history)

by_case = baseline.surplus_change(changed)
average = baseline.mean_surplus_change(changed)
```

Positive changes mean a gain to consumers. With $\alpha_c=-\beta_{price,c}>0$,
the money-metric change at fixed class weights is

\[
\Delta CS_{nt}=\sum_c h_{nc}\,
\frac{\log\sum_{j\in J^1_{nt}}e^{V^1_{njtc}}
      -\log\sum_{j\in J^0_{nt}}e^{V^0_{njtc}}}{\alpha_c}.
\]

Each class is converted to money **before** averaging. Baseline and counterfactual
must use the same fitted result, consumer/occasion IDs, and panel weights.
Alternatives may be added or removed. IDs, rather than row positions, identify
the cases being compared. Both scenarios enter the same differentiated function,
so their standard error retains the covariance between the two estimates.

`change_identified` indicates whether class weights match for every compared
case. Changing demographics or conditioning only one scenario on history can
change those weights. The resulting difference of surplus indices then depends
on arbitrary class-specific utility locations and is not an identified welfare
change. Such comparisons are flagged and produce a warning. The additional
`normalisation_sensitivity` measures sensitivity to a *common* utility-location
shift; zero does not guarantee cancellation of all class-specific constants.

`surplus` and `mean_surplus()` are normalized inclusive-value **levels**. Negative
levels need not indicate consumer harm. The latter explicitly reports
`level_identified=False`. The economically useful policy object is the change.
See [Train, chapter 3](https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf) and
the [PyBLP welfare reference](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_consumer_surpluses.html).

!!! important "When the money interpretation applies"
    The numeraire must represent expenditure and enter utility once, linearly,
    without additional price transforms or interactions. Marginal utility of
    income must remain constant over the policy change. LCL keeps probabilities
    and elasticities available for other specifications, but marks their surplus
    levels `undefined`/`NaN` and refuses monetary WTP and welfare summaries.
    Without any numeraire, inclusive values are reported in `utils`.

Include a no-purchase/outside alternative as an explicit row when consumers can
opt out. LCL does not silently add one. Otherwise probabilities are conditional
on choosing an offered alternative: they cannot by themselves forecast category
expansion, entry into the market, or all lost sales from a price increase.

## Distinguish marginal WTP from policy welfare

```python
profiles = personal.marginal_wtp("quality")
class_ratios = personal.wtp_by_class("quality")

from lcl import PartitionType, WTPRequest

groups = personal.compute_wtp(
    WTPRequest("quality", "income_band", PartitionType.CATEGORICAL),
    show=False,
)
```

For a linear quality coefficient, mean WTP is

\[
E[WTP_n]=\sum_c h_{nc}\frac{\beta_{quality,c}}{-\beta_{price,c}},
\]

which generally differs from a ratio of mean coefficients. For example, equally
likely classes with quality coefficients 1 and 4 and price coefficients −1 and
−2 have mean WTP 1.5; dividing the mean quality coefficient by the negative mean
price coefficient gives 1.667. The former averages the consumers' tradeoffs.

`marginal_wtp("quality")` also includes formula interactions and transformations.
If utility contains `quality + quality:income`, the class-specific numerator is
`beta_quality + income * beta_quality:income`. The output is one row per offered
profile. `class_sd` measures the remaining dispersion across classes at the fitted
parameters, **not** a parameter-estimation standard error. This separates
uncertainty about a consumer's tastes from uncertainty in estimated coefficients.

`compute_wtp` averages these profile derivatives equally across available
alternatives within each case, equally across cases within each consumer, and
then across consumers using `panel_weights`. For the usual linear utility, or
interactions only with fixed consumer demographics, the first two averages do not
change the value. For nonlinear attributes, the profiles used for evaluation
matter. Use `marginal_wtp()` to inspect that variation.

`wtp_by_class()` and `wtp_alt_vars_by_panel` retain **expanded-design coefficient
ratios**, useful for inspecting individual model terms. Use `marginal_wtp()` or
`compute_wtp()` for the total marginal value of a raw attribute that appears in
several terms. The conditional-logit `wtp()` method supports raw-attribute means
with the same derivative and averaging conventions.

WTP refers to a one-unit **increase**. A negative value for travel time means a
longer trip is undesirable; negate it to express willingness to pay for a unit of
time saved. A derivative for a binary attribute equals its finite 0-to-1 change
only under linearity in that attribute. For multiple simultaneous changes or
removing a product, use a logsum welfare change instead of adding marginal WTPs.

Demographic group comparisons describe predicted preferences of those groups;
they do not identify a causal effect of changing a demographic. Class membership
and utility interactions are distinct channels of heterogeneity. Computing
ratios within class removes a common utility scale, but demographic class logits
are not demographic WTP coefficients.
[Train and Weeks (2005)](https://eml.berkeley.edu/~train/papers/trainweeks.pdf),
[Scarpa, Thiene and Train (2008)](https://eml.berkeley.edu/~train/papers/ScarpaThieneTrain.pdf).

## Elasticities of individual probabilities and market demand

```python
individual = personal.elasticities(["price", "quality"])
market = personal.aggregate_elasticities(["price", "quality"])
```

`target_alts` identifies the alternative whose attribute changes; `alts` identifies
the alternative whose probability/demand responds. Aggregate elasticities refer
to a common proportional change in the target alternative's attribute wherever
it is available. They differentiate total weighted demand, including consumers
who face different choice sets. If a target product is unavailable to a consumer,
that consumer contributes zero response but still contributes to demand for the
affected product.

Raw numeric variables use the full utility-formula derivative, including
interactions and transforms. Supplying an expanded design-column name instead
holds the other design columns fixed. These are local derivatives for continuous
attributes; use two predictions for categorical changes. At zero attribute level,
a point elasticity can be zero even when the marginal effect is nonzero. At zero
predicted demand, percentage changes are undefined and are reported as `NaN`.

## Weights and uncertainty

Use `panel_weights=` on `predict` for target-population weights. It accepts a
consumer-keyed mapping, a column constant within consumer, a NumPy vector, or a
sequence in sorted prediction-consumer order. Weights must be nonnegative, with
positive mass overall and in every requested WTP group.

Market shares and mean surplus attach each consumer's weight to **each choice
occasion**. WTP groups attach the weight **once per consumer**. Unequal counts of
prediction occasions therefore affect the two aggregates differently. For a
population policy scenario, constructing one representative occasion per consumer
often makes the intended population average easiest to interpret. Quintile
partitions use unweighted sample quantiles of consumers; their cutoffs and all
demographics are treated as fixed for inference.

Aggregate methods support `se="delta"`, `se="bootstrap"`, and `se="none"`.
Both inference methods use the complete joint parameter covariance and account
for the negative-price transform, demographic priors, and any Bayesian history
update. Stored posterior WTP supports both methods.

```python
personal.market_shares(se="bootstrap", bootstrap_draws=1000, bootstrap_seed=42)
personal.compute_wtp(
    WTPRequest("quality", "income_band", PartitionType.CATEGORICAL),
    se="bootstrap", bootstrap_draws=1000, bootstrap_seed=42, show=False,
)
```

Here “bootstrap” means asymptotic normal parameter simulation in the unconstrained
parameterization, followed by structural transformation. It is **not** consumer
resampling and model refitting, and it is not a posterior over estimated model
parameters. The delta method computes $J\widehat\Sigma J'$. These SEs describe
estimated expected quantities conditional on the supplied design and histories;
they do not include random future choices, population sampling of prediction
consumers, class-count selection, or model misspecification.

Inspect convergence, information rank, class sizes, and
`denominator_diagnostics()`. A negative price constraint keeps the denominator's
sign economically consistent; it cannot create price identification. Near-zero
denominators can produce very large, skewed WTP estimates. Parameter simulation
can reveal curvature but cannot repair this problem. Singular covariance produces
unavailable SEs, not false precision.
[Daly, Hess and de Jong (2012)](https://doi.org/10.1016/j.trb.2011.10.008),
[Hole (2007)](https://doi.org/10.1002/hec.1197).

## Read and export class summaries

```python
results.summarize_betas()  # existing aggregate mean/SD and SE formatting
results.summarize_class_betas(layout="auto")
results.summarize_membership(layout="auto")

# Explicit layout for a large model:
results.summarize_class_betas(layout="dense")
results.summarize_membership(layout="dense", marginal_effects=False)
```

The aggregate utility table keeps its established formatting. Class tables use
the same estimate-over-parenthesized-SE convention. `auto` displays classes in
columns for up to eight classes, then transposes to classes in rows, including
64-class models. `wide` and `dense` override the choice. Many variables can still
require a landscape page or subsets of the returned tidy table.

Membership coefficients are log odds relative to the explicitly marked reference
class. Printed class numbers start at 1; returned `class` IDs start at 0. The
optional marginal-effect table reports derivatives with respect to **encoded
design columns**, holding other columns fixed. For factors these are not discrete
category contrasts; transformed or interacted columns are not raw-variable total
derivatives. Compare predictions on explicitly constructed demographic scenarios
when that is the question of interest.

All summary methods return Polars tables; `show=False` suppresses rendering.
Plain label characters are escaped in class-table LaTeX, while explicit `$...$`
and `\(...\)` mathematics is preserved and converted to readable terminal text.
The fragments use `booktabs` rules for insertion into the user's own `tabular`.

## Reproducible evidence

In the audit's held-out experiments, adding eight earlier choices to demographic
priors reduced log loss from **0.689 to 0.569** on synthetic data and from
**0.832 to 0.768** on Apollo's public stated-preference mode-choice data. Consumers
used for testing were excluded from estimation, and their earlier choices were
kept separate from their scored outcomes. These results illustrate the value of
personalization under the tested models; they are not a universal accuracy claim.

The [audit report](../development/counterfactual-audit.md) records independent
economic identities, held-out predictions on synthetic and public Apollo data,
and a Monte Carlo check of WTP uncertainty. These illustrate the package's
capabilities under particular specifications. Observational price effects still
need credible identification; LCL does not supply an instrument, correct omitted
quality, or solve firms' pricing equilibrium automatically.
