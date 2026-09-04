"""Counterfactual quantities written as differentiable functions of the parameters.

Choice probabilities, market shares, consumer surplus, and elasticities are all
smooth functions of the estimated coefficients, so the same delta method that
gives a coefficient its standard error gives them theirs.  Writing each as a
function of the flat *latent* parameter vector -- with the softplus transform
applied inside -- lets :mod:`lcl._delta` differentiate through the constraint
rather than requiring every call site to apply the chain rule by hand.

Aggregates carry standard errors; per-case and per-panel tables do not.  A
policy conclusion rests on the aggregate, and a Jacobian with one row per
observation is neither cheap nor usually wanted.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as onp
from jax.ops import segment_sum
from jaxtyping import Array, Float64, Int, UInt

from lcl._kernels import _choice_probabilities_and_logsum


def _betas_and_class_probs(
    results: Any,
    flat_params: Float64[Array, "all_params"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_panels: int,
) -> tuple[Float64[Array, "alt_vars classes"], Float64[Array, "panels classes"]]:
    """Return structural class betas and prior class probabilities."""
    return results._structural_betas_and_class_probs(flat_params, dems, num_panels)


def _marginal_utility_of_income(
    results: Any, betas: Float64[Array, "alt_vars classes"]
) -> Float64[Array, "classes"]:
    """Return ``-beta_numeraire`` by class, or ones when there is no numeraire."""
    numeraire_idx = getattr(results.model, "numeraire_idx", None)
    if numeraire_idx is None:
        return jnp.ones(betas.shape[1], dtype=betas.dtype)
    return -betas[numeraire_idx, :]


def choice_probabilities(
    flat_params: Float64[Array, "all_params"],
    *,
    results: Any,
    X: Float64[Array, "rows alt_vars"],
    cases: UInt[Array, "rows"],
    panels: UInt[Array, "rows"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_cases: int,
    num_panels: int,
) -> Float64[Array, "rows"]:
    """Return mixture choice probabilities for every design row."""
    betas, class_probs = _betas_and_class_probs(results, flat_params, dems, num_panels)
    probs_by_class, _ = _choice_probabilities_and_logsum(X, betas, cases, num_cases)
    return jnp.sum(class_probs[panels] * probs_by_class, axis=1)


def market_shares(
    flat_params: Float64[Array, "all_params"],
    *,
    results: Any,
    X: Float64[Array, "rows alt_vars"],
    cases: UInt[Array, "rows"],
    panels: UInt[Array, "rows"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_cases: int,
    num_panels: int,
    alt_codes: UInt[Array, "rows"],
    num_alts: int,
    row_weights: Float64[Array, "rows"],
    weight_total: float,
) -> Float64[Array, "num_alts"]:
    """Return panel-weighted predicted market shares by alternative."""
    probabilities = choice_probabilities(
        flat_params,
        results=results,
        X=X,
        cases=cases,
        panels=panels,
        dems=dems,
        num_cases=num_cases,
        num_panels=num_panels,
    )
    demand = segment_sum(probabilities * row_weights, alt_codes, num_segments=num_alts)
    return demand / weight_total


def surplus_by_case(
    flat_params: Float64[Array, "all_params"],
    *,
    results: Any,
    X: Float64[Array, "rows alt_vars"],
    cases: UInt[Array, "rows"],
    panels_of_cases: UInt[Array, "cases"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_cases: int,
    num_panels: int,
) -> Float64[Array, "cases"]:
    """Return expected consumer surplus for each choice situation."""
    betas, class_probs = _betas_and_class_probs(results, flat_params, dems, num_panels)
    _, logsum = _choice_probabilities_and_logsum(X, betas, cases, num_cases)
    surplus = logsum / _marginal_utility_of_income(results, betas)[None, :]
    return jnp.sum(class_probs[panels_of_cases] * surplus, axis=1)


def mean_surplus(
    flat_params: Float64[Array, "all_params"],
    *,
    case_weights: Float64[Array, "cases"],
    **kwargs: Any,
) -> Float64[Array, ""]:
    """Return the panel-weighted mean consumer surplus per choice situation."""
    surplus = surplus_by_case(flat_params, **kwargs)
    return jnp.sum(surplus * case_weights) / jnp.sum(case_weights)


def mean_surplus_change(
    flat_params: Float64[Array, "all_params"],
    *,
    baseline: dict[str, Any],
    counterfactual: dict[str, Any],
    case_weights: Float64[Array, "cases"],
) -> Float64[Array, ""]:
    """Return the panel-weighted mean counterfactual-minus-baseline surplus.

    Both scenarios are evaluated at the same parameter vector, so the difference
    inherits the parameter uncertainty coherently: the correlation between the
    two surplus levels is preserved rather than being lost by differencing two
    separately estimated quantities.
    """
    changed = surplus_by_case(flat_params, **counterfactual)
    base = surplus_by_case(flat_params, **baseline)
    return jnp.sum((changed - base) * case_weights) / jnp.sum(case_weights)


def build_within_case_pairs(cases: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray]:
    """Enumerate ordered row pairs within each choice situation.

    Elasticities relate the probability of alternative ``j`` to an attribute of
    alternative ``k`` in the same case.  The pair index depends only on the
    design, never on the parameters, so it is built once in NumPy and then
    gathered inside the differentiated function.

    Parameters
    ----------
    cases : numpy.ndarray
        Contiguous zero-indexed choice-situation identifier for each row.

    Returns
    -------
    affected : numpy.ndarray
        Row index of the alternative whose probability changes.
    target : numpy.ndarray
        Row index of the alternative whose attribute changes.
    """
    order = onp.argsort(cases, kind="stable")
    sorted_cases = cases[order]
    boundaries = onp.flatnonzero(
        onp.concatenate([[True], sorted_cases[1:] != sorted_cases[:-1], [True]])
    )
    affected_blocks = []
    target_blocks = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:stop]
        affected_blocks.append(onp.repeat(rows, rows.size))
        target_blocks.append(onp.tile(rows, rows.size))
    if not affected_blocks:
        empty = onp.empty(0, dtype=onp.int64)
        return empty, empty
    return onp.concatenate(affected_blocks), onp.concatenate(target_blocks)


def aggregate_elasticities(
    flat_params: Float64[Array, "all_params"],
    *,
    results: Any,
    X: Float64[Array, "rows alt_vars"],
    cases: UInt[Array, "rows"],
    panels: UInt[Array, "rows"],
    dems: Float64[Array, "panels dem_vars"] | None,
    num_cases: int,
    num_panels: int,
    design_derivative: Float64[Array, "rows alt_vars"],
    raw_values: Float64[Array, "rows"],
    affected: Int[Array, "pairs"],
    target: Int[Array, "pairs"],
    group_codes: Int[Array, "pairs"],
    num_groups: int,
    row_weights: Float64[Array, "rows"],
) -> Float64[Array, "num_groups"]:
    """Return demand-weighted own- and cross-elasticities by alternative pair.

    Parameters
    ----------
    design_derivative : Float64[Array, "rows alt_vars"]
        Row-wise derivative of every utility column with respect to the raw
        variable.  Constant in the parameters, so it is supplied rather than
        recomputed.
    raw_values : Float64[Array, "rows"]
        The raw variable's level on each row.
    affected, target : Int[Array, "pairs"]
        Row indices from :func:`build_within_case_pairs`.
    group_codes : Int[Array, "pairs"]
        Zero-indexed ``(affected alternative, target alternative)`` group.
    num_groups : int
        Number of such groups.
    row_weights : Float64[Array, "rows"]
        Panel weight on each row, used with the predicted probability to form the
        demand weight the aggregate averages over.

    Returns
    -------
    Float64[Array, "num_groups"]
        Aggregate elasticity for each alternative pair, in group-code order.
    """
    betas, class_probs = _betas_and_class_probs(results, flat_params, dems, num_panels)
    probs_by_class, _ = _choice_probabilities_and_logsum(X, betas, cases, num_cases)
    class_weights = class_probs[panels]  # (rows, classes)
    weighted_probs = class_weights * probs_by_class  # (rows, classes)
    probabilities = jnp.sum(weighted_probs, axis=1)  # (rows,)

    slope = design_derivative @ betas  # (rows, classes)
    own_term = jnp.sum(weighted_probs * slope, axis=1)  # (rows,)

    # The cross term shares the class mixture between the two alternatives, so it
    # cannot be factored into a product of marginals: sum over classes of
    # (h_c P_jc)(P_kc slope_kc).
    cross_term = jnp.sum(
        weighted_probs[affected] * (probs_by_class * slope)[target], axis=1
    )
    is_own = affected == target
    derivative = jnp.where(is_own, own_term[affected] - cross_term, -cross_term)
    elasticity = derivative * raw_values[target] / probabilities[affected]

    demand = probabilities[affected] * row_weights[affected]
    numerator = segment_sum(demand * elasticity, group_codes, num_segments=num_groups)
    denominator = segment_sum(demand, group_codes, num_segments=num_groups)
    return numerator / jnp.where(denominator > 0.0, denominator, 1.0)


__all__ = [
    "aggregate_elasticities",
    "build_within_case_pairs",
    "choice_probabilities",
    "market_shares",
    "mean_surplus",
    "mean_surplus_change",
    "surplus_by_case",
]
