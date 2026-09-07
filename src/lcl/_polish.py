"""Observed-data convergence: Aitken stopping and a final Newton polish.

EM typically approaches a local optimum linearly, so a small likelihood
increment need not imply proximity to the limit. For a geometric sequence with
rate ``r``, the remaining ascent after the latest iterate is its increment times
``r / (1 - r)``. Two tools here address convergence.

:func:`aitken_extrapolated_gap` estimates the remaining ascent from the last
three log likelihoods by summing the geometric tail, following Bohning, Dietz,
Schaub, Schlattmann and Lindsay (1994, *Ann. Inst. Statist. Math.* 46:373-388)
and McLachlan and Peel (2000, *Finite Mixture Models*, section 2.11).  Using it
as the stopping rule means the tolerance refers to the distance to the limit
rather than to one step's progress.

:func:`polish_observed_data` then takes safeguarded Newton steps directly on the
observed-data log likelihood, using the exact analytic score and Hessian that
:mod:`lcl._analytic_derivatives` already assembles for the covariance.  The
observed information and the sandwich covariance both assume the score vanishes
at the reported estimate. Polishing can improve stationarity more efficiently
than tightening the EM tolerance; the final score check still determines the
public convergence flag.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import Array, Float64

from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._em_alg_steps import (
    _compute_em_log_kernels,
    _posterior_and_loglik,
    _compute_panel_logliks,
)
from lcl._optimize import exact_newton_minimize
from lcl._params import ParamPacking
from lcl._struct import Data, DiffUnchosenChosen, EMVars

logger = logging.getLogger(__name__)


def _require_panels(data: Data) -> int:
    """Return the panel count, which every latent-class routine needs."""
    if data.num_panels is None:
        raise ValueError("Panel identifiers are required for latent-class models.")
    return data.num_panels


POLISH_DECREMENT_TOL = 1e-10
"""Newton-decrement tolerance for the polish, on the per-panel objective.

Set below the float64 round-off floor of that objective on purpose: the solver
should stop when the line search can no longer find a decrease, which is the
practical definition of a stationary point, rather than at an arbitrary
threshold above it.
"""


@lru_cache(maxsize=16)
def _compiled_polish(
    packing: ParamPacking,
    maxiter: int,
    max_step_norm: float,
    line_search_maxiter: int,
    hessian_damping: float = 0.0,
    initial_trust_radius: float = 1.0,
    accept_any_decrease: bool = False,
) -> Callable[..., Any]:
    """Return a compiled observed-data Newton solve for one static configuration.

    The solver is a single ``lax.while_loop``, so tracing it under one JIT
    boundary replaces dozens of separately compiled eager operations with one
    executable -- and a second fit of the same shape reuses it outright.

    Parameters
    ----------
    packing : :class:`~lcl._params.ParamPacking`
        Flat layout and structural transforms; frozen, so it hashes by value.
    maxiter : int
        Maximum Newton iterations.
    max_step_norm : float
        Trust-radius ceiling.
    line_search_maxiter : int
        Armijo backtracking budget.
    hessian_damping : float, default=0.0
        Initial fallback diagonal shift.
    initial_trust_radius : float, default=1.0
        Initial radius in the curvature metric.
    accept_any_decrease : bool, default=False
        Use a strict objective decrease instead of Armijo sufficiency.

    Returns
    -------
    Callable
        A compiled ``(flat_params, diff_unchosen_chosen, data, scale) -> (params, steps)``.
    """

    def run(
        flat_params: Float64[Array, "all_params"],
        diff_unchosen_chosen: DiffUnchosenChosen,
        data: Data,
        scale: Float64[Array, ""],
    ) -> tuple[Float64[Array, "all_params"], Any]:
        """Solve the observed-data score equations from ``flat_params``."""
        num_panels = _require_panels(data)

        def total_loglik(params: Float64[Array, "all_params"]) -> Float64[Array, ""]:
            """Sum the observed-data panel log likelihoods."""
            latent_betas, packed_thetas = packing.unpack(params)
            structural_betas = packing.to_structural(latent_betas)
            prior = packing.class_probs(packed_thetas, data.dems, num_panels)
            return jnp.sum(
                _compute_panel_logliks(
                    structural_betas, prior, diff_unchosen_chosen, data
                )
            )

        def value_fn(params: Float64[Array, "all_params"]) -> Float64[Array, ""]:
            """Per-panel negative observed-data log likelihood."""
            return -total_loglik(params) / scale

        def value_grad_hess_fn(
            params: Float64[Array, "all_params"],
        ) -> tuple[
            Float64[Array, ""],
            Float64[Array, "all_params"],
            Float64[Array, "all_params all_params"],
        ]:
            """Exact per-panel value, score, and Hessian of the mixture likelihood."""
            panel_scores, hessian = _panel_scores_and_hessian(
                params, diff_unchosen_chosen, data, packing
            )
            return (
                -total_loglik(params) / scale,
                -jnp.sum(panel_scores, axis=0) / scale,
                -hessian / scale,
            )

        state = exact_newton_minimize(
            value_fn,
            value_grad_hess_fn,
            flat_params,
            tol=POLISH_DECREMENT_TOL,
            maxiter=maxiter,
            max_step_norm=max_step_norm,
            line_search_maxiter=line_search_maxiter,
            damping=hessian_damping,
            initial_trust_radius=initial_trust_radius,
            accept_any_decrease=accept_any_decrease,
        )
        return state.params, state.step_num

    return filter_jit(run)


class PolishReport(NamedTuple):
    """Outcome of the observed-data Newton polish."""

    performed: bool
    iterations: int
    loglik_before: float
    loglik_after: float
    score_before: float
    score_after: float
    accepted: bool


def aitken_extrapolated_gap(logliks: list[float]) -> float:
    """Estimate the log likelihood still to be gained by running EM to convergence.

    For a linearly convergent sequence with increments ``d_t`` and rate
    ``r = d_t / d_{t-1}``, the remaining ascent after the latest iterate is the
    geometric tail ``d_t * r / (1 - r)``.

    Parameters
    ----------
    logliks : list[float]
        Observed-data log likelihoods in iteration order.  At least three are
        needed to estimate a rate.

    Returns
    -------
    float
        Estimated remaining ascent.  Returns ``inf`` when the sequence is too
        short or the rate is not a contraction, so a caller using this as a
        stopping rule keeps iterating rather than stopping on a bad estimate.
    """
    if len(logliks) < 3 or not all(math.isfinite(value) for value in logliks[-3:]):
        return float("inf")
    previous_increment = logliks[-2] - logliks[-3]
    increment = logliks[-1] - logliks[-2]
    roundoff = 64 * math.ulp(max(1.0, *(abs(value) for value in logliks[-3:])))
    if increment < -roundoff or previous_increment < -roundoff:
        return float("inf")
    if increment <= 0.0:
        # Round-off at the top of the likelihood: nothing measurable remains.
        return 0.0
    if previous_increment <= 0.0:
        return float("inf")
    rate = increment / previous_increment
    if not 0.0 < rate < 1.0:
        return float("inf")
    return increment * rate / (1.0 - rate)


@filter_jit
def _score_max_kernel(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> Float64[Array, ""]:
    """Largest absolute observed-data score component per panel.

    Compiling this discards the Hessian the derivative kernel also returns, which
    is the expensive half of that pass and is not needed for a stationarity check.
    """
    panel_scores, _ = _panel_scores_and_hessian(
        flat_params, diff_unchosen_chosen, data, packing
    )
    return jnp.max(jnp.abs(jnp.sum(panel_scores, axis=0))) / _require_panels(data)


@filter_jit
def _total_loglik_kernel(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> Float64[Array, ""]:
    """Total observed-data log likelihood at ``flat_params``."""
    latent_betas, packed_thetas = packing.unpack(flat_params)
    structural_betas = packing.to_structural(latent_betas)
    prior = packing.class_probs(packed_thetas, data.dems, _require_panels(data))
    return jnp.sum(
        _compute_panel_logliks(structural_betas, prior, diff_unchosen_chosen, data)
    )


def observed_score_max(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> float:
    """Return the largest absolute observed-data score component per panel."""
    return float(_score_max_kernel(flat_params, diff_unchosen_chosen, data, packing))


def em_vars_from_flat(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> EMVars:
    """Rebuild a complete EM state from a flat parameter vector.

    Parameters
    ----------
    flat_params : Float64[Array, "all_params"]
        Latent parameters in the canonical :class:`~lcl._params.ParamPacking`
        layout.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.
    packing : :class:`~lcl._params.ParamPacking`
        Owner of the flat layout and structural transforms.

    Returns
    -------
    :class:`~lcl._struct.EMVars`
        State with betas, membership coefficients, shares, posterior class
        probabilities, and the observed-data log likelihood all consistent with
        ``flat_params``.
    """
    num_panels = _require_panels(data)
    latent_betas, packed_thetas = packing.unpack(flat_params)
    structural_betas = packing.to_structural(latent_betas)
    prior_by_panel = packing.class_probs(packed_thetas, data.dems, num_panels)
    shares = jnp.mean(prior_by_panel, axis=0)
    # Without demographics the packed membership row holds bare log odds, and the
    # rest of the package carries that information in ``shares`` with
    # ``thetas=None``.  Preserving that convention keeps the round trip through
    # ParamPacking.pack exact.
    thetas = None if data.dems is None else packed_thetas
    posterior, loglik = _posterior_and_loglik(
        _compute_em_log_kernels(structural_betas, diff_unchosen_chosen, data),
        prior_by_panel,
    )
    return EMVars(
        latent_betas=latent_betas,
        structural_betas=structural_betas,
        thetas=thetas,
        shares=shares,
        unconditional_loglik=loglik,
        class_probs_by_panel=posterior,
    )


def polish_observed_data(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
    *,
    maxiter: int = 25,
    max_step_norm: float = 1000.0,
    line_search_maxiter: int = 40,
    hessian_damping: float = 0.0,
    initial_trust_radius: float = 1.0,
    accept_any_decrease: bool = False,
) -> tuple[Float64[Array, "all_params"], PolishReport]:
    """Drive the observed-data score to zero with safeguarded Newton steps.

    Parameters
    ----------
    flat_params : Float64[Array, "all_params"]
        Latent parameters from EM, used as the starting point.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.
    packing : :class:`~lcl._params.ParamPacking`
        Owner of the flat layout and structural transforms.
    maxiter : int, default=25
        Maximum number of Newton iterations.  Quadratic convergence from an EM
        solution normally needs a handful.
    max_step_norm : float, default=1000.0
        Trust-radius ceiling passed through to the solver.
    line_search_maxiter : int, default=40
        Armijo backtracking budget per iteration.
    hessian_damping : float, default=0.0
        Initial fallback diagonal shift passed through to the solver.
    initial_trust_radius : float, default=1.0
        Initial radius passed through to the solver.
    accept_any_decrease : bool, default=False
        Accept a strict objective decrease instead of requiring Armijo sufficiency.

    Returns
    -------
    flat_params : Float64[Array, "all_params"]
        Polished parameters, or the input unchanged when the polish did not
        improve the log likelihood.
    report : :class:`PolishReport`
        Before-and-after log likelihood and score, and whether the result was
        kept.
    """
    scale = jnp.asarray(float(max(_require_panels(data), 1)))

    def total_loglik(params: Float64[Array, "all_params"]) -> float:
        """Total observed-data log likelihood at ``params``."""
        return float(_total_loglik_kernel(params, diff_unchosen_chosen, data, packing))

    loglik_before = total_loglik(flat_params)
    score_before = observed_score_max(flat_params, diff_unchosen_chosen, data, packing)

    if maxiter <= 0:
        return flat_params, PolishReport(
            performed=False,
            iterations=0,
            loglik_before=loglik_before,
            loglik_after=loglik_before,
            score_before=score_before,
            score_after=score_before,
            accepted=False,
        )

    solve = _compiled_polish(
        packing,
        int(maxiter),
        float(max_step_norm),
        int(line_search_maxiter),
        float(hessian_damping),
        float(initial_trust_radius),
        bool(accept_any_decrease),
    )
    candidate, steps = solve(flat_params, diff_unchosen_chosen, data, scale)
    iterations = int(steps)
    loglik_after = total_loglik(candidate)

    # The line search only accepts decreases, so this should always hold; the
    # guard makes the polish unable to make a fit worse even if the analytic
    # Hessian degrades on a pathological problem.
    accepted = bool(jnp.all(jnp.isfinite(candidate))) and loglik_after >= loglik_before
    if not accepted:
        logger.warning(
            "The observed-data Newton polish did not improve the log likelihood "
            "(%.10g -> %.10g); keeping the EM solution.",
            loglik_before,
            loglik_after,
        )
        return flat_params, PolishReport(
            performed=True,
            iterations=iterations,
            loglik_before=loglik_before,
            loglik_after=loglik_before,
            score_before=score_before,
            score_after=score_before,
            accepted=False,
        )

    score_after = observed_score_max(candidate, diff_unchosen_chosen, data, packing)
    logger.info(
        "Observed-data polish: %d Newton steps, log likelihood %.10g -> %.10g, "
        "max score %.3e -> %.3e.",
        iterations,
        loglik_before,
        loglik_after,
        score_before,
        score_after,
    )
    return candidate, PolishReport(
        performed=True,
        iterations=iterations,
        loglik_before=loglik_before,
        loglik_after=loglik_after,
        score_before=score_before,
        score_after=score_after,
        accepted=True,
    )


__all__ = [
    "POLISH_DECREMENT_TOL",
    "PolishReport",
    "aitken_extrapolated_gap",
    "em_vars_from_flat",
    "observed_score_max",
    "polish_observed_data",
]
