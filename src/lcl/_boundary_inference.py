"""Conditional covariance and Gaussian critical-cone inference for LCL summaries.

Projection asymptotics: Geyer (1994), doi:10.1214/aos/1176325768;
Andrews (1999), doi:10.1111/1468-0262.00082. See docs/boundary_inference.md
for assumptions, active-set selection, and fallback interpretation.
"""

from collections.abc import Sequence
from dataclasses import replace
import logging
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import polars as pl
from jaxtyping import Array, ArrayLike, Bool, Float64, Integer
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.special import expit

from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._boundary import boundary_kkt_violation
from lcl._boundary_types import BoundarySummaryDiagnostics, CoefficientMoments
from lcl._inference import (
    _aggregate_scores,
    _invert_information,
    information_weak_directions,
)
from lcl._reporting import _model_variable_label
from lcl._results import LCLResults
from lcl._struct import Data, DiffUnchosenChosen

logger = logging.getLogger(__name__)


def boundary_covariance(
    result: LCLResults,
    flat: Float64[Array, "all_params"],
    diff: DiffUnchosenChosen,
    data: Data,
) -> Float64[Array, "all_params all_params"]:
    """Reuse one structural derivative pass for conditional and summary inference."""
    packing = result._param_packing
    if packing.numeraire_idx is None or data.num_panels is None:
        raise ValueError(
            "Boundary covariance requires a constrained coefficient and panel data."
        )
    beta, theta = packing.unpack(flat)
    structural = jnp.concatenate([packing.to_structural(beta).ravel(), theta.ravel()])
    J, H = _panel_scores_and_hessian(
        structural, diff, data, replace(packing, numeraire_idx=None)
    )
    score = np.asarray(jnp.mean(J, axis=0))
    active = np.asarray(result.boundary_parameter_indices, dtype=int)
    jac = np.ones(result.num_params)
    price_indices = packing.numeraire_idx * packing.num_classes + np.arange(
        packing.num_classes
    )
    jac[price_indices] = -expit(np.asarray(beta)[packing.numeraire_idx])
    result.boundary_kkt_violation = boundary_kkt_violation(score, active)
    result.observed_score_max = max(
        float(np.max(abs(score * jac))), result.boundary_kkt_violation
    )
    result.converged = bool(result.observed_score_max <= result.score_tol)
    if not result.converged:
        logger.warning(
            "Boundary inference requires stationarity and the structural KKT condition."
        )

    meat, groups = _centered_score_meat(
        J,
        result._cluster_ids,
        result._num_clusters,
        finite_sample_correction=result.inference.finite_sample_correction,
    )
    information = -np.asarray(H)
    free = np.setdiff1d(np.arange(result.num_params), active)
    inverse_array, diagnostics = _invert_information(
        information[np.ix_(free, free)],
        "free-parameter information conditional on boundary",
    )
    result.information_diagnostics = diagnostics
    if not diagnostics.positive_definite or diagnostics.condition_number > 1e8:
        names = result.parameter_names()
        result.information_weak_directions = information_weak_directions(
            information[np.ix_(free, free)], [names[i] for i in free]
        )
    if not diagnostics.positive_definite or not result.converged:
        return jnp.full((result.num_params, result.num_params), jnp.nan)
    inverse = np.asarray(inverse_array)
    covariance = _sandwich_covariance(
        inverse, meat[np.ix_(free, free)], result.inference.covariance
    )
    latent = np.zeros((result.num_params, result.num_params))
    latent[np.ix_(free, free)] = covariance / np.outer(jac[free], jac[free])
    result.inference_status = "conditional_on_boundary" if active.size else "regular"
    result.boundary_summary_diagnostics = {"method": result.inference_status}
    # Statistical near-boundaries matter even when a finite-sample estimate
    # is interior. Looking only at numerically binding prices would miss these.
    near_boundary = bool(active.size)
    if not active.size:
        price_sd = np.sqrt(np.maximum(np.diag(covariance)[price_indices], 0))
        distances = (
            -np.asarray(packing.to_structural(beta))[packing.numeraire_idx]
            - packing.numeraire_min_abs
        )
        near_boundary = bool(
            np.any(distances <= np.sqrt(np.log(max(groups, 3))) * price_sd)
        )
    if result.inference.boundary == "projected" and near_boundary:
        result._boundary_summary_inputs = dict(
            information=information,
            meat=meat,
            score=score * data.num_panels,
            groups=groups,
            price_indices=price_indices,
        )
    if active.size:
        logger.warning(
            "Class/prediction covariance holds %d binding price(s) fixed. "
            "Inspect beta_summary().inference_status for summary uncertainty.",
            len(active),
        )
    return jnp.asarray((latent + latent.T) / 2)


def project_upper_gaussian(
    draws: Float64[np.ndarray, "draws free_params"],
    inverse_information: Float64[np.ndarray, "free_params free_params"],
    constrained: Integer[np.ndarray, "weak_prices"] | Sequence[int],
) -> Float64[np.ndarray, "draws free_params"]:
    """Project rows in the information metric; solve only the constrained dual.

    min_h (h-z)' I (h-z), h[A] <= 0. The nonnegative dual has Hessian
    I^{-1}[A,A]; other coordinates adjust through their cross curvature.
    """
    indices = np.asarray(constrained, dtype=int)
    if not indices.size:
        return draws.copy()
    metric = inverse_information[np.ix_(indices, indices)]
    factor = np.linalg.cholesky((metric + metric.T) / 2)
    rhs = solve_triangular(factor, draws[:, indices].T, lower=True).T
    multipliers = np.empty_like(rhs)
    for row, target in enumerate(rhs):
        multipliers[row] = nnls(factor.T, target, maxiter=20 * len(indices))[0]
    projected = draws - multipliers @ inverse_information[:, indices].T
    if np.max(projected[:, indices]) > 1e-7 * max(1.0, np.max(abs(draws))):
        raise ValueError("Boundary projection failed its primal feasibility check.")
    return np.asarray(projected)


def _normal_draws(
    covariance: Float64[np.ndarray, "free_params free_params"],
    draws: int,
    seed: int,
) -> Float64[np.ndarray, "draws free_params"]:
    """Generate centered Gaussian draws without silently repairing indefiniteness."""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    tolerance = (
        100 * len(values) * np.finfo(float).eps * max(np.max(abs(values)), 1e-300)
    )
    if np.min(values) < -tolerance:
        raise ValueError("Structural sandwich covariance is not positive semidefinite.")
    root = vectors * np.sqrt(np.maximum(values, 0))
    rng = np.random.default_rng(seed)
    # Antithetic pairs improve reproducibility of means and boundary masses.
    normal = rng.normal(size=((draws + 1) // 2, len(values)))
    return np.asarray(np.concatenate((normal, -normal), axis=0)[:draws] @ root.T)


def _summary_jacobian(result: LCLResults) -> CoefficientMoments:
    """Analytic structural Jacobians of class-weighted means and variances.

    Demographics are held at their empirical distribution, as in beta_summary.
    Membership parameters and their covariances with tastes remain uncertain.
    Memory never contains draws by households by classes.
    """
    packing = result._param_packing
    beta = _structural_betas(result)
    _, theta = packing.unpack(result.flat_params)
    if result.data.num_panels is None:
        raise ValueError("Panel identifiers are required for coefficient summaries.")
    probs = np.asarray(
        packing.class_probs(theta, result.data.dems, result.data.num_panels)
    )
    shares = probs.mean(axis=0)
    design = (
        np.ones((len(probs), 1))
        if result.data.dems is None
        else np.column_stack((np.ones(len(probs)), np.asarray(result.data.dems)))
    )
    means = beta @ shares
    centered = beta - means[:, None]
    variances = centered**2 @ shares
    jac_mean = np.zeros((len(beta), result.num_params))
    jac_variance = np.zeros_like(jac_mean)
    for variable in range(len(beta)):
        start = variable * packing.num_classes
        jac_mean[variable, start : start + packing.num_classes] = shares
        jac_variance[variable, start : start + packing.num_classes] = (
            2 * shares * centered[variable]
        )
        for values, target in (
            (beta[variable], jac_mean),
            (centered[variable] ** 2, jac_variance),
        ):
            within_panel = probs @ values
            derivative = (
                design.T
                @ (probs[:, 1:] * (values[None, 1:] - within_panel[:, None]))
                / len(probs)
            )
            target[variable, packing.num_beta_params :] = derivative.ravel()
    return CoefficientMoments(means, variances, shares, jac_mean, jac_variance)


def projected_beta_summary(result: LCLResults) -> pl.DataFrame:
    """Report joint mean/SD uncertainty with a documented conditional fallback."""
    if result._boundary_summary_cache is not None:
        return result._boundary_summary_cache.clone()
    started = perf_counter()
    inputs = result._boundary_summary_inputs
    if inputs is None:
        raise ValueError(
            "Projected summary inputs were not retained by covariance estimation."
        )
    information, meat = inputs["information"], inputs["meat"]
    p = result.num_params
    active = np.asarray(result.boundary_parameter_indices, dtype=int)
    # Pointwise consistent critical-cone selection: sqrt(log G) diverges but
    # is o(sqrt G). Positive multipliers distinguish strict binding from a
    # zero-gradient boundary. This tuning is recorded, not claimed uniform.
    threshold = np.sqrt(np.log(max(inputs["groups"], 3)))
    score_sd = np.sqrt(np.maximum(np.diag(meat), np.finfo(float).tiny))
    multiplier_z = inputs["score"][active] / score_sd[active]
    strong = active[multiplier_z > threshold]
    free = np.setdiff1d(np.arange(p), strong)
    inverse_array, info = _invert_information(
        information[np.ix_(free, free)], "structural information for boundary summary"
    )
    method = "critical_cone_projection"
    fallback = None
    weak: list[int] = []
    means, variances, shares, jac_mean, jac_var = _summary_jacobian(result)
    stds = np.sqrt(np.maximum(variances, 0))
    beta = _structural_betas(result)
    identified = variances > 1e-12 * np.maximum(np.max(beta**2, axis=1), 1.0)
    if info.positive_definite:
        inverse = np.asarray(inverse_array)
        covariance = _sandwich_covariance(
            inverse, meat[np.ix_(free, free)], result.inference.covariance
        )
        unconstrained_sd = np.sqrt(np.maximum(np.diag(covariance), 0))
        positions = {int(index): i for i, index in enumerate(free)}
        identified &= _separated_spread_mask(stds, shares, covariance, free, threshold)
        distances = -beta[result.model.numeraire_idx] - result.model.numeraire_min_abs
        weak = [
            positions[int(i)]
            for cls, i in enumerate(inputs["price_indices"])
            if int(i) in positions
            and distances[cls] <= threshold * unconstrained_sd[positions[int(i)]]
        ]
        try:
            normal = _normal_draws(
                covariance,
                result.inference.boundary_draws,
                result.inference.boundary_seed,
            )
            projected = project_upper_gaussian(normal, inverse, weak)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fallback = str(error)
        if fallback is None:
            mean_se, sd_se = _propagate_summary_draws(
                projected,
                free,
                p,
                CoefficientMoments(means, variances, shares, jac_mean, jac_var),
                identified,
            )
    else:
        fallback = "Structural information is not positive definite after strict-boundary reduction."
    if fallback is not None:
        method = "conditional_on_boundary_fallback"
        covariance = np.asarray(result.cov_matrix)
        mean_se = np.sqrt(
            np.maximum(np.einsum("ip,pq,iq->i", jac_mean, covariance, jac_mean), 0)
        )
        var_se = np.sqrt(
            np.maximum(np.einsum("ip,pq,iq->i", jac_var, covariance, jac_var), 0)
        )
        sd_se = np.full(len(stds), np.nan)
        sd_se[identified] = var_se[identified] / (2 * stds[identified])
        logger.warning(
            "Boundary summary projection unavailable: %s Using explicitly conditional SEs.",
            fallback,
        )
    diagnostics: BoundarySummaryDiagnostics = dict(
        method=method,
        fallback_reason=fallback,
        active_parameters=active.tolist(),
        strict_parameters=strong.tolist(),
        weak_parameters=free[weak].tolist(),
        multiplier_z=multiplier_z.tolist(),
        selection_threshold=float(threshold),
        directional_sd_variables=[
            result.model.case_varnames[i] for i in np.flatnonzero(~identified)
        ],
        draws=result.inference.boundary_draws,
        seed=result.inference.boundary_seed,
        information=info._asdict(),
        seconds=perf_counter() - started,
    )
    result.boundary_summary_diagnostics = diagnostics
    rows: list[dict[str, str | float]] = []
    for index, variable in enumerate(result.model.case_varnames):
        rows.append(
            dict(
                variable=variable,
                label=_model_variable_label(result.model, variable),
                mean=float(means[index]),
                mean_se=float(mean_se[index]),
                sd=float(stds[index]),
                sd_se=float(sd_se[index]),
                min_class=float(beta[index].min()),
                max_class=float(beta[index].max()),
                inference_status=method,
            )
        )
    result._boundary_summary_cache = pl.DataFrame(rows)
    return result._boundary_summary_cache.clone()


def _structural_betas(result: LCLResults) -> Float64[np.ndarray, "alt_vars classes"]:
    """Read the reported coefficients without a latent-coordinate transformation."""
    if result.em_res.structural_betas is None:
        raise ValueError("Structural coefficients are required for boundary inference.")
    return np.asarray(result.em_res.structural_betas)


def _centered_score_meat(
    scores: Float64[Array, "panels all_params"],
    cluster_ids: Integer[ArrayLike, "panels"] | None,
    num_clusters: int | None,
    *,
    finite_sample_correction: bool,
) -> tuple[Float64[np.ndarray, "all_params all_params"], int]:
    """Center panel scores before grouping, without another panel-sized matrix.

    A constrained pseudo-true optimum can have nonzero expected boundary
    scores. Algebraically centering their cross-products avoids retaining a
    second panels-by-parameters array. Coarser clusters subtract their panel
    count times the common panel mean.
    """
    mean_score = np.asarray(jnp.mean(scores, axis=0))
    if cluster_ids is None:
        groups = scores.shape[0]
        meat = np.asarray(scores.T @ scores) - groups * np.outer(mean_score, mean_score)
    else:
        if num_clusters is None:
            raise ValueError("num_clusters is required with cluster_ids.")
        groups = num_clusters
        cluster_scores = _aggregate_scores(scores, cluster_ids, groups)
        counts = np.bincount(np.asarray(cluster_ids), minlength=groups)
        weighted_sum = np.asarray(cluster_scores.T @ jnp.asarray(counts))
        meat = (
            np.asarray(cluster_scores.T @ cluster_scores)
            - np.outer(weighted_sum, mean_score)
            - np.outer(mean_score, weighted_sum)
            + (counts @ counts) * np.outer(mean_score, mean_score)
        )
    if groups < 2:
        raise ValueError(
            "Boundary inference requires at least two independent clusters."
        )
    correction = groups / (groups - 1) if finite_sample_correction else 1.0
    return (meat + meat.T) * (correction / 2), groups


def _sandwich_covariance(
    inverse_information: Float64[np.ndarray, "free_params free_params"],
    score_meat: Float64[np.ndarray, "free_params free_params"],
    covariance_kind: str,
) -> Float64[np.ndarray, "free_params free_params"]:
    """Construct covariance on the selected structural parameter subspace."""
    if covariance_kind == "unadjusted":
        return inverse_information
    return np.asarray(inverse_information @ score_meat @ inverse_information)


def _separated_spread_mask(
    standard_deviations: Float64[np.ndarray, "alt_vars"],
    shares: Float64[np.ndarray, "classes"],
    covariance: Float64[np.ndarray, "free_params free_params"],
    free_indices: Integer[np.ndarray, "free_params"],
    selection_threshold: float,
) -> Bool[np.ndarray, "alt_vars"]:
    """Select ordinary SD derivatives only when class spread exceeds its noise.

    At zero population spread the sample spread is O(1/sqrt(G)), not exactly
    zero. The diverging threshold consistently selects the norm derivative
    there; it is a pointwise tuning rule, not a uniform coverage guarantee.
    """
    positions = {int(index): i for i, index in enumerate(free_indices)}
    separated = np.ones(len(standard_deviations), dtype=bool)
    for variable, spread in enumerate(standard_deviations):
        classes = [
            c for c in range(len(shares)) if variable * len(shares) + c in positions
        ]
        coordinates = [positions[variable * len(shares) + c] for c in classes]
        block = covariance[np.ix_(coordinates, coordinates)]
        weights = shares[classes]
        noise_variance = weights @ np.diag(block) - weights @ block @ weights
        separated[variable] = spread > selection_threshold * np.sqrt(
            max(float(noise_variance), 0)
        )
    return separated


def _propagate_summary_draws(
    projected: Float64[np.ndarray, "draws free_params"],
    free_indices: Integer[np.ndarray, "free_params"],
    num_params: int,
    moments: CoefficientMoments,
    separated_spread: Bool[np.ndarray, "alt_vars"],
) -> tuple[Float64[np.ndarray, "alt_vars"], Float64[np.ndarray, "alt_vars"]]:
    """Propagate joint taste/membership changes through mean and SD derivatives."""
    full_draws = np.zeros((len(projected), num_params))
    full_draws[:, free_indices] = projected
    mean_changes = full_draws @ moments.mean_jacobian.T
    variance_changes = full_draws @ moments.variance_jacobian.T
    sd_changes = np.empty_like(mean_changes)
    standard_deviations = np.sqrt(np.maximum(moments.variances, 0))
    sd_changes[:, separated_spread] = variance_changes[:, separated_spread] / (
        2 * standard_deviations[separated_spread]
    )
    for variable in np.flatnonzero(~separated_spread):
        classes = len(moments.shares)
        changes = full_draws[:, variable * classes : (variable + 1) * classes]
        average = changes @ moments.shares
        sd_changes[:, variable] = np.sqrt(
            np.maximum(((changes - average[:, None]) ** 2) @ moments.shares, 0)
        )
    return mean_changes.std(axis=0, ddof=1), sd_changes.std(axis=0, ddof=1)
