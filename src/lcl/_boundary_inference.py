"""Conditional covariance and Gaussian critical-cone inference for LCL summaries.

Projection asymptotics: Geyer (1994), doi:10.1214/aos/1176325768;
Andrews (1999), doi:10.1111/1468-0262.00082. See docs/boundary_inference.md
for assumptions, active-set selection, and fallback interpretation.
"""

from collections.abc import Sequence
import logging
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import polars as pl
from jaxtyping import Array, ArrayLike, Bool, Float64, Integer
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._boundary import boundary_kkt_violation, projected_score
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
    beta, _ = packing.unpack(flat)
    J, H = _panel_scores_and_hessian(flat, diff, data, packing)
    score = np.asarray(jnp.mean(J, axis=0))
    active = np.asarray(result.boundary_parameter_indices, dtype=int)
    price_indices = packing.numeraire_idx * packing.num_classes + np.arange(
        packing.num_classes
    )
    result.boundary_kkt_violation = boundary_kkt_violation(score, active)
    result.observed_score_max = float(
        jnp.max(
            jnp.abs(projected_score(jnp.asarray(score), flat, packing.upper_bounds()))
        )
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
    full_covariance = np.zeros((result.num_params, result.num_params))
    full_covariance[np.ix_(free, free)] = covariance
    result.inference_status = "conditional_on_boundary" if active.size else "regular"
    result.boundary_summary_diagnostics = {"method": result.inference_status}
    threshold = _selection_threshold(groups)
    multiplier_z = _multiplier_statistics(
        score * data.num_panels, information, meat, active, free, inverse
    )
    if result.inference.covariance == "unadjusted" and np.any(multiplier_z > threshold):
        logger.warning(
            "A price is selected as strictly binding, suggesting a positive "
            "population multiplier and misspecification of the constrained model. "
            "The information equality required by covariance='unadjusted' is "
            "therefore not justified. Prefer "
            "covariance='clustered'."
        )
    # Statistical near-boundaries matter even when a finite-sample estimate
    # is interior. Looking only at numerically binding prices would miss these.
    near_boundary = bool(active.size)
    if not active.size:
        price_sd = np.sqrt(np.maximum(np.diag(covariance)[price_indices], 0))
        distances = -np.asarray(beta)[packing.numeraire_idx] - packing.numeraire_min_abs
        near_boundary = bool(np.any(distances <= threshold * price_sd))
    if result.inference.boundary == "projected" and near_boundary:
        result._boundary_summary_inputs = dict(
            information=information,
            meat=meat,
            multiplier_z=multiplier_z,
            groups=groups,
            price_indices=price_indices,
        )
    if active.size:
        logger.warning(
            "Class/prediction covariance holds %d binding price(s) fixed. "
            "Inspect beta_summary().inference_status for summary uncertainty.",
            len(active),
        )
    return jnp.asarray((full_covariance + full_covariance.T) / 2)


def _selection_threshold(groups: int) -> float:
    """Return the pointwise active-set cutoff ``sqrt(log G)``.

    This is the BIC-type moment-selection constant recommended by Andrews and
    Soares (2010, Econometrica 78:119): it diverges, so zero-multiplier
    boundaries are eventually recognized, but more slowly than ``sqrt(G)``.
    """
    return float(np.sqrt(np.log(max(groups, 3))))


def _multiplier_statistics(
    total_score: Float64[np.ndarray, "all_params"],
    information: Float64[np.ndarray, "all_params all_params"],
    meat: Float64[np.ndarray, "all_params all_params"],
    active: Integer[np.ndarray, "binding_prices"],
    free: Integer[np.ndarray, "free_params"],
    free_inverse_information: Float64[np.ndarray, "free_params free_params"],
) -> Float64[np.ndarray, "binding_prices"]:
    """Standardize each estimated KKT multiplier by its own sampling SD.

    With binding prices ``A`` held at the bound and the free coordinates ``F``
    re-optimized, the multiplier ``S_A(theta_hat)`` is to first order the
    nuisance-adjusted score ``S_A - I_AF I_FF^{-1} S_F``. Its sandwich variance,
    rather than the raw score variance ``B_AA``, is the scale of a robust
    Lagrange-multiplier statistic on this fixed face. Under information equality
    the raw variance overstates it; with misspecification or clustering the
    adjustment can increase or decrease it. With weak boundaries this is a
    selection scale, not the unconditional SD of the constrained multiplier.
    """
    if not active.size:
        return np.empty(0)
    # Rows map centered structural scores to first-order multiplier changes.
    loading = np.zeros((active.size, len(total_score)))
    loading[np.arange(active.size), active] = 1.0
    loading[:, free] = -information[np.ix_(active, free)] @ free_inverse_information
    variance = np.sum((loading @ meat) * loading, axis=1)
    return total_score[active] / np.sqrt(np.maximum(variance, np.finfo(float).tiny))


def _dual_multipliers(
    targets: Float64[np.ndarray, "draws weak_prices"],
    metric: Float64[np.ndarray, "weak_prices weak_prices"],
) -> Float64[np.ndarray, "draws weak_prices"]:
    """Solve ``min_{mu >= 0} mu' M mu / 2 - mu' z`` for each row ``z``.

    Zero is optimal exactly when every component of ``z`` is nonpositive, so
    those rows skip the solver; a single constraint has the closed form
    ``max(z, 0) / M``. The Cholesky factor doubles as a positive-definiteness
    check on ``M``.
    """
    factor = np.linalg.cholesky((metric + metric.T) / 2)
    if targets.shape[1] == 1:
        return np.maximum(targets, 0.0) / metric[0, 0]
    multipliers = np.zeros_like(targets)
    rows = np.flatnonzero(np.any(targets > 0.0, axis=1))
    rhs = solve_triangular(factor, targets[rows].T, lower=True).T
    for row, target in zip(rows, rhs):
        multipliers[row] = nnls(factor.T, target, maxiter=20 * targets.shape[1])[0]
    return multipliers


def _project_with_multipliers(
    draws: Float64[np.ndarray, "draws free_params"],
    inverse_information: Float64[np.ndarray, "free_params free_params"],
    constrained: Integer[np.ndarray, "weak_prices"],
) -> tuple[
    Float64[np.ndarray, "draws free_params"], Float64[np.ndarray, "draws weak_prices"]
]:
    """Project draws onto ``h[A] <= 0`` and return the dual multipliers too."""
    multipliers = _dual_multipliers(
        draws[:, constrained],
        inverse_information[np.ix_(constrained, constrained)],
    )
    projected = draws - multipliers @ inverse_information[:, constrained].T
    if np.max(projected[:, constrained]) > 1e-7 * max(1.0, np.max(abs(draws))):
        raise ValueError("Boundary projection failed its primal feasibility check.")
    return projected, multipliers


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
    return _project_with_multipliers(draws, inverse_information, indices)[0]


def _normal_draws(
    covariance: Float64[np.ndarray, "free_params free_params"],
    draws: int,
    seed: int,
) -> Float64[np.ndarray, "draws free_params"]:
    """Generate independent mean-zero Gaussian draws, rejecting indefiniteness.

    Antithetic pairs can help estimate means but duplicate even directional
    functionals and can increase noise in the variances needed here. Independent
    rows also justify the usual ``ddof=1`` sample variance correction.
    """
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    tolerance = (
        100 * len(values) * np.finfo(float).eps * max(np.max(abs(values)), 1e-300)
    )
    if np.min(values) < -tolerance:
        raise ValueError("Structural sandwich covariance is not positive semidefinite.")
    root = vectors * np.sqrt(np.maximum(values, 0))
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(draws, len(values)))
    return np.asarray(normal @ root.T)


def _summary_jacobian(result: LCLResults) -> CoefficientMoments:
    """Analytic structural Jacobians of class-weighted means and variances.

    Demographics are held at their empirical distribution, as in beta_summary.
    Membership parameters and their covariances with tastes remain uncertain.
    Memory never contains draws by households by classes.
    """
    packing = result._param_packing
    beta = _betas(result)
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
    threshold = _selection_threshold(inputs["groups"])
    multiplier_z = inputs["multiplier_z"]
    strong = active[multiplier_z > threshold]
    free = np.setdiff1d(np.arange(p), strong)
    inverse_array, info = _invert_information(
        information[np.ix_(free, free)], "structural information for boundary summary"
    )
    method = "critical_cone_projection"
    fallback = None
    weak: list[int] = []
    simulated_dimension = 0
    moments = _summary_jacobian(result)
    stds = np.sqrt(np.maximum(moments.variances, 0))
    beta = _betas(result)
    identified = moments.variances > 1e-12 * np.maximum(np.max(beta**2, axis=1), 1.0)
    if info.positive_definite:
        inverse = np.asarray(inverse_array)
        covariance = _sandwich_covariance(
            inverse, meat[np.ix_(free, free)], result.inference.covariance
        )
        unconstrained_sd = np.sqrt(np.maximum(np.diag(covariance), 0))
        positions = {int(index): i for i, index in enumerate(free)}
        identified &= _separated_spread_mask(
            stds, moments.shares, covariance, free, threshold
        )
        distances = -beta[result.model.numeraire_idx] - result.model.numeraire_min_abs
        weak = [
            positions[int(i)]
            for cls, i in enumerate(inputs["price_indices"])
            if int(i) in positions
            and distances[cls] <= threshold * unconstrained_sd[positions[int(i)]]
        ]
        try:
            mean_se, sd_se, simulated_dimension = _projected_moment_errors(
                covariance,
                inverse,
                np.asarray(weak, dtype=int),
                free,
                moments,
                identified,
                result.inference.boundary_draws,
                result.inference.boundary_seed,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fallback = str(error)
    else:
        fallback = "Structural information is not positive definite after strict-boundary reduction."
    if fallback is not None:
        method = "conditional_on_boundary_fallback"
        covariance = np.asarray(result.cov_matrix)
        jac_mean, jac_var = moments.mean_jacobian, moments.variance_jacobian
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
        simulated_dimension=simulated_dimension,
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
                mean=float(moments.means[index]),
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


def _betas(result: LCLResults) -> Float64[np.ndarray, "alt_vars classes"]:
    """Read the fitted taste coefficients."""
    if result.em_res.betas is None:
        raise ValueError("Structural coefficients are required for boundary inference.")
    return np.asarray(result.em_res.betas)


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


def _projected_moment_errors(
    covariance: Float64[np.ndarray, "free_params free_params"],
    inverse_information: Float64[np.ndarray, "free_params free_params"],
    weak: Integer[np.ndarray, "weak_prices"],
    free_indices: Integer[np.ndarray, "free_params"],
    moments: CoefficientMoments,
    separated_spread: Bool[np.ndarray, "alt_vars"],
    draws: int,
    seed: int,
) -> tuple[Float64[np.ndarray, "alt_vars"], Float64[np.ndarray, "alt_vars"], int]:
    """SDs of first-order mean/SD changes under the projected Gaussian limit.

    The limit is ``h = z - I^{-1}[:, A] mu(z_A)`` with ``z ~ N(0, covariance)``
    and ``mu`` the dual multipliers of the weak constraints ``A``. Means and
    separated SDs are linear in ``h``. Write each such functional as
    ``L z = a z_A + e`` with ``e`` independent of ``z_A`` and hence of ``mu``:

        Var(L h) = Var(e) + Var(a z_A - L I^{-1}[:, A] mu).

    The first term is exact; only the second is simulated, from draws of
    ``z_A`` alone. Without weak constraints these SEs are therefore the
    ordinary delta method, free of simulation noise. Functionals with
    ``L I^{-1}[:, A] = 0`` also retain their exact Gaussian variance, even when
    their scores correlate with the weak prices. Zero-spread SDs use the
    directional derivative and are simulated jointly with ``z_A``. Draws have
    only as many columns as these coordinates, rather than all free parameters
    unless every free parameter is needed.
    """
    positions = {int(index): i for i, index in enumerate(free_indices)}
    classes = len(moments.shares)
    stds = np.sqrt(np.maximum(moments.variances, 0))
    linear = np.vstack(
        (
            moments.mean_jacobian[:, free_indices],
            moments.variance_jacobian[np.ix_(separated_spread, free_indices)]
            / (2 * stds[separated_spread][:, None]),
        )
    )
    linear_variance = np.sum((linear @ covariance) * linear, axis=1)

    # Free class coordinates of each zero-spread SD; strict prices stay at zero.
    directional: dict[int, Integer[np.ndarray, "free_classes 2"]] = {}
    for variable in np.flatnonzero(~separated_spread).tolist():
        flat = [variable * classes + c for c in range(classes)]
        directional[variable] = np.array(
            [(c, positions[i]) for c, i in enumerate(flat) if i in positions],
            dtype=int,
        ).reshape(-1, 2)
    simulated = np.unique(
        np.concatenate([weak, *(pairs[:, 1] for pairs in directional.values())])
    ).astype(int)
    column = {int(index): i for i, index in enumerate(simulated)}
    projected = normal = (
        _normal_draws(covariance[np.ix_(simulated, simulated)], draws, seed)
        if simulated.size
        else np.zeros((draws, 0))
    )
    if weak.size:
        weak_columns = np.array([column[int(i)] for i in weak])
        projected, multipliers = _project_with_multipliers(
            normal, inverse_information[np.ix_(simulated, simulated)], weak_columns
        )
        weak_covariance = covariance[np.ix_(weak, weak)]
        # Standardize before the pseudoinverse so units alone cannot cause a
        # small, informative price direction to be truncated. A singular score
        # covariance is allowed; information still must be positive definite.
        scales = np.sqrt(np.maximum(np.diag(weak_covariance), 0))
        scales = np.where(scales > 0, scales, 1.0)
        correlation = (weak_covariance / scales[:, None]) / scales[None, :]
        regression = ((linear @ covariance[:, weak]) / scales) @ np.linalg.pinv(
            correlation, hermitian=True
        )
        residual_variance = linear_variance - np.sum(
            (regression @ correlation) * regression, axis=1
        )
        loading = linear @ inverse_information[:, weak]
        affected = np.any(loading != 0, axis=1)
        adjustment = loading[affected] @ multipliers.T
        linear_variance[affected] = np.maximum(residual_variance[affected], 0) + np.var(
            regression[affected] @ (normal[:, weak_columns] / scales).T - adjustment,
            axis=1,
            ddof=1,
        )

    num_vars = len(stds)
    linear_se = np.sqrt(np.maximum(linear_variance, 0))
    mean_se = linear_se[:num_vars]
    sd_se = np.full(num_vars, np.nan)
    sd_se[separated_spread] = linear_se[num_vars:]
    for variable, pairs in directional.items():
        # At equal class tastes, d(SD) is the share-weighted spread of changes.
        changes = np.zeros((len(projected), classes))
        changes[:, pairs[:, 0]] = projected[:, [column[int(i)] for i in pairs[:, 1]]]
        average = changes @ moments.shares
        spread = np.sqrt(
            np.maximum(((changes - average[:, None]) ** 2) @ moments.shares, 0)
        )
        sd_se[variable] = spread.std(ddof=1)
    return mean_se, sd_se, int(simulated.size)
