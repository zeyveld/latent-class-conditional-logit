"""Estimation for latent-class conditional logit."""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import time
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as onp

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    pullback_negative_derivatives,
)
from lcl._case_utils import (
    _diff_unchosen_chosen,
    _loglik_gradient,
    _to_structural_betas,
)
from lcl._choice_model import ChoiceModel
from lcl._em_alg_startup import _get_starting_vals
from lcl._em_alg_steps import _em_step, place_em_vars
from lcl._params import ParamPacking
from lcl._polish import (
    PolishReport,
    aitken_extrapolated_gap,
    em_vars_from_flat,
    observed_score_max,
    polish_observed_data,
)
from lcl._results import LCLResults
from lcl.options import (
    DiagnosticsOptions,
    FitOptions,
    InferenceOptions,
    Options,
    OptimizationOptions,
    _resolve_options,
)
from lcl._struct import EMVars
from lcl.spec import LCLSpec, resolve_lcl_spec

logger = logging.getLogger(__name__)


class _EMRun(NamedTuple):
    """Outcome of one independent EM start."""

    em_vars: EMVars
    loglik: float
    recursions: int
    history: list[dict[str, Any]]
    criterion_met: bool
    seed: int


def _canonicalize_classes(em_vars: EMVars) -> tuple[EMVars, tuple[int, ...]]:
    """Return an observationally equivalent EM state in deterministic class order."""
    if (
        em_vars.latent_betas is None
        or em_vars.structural_betas is None
        or em_vars.shares is None
    ):
        return em_vars, ()

    structural = onp.asarray(em_vars.structural_betas)
    shares = onp.asarray(em_vars.shares)
    num_classes = structural.shape[1]
    permutation = tuple(
        sorted(
            range(num_classes),
            key=lambda class_idx: (
                *structural[:, class_idx].tolist(),
                float(shares[class_idx]),
                class_idx,
            ),
        )
    )
    perm = jnp.asarray(permutation, dtype=jnp.int32)

    thetas = em_vars.thetas
    if thetas is not None:
        full_membership = jnp.concatenate(
            [jnp.zeros((thetas.shape[0], 1), dtype=thetas.dtype), thetas], axis=1
        )
        reordered = full_membership[:, perm]
        thetas = reordered[:, 1:] - reordered[:, :1]

    posterior = em_vars.class_probs_by_panel
    if posterior is not None:
        posterior = posterior[:, perm]

    return (
        EMVars(
            latent_betas=em_vars.latent_betas[:, perm],
            structural_betas=em_vars.structural_betas[:, perm],
            thetas=thetas,
            shares=em_vars.shares[perm],
            unconditional_loglik=em_vars.unconditional_loglik,
            class_probs_by_panel=posterior,
        ),
        permutation,
    )


def _permute_em_history(
    rows: list[dict[str, Any]], permutation: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Align class-indexed EM history columns with the canonical final order."""
    if not permutation:
        return rows
    aligned: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        old_shares = [row.get(f"class_{idx}_share") for idx in range(len(permutation))]
        for new_idx, old_idx in enumerate(permutation):
            if old_shares[old_idx] is not None:
                new_row[f"class_{new_idx}_share"] = old_shares[old_idx]
        aligned.append(new_row)
    return aligned


class LatentClassConditionalLogit(ChoiceModel):
    """Specification and estimation for latent-class conditional logit models.

    This class provides the interface for defining and fitting a latent-class
    conditional logit model using an Expectation-Maximization (EM) algorithm. It
    inherits from the abstract base class `ChoiceModel` and manages the data
    ingestion, initialization, and iterative optimization of latent taste
    parameters and class membership probabilities.

    Parameters
    ----------
    num_classes : int, default=5
        The number of discrete latent classes to estimate.
    numeraire : str | None, default=None
        The name of the variable to be used as the numeraire (e.g., price or cost).
        If specified, its taste parameter is mathematically constrained to be
        strictly negative across all latent classes via a softplus transformation
        to ensure theoretically consistent willingness-to-pay calculations.

    Attributes
    ----------
    num_classes : int
        The number of discrete latent classes.
    numeraire : str | None
        The name of the numeraire variable.
    numeraire_idx : int | None
        The column index of the numeraire variable in the expanded design matrix,
        resolved during the `fit` method.
    num_vars : int
        The total number of alternative-specific variables (taste parameters),
        resolved during the `fit` method.
    num_dem_vars : int
        The total number of demographic variables, resolved during the `fit` method.
    """

    def __init__(
        self,
        num_classes: int = 5,
        numeraire: str | None = None,
        *,
        spec: LCLSpec | None = None,
        numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
    ) -> None:
        """Create an unfitted latent-class conditional-logit model specification."""
        super().__init__()
        if spec is not None:
            num_classes = spec.classes
            if (
                numeraire is not None
                and spec.numeraire is not None
                and numeraire != spec.numeraire
            ):
                raise ValueError(
                    "numeraire conflicts with the negative constraint in spec."
                )
            numeraire = numeraire or spec.numeraire
            numeraire_min_abs = spec.numeraire_min_abs

        self.spec = spec
        self.num_classes = num_classes
        self.numeraire = numeraire
        self.numeraire_min_abs = numeraire_min_abs
        self.numeraire_idx: int | None = None

    def fit(
        self,
        data: Any,
        alts_col: str | None = None,
        cases_col: str | None = None,
        panels_col: str | None = None,
        utility_formula: str | None = None,
        membership_formula: str | None = None,
        choice_col: str | None = None,
        case_varnames: Sequence[str] | None = None,
        dem_varnames: Sequence[str] | None = None,
        variable_labels: Mapping[str, str] | None = None,
        dems_data: Any | None = None,
        options: Options | None = None,
        fit_options: FitOptions | None = None,
        optimization_options: OptimizationOptions | None = None,
        inference: InferenceOptions | None = None,
        diagnostics: DiagnosticsOptions | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LCLResults:
        """Fit the latent-class conditional logit model using an EM algorithm.

        This method ingests raw data, translates it into strictly contiguous,
        zero-indexed JAX arrays (PyTrees), and executes the hardware-accelerated
        EM optimization routine.

        Parameters
        ----------
        data : Any
            The main dataset containing choice situations. Accepts a Polars DataFrame,
            Pandas DataFrame, or dictionary of arrays.
        alts_col : str
            The name of the column identifying specific alternatives within a choice
            situation.
        cases_col : str
            The name of the column grouping observations into distinct choice
            situations.
        panels_col : str
            The name of the column mapping choice situations to specific
            decision-makers (panels).
        utility_formula : str | None, default=None
            Formulaic string for the alternative-specific utility specification.
            Examples include ``"choice ~ cost + time + C(mode)"`` or, when
            ``choice_col`` supplies the outcome, ``"~ cost + time + C(mode)"``.
        membership_formula : str | None, default=None
            Right-hand-side Formulaic string for class-membership demographics,
            for example ``"~ income + C(segment)"``.  A left-hand side is not
            accepted because latent class labels are unobserved.
        choice_col : str | None, default=None
            The name of the boolean or binary column indicating chosen alternatives.
            Required when ``utility_formula`` has no left-hand side.
        case_varnames : Sequence[str] | None, default=None
            A list of alternative-specific variables to include in the utility
            specification. Required if ``utility_formula`` is not provided.
        dem_varnames : Sequence[str] | None, default=None
            A list of demographic variables used to predict latent class membership.
        variable_labels : Mapping[str, str] | None, default=None
            Optional mapping from raw DataFrame/model variable names to
            human-readable labels used in presentation tables.  Labels do not
            change model specification, constraints, prediction inputs, or WTP
            request names.
        dems_data : Any | None, default=None
            An optional, separate panel-level dataset containing demographics. If
            provided, it will be merged with the main `data` on `panels_col`.
        fit_options : FitOptions | None, optional
            Preferred EM settings, including multi-start orchestration.
        optimization_options : OptimizationOptions | None, optional
            Preferred exact-Newton M-step settings.
        inference : InferenceOptions | None, optional
            Preferred covariance and standard-error settings.
        diagnostics : DiagnosticsOptions | None, optional
            Diagnostic thresholds and switches.
        progress_callback : callable | None, optional
            Receives structured hardware, start, EM-step, and completion events.

        Notes
        -----
        Case or panel weights are not supported for latent-class estimation and
        this method takes no ``weights`` argument; passing one is a
        :class:`TypeError` rather than a silent no-op.  Weighted estimation is
        available for :class:`~lcl.conditional_logit.ConditionalLogit`.

        Returns
        -------
        :class:`~lcl._results.LCLResults`
            A container holding the estimated parameters, optimization metadata,
            information criteria, and methods for inference (standard errors,
            predictions).

        Raises
        ------
        ValueError
            If a `numeraire` was specified during class instantiation but cannot be
            found in the expanded design matrix columns.
        """
        self.spec = resolve_lcl_spec(
            spec=self.spec,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=dem_varnames,
            utility_formula=utility_formula,
            membership_formula=membership_formula,
            classes=self.num_classes,
            numeraire=self.numeraire,
            numeraire_min_abs=(
                self.numeraire_min_abs if self.numeraire is not None else None
            ),
            variable_labels=variable_labels,
        )
        alts_col = self.spec.ids.alt
        cases_col = self.spec.ids.case
        panels_col = self.spec.ids.panel
        choice_col = self.spec.ids.choice
        utility_formula = self.spec.utility_formula
        membership_formula = self.spec.membership_formula
        case_varnames = self.spec.utility
        dem_varnames = self.spec.membership
        variable_labels = self.spec.variable_labels
        self.num_classes = self.spec.classes
        self.numeraire = self.spec.numeraire
        self.numeraire_min_abs = self.spec.numeraire_min_abs

        resolved_options = _resolve_options(
            options,
            fit_options=fit_options,
            optimization_options=optimization_options,
            inference=inference,
            diagnostics=diagnostics,
        )
        fit_options = resolved_options.fit
        optimization_options = resolved_options.optimization
        inference = resolved_options.inference
        diagnostics = resolved_options.diagnostics

        parsed_data = self._ingest_data(
            data=data,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            utility_formula=utility_formula,
            membership_formula=membership_formula,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=dem_varnames,
            dems_data=dems_data,
        )

        self._pre_fit(
            parsed_data.case_varnames,
            parsed_data.dem_varnames,
            self.numeraire,
            variable_labels=variable_labels,
        )
        self.num_vars = len(self.case_varnames)
        self.num_dem_vars = len(self.dem_varnames) if self.dem_varnames else 0

        if self.numeraire:
            try:
                self.numeraire_idx = self.case_varnames.index(self.numeraire)
            except ValueError:
                raise ValueError(
                    f"Numeraire '{self.numeraire}' not found in expanded design matrix."
                )
        else:
            self.numeraire_idx = None

        data_struct, _, _ = self._setup_data(parsed_data)
        if data_struct.num_panels is None:
            raise ValueError("panels_col is required for latent-class models.")
        if self.num_classes > data_struct.num_panels:
            raise ValueError("num_classes cannot exceed the number of panels.")
        diff_unchosen_chosen = _diff_unchosen_chosen(data_struct)
        packing = ParamPacking(
            num_alt_vars=self.num_vars,
            num_classes=self.num_classes,
            num_dem_vars=self.num_dem_vars,
            numeraire_idx=self.numeraire_idx,
            numeraire_min_abs=self.numeraire_min_abs,
        )

        # Resolve a coarser clustering once, in encoded panel order, so every
        # start shares it and the results object never re-reads the raw frame.
        cluster_ids: onp.ndarray | None = None
        num_clusters: int | None = None
        cluster_column = inference.cluster_column
        if cluster_column is not None:
            cluster_ids, num_clusters = self._resolve_panel_cluster_ids(
                data, parsed_data, cluster_column, panels_col=panels_col
            )
            logger.info(
                "Clustering standard errors on %r: %s groups across %s panels.",
                cluster_column,
                num_clusters,
                data_struct.num_panels,
            )

        num_devices = fit_options.num_devices
        if num_devices > 1:
            if self.num_classes % num_devices == 0:
                message = f"Distributing {self.num_classes} classes across {num_devices} devices."
            else:
                message = f"Found {num_devices} devices; padding classes for balanced sharding."
        else:
            message = "Running beta updates on a single device."
        logger.info(message)
        if progress_callback is not None:
            progress_callback({"event": "hardware", "message": message})

        # Independent starts share one ingested dataset and one compiled EM step,
        # and the winner is kept outright rather than refit from its seed.
        best_run: _EMRun | None = None
        failures: list[str] = []
        for start_index in range(fit_options.starts):
            seed = fit_options.seed + start_index
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "start",
                        "start": start_index + 1,
                        "starts": fit_options.starts,
                        "seed": seed,
                    }
                )
            start_options = replace(fit_options, seed=seed)
            if fit_options.starts == 1:
                # A single start has nothing to fall back on, so let the original
                # exception and its traceback reach the caller unwrapped.
                run = self._run_em(
                    diff_unchosen_chosen=diff_unchosen_chosen,
                    data_struct=data_struct,
                    fit_options=start_options,
                    optimization_options=optimization_options,
                    progress_callback=progress_callback,
                )
            else:
                try:
                    run = self._run_em(
                        diff_unchosen_chosen=diff_unchosen_chosen,
                        data_struct=data_struct,
                        fit_options=start_options,
                        optimization_options=optimization_options,
                        progress_callback=progress_callback,
                    )
                except Exception as exc:  # noqa: BLE001 - reported to the caller
                    failures.append(f"seed {seed}: {exc}")
                    logger.warning("LCL start with seed %s failed: %s", seed, exc)
                    continue
            if best_run is None or run.loglik > best_run.loglik:
                best_run = run

        if best_run is None:
            detail = "; ".join(failures)
            raise RuntimeError(f"All {fit_options.starts} EM starts failed: {detail}")
        if fit_options.starts > 1:
            logger.info(
                "Selected EM start seed %s with log likelihood %.6f.",
                best_run.seed,
                best_run.loglik,
            )

        em_vars = best_run.em_vars
        em_history_rows = best_run.history
        em_recursion = best_run.recursions

        # Observed-data Newton polish.  EM is linearly convergent, so it stops
        # short of a stationary point; the observed information and the sandwich
        # covariance both assume the score vanishes at the reported estimate.
        if em_vars.latent_betas is None or em_vars.shares is None:
            raise RuntimeError("The EM run returned an incomplete parameter state.")
        flat_params = packing.pack(
            em_vars.latent_betas, em_vars.thetas, em_vars.shares
        )
        polish_report: PolishReport | None = None
        if fit_options.polish:
            if progress_callback is not None:
                progress_callback({"event": "polish", "iterations": None})
            polished, polish_report = polish_observed_data(
                flat_params,
                diff_unchosen_chosen,
                data_struct,
                packing,
                maxiter=fit_options.polish_maxiter,
                max_step_norm=optimization_options.max_step_norm,
                line_search_maxiter=optimization_options.line_search_maxiter,
            )
            if polish_report.accepted:
                em_vars = em_vars_from_flat(
                    polished, diff_unchosen_chosen, data_struct, packing
                )
            score_max = polish_report.score_after
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "polish",
                        "iterations": polish_report.iterations,
                        "score_before": polish_report.score_before,
                        "score_after": polish_report.score_after,
                    }
                )
        else:
            score_max = observed_score_max(
                flat_params, diff_unchosen_chosen, data_struct, packing
            )

        # A fit has converged when the observed-data score has actually vanished.
        # Reporting convergence from a log-likelihood change instead lets a
        # slowly crawling EM claim an optimum it has not reached.
        converged = bool(score_max <= fit_options.score_tol)
        if not converged:
            logger.warning(
                "The maximum absolute observed-data score is %.3e, above the "
                "tolerance %.3g, so the estimate is not a stationary point of the "
                "mixture likelihood. Standard errors assume it is. Consider "
                "raising max_em_iter or polish_maxiter.",
                score_max,
                fit_options.score_tol,
            )

        em_vars, class_permutation = _canonicalize_classes(em_vars)
        em_history_rows = _permute_em_history(em_history_rows, class_permutation)
        final_em_iter = max(em_recursion - 1, 0)
        optimization_history_rows = self._optimizer_snapshot(
            em_vars, diff_unchosen_chosen, data_struct, final_em_iter
        )

        estim_time_sec = time() - self._fit_start_time

        logger.info("Estimation time: %.3f seconds", estim_time_sec)
        if progress_callback is not None:
            progress_callback(
                {"event": "complete", "estimation_time_seconds": estim_time_sec}
            )

        return LCLResults(
            model_spec=self,
            em_vars=em_vars,
            estimation_data=data_struct,
            em_recursion=em_recursion,
            converged=converged,
            inference=inference,
            diagnostics_config=diagnostics,
            estim_time_sec=estim_time_sec,
            em_history=em_history_rows,
            optimization_history=optimization_history_rows,
            observed_score_max=score_max,
            score_tol=fit_options.score_tol,
            em_criterion_met=best_run.criterion_met,
            polish_report=polish_report,
            cluster_ids=cluster_ids,
            num_clusters=num_clusters,
            param_packing=packing,
        )

    def _run_em(
        self,
        *,
        diff_unchosen_chosen: Any,
        data_struct: Any,
        fit_options: FitOptions,
        optimization_options: OptimizationOptions,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> "_EMRun":
        """Run the EM recursion from one start and report where it stopped.

        The stopping rule is the Aitken-extrapolated remaining ascent per panel
        rather than the raw log-likelihood change.  EM converges linearly, so the
        raw change understates the distance to the optimum by ``1 / (1 - r)``;
        extrapolating the geometric tail makes ``em_tol`` mean what a user
        expects it to mean, and normalizing by the panel count keeps that meaning
        fixed as the sample grows.

        Parameters
        ----------
        diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
            Differenced design matrix.
        data_struct : :class:`~lcl._struct.Data`
            Encoded estimation data.
        fit_options : :class:`~lcl.options.FitOptions`
            EM settings, with ``seed`` already set for this start.
        optimization_options : :class:`~lcl.options.OptimizationOptions`
            M-step Newton settings.
        progress_callback : callable | None
            Optional progress callback.

        Returns
        -------
        _EMRun
            Final EM state, log likelihood, iteration count, history rows, and
            whether the Aitken criterion was met.
        """
        em_vars = _get_starting_vals(
            diff_unchosen_chosen,
            data_struct,
            self.num_classes,
            fit_options,
            optimization_options,
            self.numeraire_idx,
            self.numeraire_min_abs,
        )
        # Match the placement the compiled step returns, so iteration one and
        # every later iteration share a single executable.
        em_vars = place_em_vars(em_vars, fit_options.num_devices)

        num_panels = max(int(data_struct.num_panels or 1), 1)
        loglik_history: list[float] = [float(em_vars.unconditional_loglik)]
        history_rows: list[dict[str, Any]] = []
        criterion_met = False
        em_recursion = 0
        mstep_warnings = 0

        while em_recursion < fit_options.max_em_iter:
            em_vars, step_diagnostics = _em_step(
                em_vars,
                diff_unchosen_chosen,
                data_struct,
                self.num_classes,
                optimization_options,
                fit_options,
                self.numeraire_idx,
                self.numeraire_min_abs,
            )

            # One host transfer per iteration carries every scalar the loop
            # needs, so the compiled step is never interrupted more than once.
            probe = jnp.stack(
                [
                    em_vars.unconditional_loglik,
                    jnp.max(step_diagnostics.beta_newton_error),
                    step_diagnostics.membership_newton_error,
                ]
            )
            loglik, beta_error, membership_error = (
                float(value) for value in onp.asarray(probe)
            )
            loglik_history.append(loglik)
            em_recursion += 1
            if (
                max(beta_error, membership_error)
                > optimization_options.newton_decrement_tol
            ):
                mstep_warnings += 1

            history_rows.append(
                self._em_history_row(
                    em_recursion - 1,
                    em_vars,
                    loglik=loglik,
                    beta_newton_error=beta_error,
                    membership_newton_error=membership_error,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "em_step",
                        "iteration": em_recursion - 1,
                        "loglik": loglik,
                    }
                )

            if em_recursion % fit_options.check_interval == 0:
                remaining = aitken_extrapolated_gap(loglik_history)
                if remaining / num_panels <= fit_options.em_tol:
                    criterion_met = True
                    break

        if mstep_warnings:
            logger.info(
                "%s of %s EM recursions ended with an M-step Newton decrement "
                "above %.3g. That is normal early in EM and only matters if it "
                "persists at the final iterations.",
                mstep_warnings,
                em_recursion,
                optimization_options.newton_decrement_tol,
            )
        if not criterion_met:
            logger.info(
                "EM reached the maximum of %s recursions with an estimated "
                "%.3g log-likelihood units of ascent remaining.",
                fit_options.max_em_iter,
                aitken_extrapolated_gap(loglik_history),
            )

        return _EMRun(
            em_vars=em_vars,
            loglik=loglik_history[-1],
            recursions=em_recursion,
            history=history_rows,
            criterion_met=criterion_met,
            seed=fit_options.seed,
        )

    def _em_history_row(
        self,
        em_iter: int,
        em_vars: Any,
        *,
        loglik: float,
        beta_newton_error: float,
        membership_newton_error: float,
    ) -> dict[str, Any]:
        """Return one EM-history row.

        Parameters
        ----------
        em_iter : int
            EM recursion index.
        em_vars : EMVars-like
            Current EM state.
        loglik : float
            Observed-data log likelihood after this recursion.
        beta_newton_error : float
            Largest class-specific M-step Newton decrement.
        membership_newton_error : float
            Class-membership M-step Newton decrement.

        Returns
        -------
        dict[str, Any]
            Log-likelihood, M-step convergence, and class-share diagnostics.  The
            scalars arrive already materialized from the loop's single host
            transfer, so building a row costs no extra synchronization.
        """
        row: dict[str, Any] = {
            "em_iter": em_iter,
            "loglik": loglik,
            "beta_newton_error": beta_newton_error,
            "membership_newton_error": membership_newton_error,
        }
        if em_vars.shares is not None:
            for class_idx in range(self.num_classes):
                row[f"class_{class_idx}_share"] = em_vars.shares[class_idx]
        return row

    def _optimizer_snapshot(
        self,
        em_vars: Any,
        diff_unchosen_chosen: Any,
        data_struct: Any,
        em_iter: int,
    ) -> list[dict[str, Any]]:
        """Compute final class-level M-step diagnostics.

        Parameters
        ----------
        em_vars : EMVars-like
            Final EM state.
        diff_unchosen_chosen : DiffUnchosenChosen-like
            Differenced design matrix used by the conditional-logit kernels.
        data_struct : Data-like
            Encoded estimation data.
        em_iter : int
            EM recursion index associated with the final refit.

        Returns
        -------
        list[dict[str, Any]]
            One row per latent class containing first-order and scale diagnostics.
        """
        if (
            em_vars.latent_betas is None
            or em_vars.structural_betas is None
            or em_vars.class_probs_by_panel is None
            or data_struct.num_cases_per_panel is None
        ):
            return []

        class_probs_by_choice = jnp.repeat(
            em_vars.class_probs_by_panel,
            data_struct.num_cases_per_panel,
            axis=0,
            total_repeat_length=data_struct.num_cases,
        )
        rows: list[dict[str, Any]] = []
        for class_idx in range(self.num_classes):
            raw_beta = em_vars.latent_betas[:, class_idx]
            structural_beta = _to_structural_betas(
                raw_beta, self.numeraire_idx, self.numeraire_min_abs
            )
            weights = class_probs_by_choice[:, class_idx]
            (neg_loglik, score_rows), grad, hessian = _loglik_gradient(
                structural_beta, diff_unchosen_chosen, weights
            )
            grad_raw, _, _ = pullback_negative_derivatives(
                raw_beta,
                self.numeraire_idx,
                grad,
                score_rows,
                hessian,
                self.numeraire_min_abs,
            )
            gradient_scale = jnp.maximum(jnp.sum(weights), 1.0)
            rows.append(
                {
                    "em_iter": em_iter,
                    "class": class_idx,
                    "neg_loglik": float(neg_loglik),
                    "grad_norm": float(jnp.max(jnp.abs(grad_raw)) / gradient_scale),
                    "max_abs_beta": float(jnp.max(jnp.abs(structural_beta))),
                    "effective_panels": float(
                        onp.asarray(em_vars.class_probs_by_panel[:, class_idx]).sum()
                    ),
                }
            )
        return rows
