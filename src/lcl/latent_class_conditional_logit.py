"""Estimation for latent-class conditional logit."""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import time
from typing import Any

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
from lcl._em_alg_steps import _em_step
from lcl._results import LCLResults
from lcl._struct import (
    DiagnosticsOptions,
    FitOptions,
    InferenceOptions,
    OptimizationOptions,
)
from lcl.spec import LCLSpec, resolve_lcl_spec

logger = logging.getLogger(__name__)


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

        if fit_options is None:
            fit_options = FitOptions()
        if optimization_options is None:
            optimization_options = OptimizationOptions()
        if inference is None:
            inference = InferenceOptions()
        if diagnostics is None:
            diagnostics = DiagnosticsOptions()

        if fit_options.starts > 1:
            best_seed = self._select_best_start(
                data=data,
                dems_data=dems_data,
                spec=self.spec,
                fit_options=fit_options,
                optimization_options=optimization_options,
                diagnostics=diagnostics,
                progress_callback=progress_callback,
            )
            fit_options = replace(fit_options, starts=1, seed=best_seed)

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

        data_struct, weights, init_beta = self._setup_data(parsed_data)
        if data_struct.num_panels is None:
            raise ValueError("panels_col is required for latent-class models.")
        if self.num_classes > data_struct.num_panels:
            raise ValueError("num_classes cannot exceed the number of panels.")
        diff_unchosen_chosen = _diff_unchosen_chosen(data_struct)

        em_vars = _get_starting_vals(
            diff_unchosen_chosen,
            data_struct,
            self.num_classes,
            fit_options,
            optimization_options,
            self.numeraire_idx,
            self.numeraire_min_abs,
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

        logliks_list, em_recursion = [], 0
        converged = False
        standard_converged = False
        em_history_rows: list[dict[str, Any]] = []
        strict_optimization_options = OptimizationOptions(
            gradient_tol=min(optimization_options.gradient_tol, 1e-8),
            maxiter=max(optimization_options.maxiter, 500),
            hessian_damping=optimization_options.hessian_damping,
            max_step_norm=optimization_options.max_step_norm,
            line_search_maxiter=optimization_options.line_search_maxiter,
            accept_any_decrease=optimization_options.accept_any_decrease,
        )

        # Reserve one recursion for the strict phase so max_em_iter is a genuine
        # cap over every complete EM update, including the final refit.
        standard_em_limit = max(fit_options.max_em_iter - 1, 0)
        while em_recursion < standard_em_limit:
            logger.info("EM recursion: %s", em_recursion)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "em_step",
                        "iteration": em_recursion,
                        "phase": "standard",
                    }
                )

            em_vars = _em_step(
                em_vars,
                diff_unchosen_chosen,
                data_struct,
                self.num_classes,
                optimization_options,
                fit_options,
                self.numeraire_idx,
                self.numeraire_min_abs,
            )

            logliks_list.append(em_vars.unconditional_loglik)
            em_history_rows.append(
                self._em_history_row(em_recursion, em_vars, phase="standard")
            )
            em_recursion += 1

            # Force a host synchronization only at configured convergence checks.
            if em_recursion >= 5 and (em_recursion % fit_options.check_interval == 0):
                current_ll = float(em_vars.unconditional_loglik)
                past_ll = float(logliks_list[-5])

                rel_change = abs(current_ll - past_ll) / max(abs(past_ll), 1.0)
                if rel_change <= fit_options.em_tol:
                    standard_converged = True
                    break

        strict_recursions = 0
        strict_rel_change: float | None = None
        while em_recursion < fit_options.max_em_iter:
            logger.info("Strict EM recursion: %s", em_recursion)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "em_step",
                        "iteration": em_recursion,
                        "phase": "strict",
                    }
                )

            pre_refit_ll = float(em_vars.unconditional_loglik)
            em_vars = _em_step(
                em_vars,
                diff_unchosen_chosen,
                data_struct,
                self.num_classes,
                strict_optimization_options,
                fit_options,
                self.numeraire_idx,
                self.numeraire_min_abs,
            )
            post_refit_ll = float(em_vars.unconditional_loglik)
            strict_rel_change = abs(post_refit_ll - pre_refit_ll) / max(
                abs(pre_refit_ll), 1.0
            )
            em_history_rows.append(
                self._em_history_row(em_recursion, em_vars, phase="strict")
            )
            em_recursion += 1
            strict_recursions += 1

            if strict_rel_change <= fit_options.em_tol:
                converged = True
                break

            if (
                standard_converged
                and strict_recursions == 1
                and em_recursion < fit_options.max_em_iter
            ):
                logger.info(
                    "The strict final refit moved the log likelihood by %.3g "
                    "relative units, above the EM tolerance %.3g; continuing "
                    "strict EM with %s recursions remaining.",
                    strict_rel_change,
                    fit_options.em_tol,
                    fit_options.max_em_iter - em_recursion,
                )

        if standard_converged and strict_rel_change is not None and not converged:
            logger.warning(
                "The strict final refit moved the log likelihood by %.3g "
                "relative units, above the EM tolerance %.3g, and the maximum "
                "of %s EM recursions was reached.",
                strict_rel_change,
                fit_options.em_tol,
                fit_options.max_em_iter,
            )
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
        )

    @staticmethod
    def _select_best_start(
        *,
        data: Any,
        dems_data: Any | None,
        spec: LCLSpec,
        fit_options: FitOptions,
        optimization_options: OptimizationOptions,
        diagnostics: DiagnosticsOptions,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> int:
        """Evaluate independent EM starts and return the best random seed.

        Preliminary starts skip covariance work. The selected seed is then refit by
        the caller with the requested inference settings so the returned results
        contain a covariance matrix aligned to the winning optimum.

        Parameters
        ----------
        data : Any
            Long-format estimation data.
        dems_data : Any | None
            Optional panel-level demographics.
        spec : LCLSpec
            Canonical model specification shared by every start.
        fit_options : FitOptions
            EM options including the number of starts and base seed.
        optimization_options : OptimizationOptions
            M-step optimizer settings.
        diagnostics : DiagnosticsOptions
            Diagnostic configuration forwarded to candidate fits.
        progress_callback : callable | None
            Optional progress callback.

        Returns
        -------
        int
            Seed associated with the highest final training log likelihood.

        Raises
        ------
        RuntimeError
            If every requested start fails.
        """
        candidates: list[tuple[float, bool, int]] = []
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
            candidate_model = LatentClassConditionalLogit(spec=spec)
            try:
                candidate_result = candidate_model.fit(
                    data=data,
                    dems_data=dems_data,
                    fit_options=replace(fit_options, starts=1, seed=seed),
                    optimization_options=optimization_options,
                    inference=InferenceOptions(skip=True),
                    diagnostics=diagnostics,
                )
            except Exception as exc:
                failures.append(f"seed {seed}: {exc}")
                logger.warning("LCL start with seed %s failed: %s", seed, exc)
                continue
            candidates.append(
                (
                    float(candidate_result.em_res.unconditional_loglik),
                    bool(candidate_result.converged),
                    seed,
                )
            )

        if not candidates:
            detail = "; ".join(failures)
            raise RuntimeError(f"All {fit_options.starts} EM starts failed: {detail}")

        converged_candidates = [item for item in candidates if item[1]]
        selection_pool = converged_candidates or candidates
        best_loglik, _, best_seed = max(selection_pool, key=lambda item: item[0])
        logger.info(
            "Selected EM start seed %s with log likelihood %.6f.",
            best_seed,
            best_loglik,
        )
        return best_seed

    def _em_history_row(
        self, em_iter: int, em_vars: Any, *, phase: str
    ) -> dict[str, Any]:
        """Return one lazily evaluated EM-history row.

        Parameters
        ----------
        em_iter : int
            EM recursion index.
        em_vars : EMVars-like
            Current EM state.
        phase : str
            Either ``"standard"`` or ``"strict"`` for the M-step settings used.

        Returns
        -------
        dict[str, Any]
            Log-likelihood and class-share diagnostics.  JAX scalar values are
            kept lazy until results construction to avoid a host synchronization
            on every EM iteration.
        """
        row: dict[str, Any] = {
            "em_iter": em_iter,
            "phase": phase,
            "loglik": em_vars.unconditional_loglik,
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
