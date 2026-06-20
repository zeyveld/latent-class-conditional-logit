"""Estimation for latent-class conditional logit."""

import logging
from collections.abc import Callable, Sequence
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
from lcl._em_alg_steps import _em_alg
from lcl._logging import log_or_print
from lcl._results import LCLResults
from lcl._struct import (
    DiagnosticsOptions,
    EMAlgConfig,
    ErrorConfig,
    FitOptions,
    InferenceOptions,
    MleConfig,
    OptimizationOptions,
)
from lcl.spec import LCLSpec

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
        num_classes: int | LCLSpec = 5,
        numeraire: str | None = None,
        *,
        spec: LCLSpec | None = None,
        numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
    ) -> None:
        """Create an unfitted latent-class conditional-logit model specification."""
        super().__init__()
        if isinstance(num_classes, LCLSpec):
            if spec is not None:
                raise ValueError(
                    "Pass either LatentClassConditionalLogit(spec) or spec=..., not both."
                )
            spec = num_classes
            num_classes = spec.classes

        if spec is not None:
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
        formula: str | None = None,
        utility_formula: str | None = None,
        membership_formula: str | None = None,
        choice_col: str | None = None,
        case_varnames: Sequence[str] | None = None,
        dem_varnames: Sequence[str] | None = None,
        dems_data: Any | None = None,
        em_alg_config: EMAlgConfig | None = None,
        mle_config: MleConfig | None = None,
        error_config: ErrorConfig | None = None,
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
        formula : str | None, default=None
            Backward-compatible combined Formulaic string, for example
            ``"choice ~ price + time | income + C(segment)"``.  Prefer
            ``utility_formula`` and ``membership_formula`` in new code.
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
            Required if `formula` is not provided.
        case_varnames : Sequence[str] | None, default=None
            A list of alternative-specific variables to include in the utility
            specification. Required if `formula` is not provided.
        dem_varnames : Sequence[str] | None, default=None
            A list of demographic variables used to predict latent class membership.
        dems_data : Any | None, default=None
            An optional, separate panel-level dataset containing demographics. If
            provided, it will be merged with the main `data` on `panels_col`.
        em_alg_config : :class:`~lcl._struct.EMAlgConfig`, default=EMAlgConfig()
            A dataclass (PyTree) containing configuration options for the overall EM
            algorithm (e.g., maximum iterations, tolerance, hardware distribution).
        mle_config : :class:`~lcl._struct.MleConfig`, default=MleConfig()
            A dataclass (PyTree) containing optimization settings for the M-step's
            internal L-BFGS solver.

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
        if self.spec is not None:
            alts_col = alts_col or self.spec.ids.alt
            cases_col = cases_col or self.spec.ids.case
            panels_col = panels_col or self.spec.ids.panel
            choice_col = choice_col or self.spec.ids.choice
            formula = formula if formula is not None else self.spec.formula
            utility_formula = (
                utility_formula
                if utility_formula is not None
                else self.spec.utility_formula
            )
            membership_formula = (
                membership_formula
                if membership_formula is not None
                else self.spec.membership_formula
            )
            if formula is None and utility_formula is None:
                case_varnames = (
                    case_varnames if case_varnames is not None else self.spec.utility
                )
            if formula is None and membership_formula is None:
                dem_varnames = (
                    dem_varnames if dem_varnames is not None else self.spec.membership
                )
            self.num_classes = self.spec.classes
            if self.spec.numeraire is not None:
                self.numeraire = self.spec.numeraire
                self.numeraire_min_abs = self.spec.numeraire_min_abs

        if alts_col is None or cases_col is None or panels_col is None:
            raise ValueError(
                "alts_col, cases_col, and panels_col are required unless an "
                "LCLSpec is attached to the model."
            )

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if em_alg_config is None:
            em_alg_config = fit_options.to_em_config() if fit_options else EMAlgConfig()
        if mle_config is None:
            mle_config = optimization_options if optimization_options else MleConfig()
        if error_config is None:
            error_config = inference if inference is not None else ErrorConfig()
        if diagnostics is None:
            diagnostics = DiagnosticsOptions()

        parsed_data = self._ingest_data(
            data=data,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            formula=formula,
            utility_formula=utility_formula,
            membership_formula=membership_formula,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=dem_varnames,
            dems_data=dems_data,
        )

        self._pre_fit(
            parsed_data.case_varnames, parsed_data.dem_varnames, self.numeraire
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
            em_alg_config,
            mle_config,
            self.numeraire_idx,
            self.numeraire_min_abs,
        )

        num_devices = em_alg_config.num_devices
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
        em_history_rows: list[dict[str, Any]] = []
        while em_recursion < em_alg_config.maxiter:
            logger.info("EM recursion: %s", em_recursion)
            if progress_callback is not None:
                progress_callback({"event": "em_step", "iteration": em_recursion})

            em_vars = _em_alg(
                em_vars,
                diff_unchosen_chosen,
                data_struct,
                self.num_classes,
                mle_config,
                em_alg_config,
                self.numeraire_idx,
                self.numeraire_min_abs,
            )

            logliks_list.append(em_vars.unconditional_loglik)
            em_history_rows.append(self._em_history_row(em_recursion, em_vars))
            em_recursion += 1

            # Only force a host sync every `check_interval` steps
            if em_recursion >= 5 and (em_recursion % em_alg_config.check_interval == 0):
                # jax.block_until_ready() forces the sync explicitly here
                current_ll = float(em_vars.unconditional_loglik)
                past_ll = float(logliks_list[-5])

                rel_change = abs(current_ll - past_ll) / abs(past_ll)
                if rel_change <= em_alg_config.loglik_tol:
                    break

        strict_mle_config = replace(mle_config, ftol=1e-8, maxiter=500)
        em_vars = _em_alg(
            em_vars,
            diff_unchosen_chosen,
            data_struct,
            self.num_classes,
            strict_mle_config,
            em_alg_config,
            self.numeraire_idx,
            self.numeraire_min_abs,
        )
        em_history_rows.append(self._em_history_row(em_recursion, em_vars))
        optimization_history_rows = self._optimizer_snapshot(
            em_vars, diff_unchosen_chosen, data_struct, em_recursion
        )

        estim_time_sec = time() - self._fit_start_time

        log_or_print(logger, "Estimation time: %.3f seconds", estim_time_sec)
        if progress_callback is not None:
            progress_callback(
                {"event": "complete", "estimation_time_seconds": estim_time_sec}
            )

        return LCLResults(
            model_spec=self,
            em_vars=em_vars,
            estimation_data=data_struct,
            em_recursion=em_recursion,
            em_alg_config=em_alg_config,
            error_config=error_config,
            diagnostics_config=diagnostics,
            estim_time_sec=estim_time_sec,
            em_history=em_history_rows,
            optimization_history=optimization_history_rows,
        )

    def _em_history_row(self, em_iter: int, em_vars: Any) -> dict[str, Any]:
        """Return one lazily evaluated EM-history row.

        Parameters
        ----------
        em_iter : int
            EM recursion index.
        em_vars : EMVars-like
            Current EM state.

        Returns
        -------
        dict[str, Any]
            Log-likelihood and class-share diagnostics.  JAX scalar values are
            kept lazy until results construction to avoid a host synchronization
            on every EM iteration.
        """
        row: dict[str, Any] = {
            "em_iter": em_iter,
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
                raw_beta, self.numeraire_idx, grad, score_rows, hessian
            )
            rows.append(
                {
                    "em_iter": em_iter,
                    "class": class_idx,
                    "neg_loglik": float(neg_loglik),
                    "grad_norm": float(jnp.max(jnp.abs(grad_raw))),
                    "max_abs_beta": float(jnp.max(jnp.abs(structural_beta))),
                    "effective_panels": float(
                        onp.asarray(em_vars.class_probs_by_panel[:, class_idx]).sum()
                    ),
                }
            )
        return rows
