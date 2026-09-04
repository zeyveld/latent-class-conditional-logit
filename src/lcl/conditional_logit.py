"""Estimation and prediction for conditional logit."""

import logging
import warnings
from collections.abc import Mapping, Sequence
from time import time
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import jacrev
from jax.typing import ArrayLike
from jaxtyping import Array, Float64, install_import_hook
from scipy.stats import norm

# Decorate `@jaxtyped(typechecker=beartype.beartype)`
with install_import_hook("lcl", "beartype.beartype"):
    from lcl.constraints import DEFAULT_NEGATIVE_MIN_ABS
    from lcl._case_utils import (
        _diff_unchosen_chosen,
        _loglik_gradient,
        _loglik_value,
        _to_structural_betas,
    )
    from lcl._choice_model import ChoiceModel
    from lcl._diagnostics import LCLDiagnostics
    from lcl._encoding import _coerce_frame
    from lcl._kernels import _choice_probabilities_and_logsum, _diff_logit_components
    from lcl._delta import apply_delta_method, parametric_bootstrap_se
    from lcl._inference import _aggregate_scores, _robust_covariance
    from lcl._logging import log_or_print
    from lcl._optimize import _minimize
    from lcl.options import (
        InferenceOptions,
        Options,
        OptimizationOptions,
        _resolve_options,
        _resolve_weight_type,
    )
    from lcl._presentation import format_cl_coefficients
    from lcl._prediction import CLPrediction, resolve_panel_weights
    from lcl._struct import Data, OptimizeResult

logger = logging.getLogger(__name__)


class ConditionalLogit(ChoiceModel):
    """Specification and estimation for standard Multinomial Conditional Logit models.

    Unlike the Latent Class variant, this model estimates a single vector of
    homogeneous taste parameters across the entire sample.

    Parameters
    ----------
    numeraire : str | None, default=None
        The name of the variable (e.g., 'price') to use as the numeraire. If provided,
        its coefficient is bounded to be strictly negative to ensure logically
        consistent utility scaling and willingness-to-pay calculations.

    Attributes
    ----------
    numeraire_idx : int | None
        The column index of the numeraire variable in the expanded design matrix.
    """

    def __init__(
        self,
        numeraire: str | None = None,
        numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
    ) -> None:
        """Create an unfitted conditional-logit model specification."""
        super().__init__()
        self.numeraire = numeraire
        self.numeraire_min_abs = numeraire_min_abs
        self.numeraire_idx: int | None = None

    def fit(
        self,
        data: Any,
        alts_col: str,
        cases_col: str,
        panels_col: str | None = None,
        utility_formula: str | None = None,
        choice_col: str | None = None,
        case_varnames: Sequence[str] | None = None,
        variable_labels: Mapping[str, str] | None = None,
        weights: (
            str
            | Mapping[object, float | int]
            | Sequence[float | int]
            | ArrayLike
            | None
        ) = None,
        weight_type: str = "probability",
        init_beta: ArrayLike | None = None,
        options: Options | None = None,
        optimization_options: OptimizationOptions | None = None,
        inference: InferenceOptions | None = None,
    ) -> "CLResults":
        """Fit the conditional logit model via Maximum Likelihood Estimation.

        Supports both R-style formulas (via `formulaic`) and explicit lists of variables.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | ArrayLike
            The main dataset containing choice situations and alternatives in long format.
        alts_col : str
            Name of the column containing alternative identifiers.
        cases_col : str
            Name of the column grouping observations into distinct choice situations.
        panels_col : str | None, optional
            Name of the column mapping observations to specific decision-makers. If provided,
            the covariance matrix is automatically clustered at the panel level. If omitted,
            standard Huber-White robust standard errors are computed.
        utility_formula : str | None, optional
            Preferred Formulaic string for the alternative-specific utility
            specification.  If it includes a left-hand side, that outcome is used
            as the choice indicator; otherwise ``choice_col`` must be provided.
        choice_col : str | None, optional
            Name of the boolean/binary column indicating chosen alternatives.
        case_varnames : Sequence[str] | None, optional
            List of alternative-specific variables.
        variable_labels : Mapping[str, str] | None, optional
            Optional mapping from raw DataFrame/model variable names to
            human-readable labels used in printed coefficient tables.
        weights : str, Mapping, ArrayLike, or None, optional
            Case-level weights. A string names a data column that must be constant
            within case; a mapping is keyed by case ID (or ``(panel_id, case_id)``
            when case IDs repeat); a vector follows first case appearance in the
            input data and is realigned after encoding.
        weight_type : {"probability", "frequency"}, default="probability"
            How ``weights`` enter the variance, mirroring Stata's ``pweight`` and
            ``fweight``.  ``"probability"`` treats them as survey or sampling
            weights, so the score of the weighted objective for case ``i`` is
            ``w_i s_i`` and the robust meat is ``sum_i w_i^2 s_i s_i'``.
            ``"frequency"`` treats them as replication counts for collapsed data,
            giving ``sum_i w_i s_i s_i'`` and a sample size of ``sum_i w_i``.
            Point estimates and the log likelihood are identical either way; only
            robust and clustered standard errors differ, and they coincide when
            every weight is one.  Survey weights are the common case in household
            panel data, so they are the default.
        init_beta : ArrayLike | None, optional
            ``(K,)`` vector of initial taste parameters.
        optimization_options : OptimizationOptions | None, optional
            Preferred safeguarded exact-Newton settings.
        inference : InferenceOptions | None, optional
            Preferred covariance and standard-error settings.

        Returns
        -------
        :class:`~lcl.conditional_logit.CLResults`
            Results container housing coefficients, robust standard errors, and fit statistics.
        """
        resolved_options = _resolve_options(
            options,
            optimization_options=optimization_options,
            inference=inference,
        )
        optimization_options = resolved_options.optimization
        inference = resolved_options.inference

        # If no panels are provided, we substitute cases for panels purely to satisfy
        # the contiguity checks in the ingestion engine.
        _internal_panels_col = panels_col if panels_col is not None else cases_col

        parsed_data = self._ingest_data(
            data=data,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=_internal_panels_col,
            utility_formula=utility_formula,
            membership_formula=None,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=None,
            dems_data=None,
        )

        self._pre_fit(
            parsed_data.case_varnames,
            None,
            self.numeraire,
            variable_labels=variable_labels,
        )
        self.num_vars = len(self.case_varnames)

        if self.numeraire:
            try:
                self.numeraire_idx = self.case_varnames.index(self.numeraire)
            except ValueError:
                raise ValueError(
                    f"Numeraire '{self.numeraire}' not found in expanded design matrix."
                )
        else:
            self.numeraire_idx = None

        # Format data for MLE
        resolved_weight_type = _resolve_weight_type(weight_type)
        aligned_weights = self._resolve_case_weights(
            data,
            parsed_data,
            weights,
            cases_col=cases_col,
            panels_col=_internal_panels_col,
        )
        data_struct, weights_arr, init_beta_arr = self._setup_data(
            parsed=parsed_data,
            weights=aligned_weights,
            init_beta=init_beta,
        )

        diff_unchosen_chosen = _diff_unchosen_chosen(data_struct)

        # Resolve a coarser clustering to encoded panel order, then broadcast it
        # to cases: cases nest inside panels, so a grouping constant within panel
        # is well defined at either level.
        cluster_of_cases = None
        num_clusters = None
        cluster_column = inference.cluster_column
        if cluster_column is not None:
            cluster_by_panel, num_clusters = self._resolve_panel_cluster_ids(
                data,
                parsed_data,
                cluster_column,
                panels_col=_internal_panels_col,
            )
            if data_struct.panels_of_cases is None:
                raise ValueError("Panel identifiers are required for clustering.")
            cluster_of_cases = jnp.asarray(cluster_by_panel)[
                data_struct.panels_of_cases
            ]
            logger.info(
                "Clustering standard errors on %r: %s groups.",
                cluster_column,
                num_clusters,
            )

        # Estimate the conditional logit model
        optim_res = _minimize(
            _loglik_value,
            _loglik_gradient,
            init_beta_arr,
            args=(diff_unchosen_chosen, weights_arr),
            optimization_options=optimization_options,
            numeraire_idx=self.numeraire_idx,
            numeraire_min_abs=self.numeraire_min_abs,
            objective_scale=jnp.sum(weights_arr),
        )

        # Build Results
        estim_time_sec = time() - self._fit_start_time
        logger.info("Estimation time: %.3f seconds", estim_time_sec)

        return CLResults(
            model_spec=self,
            optim_res=optim_res,
            data_struct=data_struct,
            inference=inference,
            estim_time_sec=estim_time_sec,
            has_panels=panels_col is not None,
            case_weights=weights_arr,
            weight_type=resolved_weight_type,
            cluster_of_cases=cluster_of_cases,
            num_clusters=num_clusters,
        )


class CLResults:
    """Post-estimation results and inference container for Conditional Logit.

    Automatically handles the derivation of robust standard errors via the Delta Method
    if a softplus-constrained numeraire is specified in the model specification.
    """

    def __init__(
        self,
        model_spec: ConditionalLogit,
        optim_res: OptimizeResult,
        data_struct: Data,
        inference: InferenceOptions,
        estim_time_sec: float,
        has_panels: bool,
        case_weights: ArrayLike,
        weight_type: str = "probability",
        cluster_of_cases: ArrayLike | None = None,
        num_clusters: int | None = None,
    ) -> None:
        """Compute inference summaries from a fitted conditional-logit model.

        Parameters
        ----------
        model_spec : :class:`~lcl.conditional_logit.ConditionalLogit`
            Fitted model specification and variable metadata.
        optim_res : :class:`~lcl._struct.OptimizeResult`
            Optimizer output containing parameters, gradients, and Hessian inverse.
        data_struct : :class:`~lcl._struct.Data`
            Encoded estimation data.
        inference : :class:`~lcl.options.InferenceOptions`
            Covariance and standard-error configuration.
        estim_time_sec : float
            Wall-clock estimation time in seconds.
        has_panels : bool
            Whether robust covariance should cluster scores at the panel level.
        case_weights : ArrayLike
            Case weights aligned with encoded choice situations.
        weight_type : {"probability", "frequency"}, default="probability"
            Interpretation of ``case_weights`` for the robust variance.
        cluster_of_cases : ArrayLike | None, optional
            Zero-indexed cluster identifier per case, for clustering coarser than
            the panel.
        num_clusters : int | None, optional
            Number of distinct clusters implied by ``cluster_of_cases``.
        """
        self.model = model_spec
        self.data = data_struct
        self.inference = inference
        self.has_panels = has_panels
        self.case_weights = jnp.asarray(case_weights)
        self.weight_type = _resolve_weight_type(weight_type)
        self.converged = optim_res.success
        self.latent_coeff_ = optim_res.params

        # Recover structural parameters if numeraire was applied
        self.coeff_ = _to_structural_betas(
            self.latent_coeff_,
            self.model.numeraire_idx,
            self.model.numeraire_min_abs,
        )
        self.hess_inv = optim_res.hess_inv
        self.information_diagnostics = optim_res.information_diagnostics

        # Both robust branches use uncentered scores and the same finite-sample
        # multiplier; they differ only in the level at which scores are summed.
        # Clustering aggregates weighted scores first, so the two weight
        # interpretations coincide once a cluster sum has been taken.
        if inference.skip:
            self.hess_inv = jnp.full_like(self.hess_inv, jnp.nan)
            latent_cov = self.hess_inv
        elif inference.covariance in {"clustered", "robust"}:
            cluster_ids, num_groups = self._resolve_cluster_groups(
                inference,
                data_struct,
                has_panels,
                cluster_of_cases,
                num_clusters,
            )
            if cluster_ids is not None:
                if num_groups is None or num_groups < 2:
                    raise ValueError(
                        "Cluster-robust covariance requires at least two clusters."
                    )
                grad_g = _aggregate_scores(
                    optim_res.grad_n * jnp.asarray(case_weights)[:, None],
                    cluster_ids,
                    num_groups,
                )
                latent_cov = _robust_covariance(
                    self.hess_inv, grad_g, inference.finite_sample_correction
                )
            else:
                # Standard Huber-White Robust Standard Errors
                latent_cov = _robust_covariance(
                    self.hess_inv,
                    optim_res.grad_n,
                    inference.finite_sample_correction,
                    weights=case_weights,
                    weight_type=self.weight_type,
                )
        else:
            latent_cov = self.hess_inv

        # The public covariance is reported on the same scale as ``coeff_``, so
        # ``sqrt(diag(cov_matrix))`` reproduces ``stderr``.  ``latent_cov_matrix``
        # keeps the unconstrained parameterization the delta method and the
        # parametric bootstrap consume.
        self.latent_cov_matrix = latent_cov
        if self.model.numeraire_idx is not None:

            def struct_fn(
                p: Float64[Array, "alt_vars"],
            ) -> Float64[Array, "alt_vars"]:
                """Map latent coefficients to structural coefficients."""
                return _to_structural_betas(
                    p, self.model.numeraire_idx, self.model.numeraire_min_abs
                )

            jac = jacrev(struct_fn)(self.latent_coeff_)
            struct_cov = jac @ latent_cov @ jac.T
            self.cov_matrix = 0.5 * (struct_cov + struct_cov.T)
        else:
            self.cov_matrix = latent_cov
        self.stderr = jnp.sqrt(jnp.diag(self.cov_matrix))

        self.zvalues = onp.array(self.coeff_ / self.stderr, dtype=onp.float64)
        self.pvalues = 2 * norm.cdf(-onp.abs(self.zvalues))
        if self.model.numeraire_idx is not None:
            # The softplus transform makes the constrained coefficient strictly
            # negative by construction, so a test against zero is vacuous: the
            # null is excluded by the parameterization, not by the data.
            self.zvalues[self.model.numeraire_idx] = onp.nan
            self.pvalues[self.model.numeraire_idx] = onp.nan
        self.loglikelihood = -optim_res.neg_loglik
        self.estimation_message = optim_res.message
        self.total_iter = optim_res.nit
        self.estim_time_sec = estim_time_sec
        self.sample_size = data_struct.num_cases
        self.information_criterion_sample_size = (
            data_struct.num_panels
            if has_panels and data_struct.num_panels is not None
            else data_struct.num_cases
        )
        self.total_fun_eval = optim_res.nfev
        self.grad_n = optim_res.grad_n
        self.observed_score_max = float(
            jnp.max(
                jnp.abs(jnp.sum(optim_res.grad_n * self.case_weights[:, None], axis=0))
            )
        )

        # Information criteria
        self.aic = 2 * len(self.coeff_) - 2 * self.loglikelihood
        self.caic = (
            len(self.coeff_) * (jnp.log(self.information_criterion_sample_size) + 1)
            - 2 * self.loglikelihood
        )
        self.bic = (
            jnp.log(self.information_criterion_sample_size) * len(self.coeff_)
            - 2 * self.loglikelihood
        )
        self.adjusted_bic = (
            jnp.log((self.information_criterion_sample_size + 2) / 24)
            * len(self.coeff_)
            - 2 * self.loglikelihood
        )
        alternatives_per_case = jnp.bincount(
            data_struct.cases, length=data_struct.num_cases
        )
        self.null_loglikelihood = -jnp.sum(
            self.case_weights * jnp.log(alternatives_per_case)
        )
        self.mcfadden_r2 = 1.0 - self.loglikelihood / self.null_loglikelihood

        if not self.converged:
            logger.warning(
                "The optimization did not converge after %s iterations. Message: %s",
                self.total_iter,
                optim_res.message,
            )

    @staticmethod
    def _resolve_cluster_groups(
        inference: InferenceOptions,
        data_struct: Data,
        has_panels: bool,
        cluster_of_cases: ArrayLike | None,
        num_clusters: int | None,
    ) -> tuple[ArrayLike | None, int | None]:
        """Return the case-level grouping used to sum scores, if any.

        Returns ``(None, None)`` for the unclustered sandwich so the caller falls
        through to case-level Huber-White.
        """
        if inference.covariance != "clustered":
            return None, None
        if cluster_of_cases is not None:
            return cluster_of_cases, num_clusters
        if (
            has_panels
            and data_struct.panels_of_cases is not None
            and data_struct.num_panels is not None
        ):
            return data_struct.panels_of_cases, data_struct.num_panels
        return None, None

    @property
    def flat_params(self) -> Array:
        """Latent parameter vector, aligned with :attr:`latent_cov_matrix`."""
        return self.latent_coeff_

    @property
    def covariance_available(self) -> bool:
        """Report whether a usable covariance matrix was estimated."""
        return bool(onp.all(onp.isfinite(onp.asarray(self.cov_matrix))))

    def _apply_delta_method(
        self,
        func: Any,
        flat_params: Array,
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        """Apply the delta method to a function of the latent coefficients."""
        return apply_delta_method(func, flat_params, self.latent_cov_matrix, **kwargs)

    def _parametric_bootstrap_se(
        self,
        func: Any,
        flat_params: Array,
        *,
        draws: int = 500,
        seed: int = 0,
        **kwargs: Any,
    ) -> Array:
        """Estimate nonlinear standard errors from asymptotic parameter draws."""
        return parametric_bootstrap_se(
            func,
            flat_params,
            self.latent_cov_matrix,
            draws=draws,
            seed=seed,
            **kwargs,
        )

    def _structural_betas_and_class_probs(
        self,
        flat_params: Array,
        dems: Array | None,
        num_panels: int,
    ) -> tuple[Array, Array]:
        """Return structural betas and class probabilities for one homogeneous class.

        A conditional logit is the one-class case of the mixture, so presenting it
        that way lets the counterfactual inference in
        :mod:`lcl._prediction_inference` serve both estimators unchanged.
        """
        betas = _to_structural_betas(
            flat_params, self.model.numeraire_idx, self.model.numeraire_min_abs
        )
        return betas[:, None], jnp.ones((num_panels, 1), dtype=betas.dtype)

    def coefficient_table(self) -> pl.DataFrame:
        """Return conditional-logit coefficients with presentation labels.

        Returns
        -------
        pl.DataFrame
            One row per alternative-specific variable with raw variable names,
            display labels, estimates, standard errors, z-values, and p-values.
        """
        rows = []
        for coeff_idx, variable in enumerate(self.model.case_varnames):
            rows.append(
                {
                    "variable": variable,
                    "label": self.model.variable_label(variable),
                    "estimate": float(self.coeff_[coeff_idx]),
                    "std_error": float(self.stderr[coeff_idx]),
                    "z_value": float(self.zvalues[coeff_idx]),
                    "p_value": float(self.pvalues[coeff_idx]),
                }
            )
        return pl.DataFrame(rows)

    def parameter_names(self) -> list[str]:
        """Return names aligned with covariance rows and columns."""
        return list(self.model.case_varnames)

    @property
    def convergence(self) -> bool:
        """Deprecated alias for :attr:`converged`."""
        warnings.warn(
            "CLResults.convergence is deprecated; use converged.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.converged

    @property
    def covariance(self) -> Array:
        """Deprecated alias for :attr:`cov_matrix`."""
        warnings.warn(
            "CLResults.covariance is deprecated; use cov_matrix.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cov_matrix

    @property
    def abic(self) -> Array:
        """Deprecated alias for :attr:`adjusted_bic`."""
        warnings.warn(
            "CLResults.abic is deprecated; use adjusted_bic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.adjusted_bic

    def summarize_betas(
        self,
        header: tuple[str, str, str] = ("Variable", "Estimate", "Std. Error"),
        num_decimals: int = 3,
        *,
        show: bool = True,
    ) -> pl.DataFrame:
        """Print and return a table of parameter estimates and standard errors.

        Parameters
        ----------
        header : tuple[str, str, str], default=("Variable", "Estimate", "Std. Error")
            Column labels used for printed LaTeX and terminal tables.
        num_decimals : int, default=3
            Number of decimal places used in printed tables.
        show : bool, default=True
            Emit LaTeX and terminal renderings. Set to ``False`` for computation-only
            use.

        Returns
        -------
        pl.DataFrame
            Tidy coefficient table.  The ``variable`` column preserves raw model
            names, while ``label`` contains presentation labels.
        """
        table_df = self.coefficient_table()
        if show:
            log_or_print(
                logger,
                "%s",
                format_cl_coefficients(table_df, header, num_decimals),
            )
        return table_df

    def summarize(self, num_decimals: int = 3, *, show: bool = True) -> pl.DataFrame:
        """Alias for :meth:`summarize_betas`."""
        return self.summarize_betas(num_decimals=num_decimals, show=show)

    def loglik(self, data: Any, *, per_case: bool = False) -> float | pl.DataFrame:
        """Score observed choices with the fitted conditional-logit encoder."""
        parsed = self.model._transform_data(data, require_choice=True)
        data_struct, weights, _ = self.model._setup_data(parsed)
        differenced = _diff_unchosen_chosen(data_struct)
        log_probabilities, _ = _diff_logit_components(
            differenced.X,
            self.coeff_,
            differenced.cases,
            differenced.num_cases,
        )
        if not per_case:
            return float(jnp.sum(log_probabilities * weights))
        if parsed.original_cases is None:
            raise ValueError("Original case identifiers are unavailable.")
        first_case_rows = onp.asarray(data_struct.cases) != onp.roll(
            onp.asarray(data_struct.cases), 1
        )
        first_case_rows[0] = True
        return pl.DataFrame(
            {
                "case": onp.asarray(parsed.original_cases[first_case_rows]),
                "log_likelihood": onp.asarray(log_probabilities),
            }
        )

    def diagnostics(self) -> LCLDiagnostics:
        """Return convergence, fit, score, and information diagnostics."""
        rows: list[dict[str, object]] = [
            {
                "section": "fit",
                "check": "converged",
                "value": bool(self.converged),
                "status": "ok" if self.converged else "warning",
                "message": "Conditional-logit optimizer convergence flag.",
            },
            {
                "section": "fit",
                "check": "observed_score_max",
                "value": self.observed_score_max,
                "status": "warning" if self.observed_score_max > 1e-4 else "ok",
                "message": "Maximum absolute component of the weighted score.",
            },
            {
                "section": "fit",
                "check": "mcfadden_r2",
                "value": float(self.mcfadden_r2),
                "status": "ok",
                "message": "McFadden pseudo-R-squared against equal choice shares.",
            },
        ]
        if self.information_diagnostics is not None:
            info = self.information_diagnostics
            rows.extend(
                [
                    {
                        "section": "inference",
                        "check": "information_rank",
                        "value": info.rank,
                        "status": "warning" if info.rank_deficient else "ok",
                        "message": "Numerical rank of the information matrix.",
                    },
                    {
                        "section": "inference",
                        "check": "information_condition_number",
                        "value": info.condition_number,
                        "status": (
                            "warning"
                            if not info.positive_definite
                            or info.condition_number > 1e12
                            else "ok"
                        ),
                        "message": "Condition number of the information matrix.",
                    },
                ]
            )
        return LCLDiagnostics(pl.DataFrame(rows))

    def predict(
        self,
        data: Any,
        *,
        alts_col: str | None = None,
        cases_col: str | None = None,
        panels_col: str | None = None,
        panel_weights: str | Mapping[object, float] | Sequence[float] | None = None,
    ) -> CLPrediction:
        """Predict conditional choice probabilities for a given set of alternatives.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame
            The counterfactual dataset. Must contain all variables specified in
            the original model (including expanded dummy columns if a formula was used).
        alts_col : str
            Name of the column containing alternative identifiers.
        cases_col : str
            Name of the column grouping observations into distinct choice situations.
        panels_col : str | None, optional
            Name of the column mapping observations to specific decision-makers.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the computed out-of-sample choice probabilities.
        """
        if alts_col is not None or cases_col is not None:
            warnings.warn(
                "alts_col and cases_col are no longer needed by predict(); the "
                "fitted encoder supplies identifier columns.",
                DeprecationWarning,
                stacklevel=2,
            )
        parsed = self.model._transform_data(data)
        if (
            parsed.original_alts is None
            or parsed.original_cases is None
            or parsed.original_panels is None
        ):
            raise ValueError("Original prediction identifiers are unavailable.")
        data_struct, _, _ = self.model._setup_data(parsed)
        probs, logsum = _choice_probabilities_and_logsum(
            data_struct.X,
            self.coeff_[:, None],
            data_struct.cases,
            data_struct.num_cases,
        )
        predicted_probs = pl.DataFrame(
            {
                "panels": parsed.original_panels,
                "cases": parsed.original_cases,
                "alts": parsed.original_alts,
                "choice_probs": onp.asarray(probs[:, 0], dtype=onp.float64),
            }
        )
        first_case_rows = onp.asarray(data_struct.cases) != onp.roll(
            onp.asarray(data_struct.cases), 1
        )
        first_case_rows[0] = True
        marginal_utility_income = (
            1.0
            if self.model.numeraire_idx is None
            else float(-self.coeff_[self.model.numeraire_idx])
        )
        surplus = pl.DataFrame(
            {
                "panels": onp.asarray(parsed.original_panels[first_case_rows]),
                "cases": onp.asarray(parsed.original_cases[first_case_rows]),
                "surplus": onp.asarray(logsum[:, 0] / marginal_utility_income),
            }
        )
        if data_struct.panels is None or data_struct.num_panels is None:
            raise ValueError("Panel identifiers are required for prediction.")
        first_panel_rows = onp.asarray(data_struct.panels) != onp.roll(
            onp.asarray(data_struct.panels), 1
        )
        first_panel_rows[0] = True
        panel_ids = onp.asarray(parsed.original_panels[first_panel_rows])
        wtp_variables = [
            variable
            for idx, variable in enumerate(self.model.case_varnames)
            if idx != self.model.numeraire_idx
        ]
        if self.model.numeraire_idx is None:
            wtp_values = onp.empty((data_struct.num_panels, 0))
            wtp_variables = []
        else:
            ratios = (
                onp.delete(onp.asarray(self.coeff_), self.model.numeraire_idx)
                / marginal_utility_income
            )
            wtp_values = onp.repeat(ratios[None, :], data_struct.num_panels, axis=0)
        wtp_by_panel = pl.DataFrame(wtp_values, schema=wtp_variables).with_columns(
            pl.Series("panels", panel_ids)
        )
        encoder = self.model._encoder
        if encoder is None:
            raise ValueError("The fitted data encoder is unavailable.")
        raw_data = _coerce_frame(data).sort(
            list(
                dict.fromkeys([encoder.panels_col, encoder.cases_col, encoder.alts_col])
            )
        )
        resolved_panel_weights = resolve_panel_weights(
            panel_weights, panel_ids, raw_data, encoder.panels_col
        )
        return CLPrediction(
            predicted_probs_df=predicted_probs,
            surplus_df=surplus,
            wtp_alt_vars_by_panel_df=wtp_by_panel,
            predict_data=data_struct,
            results=self,
            class_probs_by_panel=jnp.ones((data_struct.num_panels, 1)),
            partition_data_df=None,
            original_alts=parsed.original_alts,
            original_cases=parsed.original_cases,
            original_panels=parsed.original_panels,
            raw_prediction_data=raw_data,
            panel_weights=resolved_panel_weights,
        )
