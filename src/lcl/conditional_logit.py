"""Estimation and prediction for conditional logit."""

import logging
from collections.abc import Mapping, Sequence
from time import time
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import jacrev
from jax.ops import segment_sum
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
    from lcl._kernels import _choice_probabilities_and_logsum
    from lcl._logging import log_or_print
    from lcl._optimize import _minimize
    from lcl._presentation import format_cl_coefficients
    from lcl._struct import (
        Data,
        InferenceOptions,
        OptimizationOptions,
        OptimizeResult,
    )
from lcl.utils import _robust_covariance

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
        init_beta: ArrayLike | None = None,
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
            Case-level importance weights. A string names a data column that must
            be constant within case; a mapping is keyed by case ID (or
            ``(panel_id, case_id)`` when case IDs repeat); a vector follows first
            case appearance in the input data and is realigned after encoding.
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
        if optimization_options is None:
            optimization_options = OptimizationOptions()
        if inference is None:
            inference = InferenceOptions()

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
        inference : :class:`~lcl._struct.InferenceOptions`
            Covariance and standard-error configuration.
        estim_time_sec : float
            Wall-clock estimation time in seconds.
        has_panels : bool
            Whether robust covariance should cluster scores at the panel level.
        """
        self.model = model_spec
        self.data = data_struct
        self.convergence = optim_res.success
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
        if inference.skip:
            self.hess_inv = jnp.full_like(self.hess_inv, jnp.nan)
            self.covariance = self.hess_inv
        elif inference.covariance == "clustered":
            if (
                has_panels
                and data_struct.panels_of_cases is not None
                and data_struct.num_panels is not None
            ):
                # Cluster-robust standard errors
                G = data_struct.num_panels
                if G < 2:
                    raise ValueError(
                        "Cluster-robust covariance requires at least two panels."
                    )
                grad_g = segment_sum(
                    optim_res.grad_n,
                    data_struct.panels_of_cases,
                    num_segments=G,
                )
                self.covariance = _robust_covariance(
                    self.hess_inv, grad_g, inference.finite_sample_correction
                )
            else:
                # Standard Huber-White Robust Standard Errors
                self.covariance = _robust_covariance(
                    self.hess_inv, optim_res.grad_n, inference.finite_sample_correction
                )
        else:
            self.covariance = self.hess_inv

        # Apply delta method for standard errors if numeraire (softplus) is used
        if self.model.numeraire_idx is not None:

            def struct_fn(
                p: Float64[Array, "alt_vars"],
            ) -> Float64[Array, "alt_vars"]:
                """Map latent coefficients to structural coefficients."""
                return _to_structural_betas(
                    p, self.model.numeraire_idx, self.model.numeraire_min_abs
                )

            jac = jacrev(struct_fn)(self.latent_coeff_)
            struct_cov = jac @ self.covariance @ jac.T
            self.stderr = jnp.sqrt(jnp.diag(struct_cov))
        else:
            self.stderr = jnp.sqrt(jnp.diag(self.covariance))

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
        self.abic = (
            jnp.log((self.information_criterion_sample_size + 2) / 24)
            * len(self.coeff_)
            - 2 * self.loglikelihood
        )

        if not self.convergence:
            logger.warning(
                "The optimization did not converge after %s iterations. Message: %s",
                self.total_iter,
                optim_res.message,
            )

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

    def predict(
        self,
        data: Any,
        alts_col: str,
        cases_col: str,
        panels_col: str | None = None,
    ) -> pl.DataFrame:
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
        parsed = self.model._transform_data(data)
        probs, _ = _choice_probabilities_and_logsum(
            parsed.X,
            self.coeff_[:, None],
            parsed.cases,
            int(jnp.max(parsed.cases)) + 1,
        )

        result_dict = {
            "cases": parsed.original_cases,
            "alts": parsed.original_alts,
            "choice_probs": onp.array(probs[:, 0], dtype=onp.float64),
        }

        if panels_col is not None:
            result_dict["panels"] = parsed.original_panels

        return pl.DataFrame(result_dict)
