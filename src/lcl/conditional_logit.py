"""Estimation and prediction for conditional logit."""

import logging
from collections.abc import Sequence
from time import time
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array, jacrev
from jax.ops import segment_sum
from jax.typing import ArrayLike
from jaxtyping import Float64, install_import_hook
from pylatexenc.latex2text import LatexNodes2Text  # type: ignore
from scipy.stats import t
from tabulate import tabulate

# Decorate `@jaxtyped(typechecker=beartype.beartype)`
with install_import_hook("lcl", "beartype.beartype"):
    from lcl._case_utils import (
        _diff_unchosen_chosen,
        _loglik_gradient,
        _to_structural_betas,
    )
    from lcl._choice_model import ChoiceModel
    from lcl._kernels import _choice_probabilities_and_logsum
    from lcl._logging import log_or_print
    from lcl._optimize import _minimize
    from lcl._struct import Data, ErrorConfig, MleConfig, OptimizeResult
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

    def __init__(self, numeraire: str | None = None) -> None:
        """Create an unfitted conditional-logit model specification."""
        super().__init__()
        self.numeraire = numeraire
        self.numeraire_idx: int | None = None

    def fit(
        self,
        data: Any,
        alts_col: str,
        cases_col: str,
        panels_col: str | None = None,
        formula: str | None = None,
        choice_col: str | None = None,
        case_varnames: Sequence[str] | None = None,
        weights: ArrayLike | None = None,
        init_beta: ArrayLike | None = None,
        mle_config: MleConfig | None = None,
        error_config: ErrorConfig | None = None,
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
        formula : str | None, optional
            R-style formula string (e.g., "choice ~ price + C(brand)").
            If provided, `choice_col` and `case_varnames` are ignored.
        choice_col : str | None, optional
            Name of the boolean/binary column indicating chosen alternatives.
        case_varnames : Sequence[str] | None, optional
            List of alternative-specific variables.
        weights : ArrayLike | None, optional
            ``(Nc,)`` vector of choice situation importance weights.
        init_beta : ArrayLike | None, optional
            ``(K,)`` vector of initial taste parameters.
        mle_config : :class:`~lcl._struct.MleConfig`, optional
            Configuration for the L-BFGS optimization routine.
        error_config : :class:`~lcl._struct.ErrorConfig`, optional
            Configuration determining the robust covariance estimation strategy.

        Returns
        -------
        :class:`~lcl.conditional_logit.CLResults`
            Results container housing coefficients, robust standard errors, and fit statistics.
        """
        if mle_config is None:
            mle_config = MleConfig()
        if error_config is None:
            error_config = ErrorConfig()

        # If no panels are provided, we substitute cases for panels purely to satisfy
        # the contiguity checks in the ingestion engine.
        _internal_panels_col = panels_col if panels_col is not None else cases_col

        parsed_data = self._ingest_data(
            data=data,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=_internal_panels_col,
            formula=formula,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=None,  # Standard CL does not take demographics
            dems_data=None,
        )

        self._pre_fit(parsed_data.case_varnames, None, self.numeraire)
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
        data_struct, weights_arr, init_beta_arr = self._setup_data(
            parsed=parsed_data,
            weights=weights,
            init_beta=init_beta,
        )

        diff_unchosen_chosen = _diff_unchosen_chosen(data_struct)

        # Estimate the conditional logit model
        optim_res = _minimize(
            _loglik_gradient,
            init_beta_arr,
            args=(diff_unchosen_chosen, weights_arr),
            mle_config=mle_config,
            numeraire_idx=self.numeraire_idx,
        )

        # Build Results
        estim_time_sec = time() - self._fit_start_time
        log_or_print(logger, "Estimation time: %.3f seconds", estim_time_sec)

        return CLResults(
            model_spec=self,
            optim_res=optim_res,
            data_struct=data_struct,
            error_config=error_config,
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
        error_config: ErrorConfig,
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
        error_config : :class:`~lcl._struct.ErrorConfig`
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
        self.coeff_ = _to_structural_betas(self.latent_coeff_, self.model.numeraire_idx)
        self.hess_inv = optim_res.hess_inv

        if error_config.skip_std_errs:
            self.hess_inv = jnp.eye(len(optim_res.params))

        # Covariance Calculation (Cluster-Robust, Standard Robust, or Unadjusted)
        if error_config.robust:
            if (
                has_panels
                and data_struct.panels_of_cases is not None
                and data_struct.num_panels is not None
            ):
                # Cluster-robust standard errors
                grad_g = segment_sum(
                    optim_res.grad_n,
                    data_struct.panels_of_cases,
                    num_segments=data_struct.num_panels,
                )
                B = grad_g.T @ grad_g
                robust_cov = self.hess_inv @ B @ self.hess_inv
                G = data_struct.num_panels
                self.covariance = robust_cov * (G / (G - 1))
            else:
                # Standard Huber-White Robust Standard Errors
                self.covariance = _robust_covariance(self.hess_inv, optim_res.grad_n)
        else:
            self.covariance = self.hess_inv

        # Apply delta method for standard errors if numeraire (softplus) is used
        if self.model.numeraire_idx is not None:

            def struct_fn(
                p: Float64[Array, "alt_vars"],
            ) -> Float64[Array, "alt_vars"]:
                """Map latent coefficients to structural coefficients."""
                return _to_structural_betas(p, self.model.numeraire_idx)

            jac = jacrev(struct_fn)(self.latent_coeff_)
            struct_cov = jac @ self.covariance @ jac.T
            self.stderr = jnp.sqrt(jnp.diag(struct_cov))
        else:
            self.stderr = jnp.sqrt(jnp.diag(self.covariance))

        self.zvalues = self.coeff_ / self.stderr
        self.pvalues = 2 * t.cdf(-onp.abs(self.zvalues), df=data_struct.num_cases)
        self.loglikelihood = -optim_res.neg_loglik
        self.estimation_message = optim_res.message
        self.total_iter = optim_res.nit
        self.estim_time_sec = estim_time_sec
        self.sample_size = data_struct.num_cases
        self.total_fun_eval = optim_res.nfev
        self.grad_n = optim_res.grad_n

        # Information criteria
        self.aic = 2 * len(self.coeff_) - 2 * self.loglikelihood
        self.caic = (
            len(self.coeff_) * (jnp.log(data_struct.num_cases) + 1)
            - 2 * self.loglikelihood
        )
        self.bic = (
            jnp.log(data_struct.num_cases) * len(self.coeff_) - 2 * self.loglikelihood
        )
        self.abic = (
            jnp.log((data_struct.num_cases + 2) / 24) * len(self.coeff_)
            - 2 * self.loglikelihood
        )

        if not self.convergence:
            logger.warning(
                "The optimization did not converge after %s iterations. Message: %s",
                self.total_iter,
                optim_res.message,
            )

    def summarize_betas(
        self,
        header: tuple[str, str, str] = ("Variable", "Estimate", "Std. Error"),
        num_decimals: int = 3,
    ) -> None:
        """Print LaTeX and plain-text tables summarizing parameter estimates and standard errors."""
        body_rows, data_clean = [], []
        converter = LatexNodes2Text(math_mode="text")
        header_clean = [converter.latex_to_text(col) for col in header]

        for coeff_idx, coeff_nm in enumerate(self.model.case_varnames):
            body_rows.append(
                f"{coeff_nm} & {self.coeff_[coeff_idx]:.{num_decimals}f} & {self.stderr[coeff_idx]:.{num_decimals}f} \\\\"
            )
            var_clean = converter.latex_to_text(coeff_nm)
            data_clean.append(
                (
                    var_clean,
                    f"{self.coeff_[coeff_idx]:.{num_decimals}f}",
                    f"{self.stderr[coeff_idx]:.{num_decimals}f}",
                )
            )

        latex_string = "\n".join(
            [r"\toprule", " & ".join(header) + r" \\", r"\midrule", "%"]
            + body_rows
            + ["%", r"\bottomrule "]
        )
        table_preview = tabulate(
            data_clean,
            headers=header_clean,
            tablefmt="simple_outline",
            floatfmt=f".{num_decimals}f",
        )
        log_or_print(
            logger,
            "\n--- LaTeX Output ---\n\n%s\n\n--- Table preview ---\n\n%s",
            latex_string,
            table_preview,
        )

    def summarize(self, num_decimals: int = 3) -> None:
        """Alias for :meth:`summarize_betas`."""
        self.summarize_betas(num_decimals=num_decimals)

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
