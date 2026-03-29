"""Estimation and prediction for conditional logit."""

from time import time
from typing import Sequence

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array, jacrev
from jax.ops import segment_sum
from jax.typing import ArrayLike
from jaxtyping import Float64, install_import_hook
from pylatexenc.latex2text import LatexNodes2Text
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
    from lcl._optimize import _minimize
    from lcl._struct import Data, ErrorConfig, MleConfig, OptimizeResult
    from lcl.utils import _robust_covariance


class ConditionalLogit(ChoiceModel):
    """Specification and estimation for standard Multinomial Conditional Logit models.

    Unlike the Latent Class variant, this model estimates a single vector of
    homogenous taste parameters across the entire sample.
    """

    def __init__(
        self, numeraire: str | None = None, numeraire_idx: int | None = None
    ) -> None:
        super().__init__()
        self.numeraire = numeraire
        self.numeraire_idx = numeraire_idx

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        case_varnames: Sequence[str],
        alts: ArrayLike,
        cases: ArrayLike,
        panels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        init_beta: ArrayLike | None = None,
        mle_config: MleConfig = MleConfig(),
        error_config: ErrorConfig = ErrorConfig(),
    ) -> "CLResults":
        """Fit the conditional logit model via Maximum Likelihood Estimation.

        Parameters
        ----------
        X : ArrayLike
            ``(N, K)`` design matrix of alternative-specific characteristics in long format.
        y : ArrayLike
            ``(N,)`` boolean array indicating chosen alternatives.
        case_varnames : Sequence[str]
            List of variable names corresponding to the columns of ``X``.
        alts : ArrayLike
            ``(N,)`` array of alternative identifiers.
        cases : ArrayLike
            ``(N,)`` array grouping observations into distinct choice situations.
        panels : ArrayLike | None, optional
            ``(N,)`` array mapping observations to specific decision-makers. If provided,
            the covariance matrix is automatically clustered at the panel level.
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

        self._pre_fit(case_varnames, None, self.numeraire)
        self.num_vars = len(case_varnames)

        if self.numeraire:
            self.numeraire_idx = self.case_varnames.index(self.numeraire)
        else:
            self.numeraire_idx = None

        # Format data for MLE
        data, weights, init_beta = self._setup_data(
            X=X,
            dems=None,
            y=y,
            cases=cases,
            panels=panels,
            alts=alts,
            weights=weights,
            init_beta=init_beta,
        )

        diff_unchosen_chosen = _diff_unchosen_chosen(data)

        # Estimate the conditional logit model
        optim_res = _minimize(
            _loglik_gradient,
            init_beta,
            args=(diff_unchosen_chosen, weights),
            mle_config=mle_config,
            numeraire_idx=self.numeraire_idx,
        )

        if error_config.skip_std_errs:
            optim_res.hess_inv = jnp.eye(len(optim_res.params))

        self.hess_inv = optim_res.hess_inv

        # Covariance Calculation (Cluster-Robust, Standard Robust, or Unadjusted)
        if error_config.robust:
            if data.panels_of_cases is not None and data.num_panels is not None:
                # If provided panels, compute cluster-robust standard errors
                # (StataCorp. 2025. Stata 19 Base Reference Manual. College Station, TX: Stata Press.)

                # Sum the case-level gradients up to the panel level
                grad_g = segment_sum(
                    optim_res.grad_n, data.panels_of_cases, num_segments=data.num_panels
                )

                # Outer product of panel-level gradients
                B = grad_g.T @ grad_g
                robust_cov = self.hess_inv @ B @ self.hess_inv

                # Stata finite-sample cluster correction
                G = data.num_panels
                self.covariance = robust_cov * (G / (G - 1))
            else:
                # Otherwise, compute Standard Huber-White Robust Standard Errors
                self.covariance = _robust_covariance(self.hess_inv, optim_res.grad_n)
        else:
            self.covariance = self.hess_inv

        # Apply delta method for standard errors if numeraire (softplus) is used

        estim_time_sec = time() - self._fit_start_time

        return CLResults(
            model_spec=self,
            optim_res=optim_res,
            data=data,
            error_config=error_config,
            estim_time_sec=estim_time_sec,
        )


class CLResults:
    """Post-estimation results and inference container for Conditional Logit.

    Automatically handles the derivation of standard errors via the Delta Method
    if a softplus-constrained numeraire is specified in the model specification.
    """

    def __init__(
        self,
        model_spec: "ConditionalLogit",
        optim_res: OptimizeResult,
        data: Data,
        error_config: ErrorConfig,
        estim_time_sec: float,
    ) -> None:
        self.model = model_spec
        self.data = data
        self.convergence = optim_res.success
        self.latent_coeff_ = optim_res.params

        # Recover structural parameters if numeraire was applied
        self.coeff_ = _to_structural_betas(self.latent_coeff_, self.model.numeraire_idx)
        self.hess_inv = optim_res.hess_inv

        if error_config.robust:
            self.covariance = _robust_covariance(optim_res.hess_inv, optim_res.grad_n)
        else:
            self.covariance = optim_res.hess_inv

        # Apply delta method for standard errors if numeraire (softplus) is used
        if self.model.numeraire_idx is not None:

            def struct_fn(p) -> Float64[Array, "..."]:
                return _to_structural_betas(p, self.model.numeraire_idx)

            jac = jacrev(struct_fn)(self.latent_coeff_)
            struct_cov = jac @ self.covariance @ jac.T
            self.stderr = jnp.sqrt(jnp.diag(struct_cov))
        else:
            self.stderr = jnp.sqrt(jnp.diag(self.covariance))

        self.zvalues = self.coeff_ / self.stderr
        self.pvalues = 2 * t.cdf(-onp.abs(self.zvalues), df=data.num_cases)
        self.loglikelihood = -optim_res.neg_loglik
        self.estimation_message = optim_res.message
        self.total_iter = optim_res.nit
        self.estim_time_sec = estim_time_sec
        self.sample_size = data.num_cases
        self.total_fun_eval = optim_res.nfev
        self.grad_n = optim_res.grad_n

        # Information criteria
        self.aic = 2 * len(self.coeff_) - 2 * self.loglikelihood
        self.caic = (
            len(self.coeff_) * (jnp.log(data.num_cases) + 1) - 2 * self.loglikelihood
        )
        self.bic = jnp.log(data.num_cases) * len(self.coeff_) - 2 * self.loglikelihood
        self.abic = (
            jnp.log((data.num_cases + 2) / 24) * len(self.coeff_)
            - 2 * self.loglikelihood
        )

        if not self.convergence:
            print(
                "\n".join(
                    [
                        f"**** The optimization did not converge after {self.total_iter} iterations. ****",
                        f"Message: {optim_res.message}",
                    ]
                )
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
        print("\n--- LaTeX Output ---\n")
        print(latex_string)
        print("\n--- Table preview ---\n")
        print(
            tabulate(
                data_clean,
                headers=header_clean,
                tablefmt="simple_outline",
                floatfmt=f".{num_decimals}f",
            )
        )

    def predict(
        self,
        X: ArrayLike,
        case_varnames: Sequence[str],
        alts: ArrayLike,
        cases: ArrayLike,
        panels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> pl.DataFrame:
        """Predict conditional choice probabilities for a given set of alternatives.

        Parameters
        ----------
        X : ArrayLike
            ``(N, K)`` counterfactual design matrix.
        case_varnames : Sequence[str]
            Must strictly match the specification stored in `self.model.case_varnames`.
        alts : ArrayLike
            ``(N,)`` vector of alternative IDs.
        cases : ArrayLike
            ``(N,)`` vector of choice situation IDs.
        panels : ArrayLike | None, optional
            ``(N,)`` vector mapping observations to specific decision-makers.
        weights : ArrayLike | None, optional
            ``(Nc,)`` vector of importance weights.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the computed out-of-sample choice probabilities.
        """
        data, *_ = self.model._setup_data(
            X=X,
            dems=None,
            y=None,
            cases=cases,
            panels=panels,
            alts=alts,
            weights=weights,
        )

        if not onp.array_equal(onp.array(case_varnames), self.model.case_varnames):
            raise ValueError(
                "The provided `case_varnames` yield coefficient names that are inconsistent with those stored "
                "in `self.model.case_varnames`"
            )

        eV = jnp.exp(jnp.clip(data.X.dot(self.coeff_), a_max=700.0))
        sum_eV = segment_sum(eV, data.cases, num_segments=data.num_cases)
        probs = eV / sum_eV[data.cases]

        result_dict = {
            "cases": onp.array(cases),
            "alts": onp.array(alts),
            "choice_probs": onp.array(probs, dtype=onp.float64),
        }

        if panels is not None:
            result_dict["panels"] = onp.array(panels)

        return pl.DataFrame(result_dict)


# EOF
