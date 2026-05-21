"""In-sample estimation results and inference."""

import logging

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array, hessian, jacrev
from jax.tree_util import Partial
from jax.typing import ArrayLike
from jaxtyping import Float64
from pylatexenc.latex2text import LatexNodes2Text
from tabulate import tabulate

from lcl._case_utils import _diff_unchosen_chosen, _to_structural_betas
from lcl._em_alg_steps import (
    _compute_conditional_class_probs,
    _compute_panel_logliks,
)
from lcl._kernels import _choice_probabilities_and_logsum, _class_membership_probs
from lcl._logging import log_or_print
from lcl._prediction import LCLPrediction
from lcl._struct import Data, EMVars, ErrorConfig, ParsedData, PastChoicesData


logger = logging.getLogger(__name__)


class LCLResults:
    """Post-estimation results and inference container.

    Computes robust sandwich covariance matrices (clustered at the decision-maker level)
    and handles the extraction of population-level moments via the Delta Method.

    Attributes
    ----------
    cov_matrix : Float64[Array, "all_params all_params"]
        Robust cluster-adjusted covariance matrix, strictly aligned with the Stata
        finite-sample correction multiplier :math:`(G / (G - 1))`.
    caic : float
        Consistent Akaike Information Criterion (Bozdogan, 1987).
    bic : float
        Bayesian Information Criterion (Schwarz, 1978).
    adjusted_bic : float
        Sample-size adjusted BIC (Sclove, 1987).
    """

    def __init__(
        self,
        model_spec,
        em_vars: EMVars,
        estimation_data: Data,
        em_recursion: int,
        em_alg_config,
        error_config: ErrorConfig | None,
        estim_time_sec: float,
    ) -> None:
        self.model = model_spec
        self.em_res = em_vars
        self.data = estimation_data
        self.total_recursions = em_recursion
        self.converged = em_recursion < (em_alg_config.maxiter - 1)
        self.estim_time_sec = estim_time_sec
        self.error_config = error_config if error_config is not None else ErrorConfig()

        self.flat_params = self._pack_params()

        # Calculate degrees of freedom
        num_beta_params = self.em_res.latent_betas.size
        if self.em_res.thetas is not None:
            num_theta_params = self.em_res.thetas.size
        else:
            num_theta_params = self.model.num_classes - 1

        self.num_params = num_beta_params + num_theta_params
        self.cov_matrix = self._compute_covariance()

        # Compute information criteria
        self.caic = (
            jnp.log(self.data.num_panels) + 1
        ) * self.num_params - 2 * self.em_res.unconditional_loglik
        self.bic = (
            jnp.log(self.data.num_panels) * self.num_params
            - 2 * self.em_res.unconditional_loglik
        )
        n_star = (self.num_params + 2) / 24
        self.adjusted_bic = (
            jnp.log(self.data.num_panels) * n_star
            - 2 * self.em_res.unconditional_loglik
        )
        logger.info(
            "Information criteria: CAIC=%.1f, BIC=%.1f, adjusted BIC=%.1f",
            self.caic,
            self.bic,
            self.adjusted_bic,
        )

        if not self.converged:
            logger.warning(
                "Optimization did not converge after %s iterations.",
                self.total_recursions,
            )

    def __repr__(self) -> str:
        status = "Converged" if self.converged else "Did Not Converge"
        return " | ".join(
            [
                f"<LCLResults: {self.model.num_classes} Classes",
                f"{status}",
                f"Log likelihood: {self.em_res.unconditional_loglik:.1f}>",
                f"CAIC: {self.caic:.1f}",
                f"BIC: {self.bic:.1f}",
                f"Adj. BIC: {self.adjusted_bic:.1f}>",
            ]
        )

    def _pack_params(self) -> Float64[Array, "all_params"]:
        """Flatten structural parameters and class memberships for Hessian calculation."""
        latent_betas_flat = self.em_res.latent_betas.ravel()
        if self.em_res.thetas is not None:
            theta_flat = self.em_res.thetas.ravel()
        else:
            shares = jnp.clip(self.em_res.shares, 1e-10)
            shares = shares / shares.sum()
            theta_flat = jnp.log(shares[1:] / shares[0]).ravel()
        return jnp.concatenate([latent_betas_flat, theta_flat])

    def _unpack_params(
        self, flat_params: Float64[Array, "all_params"]
    ) -> tuple[
        Float64[Array, "alt_vars classes"],
        Float64[Array, "dem_vars_plus_one classes_minus_one"],
    ]:
        """Reconstruct parameter matrices from the flattened array."""
        num_beta_params = self.model.num_vars * self.model.num_classes
        latent_betas = flat_params[:num_beta_params].reshape(
            self.model.num_vars, self.model.num_classes
        )

        thetas_flat = flat_params[num_beta_params:]
        if self.model.num_dem_vars > 0:
            thetas = thetas_flat.reshape(
                self.model.num_dem_vars + 1, self.model.num_classes - 1
            )
        else:
            thetas = thetas_flat.reshape(1, self.model.num_classes - 1)

        return latent_betas, thetas

    def _get_class_probs(
        self,
        thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "panels classes"]:
        """Extract unconditional class probabilities (via fractional response)."""
        return _class_membership_probs(thetas, dems, num_panels)

    def _compute_covariance(self) -> Float64[Array, "all_params all_params"]:
        if self.error_config.skip_std_errs:
            return jnp.full((self.num_params, self.num_params), jnp.nan)

        logger.info("Computing LCL covariance matrix.")
        diff_unchosen_chosen = _diff_unchosen_chosen(self.data)
        H = hessian(self._full_loglik_fn)(
            self.flat_params, diff_unchosen_chosen, self.data
        )
        H_inv = jnp.array(onp.linalg.pinv(onp.array(-H)))

        if not self.error_config.robust:
            return H_inv

        J = jacrev(self._panel_loglik_fn)(
            self.flat_params, diff_unchosen_chosen, self.data
        )
        B = J.T @ J

        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required for clustered covariance.")
        G = self.data.num_panels
        return (H_inv @ B @ H_inv) * (G / (G - 1))

    def _panel_loglik_fn(
        self, flat_params, diff_unchosen_chosen, data
    ) -> Float64[Array, "panels"]:
        """Compute the log-likelihood for each panel (used to build the Jacobian)."""
        latent_betas, thetas = self._unpack_params(flat_params)
        structural_betas = _to_structural_betas(
            latent_betas, getattr(self.model, "numeraire_idx", None)
        )
        class_probs = self._get_class_probs(thetas, data.dems, data.num_panels)
        return _compute_panel_logliks(
            structural_betas, class_probs, diff_unchosen_chosen, data
        )

    def _full_loglik_fn(
        self, flat_params, diff_unchosen_chosen, data
    ) -> Float64[Array, ""]:
        """Re-sums the panel log-likelihoods to a scalar for the Hessian."""
        return jnp.sum(self._panel_loglik_fn(flat_params, diff_unchosen_chosen, data))

    def _apply_delta_method(
        self, func, flat_params: Float64[Array, "all_params"], *args, **kwargs
    ) -> tuple[Float64[Array, "..."], Float64[Array, "..."]]:
        """Apply the Delta Method to derive standard errors for non-linear parameter functions."""
        target_func = Partial(func, *args, **kwargs)
        val = target_func(flat_params)
        jac = jacrev(target_func)(flat_params)

        if val.ndim == 0:
            variance = jac.T @ self.cov_matrix @ jac
        else:
            variance = jnp.einsum("kp,pq,kq->k", jac, self.cov_matrix, jac)

        return val, jnp.sqrt(jnp.maximum(variance, 0.0))

    def _calc_population_mean_betas(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "alt_vars"]:
        """Compute the expectation of the structural taste parameters across the population."""
        latent_betas, thetas = self._unpack_params(flat_params)

        class_probs = self._get_class_probs(thetas, dems, num_panels)
        avg_shares = jnp.mean(class_probs, axis=0)

        structural_betas = _to_structural_betas(
            latent_betas, getattr(self.model, "numeraire_idx", None)
        )
        return structural_betas @ avg_shares

    def _calc_population_std_betas(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "alt_vars"]:
        """Compute the population variance of the structural taste parameters."""
        latent_betas, thetas = self._unpack_params(flat_params)

        numeraire_idx = getattr(self.model, "numeraire_idx", None)

        class_probs = self._get_class_probs(thetas, dems, num_panels)
        avg_shares = jnp.mean(class_probs, axis=0)

        structural_betas = _to_structural_betas(latent_betas, numeraire_idx)

        mean_betas = structural_betas @ avg_shares
        diff_sq = (structural_betas - mean_betas[:, None]) ** 2
        var_betas = diff_sq @ avg_shares

        return jnp.sqrt(jnp.maximum(var_betas, 1e-250))

    def summarize_betas(
        self,
        header: tuple[str, str, str] = (
            "Variable",
            r"Means (\beta's)",
            r"Standard deviations (\sigma's)",
        ),
        num_decimals: int = 3,
    ) -> None:
        """Output population-level moments with Delta Method standard errors to the console and LaTeX."""
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required to summarize LCL results.")

        means, se_means = self._apply_delta_method(
            self._calc_population_mean_betas,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )
        stds, se_stds = self._apply_delta_method(
            self._calc_population_std_betas,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )

        body_rows, data_clean = [], []
        converter = LatexNodes2Text(math_mode="text")
        header_clean = [converter.latex_to_text(col) for col in header]

        for coeff_idx, coeff_nm in enumerate(self.model.case_varnames):
            body_rows.append(
                f"{coeff_nm} & {means[coeff_idx]:.{num_decimals}f} & {stds[coeff_idx]:.{num_decimals}f} \\\\"
            )
            body_rows.append(
                f" & ({se_means[coeff_idx]:.{num_decimals}f}) & ({se_stds[coeff_idx]:.{num_decimals}f}) \\\\"
            )
            var_clean = converter.latex_to_text(coeff_nm)
            data_clean.append(
                (
                    var_clean,
                    f"{means[coeff_idx]:.{num_decimals}f}",
                    f"{stds[coeff_idx]:.{num_decimals}f}",
                )
            )
            data_clean.append(
                (
                    "",
                    f"({se_means[coeff_idx]:.{num_decimals}f})",
                    f"({se_stds[coeff_idx]:.{num_decimals}f})",
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
        X: ArrayLike | None = None,
        alts: ArrayLike | None = None,
        cases: ArrayLike | None = None,
        panels: ArrayLike | None = None,
        dems: ArrayLike | None = None,
        past_choices: PastChoicesData | None = None,
        data: object | None = None,
        dems_data: object | None = None,
    ) -> LCLPrediction:
        """Generate out-of-sample predictions and counterfactual inclusive values (consumer surplus)."""

        if data is not None:
            parsed_predict = self.model._transform_data(data, dems_data=dems_data)
        else:
            if X is None or alts is None or cases is None or panels is None:
                raise ValueError(
                    "Provide either data=... or X, alts, cases, and panels."
                )
            parsed_predict = _parsed_prediction_arrays(
                X=X,
                dems=dems,
                alts=alts,
                cases=cases,
                panels=panels,
                case_varnames=self.model.case_varnames,
                dem_varnames=self.model.dem_varnames,
            )
        data, *_ = self.model._setup_data(parsed_predict)

        if past_choices is not None:
            parsed_past = _parsed_prediction_arrays(
                X=past_choices.X,
                dems=past_choices.dems,
                alts=past_choices.alts,
                cases=past_choices.cases,
                panels=past_choices.panels,
                y=past_choices.y,
                case_varnames=self.model.case_varnames,
                dem_varnames=self.model.dem_varnames,
            )
            data_past, *_ = self.model._setup_data(parsed_past)
            diff_unchosen_chosen_past = _diff_unchosen_chosen(data_past)
            class_probs_by_panel, _ = _compute_conditional_class_probs(
                structural_betas=self.em_res.structural_betas,
                thetas=self.em_res.thetas,
                shares=self.em_res.shares,
                diff_unchosen_chosen=diff_unchosen_chosen_past,
                data=data_past,
            )
        elif self.em_res.thetas is not None and data.dems is not None:
            class_probs_by_panel = self._get_class_probs(
                self.em_res.thetas, data.dems, data.num_panels
            )
        else:
            class_probs_by_panel = jnp.repeat(
                self.em_res.shares[None, :], data.num_panels, axis=0
            )

        choice_probs_by_class, log_sum_exp_utility = _choice_probabilities_and_logsum(
            data.X,
            self.em_res.structural_betas,
            data.cases,
            data.num_cases,
        )

        # Ensure alpha (marginal utility of income) is correctly signed
        numeraire_idx = getattr(self.model, "numeraire_idx", None)
        if numeraire_idx is None:
            marginal_utility_income = jnp.ones(self.model.num_classes)
        else:
            marginal_utility_income = -self.em_res.structural_betas[numeraire_idx, :]

        surplus_by_class = (
            log_sum_exp_utility / marginal_utility_income[None, :]
        ).squeeze()

        if numeraire_idx is not None:
            betas_sans_numeraire = jnp.delete(
                self.em_res.structural_betas, numeraire_idx, axis=0
            )
            wtp_alt_vars_by_class = betas_sans_numeraire / marginal_utility_income
            wtp_alt_vars_by_panel = class_probs_by_panel @ wtp_alt_vars_by_class.T
            schema = [
                var for var in self.model.case_varnames if var != self.model.numeraire
            ]
        else:
            wtp_alt_vars_by_panel = jnp.empty((data.num_panels, 0))
            schema = []

        panel_first_rows = data.panels != jnp.roll(data.panels, shift=1)
        panel_first_rows = panel_first_rows.at[0].set(True)
        panels_unique = onp.array(parsed_predict.original_panels[panel_first_rows])
        wtp_alt_vars_by_panel_df = pl.DataFrame(
            onp.array(wtp_alt_vars_by_panel), schema=schema
        ).with_columns(pl.Series("panels", panels_unique))

        if data.num_cases_per_panel is None or data.panels_of_cases is None:
            raise ValueError(
                "Panel identifiers are required for latent-class prediction."
            )
        conditional_surplus = jnp.einsum(
            "np,np->n",
            class_probs_by_panel[data.panels_of_cases],
            surplus_by_class,
        )

        unconditional_choice_probs = jnp.sum(
            class_probs_by_panel[data.panels] * choice_probs_by_class, axis=1
        )

        predicted_probs_df = pl.DataFrame(
            {
                "panels": parsed_predict.original_panels,
                "cases": parsed_predict.original_cases,
                "alts": parsed_predict.original_alts,
                "choice_probs": onp.array(
                    unconditional_choice_probs, dtype=onp.float64
                ),
            }
        )

        if data.panels is None:
            raise ValueError(
                "Panel identifiers are required for latent-class prediction."
            )
        first_case_rows = data.cases != jnp.roll(data.cases, shift=1)
        surplus_df = pl.DataFrame(
            {
                "panels": onp.array(parsed_predict.original_panels[first_case_rows]),
                "cases": onp.array(parsed_predict.original_cases[first_case_rows]),
                "surplus": onp.array(conditional_surplus, dtype=onp.float64),
            }
        )

        return LCLPrediction(
            predicted_probs_df=predicted_probs_df,
            surplus_df=surplus_df,
            wtp_alt_vars_by_panel_df=wtp_alt_vars_by_panel_df,
            predict_data=data,
            results=self,
            class_probs_by_panel=class_probs_by_panel,
        )


def _parsed_prediction_arrays(
    X: ArrayLike,
    dems: ArrayLike | None,
    alts: ArrayLike,
    cases: ArrayLike,
    panels: ArrayLike,
    case_varnames: list[str],
    dem_varnames: list[str] | None,
    y: ArrayLike | None = None,
) -> ParsedData:
    X_np = onp.asarray(X)
    alts_np = onp.asarray(alts)
    cases_np = onp.asarray(cases)
    panels_np = onp.asarray(panels)
    order = onp.lexsort((alts_np, cases_np, panels_np))

    X_sorted = X_np[order]
    alts_sorted = alts_np[order]
    cases_sorted = cases_np[order]
    panels_sorted = panels_np[order]
    y_sorted = None if y is None else onp.asarray(y)[order]

    panel_ids, panel_seq = onp.unique(panels_sorted, return_inverse=True)
    alt_ids, alt_seq = onp.unique(alts_sorted, return_inverse=True)
    _ = alt_ids

    case_seq = onp.empty_like(cases_sorted, dtype=onp.uint32)
    case_lookup: dict[tuple[object, object], int] = {}
    for idx, key in enumerate(zip(panels_sorted.tolist(), cases_sorted.tolist())):
        if key not in case_lookup:
            case_lookup[key] = len(case_lookup)
        case_seq[idx] = case_lookup[key]

    dems_array = None
    if dems is not None:
        dems_np = onp.asarray(dems)
        if dems_np.shape[0] != panel_ids.shape[0]:
            raise ValueError("dems must have one row per unique panel.")
        dems_array = jnp.array(dems_np, dtype="float64")

    return ParsedData(
        X=jnp.array(X_sorted, dtype="float64"),
        dems=dems_array,
        y=None if y_sorted is None else jnp.array(y_sorted, dtype="bool"),
        cases=jnp.array(case_seq, dtype="uint32"),
        panels=jnp.array(panel_seq, dtype="uint32"),
        alts=jnp.array(alt_seq, dtype="uint32"),
        case_varnames=case_varnames,
        dem_varnames=dem_varnames,
        original_alts=alts_sorted,
        original_cases=cases_sorted,
        original_panels=panels_sorted,
    )


# EOF
