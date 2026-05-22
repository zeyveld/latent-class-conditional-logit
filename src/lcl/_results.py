"""In-sample estimation results and inference."""

import logging
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

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
from lcl._struct import (
    Data,
    DiffUnchosenChosen,
    EMAlgConfig,
    EMVars,
    ErrorConfig,
    ParsedData,
    PastChoicesData,
)


logger = logging.getLogger(__name__)


@runtime_checkable
class _PastChoicesParser(Protocol):
    """Fitted model interface needed to encode historical choices."""

    case_varnames: list[str]
    dem_varnames: list[str] | None

    def _transform_data(
        self,
        data: object,
        dems_data: object | None = None,
        require_choice: bool = False,
    ) -> ParsedData:
        """Transform raw choice data with the fitted empirical specification."""
        ...


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
        model_spec: Any,
        em_vars: EMVars,
        estimation_data: Data,
        em_recursion: int,
        em_alg_config: EMAlgConfig,
        error_config: ErrorConfig | None,
        estim_time_sec: float,
    ) -> None:
        """Build a latent-class results object and compute inference artifacts.

        Parameters
        ----------
        model_spec : Any
            Fitted model specification. Kept broad to avoid a runtime circular import
            with :class:`~lcl.latent_class_conditional_logit.LatentClassConditionalLogit`.
        em_vars : :class:`~lcl._struct.EMVars`
            Final EM state containing parameters, probabilities, and log likelihood.
        estimation_data : :class:`~lcl._struct.Data`
            Encoded estimation data.
        em_recursion : int
            Number of EM recursions completed before termination.
        em_alg_config : :class:`~lcl._struct.EMAlgConfig`
            EM convergence and iteration configuration.
        error_config : :class:`~lcl._struct.ErrorConfig` | None
            Covariance and standard-error configuration.
        estim_time_sec : float
            Wall-clock estimation time in seconds.
        """
        self.model = model_spec
        self.em_res = em_vars
        self.data = estimation_data
        self.total_recursions = em_recursion
        self.converged = em_recursion < (em_alg_config.maxiter - 1)
        self.estim_time_sec = estim_time_sec
        self.error_config = error_config if error_config is not None else ErrorConfig()
        if self.em_res.latent_betas is None:
            raise ValueError("Latent betas are required to construct LCL results.")
        if self.em_res.structural_betas is None:
            raise ValueError("Structural betas are required to construct LCL results.")
        if self.em_res.shares is None:
            raise ValueError("Class shares are required to construct LCL results.")
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required for LCL results.")

        self.flat_params = self._pack_params()

        # Calculate degrees of freedom
        latent_betas = self.em_res.latent_betas
        num_beta_params = latent_betas.size
        if self.em_res.thetas is not None:
            num_theta_params = self.em_res.thetas.size
        else:
            num_theta_params = self.model.num_classes - 1

        self.num_params = num_beta_params + num_theta_params
        self.cov_matrix = self._compute_covariance()

        # Compute information criteria
        num_panels = self.data.num_panels
        self.caic = (
            jnp.log(num_panels) + 1
        ) * self.num_params - 2 * self.em_res.unconditional_loglik
        self.bic = (
            jnp.log(num_panels) * self.num_params - 2 * self.em_res.unconditional_loglik
        )
        n_star = (self.num_params + 2) / 24
        self.adjusted_bic = (
            jnp.log(num_panels) * n_star - 2 * self.em_res.unconditional_loglik
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
        """Return a compact, human-readable summary of fit quality."""
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
        latent_betas = self.em_res.latent_betas
        if latent_betas is None:
            raise ValueError("Latent betas are required to pack parameters.")
        latent_betas_flat = latent_betas.ravel()
        if self.em_res.thetas is not None:
            theta_flat = self.em_res.thetas.ravel()
        else:
            if self.em_res.shares is None:
                raise ValueError("Class shares are required to pack parameters.")
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
        """Compute Hessian-based covariance, optionally with clustered sandwich correction.

        Returns
        -------
        Float64[Array, "all_params all_params"]
            Covariance matrix aligned with the flattened parameter vector.
        """
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
        self,
        flat_params: Float64[Array, "all_params"],
        diff_unchosen_chosen: DiffUnchosenChosen,
        data: Data,
    ) -> Float64[Array, "panels"]:
        """Compute the log-likelihood for each panel (used to build the Jacobian)."""
        latent_betas, thetas = self._unpack_params(flat_params)
        structural_betas = _to_structural_betas(
            latent_betas, getattr(self.model, "numeraire_idx", None)
        )
        if data.num_panels is None:
            raise ValueError("Panel identifiers are required for LCL log-likelihoods.")
        class_probs = self._get_class_probs(thetas, data.dems, data.num_panels)
        return _compute_panel_logliks(
            structural_betas, class_probs, diff_unchosen_chosen, data
        )

    def _full_loglik_fn(
        self,
        flat_params: Float64[Array, "all_params"],
        diff_unchosen_chosen: DiffUnchosenChosen,
        data: Data,
    ) -> Float64[Array, ""]:
        """Re-sums the panel log-likelihoods to a scalar for the Hessian."""
        return jnp.sum(self._panel_loglik_fn(flat_params, diff_unchosen_chosen, data))

    def _apply_delta_method(
        self,
        func: Callable[..., Float64[Array, "..."]],
        flat_params: Float64[Array, "all_params"],
        *args: object,
        **kwargs: object,
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
        past_choices: object | None = None,
        data: object | None = None,
        dems_data: object | None = None,
        past_choices_dems_data: object | None = None,
    ) -> LCLPrediction:
        """Generate out-of-sample latent-class predictions.

        Prediction can be requested either with raw tabular data, which is encoded
        using the fitted model specification, or with already-constructed arrays.
        When historical choices are supplied through ``past_choices``, class
        membership probabilities are updated with Bayes' rule before computing
        counterfactual choice probabilities, consumer surplus, and willingness to pay.

        Parameters
        ----------
        X : ArrayLike | None, optional
            Alternative-specific design matrix for array-style prediction. Ignored
            when ``data`` is provided.
        alts : ArrayLike | None, optional
            Alternative identifiers aligned to rows of ``X``.
        cases : ArrayLike | None, optional
            Choice-situation identifiers aligned to rows of ``X``.
        panels : ArrayLike | None, optional
            Decision-maker identifiers aligned to rows of ``X``.
        dems : ArrayLike | None, optional
            Panel-level demographics for array-style prediction.
        past_choices : PastChoicesData or tabular data, optional
            Historical choices used to condition latent-class membership probabilities.
            Pass a :class:`~lcl._struct.PastChoicesData` instance for array-style
            inputs, or a Polars/Pandas/DataFrame-like object containing the fitted
            model's alternative, case, panel, choice, alternative-specific, and
            demographic columns.
        data : object | None, optional
            Long-format prediction data. If provided, the fitted encoder parses this
            data using the original empirical specification.
        dems_data : object | None, optional
            Optional panel-level demographics to merge into ``data`` during prediction.
        past_choices_dems_data : object | None, optional
            Optional panel-level demographics to merge into tabular ``past_choices``.
            This argument is not used with :class:`~lcl._struct.PastChoicesData`.

        Returns
        -------
        :class:`~lcl._prediction.LCLPrediction`
            Prediction results, including choice probabilities, consumer surplus,
            panel-level WTP values, and the class probabilities used for prediction.

        Raises
        ------
        ValueError
            If required prediction identifiers are missing, if fitted latent-class
            parameters are unavailable, or if ``past_choices_dems_data`` is provided
            without tabular ``past_choices``.
        """
        if past_choices is None and past_choices_dems_data is not None:
            raise ValueError(
                "past_choices_dems_data can only be used when past_choices is provided."
            )
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
        predict_data = cast(Data, self.model._setup_data(parsed_predict)[0])
        if predict_data.num_panels is None or predict_data.panels is None:
            raise ValueError(
                "Panel identifiers are required for latent-class prediction."
            )
        structural_betas = self.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required for prediction.")
        shares = self.em_res.shares
        if shares is None:
            raise ValueError("Class shares are required for prediction.")

        if past_choices is not None:
            parsed_past = _parse_past_choices(
                model=self.model,
                past_choices=past_choices,
                past_choices_dems_data=past_choices_dems_data,
            )
            data_past = cast(Data, self.model._setup_data(parsed_past)[0])
            diff_unchosen_chosen_past = _diff_unchosen_chosen(data_past)
            class_probs_by_panel, _ = _compute_conditional_class_probs(
                structural_betas=structural_betas,
                thetas=self.em_res.thetas,
                shares=shares,
                diff_unchosen_chosen=diff_unchosen_chosen_past,
                data=data_past,
            )
        elif self.em_res.thetas is not None and predict_data.dems is not None:
            class_probs_by_panel = self._get_class_probs(
                self.em_res.thetas, predict_data.dems, predict_data.num_panels
            )
        else:
            class_probs_by_panel = jnp.repeat(
                shares[None, :], predict_data.num_panels, axis=0
            )

        choice_probs_by_class, log_sum_exp_utility = _choice_probabilities_and_logsum(
            predict_data.X,
            structural_betas,
            predict_data.cases,
            predict_data.num_cases,
        )

        # Ensure alpha (marginal utility of income) is correctly signed
        numeraire_idx = getattr(self.model, "numeraire_idx", None)
        if numeraire_idx is None:
            marginal_utility_income = jnp.ones(self.model.num_classes)
        else:
            marginal_utility_income = -structural_betas[numeraire_idx, :]

        surplus_by_class = (
            log_sum_exp_utility / marginal_utility_income[None, :]
        ).squeeze()

        if numeraire_idx is not None:
            betas_sans_numeraire = jnp.delete(structural_betas, numeraire_idx, axis=0)
            wtp_alt_vars_by_class = betas_sans_numeraire / marginal_utility_income
            wtp_alt_vars_by_panel = class_probs_by_panel @ wtp_alt_vars_by_class.T
            schema = [
                var for var in self.model.case_varnames if var != self.model.numeraire
            ]
        else:
            wtp_alt_vars_by_panel = jnp.empty((predict_data.num_panels, 0))
            schema = []

        panel_first_rows = predict_data.panels != jnp.roll(predict_data.panels, shift=1)
        panel_first_rows = panel_first_rows.at[0].set(True)
        panels_unique = onp.array(parsed_predict.original_panels[panel_first_rows])
        wtp_alt_vars_by_panel_df = pl.DataFrame(
            onp.array(wtp_alt_vars_by_panel), schema=schema
        ).with_columns(pl.Series("panels", panels_unique))

        if (
            predict_data.num_cases_per_panel is None
            or predict_data.panels_of_cases is None
        ):
            raise ValueError(
                "Panel identifiers are required for latent-class prediction."
            )
        conditional_surplus = jnp.einsum(
            "np,np->n",
            class_probs_by_panel[predict_data.panels_of_cases],
            surplus_by_class,
        )

        unconditional_choice_probs = jnp.sum(
            class_probs_by_panel[predict_data.panels] * choice_probs_by_class, axis=1
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

        first_case_rows = predict_data.cases != jnp.roll(predict_data.cases, shift=1)
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
            predict_data=predict_data,
            results=self,
            class_probs_by_panel=class_probs_by_panel,
        )


def _parse_past_choices(
    model: _PastChoicesParser,
    past_choices: object,
    past_choices_dems_data: object | None,
) -> ParsedData:
    """Parse historical choices for prediction-time class updating.

    Parameters
    ----------
    model : _PastChoicesParser
        Fitted model specification that owns the trained encoder and variable
        metadata.
    past_choices : PastChoicesData or object
        Historical choices. ``PastChoicesData`` supports array-style callers; any
        other object is treated as tabular data and transformed with the fitted
        encoder.
    past_choices_dems_data : object | None
        Optional panel-level demographics to join to tabular ``past_choices``.

    Returns
    -------
    :class:`~lcl._struct.ParsedData`
        Encoded historical choices with validated choice indicators.

    Raises
    ------
    ValueError
        If ``past_choices_dems_data`` is supplied with ``PastChoicesData``.
    """
    if isinstance(past_choices, PastChoicesData):
        if past_choices_dems_data is not None:
            raise ValueError(
                "past_choices_dems_data is only supported when past_choices is "
                "provided as tabular data."
            )
        return _parsed_prediction_arrays(
            X=past_choices.X,
            dems=past_choices.dems,
            alts=past_choices.alts,
            cases=past_choices.cases,
            panels=past_choices.panels,
            y=past_choices.y,
            case_varnames=model.case_varnames,
            dem_varnames=model.dem_varnames,
        )

    return model._transform_data(
        past_choices,
        dems_data=past_choices_dems_data,
        require_choice=True,
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
    """Parse array-style prediction inputs into the shared encoded-data container.

    Parameters
    ----------
    X : ArrayLike
        Alternative-specific design matrix in long format.
    dems : ArrayLike | None
        Optional panel-level demographic matrix, one row per unique panel.
    alts : ArrayLike
        Alternative identifiers aligned to rows of ``X``.
    cases : ArrayLike
        Choice-situation identifiers aligned to rows of ``X``.
    panels : ArrayLike
        Panel identifiers aligned to rows of ``X``.
    case_varnames : list[str]
        Names corresponding to columns of ``X``.
    dem_varnames : list[str] | None
        Names corresponding to columns of ``dems``.
    y : ArrayLike | None, optional
        Optional choice indicators for historical-choice updating.

    Returns
    -------
    :class:`~lcl._struct.ParsedData`
        Sorted arrays with contiguous zero-indexed IDs and original labels preserved.
    """
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
    # Cases are only unique within panel for some user datasets, so key on both IDs.
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
