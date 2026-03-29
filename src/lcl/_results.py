"""In-sample estimation results and inference."""

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array, hessian, jacrev, lax
from jax.nn import softmax
from jax.ops import segment_sum
from jax.tree_util import Partial
from jax.typing import ArrayLike
from jaxtyping import Float64
from pylatexenc.latex2text import LatexNodes2Text
from tabulate import tabulate

from lcl._case_utils import _diff_unchosen_chosen, _to_structural_betas
from lcl._em_alg_steps import (
    _compute_conditional_class_probs,
    _compute_panel_logliks,
    _compute_probs_and_exp_utility,
)
from lcl._prediction import LCLPrediction
from lcl._struct import Data, EMVars, PastChoicesData


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
        estim_time_sec: float,
    ) -> None:
        self.model = model_spec
        self.em_res = em_vars
        self.data = estimation_data
        self.total_recursions = em_recursion
        self.converged = em_recursion < (em_alg_config.maxiter - 1)
        self.estim_time_sec = estim_time_sec

        print(
            "\n".join(
                [
                    "\nComputing clustered covariance matrix per steps in Stata's reference manual...",
                    "(StataCorp. 2025. Stata 19 Base Reference Manual. College Station, TX: Stata Press.)\n",
                ]
            )
        )

        self.flat_params = self._pack_params()
        diff_unchosen_chosen = _diff_unchosen_chosen(self.data)

        # Obtain inverse Hessian
        H = hessian(self._full_loglik_fn)(
            self.flat_params, diff_unchosen_chosen, self.data
        )
        H_inv = jnp.linalg.pinv(-H)

        # Take outer product of panel-level gradients
        J = jacrev(self._panel_loglik_fn)(
            self.flat_params, diff_unchosen_chosen, self.data
        )
        B = J.T @ J

        # The Sandwich!
        robust_cov = H_inv @ B @ H_inv

        # Finite-sample cluster correction (G / (G - 1))
        # See Stata manual
        assert self.data.num_panels is not None
        G = self.data.num_panels
        self.cov_matrix = robust_cov * (G / (G - 1))

        # Calculate degrees of freedom
        num_beta_params = self.em_res.latent_betas.size
        if self.em_res.thetas is not None:
            num_theta_params = self.em_res.thetas.size
        else:
            num_theta_params = (
                self.model.num_classes - 1
            )  # Only C-1 shares are identified

        self.num_params = num_beta_params + num_theta_params

        assert self.data.num_panels is not None

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
        print(
            "\n".join(
                [
                    "\nInformation criteria:",
                    f"  Consistent Aikake information criterion (CAIC; see Bozdogan [1987]): {self.caic:.1f}",
                    f"  Bayesian information criterion (BIC; see Schwartz [1978]): {self.bic:.1f}",
                    f"  Adjusted BIC (see Sclove [1987]): {self.adjusted_bic:.1f}",
                ]
            )
        )

        if not self.converged:
            print(
                f"**** Optimization did not converge after {self.total_recursions} iterations. ****"
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
            theta_flat = jnp.log(shares[:-1] / shares[-1]).ravel()
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
        if dems is not None:
            V = thetas[None, 0] + dems @ thetas[1:]
        else:
            V = jnp.repeat(thetas, num_panels, axis=0)

        # Prepend a column of zeros for the reference class
        V_ref = jnp.zeros((num_panels, 1))
        V_full = jnp.concatenate([V_ref, V], axis=1)
        # Softmax uses log-sum-exp trick to avoid numerical overflow
        return softmax(V_full, axis=1)

    def _panel_loglik_fn(
        self, flat_params, diff_unchosen_chosen, data
    ) -> Float64[Array, "panels"]:
        """Compute the log-likelihood for each panel (used to build the Jacobian)."""
        latent_betas, thetas = self._unpack_params(flat_params)
        structural_betas = _to_structural_betas(latent_betas, self.model.numeraire_idx)
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

        return val, jnp.sqrt(variance)

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

        structural_betas = _to_structural_betas(latent_betas, self.model.numeraire_idx)
        return structural_betas @ avg_shares

    def _calc_population_std_betas(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "alt_vars"]:
        """Compute the population variance of the structural taste parameters."""
        latent_betas, thetas = self._unpack_params(flat_params)
        if self.model.numeraire:
            numeraire_idx = self.model.case_varnames.index(self.model.numeraire)
        else:
            numeraire_idx = None

        class_probs = self._get_class_probs(thetas, dems, num_panels)
        avg_shares = jnp.mean(class_probs, axis=0)

        # Get structural betas
        if self.model.numeraire:
            numeraire_idx = self.model.case_varnames.index(self.model.numeraire)
        else:
            numeraire_idx = None
        structural_betas = _to_structural_betas(latent_betas, numeraire_idx)

        mean_betas = structural_betas @ avg_shares
        diff_sq = (structural_betas - mean_betas[:, None]) ** 2
        var_betas = diff_sq @ avg_shares

        return jnp.sqrt(jnp.clip(var_betas, a_min=1e-250))

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
        assert self.data.num_panels is not None

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
                # (var_clean, float(means[coeff_idx]), float(stds[coeff_idx]))
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
        alts: ArrayLike,
        cases: ArrayLike,
        panels: ArrayLike,
        dems: ArrayLike | None = None,
        past_choices: PastChoicesData | None = None,
    ) -> LCLPrediction:
        """Generate out-of-sample predictions and counterfactual inclusive values (consumer surplus).

        If a decision-maker's historical choice sequence is provided via `past_choices`,
        the model utilizes Bayesian updating to rigorously re-weight their conditional
        class membership probabilities before generating predictions.

        Parameters
        ----------
        X : ArrayLike
            ``(N, K)`` counterfactual design matrix.
        alts : ArrayLike
            ``(N,)`` vector of alternative IDs.
        cases : ArrayLike
            ``(N,)`` vector of choice situation IDs.
        panels : ArrayLike
            ``(N,)`` vector mapping observations to specific decision-makers.
        dems : ArrayLike | None, optional
            ``(N, D)`` matrix of demographic variables.
        past_choices : :class:`~lcl._struct.PastChoicesData` | None, optional
            Container housing the decision-maker's observed choice history for Bayesian updating.

        Returns
        -------
        :class:`~lcl._prediction.LCLPrediction`
            Container housing probabilities, surplus, and WTP evaluators.
        """
        data, *_ = self.model._setup_data(
            X=X,
            dems=dems,
            y=None,
            cases=cases,
            panels=panels,
            alts=alts,
            weights=None,
            init_beta=None,
        )

        if past_choices is not None:
            data_past, *_ = self.model._setup_data(
                X=past_choices.X,
                dems=past_choices.dems,
                y=past_choices.y,
                cases=past_choices.cases,
                panels=past_choices.panels,
                alts=past_choices.alts,
            )
            diff_unchosen_chosen_past = _diff_unchosen_chosen(data_past)
            class_probs_by_panel, _ = _compute_conditional_class_probs(
                structural_betas=self.em_res.structural_betas,
                thetas=self.em_res.thetas,
                shares=self.em_res.shares,
                diff_unchosen_chosen=diff_unchosen_chosen_past,
                data=data_past,
            )
        else:
            assert data.num_panels is not None
            class_probs_by_panel = jnp.repeat(
                self.em_res.shares[None, :], data.num_panels, axis=0
            )

        choice_probs_by_class, exp_utility_by_class = lax.map(
            lambda _beta: _compute_probs_and_exp_utility(_beta, data=data),
            self.em_res.structural_betas.T,
        )

        conditional_choice_probs = (
            class_probs_by_panel[panels] * choice_probs_by_class.T
        )
        sum_exp_utility = segment_sum(
            exp_utility_by_class.T, cases, num_segments=data.num_cases
        )
        log_sum_exp_utility = jnp.log(jnp.clip(sum_exp_utility, a_min=1e-250))

        if self.model.numeraire is None:
            numeraire_coeff_by_class = jnp.ones(self.model.num_classes)
        else:
            numeraire_coeff_by_class = self.em_res.structural_betas[
                self.model.numeraire_idx, :
            ]

        surplus_by_class = (
            log_sum_exp_utility / numeraire_coeff_by_class[None, :]
        ).squeeze()

        # WTP calculation setup
        if self.model.numeraire_idx is not None:
            betas_sans_numeraire = jnp.delete(
                self.em_res.structural_betas, self.model.numeraire_idx, axis=0
            )
            wtp_alt_vars_by_class = betas_sans_numeraire / numeraire_coeff_by_class
            wtp_alt_vars_by_panel = class_probs_by_panel @ wtp_alt_vars_by_class.T
            schema = [
                var for var in self.model.case_varnames if var != self.model.numeraire
            ]
        else:
            wtp_alt_vars_by_panel = jnp.empty((data.num_panels, 0))
            schema = []

        panels_unique = onp.array(panels[panels != jnp.roll(panels, shift=1)])
        wtp_alt_vars_by_panel_df = pl.DataFrame(
            onp.array(wtp_alt_vars_by_panel), schema=schema
        ).with_columns(pl.Series("panels", panels_unique, dtype=pl.UInt32))

        assert data.num_cases_per_panel is not None
        conditional_surplus = jnp.einsum(
            "np,np->n",
            jnp.repeat(class_probs_by_panel, data.num_cases_per_panel, axis=0),
            surplus_by_class,
        )

        predicted_probs_df = pl.DataFrame(
            {
                "panels": onp.array(panels),
                "cases": onp.array(cases),
                "alts": onp.array(alts),
                "choice_probs": onp.array(conditional_choice_probs, dtype=onp.float64),
            }
        )

        assert data.panels is not None and isinstance(cases, Array)
        surplus_df = pl.DataFrame(
            {
                "panels": onp.array(
                    data.panels[data.cases != jnp.roll(data.cases, shift=1)]
                ),
                "cases": onp.array(cases[data.cases != jnp.roll(data.cases, shift=1)]),
                "surplus": onp.array(conditional_surplus, dtype=onp.float64),
            }
        )

        return LCLPrediction(
            predicted_probs_df, surplus_df, wtp_alt_vars_by_panel_df, data, self
        )


# EOF
