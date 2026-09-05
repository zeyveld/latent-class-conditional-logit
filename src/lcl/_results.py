"""In-sample estimation results and inference."""

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import jacfwd
from jax.typing import ArrayLike
from jaxtyping import Array, Float64

from lcl._analytic_derivatives import _panel_scores_and_hessian
from lcl._case_utils import _diff_unchosen_chosen
from lcl._delta import apply_delta_method, parametric_bootstrap_se
from lcl._diagnostics import LCLDiagnostics
from lcl._em_alg_steps import (
    _compute_conditional_class_probs,
    _compute_panel_logliks,
)
from lcl._encoding import _coerce_frame
from lcl._jax_compat import cpu_device, device_put_array_leaves
from lcl._inference import _aggregate_scores, _invert_information, _symmetrize
from lcl._kernels import _choice_probabilities_and_logsum
from lcl._logging import log_or_print
from lcl._params import ParamPacking
from lcl._polish import PolishReport
from lcl._prediction import LCLPrediction, resolve_panel_weights
from lcl._predict_inputs import (
    _parse_past_choices,
    _parsed_prediction_arrays,
    _prediction_partition_data,
    _validate_past_choice_panels,
)
from lcl._labels import numeraire_enters_linearly
from lcl._reporting import _history_frame, _model_variable_label
from lcl._presentation import (
    format_class_coefficients,
    format_lcl_beta_summary,
    format_membership_coefficients,
    format_membership_marginal_effects,
)
from lcl.options import DiagnosticsOptions, InferenceOptions
from lcl._struct import Data, DiffUnchosenChosen, EMVars

logger = logging.getLogger(__name__)

DEGENERATE_VARIANCE_RTOL = 1e-12
"""Relative floor below which between-class spread counts as unidentified."""


class LCLResults:
    """Post-estimation results and inference container.

    Computes robust sandwich covariance matrices (clustered at the decision-maker level)
    and handles the extraction of population-level moments via the Delta Method.

    Attributes
    ----------
    cov_matrix : Float64[Array, "all_params all_params"]
        Covariance of the *reported* (structural) parameters, aligned row for row
        with :meth:`parameter_names`.  Constrained coefficients appear on the
        scale printed by :meth:`class_coefficients`, so ``sqrt(diag(cov_matrix))``
        matches the published standard errors.  Clustered covariance uses the
        Stata maximum-likelihood multiplier :math:`(G / (G - 1))`.
    latent_cov_matrix : Float64[Array, "all_params all_params"]
        Covariance in the unconstrained parameterization the optimizer works in.
        This is the matrix the delta method consumes: target functions apply the
        softplus transform internally, so its Jacobian is differentiated rather
        than applied twice.
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
        converged: bool,
        inference: InferenceOptions | None,
        estim_time_sec: float,
        diagnostics_config: DiagnosticsOptions | None = None,
        em_history: list[dict[str, Any]] | None = None,
        optimization_history: list[dict[str, Any]] | None = None,
        observed_score_max: float = float("nan"),
        score_tol: float = 1e-4,
        em_criterion_met: bool = False,
        polish_report: PolishReport | None = None,
        cluster_ids: ArrayLike | None = None,
        num_clusters: int | None = None,
        param_packing: ParamPacking | None = None,
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
        converged : bool
            Whether the explicit EM stopping criterion was satisfied.
        inference : :class:`~lcl.options.InferenceOptions` | None
            Covariance and standard-error configuration.
        estim_time_sec : float
            Wall-clock estimation time in seconds.
        diagnostics_config : :class:`~lcl.options.DiagnosticsOptions` | None
            Thresholds and switches for public diagnostics.
        em_history : list[dict[str, Any]] | None
            EM log-likelihood and class-share history.
        optimization_history : list[dict[str, Any]] | None
            Final class-level M-step diagnostics.
        observed_score_max : float, optional
            Largest absolute component of the observed-data score at the reported
            estimate.  Recomputed here when covariance estimation runs.
        score_tol : float, default=1e-4
            Stationarity tolerance used for the ``converged`` flag, reused by
            :meth:`diagnostics` so the two can never disagree.
        em_criterion_met : bool, default=False
            Whether the Aitken EM stopping criterion was satisfied before the
            iteration cap.
        polish_report : :class:`~lcl._polish.PolishReport` | None, optional
            Outcome of the observed-data Newton polish.
        cluster_ids : ArrayLike | None, optional
            Zero-indexed cluster identifier per panel, for clustering coarser than
            the decision-maker.
        num_clusters : int | None, optional
            Number of distinct clusters implied by ``cluster_ids``.
        param_packing : :class:`~lcl._params.ParamPacking` | None, optional
            Reuse of the packing built during estimation.
        """
        self.model = model_spec
        self.em_res = em_vars
        self.data = estimation_data
        self.total_recursions = em_recursion
        self.converged = converged
        self.estim_time_sec = estim_time_sec
        self.inference = inference if inference is not None else InferenceOptions()
        self.diagnostics_config = (
            diagnostics_config
            if diagnostics_config is not None
            else DiagnosticsOptions()
        )
        self.em_history_ = _history_frame(em_history)
        self.optimization_history_ = _history_frame(optimization_history)
        if self.em_res.latent_betas is None:
            raise ValueError("Latent betas are required to construct LCL results.")
        if self.em_res.structural_betas is None:
            raise ValueError("Structural betas are required to construct LCL results.")
        if self.em_res.shares is None:
            raise ValueError("Class shares are required to construct LCL results.")
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required for LCL results.")

        self._param_packing = param_packing or ParamPacking(
            num_alt_vars=self.model.num_vars,
            num_classes=self.model.num_classes,
            num_dem_vars=self.model.num_dem_vars,
            numeraire_idx=self.model.numeraire_idx,
            numeraire_min_abs=self.model.numeraire_min_abs,
        )
        self.flat_params = self._pack_params()
        self.num_params = self._param_packing.num_params
        self.score_tol = float(score_tol)
        self.em_criterion_met = bool(em_criterion_met)
        self.polish_report = polish_report
        self._cluster_ids = (
            None if cluster_ids is None else jnp.asarray(cluster_ids, dtype=jnp.int32)
        )
        self._num_clusters = num_clusters
        # Populated by _compute_covariance; stays None when inference is skipped.
        self.information_diagnostics: Any = None
        self.observed_score_max = float(observed_score_max)
        self.latent_cov_matrix = self._compute_covariance()
        self.cov_matrix = self._structural_covariance(self.latent_cov_matrix)

        # Compute information criteria
        num_panels = self.data.num_panels
        self.aic = 2 * self.num_params - 2 * self.em_res.unconditional_loglik
        self.aic3 = 3 * self.num_params - 2 * self.em_res.unconditional_loglik
        self.caic = (
            jnp.log(num_panels) + 1
        ) * self.num_params - 2 * self.em_res.unconditional_loglik
        self.bic = (
            jnp.log(num_panels) * self.num_params - 2 * self.em_res.unconditional_loglik
        )
        self.adjusted_bic = (
            jnp.log((num_panels + 2) / 24) * self.num_params
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
        """Return a compact, human-readable summary of fit quality."""
        status = "Converged" if self.converged else "Did Not Converge"
        parts = [
            f"<LCLResults: {self.model.num_classes} Classes",
            f"{status}",
            f"Log likelihood: {self.em_res.unconditional_loglik:.1f}",
            f"CAIC: {self.caic:.1f}",
            f"BIC: {self.bic:.1f}",
            f"Adj. BIC: {self.adjusted_bic:.1f}",
        ]
        if not self.inference.skip and not self.covariance_available:
            parts.append("Covariance unavailable")
        return " | ".join(parts) + ">"

    def parameter_names(self) -> list[str]:
        """Return names aligned with rows and columns of ``cov_matrix``."""
        names = [
            f"class_{class_idx}:{variable}"
            for variable in self.model.case_varnames
            for class_idx in range(self.model.num_classes)
        ]
        membership_rows = ["Intercept", *(self.model.dem_varnames or [])]
        names.extend(
            f"membership_class_{class_idx}:{variable}"
            for variable in membership_rows
            for class_idx in range(1, self.model.num_classes)
        )
        if len(names) != self.num_params:
            raise RuntimeError(
                "Parameter-name layout does not match covariance packing."
            )
        return names

    @property
    def convergence(self) -> bool:
        """Deprecated alias for :attr:`converged`."""
        warnings.warn(
            "LCLResults.convergence is deprecated; use converged.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.converged

    @property
    def covariance(self) -> Array:
        """Deprecated alias for :attr:`cov_matrix`."""
        warnings.warn(
            "LCLResults.covariance is deprecated; use cov_matrix.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cov_matrix

    @property
    def abic(self) -> Array:
        """Deprecated alias for :attr:`adjusted_bic`."""
        warnings.warn(
            "LCLResults.abic is deprecated; use adjusted_bic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.adjusted_bic

    def _pack_params(self) -> Float64[Array, "all_params"]:
        """Flatten structural parameters and class memberships for Hessian calculation."""
        latent_betas = self.em_res.latent_betas
        if latent_betas is None:
            raise ValueError("Latent betas are required to pack parameters.")
        return self._param_packing.pack(
            latent_betas,
            self.em_res.thetas,
            self.em_res.shares,
        )

    def _unpack_params(
        self, flat_params: Float64[Array, "all_params"]
    ) -> tuple[
        Float64[Array, "alt_vars classes"],
        Float64[Array, "dem_vars_plus_one classes_minus_one"],
    ]:
        """Reconstruct parameter matrices from the flattened array."""
        return self._param_packing.unpack(flat_params)

    def _get_class_probs(
        self,
        thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "panels classes"]:
        """Extract unconditional class probabilities (via fractional response)."""
        return self._param_packing.class_probs(thetas, dems, num_panels)

    def _compute_covariance(self) -> Float64[Array, "all_params all_params"]:
        """Compute covariance on CPU, optionally with clustered sandwich correction.

        The EM algorithm may leave fitted arrays committed to a GPU or sharded
        accelerator placement, so this method explicitly moves the inference
        inputs to CPU and runs the differencing, Hessian, pseudo-inverse, and
        score work there.  Panel scores and the observed-information Hessian
        come from the closed-form mixture derivatives in
        :mod:`lcl._analytic_derivatives` (Fisher identity and Louis/Oakes
        observed information), which match ``jax.hessian``/``jax.jacfwd`` of the
        panel log likelihood to machine precision while touching the data once
        instead of once per parameter.  The analytic path is much leaner than
        autodiff but still holds all-class case statistics and the panel-by-
        parameter score matrix at once, so its peak memory remains above the
        class-local EM and M-step kernels; CPU placement is therefore intentional.

        Returns
        -------
        Float64[Array, "all_params all_params"]
            Covariance in the latent (unconstrained) parameterization.
        """
        cpu = cpu_device()
        if self.inference.skip:
            with jax.default_device(cpu):
                return jnp.full((self.num_params, self.num_params), jnp.nan)
        if self.inference.covariance == "robust":
            raise ValueError(
                "Case-level robust covariance is not valid for an LCL likelihood "
                "whose latent class is shared within panel. Use covariance='clustered' "
                "for panel-clustered inference or 'unadjusted'."
            )

        logger.info("Computing LCL covariance matrix.")
        with jax.default_device(cpu):
            flat_params = device_put_array_leaves(self.flat_params, cpu)
            data = device_put_array_leaves(self.data, cpu)
            diff_unchosen_chosen = device_put_array_leaves(
                _diff_unchosen_chosen(data), cpu
            )

            J, H = _panel_scores_and_hessian(
                flat_params, diff_unchosen_chosen, data, self._param_packing
            )
            self.observed_score_max = float(jnp.max(jnp.abs(jnp.sum(J, axis=0))))
            H_inv, diagnostics = _invert_information(
                -H, label="latent-class observed information matrix"
            )
            self.information_diagnostics = diagnostics
            H_inv = jax.device_put(H_inv, cpu)

            if self.inference.covariance == "unadjusted":
                return _symmetrize(H_inv)

            if data.num_panels is None:
                raise ValueError(
                    "Panel identifiers are required for clustered covariance."
                )
            # Each panel contributes exactly one score row to the mixture
            # likelihood, so panel clustering aggregates nothing further.  A
            # coarser grouping sums those rows first and counts its own groups.
            if self._cluster_ids is None:
                cluster_scores = J
                G = data.num_panels
            else:
                if self._num_clusters is None:
                    raise ValueError("num_clusters is required with cluster_ids.")
                cluster_scores = _aggregate_scores(
                    J, self._cluster_ids, self._num_clusters
                )
                G = self._num_clusters
            if G < 2:
                raise ValueError(
                    "Cluster-robust covariance requires at least two clusters."
                )
            B = cluster_scores.T @ cluster_scores
            correction = G / (G - 1) if self.inference.finite_sample_correction else 1.0
            return _symmetrize((H_inv @ B @ H_inv) * correction)

    def _structural_from_latent(
        self, flat_params: Float64[Array, "all_params"]
    ) -> Float64[Array, "all_params"]:
        """Map latent parameters to the scale the results tables report."""
        latent_betas, thetas = self._param_packing.unpack(flat_params)
        structural_betas = self._param_packing.to_structural(latent_betas)
        return jnp.concatenate([structural_betas.ravel(), thetas.ravel()])

    def _structural_covariance(
        self, latent_cov: Float64[Array, "all_params all_params"]
    ) -> Float64[Array, "all_params all_params"]:
        """Push the latent covariance through to the reported parameter scale.

        Without this the public covariance would sit in a different
        parameterization from the coefficients whose names label its rows, and
        from :attr:`~lcl.conditional_logit.CLResults.cov_matrix`, which the shared
        results protocol advertises as the same object.
        """
        if self._param_packing.numeraire_idx is None:
            return latent_cov
        cpu = cpu_device()
        with jax.default_device(cpu):
            flat_params = device_put_array_leaves(self.flat_params, cpu)
            jacobian = jacfwd(self._structural_from_latent)(flat_params)
            return _symmetrize(jacobian @ device_put_array_leaves(latent_cov, cpu) @ jacobian.T)

    @property
    def covariance_available(self) -> bool:
        """Report whether a usable covariance matrix was estimated."""
        return bool(onp.all(onp.isfinite(onp.asarray(self.cov_matrix))))

    def _panel_loglik_fn(
        self,
        flat_params: Float64[Array, "all_params"],
        diff_unchosen_chosen: DiffUnchosenChosen,
        data: Data,
    ) -> Float64[Array, "panels"]:
        """Compute the log-likelihood for each panel (used to build the Jacobian)."""
        latent_betas, thetas = self._unpack_params(flat_params)
        structural_betas = self._param_packing.to_structural(latent_betas)
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

    def loglik(
        self,
        data: object,
        dems_data: object | None = None,
        *,
        per_panel: bool = False,
    ) -> float | pl.DataFrame:
        """Score observed choices with the fitted empirical specification.

        The fitted encoder is reused, so Formulaic categorical levels and expanded
        columns retain their training-time meaning.

        Parameters
        ----------
        data : object
            Long-format data containing one observed choice per case.
        dems_data : object | None, optional
            Optional panel-level demographics joined by the fitted panel ID column.
        per_panel : bool, default=False
            Return a panel-level table instead of the total log likelihood.

        Returns
        -------
        float or pl.DataFrame
            Total log likelihood when ``per_panel=False``. Otherwise, a table with
            the original panel IDs and their log-likelihood contributions.
        """
        parsed = self.model._transform_data(
            data,
            dems_data=dems_data,
            require_choice=True,
        )
        data_struct = cast(Data, self.model._setup_data(parsed)[0])
        if data_struct.num_panels is None or data_struct.panels is None:
            raise ValueError("Panel identifiers are required to score LCL data.")
        diff = _diff_unchosen_chosen(data_struct)
        panel_values = self._panel_loglik_fn(self.flat_params, diff, data_struct)

        if not per_panel:
            return float(jnp.sum(panel_values))

        first_panel_rows = data_struct.panels != jnp.roll(data_struct.panels, shift=1)
        first_panel_rows = first_panel_rows.at[0].set(True)
        return pl.DataFrame(
            {
                "panel": onp.asarray(parsed.original_panels[first_panel_rows]),
                "log_likelihood": onp.asarray(panel_values, dtype=onp.float64),
            }
        )

    def _apply_delta_method(
        self,
        func: Callable[..., Float64[Array, "..."]],
        flat_params: Float64[Array, "all_params"],
        **kwargs: Any,
    ) -> tuple[Float64[Array, "..."], Float64[Array, "..."]]:
        """Apply the delta method to a function of the latent parameters.

        Extras are keyword-only: the parameter vector is the sole positional
        argument, so nothing can displace it at a call site.

        Parameters
        ----------
        func : Callable
            Target taking the flat latent parameter vector.
        flat_params : Float64[Array, "all_params"]
            Latent parameters.
        **kwargs
            Extra keyword arguments bound into ``func``.

        Returns
        -------
        tuple[Array, Array]
            Value and delta-method standard errors.
        """
        return apply_delta_method(
            func, flat_params, self.latent_cov_matrix, **kwargs
        )

    def _parametric_bootstrap_se(
        self,
        func: Callable[..., Float64[Array, "..."]],
        flat_params: Float64[Array, "all_params"],
        *,
        draws: int = 500,
        seed: int = 0,
        **kwargs: Any,
    ) -> Float64[Array, "..."]:
        """Estimate nonlinear standard errors from asymptotic parameter draws."""
        return parametric_bootstrap_se(
            func,
            flat_params,
            self.latent_cov_matrix,
            draws=draws,
            seed=seed,
            **kwargs,
        )

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

        structural_betas = self._param_packing.to_structural(latent_betas)
        return structural_betas @ avg_shares

    def _calc_population_var_betas(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "alt_vars"]:
        """Compute the between-class variance of the structural taste parameters.

        The delta method is applied to the variance rather than to its square
        root because the square root is not differentiable where the variance
        vanishes -- which is exactly the case of interest, a variable whose
        coefficient does not differ across classes.  :meth:`beta_summary`
        converts to a standard deviation and applies the chain rule explicitly,
        so it can report ``NaN`` at the degenerate point instead of a spurious
        zero.
        """
        latent_betas, thetas = self._unpack_params(flat_params)

        class_probs = self._get_class_probs(thetas, dems, num_panels)
        avg_shares = jnp.mean(class_probs, axis=0)

        structural_betas = self._param_packing.to_structural(latent_betas)

        mean_betas = structural_betas @ avg_shares
        diff_sq = (structural_betas - mean_betas[:, None]) ** 2
        return diff_sq @ avg_shares

    def _calc_population_std_betas(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "alt_vars"]:
        """Return the between-class standard deviation of the taste parameters."""
        return jnp.sqrt(
            jnp.clip(
                self._calc_population_var_betas(flat_params, dems, num_panels),
                min=0.0,
            )
        )

    def _structural_betas_and_class_probs(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> tuple[
        Float64[Array, "alt_vars classes"], Float64[Array, "panels classes"]
    ]:
        """Return structural class betas and prior class probabilities.

        The one hook the differentiable counterfactual quantities in
        :mod:`lcl._prediction_inference` need from a fitted model.  A conditional
        logit implements it with a single class, so those quantities are written
        once and serve both estimators.
        """
        latent_betas, thetas = self._unpack_params(flat_params)
        return (
            self._param_packing.to_structural(latent_betas),
            self._get_class_probs(thetas, dems, num_panels),
        )

    def _calc_structural_betas(
        self, flat_params: Float64[Array, "all_params"]
    ) -> Float64[Array, "alt_vars classes"]:
        """Transform packed utility coefficients to their reported scale."""
        latent_betas, _ = self._unpack_params(flat_params)
        return self._param_packing.to_structural(latent_betas)

    def _calc_membership_coefficients(
        self, flat_params: Float64[Array, "all_params"]
    ) -> Float64[Array, "dem_vars_plus_one classes_minus_one"]:
        """Extract nonbaseline membership logits from packed parameters."""
        _, thetas = self._unpack_params(flat_params)
        return thetas

    def _calc_class_shares(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "classes"]:
        """Average prior membership probabilities across the sample."""
        _, thetas = self._unpack_params(flat_params)
        return jnp.mean(self._get_class_probs(thetas, dems, num_panels), axis=0)

    def class_coefficients(self) -> pl.DataFrame:
        """Return class-specific structural coefficients.

        Returns
        -------
        pl.DataFrame
            Long-format table with one row per variable and latent class.  The
            ``variable`` column preserves raw model names; ``label`` contains
            human-readable presentation labels.
        """
        structural_betas, standard_errors = self._apply_delta_method(
            self._calc_structural_betas, self.flat_params
        )
        rows = []
        beta_array = onp.asarray(structural_betas)
        se_array = onp.asarray(standard_errors)
        for var_idx, variable in enumerate(self.model.case_varnames):
            for class_idx in range(self.model.num_classes):
                rows.append(
                    {
                        "variable": variable,
                        "label": _model_variable_label(self.model, variable),
                        "class": class_idx,
                        "coefficient": float(beta_array[var_idx, class_idx]),
                        "std_error": float(se_array[var_idx, class_idx]),
                        "constrained": variable == self.model.numeraire,
                    }
                )
        return pl.DataFrame(rows)

    def membership_coefficients(self) -> pl.DataFrame:
        """Return nonbaseline class-membership coefficients with standard errors.

        Class 0 is the reference category and therefore has no separately
        estimated membership coefficients.
        """
        coefficients, standard_errors = self._apply_delta_method(
            self._calc_membership_coefficients, self.flat_params
        )
        coefficient_array = onp.asarray(coefficients)
        se_array = onp.asarray(standard_errors)
        variables = ["Intercept", *(self.model.dem_varnames or [])]
        rows = []
        for variable_idx, variable in enumerate(variables):
            for class_idx in range(1, self.model.num_classes):
                rows.append(
                    {
                        "variable": variable,
                        "label": _model_variable_label(self.model, variable),
                        "class": class_idx,
                        "reference_class": 0,
                        "coefficient": float(
                            coefficient_array[variable_idx, class_idx - 1]
                        ),
                        "std_error": float(se_array[variable_idx, class_idx - 1]),
                    }
                )
        return pl.DataFrame(rows)

    def _calc_membership_marginal_effects(
        self,
        flat_params: Float64[Array, "all_params"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "dem_vars classes"]:
        """Average over panels of ``d pi_ns / d z_nm`` for each demographic.

        For a multinomial logit membership model the panel-level derivative is
        ``pi_ns (theta_sm - sum_r pi_nr theta_rm)``.  Averaging it over the
        sample is the "average marginal effect", which is what a reader wants
        from a class-membership regression: it is in probability units, it does
        not depend on which class was chosen as the reference, and it sums to
        zero across classes because the probabilities sum to one.
        """
        _, thetas = self._unpack_params(flat_params)
        class_probs = self._get_class_probs(thetas, dems, num_panels)
        # Restore the reference class's implicit row of zeros.
        coefficients = jnp.concatenate(
            [jnp.zeros((thetas.shape[0], 1), dtype=thetas.dtype), thetas], axis=1
        )[1:, :]  # drop the intercept row: it is not a marginal effect
        weighted = class_probs @ coefficients.T  # (panels, dem_vars)
        # d pi_ns / d z_nm = pi_ns (theta_sm - sum_r pi_nr theta_rm)
        derivative = class_probs[:, None, :] * (
            coefficients[None, :, :] - weighted[:, :, None]
        )
        return jnp.mean(derivative, axis=0)

    def membership_marginal_effects(self) -> pl.DataFrame:
        """Return average marginal effects of demographics on class membership.

        The membership coefficients are log-odds against an arbitrary reference
        class, which makes them awkward to read and impossible to compare across
        fits that landed on a different reference.  The average marginal effect
        ``(1/N) sum_n d P(class = s | z_n) / d z_nm`` is free of that
        normalisation, is denominated in probability points, and sums to zero
        across classes for each demographic variable.

        Returns
        -------
        pl.DataFrame
            One row per (demographic variable, class) with the average marginal
            effect and its Delta-method standard error.

        Raises
        ------
        ValueError
            If the model was fitted without a demographic membership regression.

        Notes
        -----
        For a binary demographic this is the average of a derivative, not a
        discrete difference in probabilities; the two agree only approximately.
        """
        if self.em_res.thetas is None:
            raise ValueError(
                "This model has no class-membership regression, so demographics "
                "have no marginal effect on class membership."
            )
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required for marginal effects.")
        effects, standard_errors = self._apply_delta_method(
            self._calc_membership_marginal_effects,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )
        effect_array = onp.asarray(effects)
        se_array = onp.asarray(standard_errors)
        rows = []
        for variable_idx, variable in enumerate(self.model.dem_varnames or []):
            for class_idx in range(self.model.num_classes):
                rows.append(
                    {
                        "variable": variable,
                        "label": _model_variable_label(self.model, variable),
                        "class": class_idx,
                        "marginal_effect": float(
                            effect_array[variable_idx, class_idx]
                        ),
                        "std_error": float(se_array[variable_idx, class_idx]),
                    }
                )
        return pl.DataFrame(rows)

    def class_shares(self) -> pl.DataFrame:
        """Return aggregate latent-class shares.

        Returns
        -------
        pl.DataFrame
            One row per latent class with aggregate class share and effective
            panel mass.
        """
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required for class shares.")
        shares, share_se = self._apply_delta_method(
            self._calc_class_shares,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )
        shares_array = onp.asarray(shares)
        share_se_array = onp.asarray(share_se)
        rows = []
        posterior = self.em_res.class_probs_by_panel
        posterior_arr = onp.asarray(posterior) if posterior is not None else None
        for class_idx, share in enumerate(shares_array):
            row = {
                "class": class_idx,
                "share": float(share),
                "std_error": float(share_se_array[class_idx]),
            }
            if posterior_arr is not None:
                row["effective_panels"] = float(posterior_arr[:, class_idx].sum())
            rows.append(row)
        return pl.DataFrame(rows)

    def classification_diagnostics(self) -> pl.DataFrame:
        """Summarize posterior separation and modal classification by class."""
        posterior = self.em_res.class_probs_by_panel
        if posterior is None:
            raise ValueError("Posterior class probabilities are required.")
        probabilities = onp.asarray(posterior, dtype=onp.float64)
        modal = onp.argmax(probabilities, axis=1)
        entropy = -onp.sum(probabilities * onp.log(onp.maximum(probabilities, 1e-300)))
        entropy_r2 = 1.0 - entropy / (
            probabilities.shape[0] * onp.log(self.model.num_classes)
        )
        prior_shares = probabilities.mean(axis=0)
        rows = []
        for class_idx in range(self.model.num_classes):
            selected = modal == class_idx
            modal_count = int(selected.sum())
            average_posterior = (
                float(probabilities[selected, class_idx].mean())
                if modal_count
                else float("nan")
            )
            prior = float(prior_shares[class_idx])
            if modal_count and 0.0 < average_posterior < 1.0 and 0.0 < prior < 1.0:
                occ = (average_posterior / (1.0 - average_posterior)) / (
                    prior / (1.0 - prior)
                )
            else:
                occ = float("nan")
            rows.append(
                {
                    "class": class_idx,
                    "modal_panels": modal_count,
                    "modal_share": modal_count / probabilities.shape[0],
                    "average_posterior": average_posterior,
                    "odds_correct_classification": occ,
                    "entropy_r2": float(entropy_r2),
                }
            )
        return pl.DataFrame(rows)

    def beta_summary(self) -> pl.DataFrame:
        """Return population-level coefficient moments with Delta-method SEs.

        Returns
        -------
        pl.DataFrame
            Raw variables, display labels, mean coefficients, standard deviations
            across classes, Delta-method standard errors, and class-specific extrema.
        """
        if self.data.num_panels is None:
            raise ValueError("Panel identifiers are required to summarize LCL results.")

        means, se_means = self._apply_delta_method(
            self._calc_population_mean_betas,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )
        variances, se_variances = self._apply_delta_method(
            self._calc_population_var_betas,
            self.flat_params,
            dems=self.data.dems,
            num_panels=self.data.num_panels,
        )
        structural = onp.asarray(self.em_res.structural_betas)

        # sd = sqrt(var), so se(sd) = se(var) / (2 sd) -- but only where the
        # variance is separated from zero.  A variable whose coefficient is
        # common to every class has no identified spread, and the derivative of
        # the square root there is unbounded, so the standard error is NaN rather
        # than the zero a floored square root would silently report.
        variance_array = onp.asarray(variances, dtype=onp.float64)
        se_variance_array = onp.asarray(se_variances, dtype=onp.float64)
        scale = onp.maximum(onp.max(structural**2, axis=1), 1.0)
        identified = variance_array > DEGENERATE_VARIANCE_RTOL * scale
        stds = onp.sqrt(onp.maximum(variance_array, 0.0))
        with onp.errstate(divide="ignore", invalid="ignore"):
            se_stds = onp.where(
                identified, se_variance_array / (2.0 * stds), onp.nan
            )
        if not bool(onp.all(identified)):
            degenerate = [
                variable
                for variable, keep in zip(self.model.case_varnames, identified)
                if not keep
            ]
            logger.warning(
                "Between-class spread is not identified for %s: the coefficient "
                "is common to every class, so its standard deviation has no "
                "standard error.",
                ", ".join(degenerate),
            )
        rows = []
        for idx, variable in enumerate(self.model.case_varnames):
            rows.append(
                {
                    "variable": variable,
                    "label": _model_variable_label(self.model, variable),
                    "mean": float(means[idx]),
                    "mean_se": float(se_means[idx]),
                    "sd": float(stds[idx]),
                    "sd_se": float(se_stds[idx]),
                    "min_class": float(onp.min(structural[idx, :])),
                    "max_class": float(onp.max(structural[idx, :])),
                }
            )
        return pl.DataFrame(rows)

    def summarize_betas(
        self,
        header: tuple[str, str, str] = (
            "Variable",
            r"Means (\beta's)",
            r"Standard deviations (\sigma's)",
        ),
        num_decimals: int = 3,
        *,
        show: bool = True,
    ) -> pl.DataFrame:
        """Print and return population-level coefficient moments.

        Parameters
        ----------
        header : tuple[str, str, str], default=("Variable", ...)
            Column labels used in the printed LaTeX and terminal tables.
        num_decimals : int, default=3
            Number of decimal places used in printed tables.
        show : bool, default=True
            Emit LaTeX and terminal renderings. Set to ``False`` for
            computation-only use.

        Returns
        -------
        pl.DataFrame
            Tidy coefficient-moment table.  The ``variable`` column preserves raw
            model names; ``label`` contains presentation labels used for printing.
        """
        summary_df = self.beta_summary()
        if show:
            log_or_print(
                logger,
                "%s",
                format_lcl_beta_summary(summary_df, header, num_decimals),
            )
        return summary_df

    def summarize(self, num_decimals: int = 3, *, show: bool = True) -> pl.DataFrame:
        """Alias for :meth:`summarize_betas`."""
        return self.summarize_betas(num_decimals=num_decimals, show=show)

    def summarize_class_betas(
        self,
        num_decimals: int = 3,
        *,
        layout: Literal["auto", "wide", "dense"] = "auto",
        include_shares: bool = True,
        show: bool = True,
    ) -> pl.DataFrame:
        """Print and return class-specific utility coefficients.

        :meth:`summarize_betas` reports the population mean and standard
        deviation of each coefficient; this method reports the class-specific
        coefficients those moments are computed from, in the same two-line
        estimate-over-standard-error cell.

        Parameters
        ----------
        num_decimals : int, default=3
            Decimal places used in the printed tables.
        layout : {"auto", "wide", "dense"}, default="auto"
            ``"wide"`` puts one variable per row and one class per column, which
            reads best for a handful of classes.  ``"dense"`` transposes to one
            row per class, which keeps its width bounded by the number of
            utility variables and therefore stays readable at many classes.
            ``"auto"`` switches to ``"dense"`` at nine classes.
        include_shares : bool, default=True
            Append each class's aggregate share to the printed table.
        show : bool, default=True
            Emit LaTeX and terminal renderings.

        Returns
        -------
        pl.DataFrame
            The long-format frame from :meth:`class_coefficients`, unchanged, so
            printing and programmatic use share one source of truth.
        """
        table = self.class_coefficients()
        if show:
            shares = None
            share_standard_errors = None
            if include_shares:
                share_frame = self.class_shares()
                shares = share_frame["share"].to_list()
                share_standard_errors = share_frame["std_error"].to_list()
            log_or_print(
                logger,
                "%s",
                format_class_coefficients(
                    table,
                    self.model.num_classes,
                    num_decimals,
                    layout=layout,
                    shares=shares,
                    share_standard_errors=share_standard_errors,
                ),
            )
        return table

    def summarize_membership(
        self,
        num_decimals: int = 3,
        *,
        layout: Literal["auto", "wide", "dense"] = "auto",
        include_shares: bool = True,
        marginal_effects: bool = True,
        show: bool = True,
    ) -> pl.DataFrame:
        """Print and return the class-membership demographic regression.

        The reference class is printed as an explicit row (or column) of zeros
        rather than omitted, and the rendering carries the normalisation note, so
        a reader can see which class the log-odds are measured against without
        consulting the documentation.

        Parameters
        ----------
        num_decimals : int, default=3
            Decimal places used in the printed tables.
        layout : {"auto", "wide", "dense"}, default="auto"
            ``"dense"`` gives one row per class and one column per demographic
            variable, the layout that scales to many classes; ``"wide"``
            transposes it.  ``"auto"`` switches to ``"dense"`` at nine classes.
        include_shares : bool, default=True
            Show each class's aggregate share beside its coefficients.
        marginal_effects : bool, default=True
            Also print :meth:`membership_marginal_effects`, which restates the
            same regression in probability points and without the reference-class
            normalisation.
        show : bool, default=True
            Emit LaTeX and terminal renderings.

        Returns
        -------
        pl.DataFrame
            The long-format frame from :meth:`membership_coefficients`.

        Raises
        ------
        ValueError
            If the fitted model has no demographic class-membership regression.
        """
        if self.em_res.thetas is None:
            raise ValueError(
                "This model has no class-membership regression: it was fitted "
                "without demographics, so class shares are constant across "
                "panels. Use class_shares() instead."
            )
        table = self.membership_coefficients()
        if show:
            shares = (
                self.class_shares()["share"].to_list() if include_shares else None
            )
            log_or_print(
                logger,
                "%s",
                format_membership_coefficients(
                    table,
                    self.model.num_classes,
                    num_decimals,
                    layout=layout,
                    shares=shares,
                ),
            )
            if marginal_effects:
                log_or_print(
                    logger,
                    "%s",
                    format_membership_marginal_effects(
                        self.membership_marginal_effects(),
                        self.model.num_classes,
                        num_decimals,
                        layout=layout,
                    ),
                )
        return table

    def spec_summary(self) -> str:
        """Return a human-readable model specification summary."""
        spec = getattr(self.model, "spec", None)
        if spec is not None:
            return "\n".join(spec.summary_lines())

        lines = [
            "Latent-class conditional logit",
            f"Classes: {self.model.num_classes}",
            "",
            "Utility variables:",
        ]
        for variable in self.model.case_varnames:
            suffix = ""
            if variable == self.model.numeraire:
                suffix = (
                    f" [negative, min_abs={self._param_packing.numeraire_min_abs:g}]"
                )
            label = _model_variable_label(self.model, variable)
            variable_text = label if label == variable else f"{label} ({variable})"
            lines.append(f"  {variable_text}{suffix}")
        lines.append("")
        lines.append("Class-membership variables:")
        if self.model.dem_varnames:
            for variable in self.model.dem_varnames:
                label = _model_variable_label(self.model, variable)
                variable_text = label if label == variable else f"{label} ({variable})"
                lines.append(f"  {variable_text}")
        else:
            lines.append("  none")
        return "\n".join(lines)

    def diagnostics(self) -> LCLDiagnostics:
        """Return structured model diagnostics."""
        rows: list[dict[str, object]] = [
            {
                "section": "fit",
                "check": "converged",
                "value": bool(self.converged),
                "status": "ok" if self.converged else "warning",
                "message": (
                    "Whether the estimate is a stationary point of the mixture "
                    "likelihood, judged by observed_score_max against score_tol."
                ),
            },
            {
                "section": "fit",
                "check": "log_likelihood",
                "value": float(self.em_res.unconditional_loglik),
                "status": "ok",
                "message": "Final unconditional log likelihood.",
            },
            {
                "section": "fit",
                "check": "observed_score_max",
                "value": self.observed_score_max,
                "status": (
                    "warning"
                    if onp.isfinite(self.observed_score_max)
                    and self.observed_score_max > self.score_tol
                    else "ok"
                ),
                "message": (
                    "Maximum absolute component of the final observed-data score, "
                    f"against the score_tol of {self.score_tol:.3g}. This is the "
                    "same test that sets the converged flag, so the two agree by "
                    "construction."
                ),
            },
            {
                "section": "fit",
                "check": "em_criterion_met",
                "value": bool(self.em_criterion_met),
                "status": "ok" if self.em_criterion_met else "warning",
                "message": (
                    "Whether the Aitken-extrapolated EM criterion was met before "
                    "the iteration cap."
                ),
            },
            {
                "section": "data",
                "check": "panels",
                "value": int(self.data.num_panels or 0),
                "status": "ok",
                "message": "Number of decision-maker panels.",
            },
            {
                "section": "data",
                "check": "cases",
                "value": int(self.data.num_cases),
                "status": "ok",
                "message": "Number of choice situations.",
            },
        ]

        if self.polish_report is not None:
            report = self.polish_report
            rows.append(
                {
                    "section": "fit",
                    "check": "polish_loglik_gain",
                    "value": float(report.loglik_after - report.loglik_before),
                    "status": "ok",
                    "message": (
                        f"Log likelihood gained by {report.iterations} "
                        "observed-data Newton step(s) after EM. A large value "
                        "means EM stopped well short of the optimum."
                    ),
                }
            )
            rows.append(
                {
                    "section": "fit",
                    "check": "polish_score_reduction",
                    "value": float(report.score_before),
                    "status": "ok",
                    "message": (
                        "Observed-data score before the polish, for comparison "
                        "with observed_score_max after it."
                    ),
                }
            )

        rows.append(
            {
                "section": "inference",
                "check": "covariance_available",
                "value": bool(self.inference.skip or self.covariance_available),
                "status": (
                    "ok"
                    if self.inference.skip or self.covariance_available
                    else "warning"
                ),
                "message": (
                    "Whether a finite covariance matrix was estimated. When false, "
                    "every standard error in this results object is NaN."
                ),
            }
        )

        if self.diagnostics_config.check_separation and self.data.num_panels:
            prior = onp.asarray(
                self._get_class_probs(
                    self._unpack_params(self.flat_params)[1],
                    self.data.dems,
                    self.data.num_panels,
                )
            )
            min_prior = float(prior.min())
            separated = min_prior <= self.diagnostics_config.separation_threshold
            rows.append(
                {
                    "section": "latent_class",
                    "check": "min_membership_probability",
                    "value": min_prior,
                    "status": "warning" if separated else "ok",
                    "message": (
                        "Smallest prior class-membership probability over panels. "
                        "A value at zero means some demographic cell never belongs "
                        "to that class, so its membership coefficients are "
                        "unbounded and the observed information is singular in "
                        "that direction. Merge the cell, drop the variable, or fit "
                        "fewer classes."
                    ),
                }
            )

        if self.information_diagnostics is not None and (
            self.diagnostics_config.check_collinearity
        ):
            info = self.information_diagnostics
            rows.append(
                {
                    "section": "inference",
                    "check": "information_rank",
                    "value": float(info.rank),
                    "status": "warning" if info.rank_deficient else "ok",
                    "message": (
                        f"Numerical rank of the observed information out of "
                        f"{info.num_params} parameters. A deficient rank means "
                        "some standard errors are not identified."
                    ),
                }
            )
            rows.append(
                {
                    "section": "inference",
                    "check": "information_condition_number",
                    "value": float(info.condition_number),
                    "status": (
                        "warning"
                        if not info.positive_definite or info.condition_number > 1e12
                        else "ok"
                    ),
                    "message": (
                        "Ratio of largest to smallest eigenvalue of the observed "
                        "information. Large values indicate weakly identified "
                        "parameter directions."
                    ),
                }
            )
            rows.append(
                {
                    "section": "inference",
                    "check": "information_min_eigenvalue",
                    "value": float(info.smallest_eigenvalue),
                    "status": "ok" if info.positive_definite else "warning",
                    "message": (
                        "Smallest eigenvalue of the observed information. Values "
                        "at or below zero indicate a saddle point rather than a "
                        "maximum."
                    ),
                }
            )

        if self.em_res.class_probs_by_panel is not None:
            posterior = onp.asarray(self.em_res.class_probs_by_panel)
            entropy = -onp.sum(
                posterior * onp.log(onp.maximum(posterior, 1e-300)), axis=1
            )
            rows.append(
                {
                    "section": "latent_class",
                    "check": "posterior_entropy_mean",
                    "value": float(entropy.mean()),
                    "status": "ok",
                    "message": "Mean entropy of posterior class membership.",
                }
            )

        shares_df = self.class_shares()
        min_share = float(cast(float, shares_df["share"].min()))
        rows.append(
            {
                "section": "latent_class",
                "check": "min_class_share",
                "value": min_share,
                "status": "warning" if min_share < 0.01 else "ok",
                "message": "Small classes can indicate weakly identified local optima.",
            }
        )
        if "effective_panels" in shares_df.columns:
            rows.append(
                {
                    "section": "latent_class",
                    "check": "min_effective_panels",
                    "value": float(cast(float, shares_df["effective_panels"].min())),
                    "status": "ok",
                    "message": "Smallest posterior panel mass across classes.",
                }
            )

        structural = onp.asarray(self.em_res.structural_betas)
        max_abs_beta = float(onp.max(onp.abs(structural)))
        rows.append(
            {
                "section": "coefficients",
                "check": "max_abs_beta",
                "value": max_abs_beta,
                "status": (
                    "warning"
                    if (
                        self.diagnostics_config.warn_large_coefficients
                        and max_abs_beta
                        > self.diagnostics_config.large_coefficient_threshold
                    )
                    else "ok"
                ),
                "message": "Largest absolute structural coefficient.",
            }
        )
        numeraire_idx = getattr(self.model, "numeraire_idx", None)
        if numeraire_idx is not None:
            min_abs_numeraire = float(onp.min(onp.abs(structural[numeraire_idx, :])))
            threshold = self.diagnostics_config.near_zero_numeraire_threshold
            rows.append(
                {
                    "section": "coefficients",
                    "check": "min_abs_numeraire",
                    "value": min_abs_numeraire,
                    "status": (
                        "warning"
                        if (
                            self.diagnostics_config.warn_near_zero_numeraire
                            and min_abs_numeraire < threshold
                        )
                        else "ok"
                    ),
                    "message": "Small numeraires can dominate WTP/tradeoff ratios.",
                }
            )
            linear = numeraire_enters_linearly(self.model)
            rows.append(
                {
                    "section": "coefficients",
                    "check": "numeraire_enters_linearly",
                    "value": float(linear),
                    "status": "ok" if linear else "warning",
                    "message": (
                        "Whether the numeraire enters utility as a bare column. "
                        "Consumer surplus and willingness to pay divide by "
                        "-dV/d(numeraire), which equals -beta only when the "
                        "numeraire is untransformed. Under a transform such as "
                        "log(price) the ratio is per unit of the transform, not "
                        "per dollar, and the constant-marginal-utility-of-income "
                        "assumption behind the log-sum welfare formula no longer "
                        "holds."
                    ),
                }
            )

        return LCLDiagnostics(pl.DataFrame(rows))

    def diagnose(self) -> LCLDiagnostics:
        """Alias for :meth:`diagnostics`."""
        return self.diagnostics()

    def convergence_report(self) -> str:
        """Return a compact convergence and diagnostic report."""
        diagnostics = self.diagnostics().to_frame()
        warnings = diagnostics.filter(pl.col("status") != "ok")
        lines = [
            f"Converged: {self.converged}",
            f"EM recursions: {self.total_recursions}",
            f"EM criterion met: {self.em_criterion_met}",
            f"Final log likelihood: {float(self.em_res.unconditional_loglik):.6g}",
            f"Max observed-data score: {self.observed_score_max:.3e} "
            f"(tolerance {self.score_tol:.3g})",
            f"Warnings: {warnings.height}",
        ]
        if self.polish_report is not None:
            report = self.polish_report
            lines.append(
                f"Observed-data polish: {report.iterations} Newton step(s), "
                f"log likelihood {report.loglik_before:.6f} -> "
                f"{report.loglik_after:.6f}, score {report.score_before:.3e} -> "
                f"{report.score_after:.3e}"
            )
        if self.em_history_.height:
            last = self.em_history_.tail(1).row(0, named=True)
            lines.append(f"Last EM history row: {last}")
        return "\n".join(lines)

    def audit_report(self) -> str:
        """Return a text audit report for replication materials."""
        diagnostics_table = self.diagnostics().to_frame()
        return "\n\n".join(
            [
                "1. Model Specification\n" + self.spec_summary(),
                "2. Fit Statistics\n"
                + "\n".join(
                    [
                        f"Log likelihood: {float(self.em_res.unconditional_loglik):.6g}",
                        f"CAIC: {float(self.caic):.6g}",
                        f"BIC: {float(self.bic):.6g}",
                        f"Adjusted BIC: {float(self.adjusted_bic):.6g}",
                        f"Estimation seconds: {self.estim_time_sec:.3f}",
                    ]
                ),
                "3. Class Shares\n" + str(self.class_shares()),
                "4. Diagnostics\n" + str(diagnostics_table),
            ]
        )

    def predict(
        self,
        data: object | None = None,
        *,
        X: ArrayLike | None = None,
        alts: ArrayLike | None = None,
        cases: ArrayLike | None = None,
        panels: ArrayLike | None = None,
        dems: ArrayLike | None = None,
        dem_panel_ids: ArrayLike | None = None,
        past_choices: object | None = None,
        dems_data: object | None = None,
        past_choices_dems_data: object | None = None,
        panel_weights: str | Mapping[object, float] | Sequence[float] | None = None,
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
            Panel-level demographics for array-style prediction. When
            ``dem_panel_ids`` is omitted, rows must be in sorted unique panel-ID
            order.
        dem_panel_ids : ArrayLike | None, optional
            Panel IDs aligned with rows of ``dems``. The parser uses these IDs to
            validate and reorder demographic rows.
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
        partition_data_df = None
        raw_prediction_data = None
        if data is not None:
            parsed_predict = self.model._transform_data(data, dems_data=dems_data)
            encoder = getattr(self.model, "_encoder", None)
            if encoder is not None:
                raw_prediction_data = _coerce_frame(data).sort(
                    list(
                        dict.fromkeys(
                            [encoder.panels_col, encoder.cases_col, encoder.alts_col]
                        )
                    )
                )
                partition_data_df = _prediction_partition_data(
                    data, dems_data, encoder.panels_col
                )
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
                dem_panel_ids=dem_panel_ids,
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
        if self.em_res.thetas is not None and predict_data.dems is None:
            raise ValueError(
                "dems is required for array-style prediction because the fitted "
                "class-membership model uses demographics. Pass dem_panel_ids to "
                "validate their panel alignment."
            )

        # Retained for posterior-updated WTP inference, which differentiates
        # through the Bayes update rather than freezing the posterior.
        data_past: Data | None = None
        diff_unchosen_chosen_past: DiffUnchosenChosen | None = None
        if past_choices is not None:
            parsed_past = _parse_past_choices(
                model=self.model,
                past_choices=past_choices,
                past_choices_dems_data=past_choices_dems_data,
            )
            _validate_past_choice_panels(parsed_past, parsed_predict)
            data_past = cast(Data, self.model._setup_data(parsed_past)[0])
            diff_unchosen_chosen_past = _diff_unchosen_chosen(data_past)
            class_probs_by_panel, _ = _compute_conditional_class_probs(
                structural_betas=structural_betas,
                thetas=self.em_res.thetas,
                shares=shares,
                diff_unchosen_chosen=diff_unchosen_chosen_past,
                data=data_past,
            )
            class_probabilities_source = "posterior"
        elif self.em_res.thetas is not None and predict_data.dems is not None:
            class_probs_by_panel = self._get_class_probs(
                self.em_res.thetas, predict_data.dems, predict_data.num_panels
            )
            class_probabilities_source = "prior"
        else:
            class_probs_by_panel = jnp.repeat(
                shares[None, :], predict_data.num_panels, axis=0
            )
            class_probabilities_source = "prior"

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

        surplus_by_class = log_sum_exp_utility / marginal_utility_income[None, :]

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
        encoder = getattr(self.model, "_encoder", None)
        resolved_panel_weights = resolve_panel_weights(
            panel_weights,
            panels_unique,
            raw_prediction_data,
            encoder.panels_col if encoder is not None else "panels",
        )
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
        first_case_rows = first_case_rows.at[0].set(True)
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
            class_probabilities_source=class_probabilities_source,
            partition_data_df=partition_data_df,
            original_alts=parsed_predict.original_alts,
            original_cases=parsed_predict.original_cases,
            original_panels=parsed_predict.original_panels,
            raw_prediction_data=raw_prediction_data,
            panel_weights=resolved_panel_weights,
            past_diff_unchosen_chosen=diff_unchosen_chosen_past,
            past_data=data_past,
        )
