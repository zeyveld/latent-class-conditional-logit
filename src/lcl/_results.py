"""In-sample estimation results and inference."""

import logging
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import hessian, jacfwd, jacrev
from jax.tree_util import Partial
from jax.typing import ArrayLike
from jaxtyping import Array, Float64
from pylatexenc.latex2text import LatexNodes2Text
from tabulate import tabulate

from lcl._case_utils import _diff_unchosen_chosen, _to_structural_betas
from lcl._diagnostics import LCLDiagnostics
from lcl._em_alg_steps import (
    _compute_conditional_class_probs,
    _compute_panel_logliks,
)
from lcl._encoding import _coerce_frame
from lcl._jax_compat import cpu_device, device_put_array_leaves
from lcl._kernels import _choice_probabilities_and_logsum, _class_membership_probs
from lcl._logging import log_or_print
from lcl._prediction import LCLPrediction
from lcl._struct import (
    Data,
    DiagnosticsOptions,
    DiffUnchosenChosen,
    EMAlgConfig,
    EMVars,
    ErrorConfig,
    ParsedData,
    PastChoicesData,
)

logger = logging.getLogger(__name__)


def _symmetrize(
    matrix: Float64[Array, "all_params all_params"],
) -> Float64[Array, "all_params all_params"]:
    """Remove tiny numerical asymmetry from covariance-style matrices."""
    return 0.5 * (matrix + matrix.T)


def _history_frame(rows: list[dict[str, Any]] | None) -> pl.DataFrame:
    """Convert diagnostic history rows containing JAX scalars to Polars."""
    if not rows:
        return pl.DataFrame()
    clean_rows: list[dict[str, object]] = []
    for row in rows:
        clean_row: dict[str, object] = {}
        for key, value in row.items():
            arr = onp.asarray(value)
            clean_row[key] = arr.item() if arr.shape == () else arr.tolist()
        clean_rows.append(clean_row)
    return pl.DataFrame(clean_rows)


def _panel_constant_columns(data: object, panel_col: str) -> pl.DataFrame:
    """Return raw columns that are constant within each prediction panel.

    Parameters
    ----------
    data : object
        Tabular prediction data.
    panel_col : str
        Column identifying decision-makers.

    Returns
    -------
    pl.DataFrame
        One row per panel, keyed by ``"panels"``, containing only columns whose
        values do not vary within panel.
    """
    df = _coerce_frame(data)
    candidate_cols = [col for col in df.columns if col != panel_col]
    if not candidate_cols:
        return (
            df.select(panel_col)
            .unique(maintain_order=True)
            .rename({panel_col: "panels"})
        )

    max_unique_by_col = (
        df.group_by(panel_col)
        .agg([pl.col(col).n_unique().alias(col) for col in candidate_cols])
        .select(pl.exclude(panel_col).max())
        .row(0)
    )
    constant_cols = [
        col
        for col, max_unique in zip(candidate_cols, max_unique_by_col)
        if max_unique <= 1
    ]
    return (
        df.select([panel_col, *constant_cols])
        .unique(subset=[panel_col], maintain_order=True)
        .rename({panel_col: "panels"})
    )


def _prediction_partition_data(
    data: object,
    dems_data: object | None,
    panel_col: str,
) -> pl.DataFrame:
    """Build panel-level WTP partition data from raw prediction inputs."""
    partition_df = _panel_constant_columns(data, panel_col)
    if dems_data is None:
        return partition_df

    dems_partition_df = _panel_constant_columns(dems_data, panel_col)
    dems_cols = [
        col
        for col in dems_partition_df.columns
        if col == "panels" or col not in partition_df.columns
    ]
    if len(dems_cols) == 1:
        return partition_df
    return partition_df.join(
        dems_partition_df.select(dems_cols), on="panels", how="left"
    )


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
        diagnostics_config: DiagnosticsOptions | None = None,
        em_history: list[dict[str, Any]] | None = None,
        optimization_history: list[dict[str, Any]] | None = None,
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
        diagnostics_config : :class:`~lcl._struct.DiagnosticsOptions` | None
            Thresholds and switches for public diagnostics.
        em_history : list[dict[str, Any]] | None
            EM log-likelihood and class-share history.
        optimization_history : list[dict[str, Any]] | None
            Final class-level M-step diagnostics.
        """
        self.model = model_spec
        self.em_res = em_vars
        self.data = estimation_data
        self.total_recursions = em_recursion
        self.converged = em_recursion < (em_alg_config.maxiter - 1)
        self.estim_time_sec = estim_time_sec
        self.error_config = error_config if error_config is not None else ErrorConfig()
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
        """Compute covariance on CPU, optionally with clustered sandwich correction.

        The EM algorithm may leave fitted arrays committed to a GPU or sharded
        accelerator placement.  Robust inference builds dense Hessians/Jacobians,
        so this method explicitly moves only the inference inputs to CPU and runs
        all derived differencing, Hessian, pseudo-inverse, and score-Jacobian work
        there.  The robust score Jacobian maps parameters to panel contributions;
        because that Jacobian is usually tall, forward-mode AD is the better default
        than reverse-mode AD for this calculation.

        Returns
        -------
        Float64[Array, "all_params all_params"]
            Covariance matrix aligned with the flattened parameter vector.
        """
        cpu = cpu_device()
        if self.error_config.skip_std_errs:
            with jax.default_device(cpu):
                return jnp.full((self.num_params, self.num_params), jnp.nan)

        logger.info("Computing LCL covariance matrix.")
        with jax.default_device(cpu):
            flat_params = device_put_array_leaves(self.flat_params, cpu)
            data = device_put_array_leaves(self.data, cpu)
            diff_unchosen_chosen = device_put_array_leaves(
                _diff_unchosen_chosen(data), cpu
            )

            H = hessian(self._full_loglik_fn)(flat_params, diff_unchosen_chosen, data)
            H_inv = jax.device_put(onp.linalg.pinv(onp.asarray(-H)), cpu)

            if not self.error_config.robust:
                return _symmetrize(H_inv)

            J = jacfwd(self._panel_loglik_fn)(flat_params, diff_unchosen_chosen, data)
            B = J.T @ J

            if data.num_panels is None:
                raise ValueError(
                    "Panel identifiers are required for clustered covariance."
                )
            G = data.num_panels
            return _symmetrize((H_inv @ B @ H_inv) * (G / (G - 1)))

    def _panel_loglik_fn(
        self,
        flat_params: Float64[Array, "all_params"],
        diff_unchosen_chosen: DiffUnchosenChosen,
        data: Data,
    ) -> Float64[Array, "panels"]:
        """Compute the log-likelihood for each panel (used to build the Jacobian)."""
        latent_betas, thetas = self._unpack_params(flat_params)
        structural_betas = _to_structural_betas(
            latent_betas,
            getattr(self.model, "numeraire_idx", None),
            getattr(self.model, "numeraire_min_abs", 1e-5),
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
        """Apply the Delta Method on CPU for non-linear parameter functions.

        The target functions used for summaries and WTP inference generally return
        scalars or short vectors, so reverse-mode AD remains appropriate here even
        though the robust covariance score Jacobian uses ``jacfwd``.
        """
        cpu = cpu_device()
        with jax.default_device(cpu):
            flat_params_cpu = device_put_array_leaves(flat_params, cpu)
            args_cpu = device_put_array_leaves(args, cpu)
            kwargs_cpu = device_put_array_leaves(kwargs, cpu)
            cov_matrix = device_put_array_leaves(self.cov_matrix, cpu)

            target_func = Partial(func, *args_cpu, **kwargs_cpu)
            val = target_func(flat_params_cpu)
            jac = jacrev(target_func)(flat_params_cpu)

            if val.ndim == 0:
                variance = jac.T @ cov_matrix @ jac
            else:
                variance = jnp.einsum("kp,pq,kq->k", jac, cov_matrix, jac)

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
            latent_betas,
            getattr(self.model, "numeraire_idx", None),
            getattr(self.model, "numeraire_min_abs", 1e-5),
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

        structural_betas = _to_structural_betas(
            latent_betas,
            numeraire_idx,
            getattr(self.model, "numeraire_min_abs", 1e-5),
        )

        mean_betas = structural_betas @ avg_shares
        diff_sq = (structural_betas - mean_betas[:, None]) ** 2
        var_betas = diff_sq @ avg_shares

        return jnp.sqrt(jnp.maximum(var_betas, 1e-250))

    def class_coefficients(self) -> pl.DataFrame:
        """Return class-specific structural coefficients.

        Returns
        -------
        pl.DataFrame
            Long-format table with one row per variable and latent class.
        """
        structural_betas = self.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required.")
        rows = []
        beta_array = onp.asarray(structural_betas)
        for var_idx, variable in enumerate(self.model.case_varnames):
            for class_idx in range(self.model.num_classes):
                rows.append(
                    {
                        "variable": variable,
                        "class": class_idx,
                        "coefficient": float(beta_array[var_idx, class_idx]),
                        "constrained": variable == self.model.numeraire,
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
        if self.em_res.shares is None:
            raise ValueError("Class shares are required.")
        shares = onp.asarray(self.em_res.shares)
        rows = []
        posterior = self.em_res.class_probs_by_panel
        posterior_arr = onp.asarray(posterior) if posterior is not None else None
        for class_idx, share in enumerate(shares):
            row = {"class": class_idx, "share": float(share)}
            if posterior_arr is not None:
                row["effective_panels"] = float(posterior_arr[:, class_idx].sum())
            rows.append(row)
        return pl.DataFrame(rows)

    def beta_summary(self) -> pl.DataFrame:
        """Return population-level coefficient moments with Delta-method SEs.

        Returns
        -------
        pl.DataFrame
            Variables, mean coefficients, standard deviations across classes,
            Delta-method standard errors, and class-specific extrema.
        """
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
        structural = onp.asarray(self.em_res.structural_betas)
        rows = []
        for idx, variable in enumerate(self.model.case_varnames):
            rows.append(
                {
                    "variable": variable,
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
    ) -> pl.DataFrame:
        """Print and return population-level moments with Delta-method SEs."""
        summary_df = self.beta_summary()
        body_rows, data_clean = [], []
        converter = LatexNodes2Text(math_mode="text")
        header_clean = [converter.latex_to_text(col) for col in header]

        for row in summary_df.iter_rows(named=True):
            coeff_nm = str(row["variable"])
            body_rows.append(
                f"{coeff_nm} & {float(row['mean']):.{num_decimals}f} & {float(row['sd']):.{num_decimals}f} \\\\"
            )
            body_rows.append(
                f" & ({float(row['mean_se']):.{num_decimals}f}) & ({float(row['sd_se']):.{num_decimals}f}) \\\\"
            )
            var_clean = converter.latex_to_text(coeff_nm)
            data_clean.append(
                (
                    var_clean,
                    f"{float(row['mean']):.{num_decimals}f}",
                    f"{float(row['sd']):.{num_decimals}f}",
                )
            )
            data_clean.append(
                (
                    "",
                    f"({float(row['mean_se']):.{num_decimals}f})",
                    f"({float(row['sd_se']):.{num_decimals}f})",
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
        return summary_df

    def summarize(self, num_decimals: int = 3) -> pl.DataFrame:
        """Alias for :meth:`summarize_betas`."""
        return self.summarize_betas(num_decimals=num_decimals)

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
                    " [negative, "
                    f"min_abs={getattr(self.model, 'numeraire_min_abs', 1e-5):g}]"
                )
            lines.append(f"  {variable}{suffix}")
        lines.append("")
        lines.append("Class-membership variables:")
        if self.model.dem_varnames:
            lines.extend(f"  {variable}" for variable in self.model.dem_varnames)
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
                "message": "EM convergence flag.",
            },
            {
                "section": "fit",
                "check": "log_likelihood",
                "value": float(self.em_res.unconditional_loglik),
                "status": "ok",
                "message": "Final unconditional log likelihood.",
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
            f"Final log likelihood: {float(self.em_res.unconditional_loglik):.6g}",
            f"Warnings: {warnings.height}",
        ]
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
        partition_data_df = None
        if data is not None:
            parsed_predict = self.model._transform_data(data, dems_data=dems_data)
            encoder = getattr(self.model, "_encoder", None)
            if encoder is not None:
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
            class_probabilities_source=class_probabilities_source,
            partition_data_df=partition_data_df,
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
