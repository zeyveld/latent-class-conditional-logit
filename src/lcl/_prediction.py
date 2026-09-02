"""Out-of-sample prediction, elasticities, and willingness-to-pay (WTP) analysis."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
from jaxtyping import Array, Float64, Int

from lcl._elasticities import compute_elasticities, elasticity_design_derivative
from lcl._logging import log_or_print
from lcl._presentation import format_wtp_table
from lcl.options import PartitionType, WTPRequest
from lcl._struct import Data
from lcl._wtp_partitions import (
    _apply_wtp_partition,
    _coerce_partition_data,
    _flatten_wtp_requests,
    _partition_columns,
    _partition_label,
)

logger = logging.getLogger(__name__)


class _PredictionBase:
    """Shared probability, surplus, and elasticity prediction behavior.

    Provides methods to analyze decision-maker behavior under new choice sets or
    policy changes. Utilizes the Delta Method to compute rigorous analytical standard
    errors for non-linear combinations of parameters (e.g., marginal WTP) across
    dynamically defined demographic partitions.

    Attributes
    ----------
    predicted_probs : pl.DataFrame
        DataFrame of out-of-sample choice probabilities for each alternative.
    surplus : pl.DataFrame
        DataFrame of expected consumer surplus (inclusive value) per choice situation.
    wtp_alt_vars_by_panel : pl.DataFrame
        DataFrame of expected marginal WTP for each alternative-specific characteristic,
        calculated at the individual decision-maker level.
    predict_data : :class:`~lcl._struct.Data`
        The parsed design matrices corresponding to the counterfactual scenarios.
    results : :class:`~lcl._results.LCLResults`
        Reference to the parent estimation results, required for Delta Method covariance
        calculations and parameter unpacking.
    class_probs_by_panel : Array | None
        Posterior (or prior) probabilities of latent class membership used to generate
        these predictions. If historical choices were provided during prediction, these
        represent the Bayesian-updated posteriors.
    partition_data : pl.DataFrame | None
        Panel-level columns from raw prediction data that are constant within panel
        and can be used for WTP partitions.
    """

    def __init__(
        self,
        predicted_probs_df: pl.DataFrame,
        surplus_df: pl.DataFrame,
        wtp_alt_vars_by_panel_df: pl.DataFrame,
        predict_data: Data,
        results: Any,
        class_probs_by_panel: Float64[Array, "panels classes"] | None = None,
        class_probabilities_source: str = "prior",
        partition_data_df: pl.DataFrame | None = None,
        original_alts: Any | None = None,
        original_cases: Any | None = None,
        original_panels: Any | None = None,
        raw_prediction_data: pl.DataFrame | None = None,
        panel_weights: Sequence[float] | onp.ndarray | None = None,
    ) -> None:
        """Store prediction outputs and references needed for post-processing.

        Parameters
        ----------
        predicted_probs_df : pl.DataFrame
            Long-format alternative choice probabilities.
        surplus_df : pl.DataFrame
            Case-level consumer surplus estimates.
        wtp_alt_vars_by_panel_df : pl.DataFrame
            Panel-level marginal WTP values for non-numeraire variables.
        predict_data : :class:`~lcl._struct.Data`
            Encoded data used to generate the predictions.
        results : Any
            Parent results object. Kept broad to support both latent-class and
            conditional-logit result containers without a circular import.
        class_probs_by_panel : Float64[Array, "panels classes"] | None, optional
            Class probabilities used to marginalize class-specific predictions.
        class_probabilities_source : str, default="prior"
            ``"posterior"`` when prediction used historical choices, otherwise
            ``"prior"``.
        partition_data_df : pl.DataFrame | None, optional
            Panel-level raw prediction columns available for WTP partitions.
        """
        self.predicted_probs = predicted_probs_df
        self.surplus_units = (
            "money"
            if getattr(results.model, "numeraire_idx", None) is not None
            else "utils"
        )
        self.surplus = surplus_df.with_columns(
            pl.lit(self.surplus_units).alias("surplus_units")
        )
        self.wtp_alt_vars_by_panel = wtp_alt_vars_by_panel_df
        self.predict_data = predict_data
        self.results = results
        self.class_probs_by_panel = class_probs_by_panel
        self.class_probabilities_source = class_probabilities_source
        self.partition_data = partition_data_df
        self.original_alts = (
            onp.asarray(original_alts)
            if original_alts is not None
            else onp.asarray(predict_data.alts)
        )
        self.original_cases = (
            onp.asarray(original_cases)
            if original_cases is not None
            else onp.asarray(predict_data.cases)
        )
        self.original_panels = (
            onp.asarray(original_panels)
            if original_panels is not None
            else (
                None
                if predict_data.panels is None
                else onp.asarray(predict_data.panels)
            )
        )
        self.raw_prediction_data = raw_prediction_data
        num_panels = predict_data.num_panels
        if num_panels is None:
            raise ValueError("Panel identifiers are required for prediction.")
        weights = (
            onp.ones(num_panels, dtype=onp.float64)
            if panel_weights is None
            else onp.asarray(panel_weights, dtype=onp.float64)
        )
        if weights.shape != (num_panels,):
            raise ValueError(
                "panel_weights must contain one value per prediction panel."
            )
        if not onp.all(onp.isfinite(weights)) or onp.any(weights < 0.0):
            raise ValueError("panel_weights must be finite and nonnegative.")
        if not onp.any(weights > 0.0):
            raise ValueError("At least one panel weight must be positive.")
        self.panel_weights = weights

    def _elasticity_design_derivative(
        self, variable: str
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return raw values and row-wise derivatives of every design column."""
        return elasticity_design_derivative(self, variable)

    def elasticities(self, vars: str | Iterable[str]) -> pl.DataFrame:
        """Compute full matrices of own- and cross-elasticities for continuous features.

        Analytically calculates the percentage change in the probability of choosing
        alternative J given a one-percent change in a continuous attribute of
        alternative K. The method handles both conditional (latent class) and
        unconditional (standard conditional logit) probability matrices via a vectorized
        cartesian expansion across choice situations.

        Parameters
        ----------
        vars : str | Iterable[str]
            The name(s) of the continuous variable(s) for which to compute the
            elasticities (e.g., "price", ["price", "travel_time"]).

        Returns
        -------
        pl.DataFrame
            A DataFrame in long format mapping the target alternative (`target_alts`,
            whose attribute is changing) to the affected alternative (`alts`, whose
            probability is changing). Includes choice situation IDs, decision-maker
            panel IDs (if applicable), and the computed point elasticities.

        Raises
        ------
        ValueError
            If latent class probabilities are missing, or if a requested variable
            is not found in the estimated model specification.
        """
        return compute_elasticities(self, vars)

    def market_shares(self) -> pl.DataFrame:
        """Return panel-weighted predicted market shares by alternative."""
        data = self.predict_data
        if data.panels is None:
            raise ValueError("Panel identifiers are required for market shares.")
        row_weights = self.panel_weights[onp.asarray(data.panels)]
        first_case_rows = onp.asarray(data.cases) != onp.roll(
            onp.asarray(data.cases), 1
        )
        first_case_rows[0] = True
        denominator = float(row_weights[first_case_rows].sum())
        frame = self.predicted_probs.with_columns(
            pl.Series("_panel_weight", row_weights)
        ).with_columns(
            (pl.col("choice_probs") * pl.col("_panel_weight")).alias("_demand")
        )
        return (
            frame.group_by("alts", maintain_order=True)
            .agg(pl.col("_demand").sum())
            .with_columns((pl.col("_demand") / denominator).alias("market_share"))
            .select("alts", "market_share")
            .sort("alts")
        )

    def surplus_change(self, counterfactual: "_PredictionBase") -> pl.DataFrame:
        """Return identified counterfactual-minus-baseline surplus changes."""
        if self.surplus_units != counterfactual.surplus_units:
            raise ValueError("Baseline and counterfactual surplus units do not match.")
        keys = ["cases"]
        if "panels" in self.surplus.columns:
            keys.insert(0, "panels")
        baseline = self.surplus.select(
            *keys, pl.col("surplus").alias("_baseline_surplus")
        )
        changed = counterfactual.surplus.select(
            *keys, pl.col("surplus").alias("_counterfactual_surplus")
        )
        joined = baseline.join(changed, on=keys, how="inner", validate="1:1")
        if joined.height != baseline.height or joined.height != changed.height:
            raise ValueError(
                "Baseline and counterfactual predictions must contain identical cases."
            )
        return joined.select(
            *keys,
            (pl.col("_counterfactual_surplus") - pl.col("_baseline_surplus")).alias(
                "surplus_change"
            ),
            pl.lit(self.surplus_units).alias("surplus_units"),
        )

    def aggregate_elasticities(self, vars: str | Iterable[str]) -> pl.DataFrame:
        """Aggregate elasticities using each row's weighted demand contribution."""
        variables = [vars] if isinstance(vars, str) else list(vars)
        elasticities = self.elasticities(variables)
        join_keys = ["cases", "alts"]
        if "panels" in elasticities.columns:
            join_keys.insert(0, "panels")
        probability_columns = self.predicted_probs.select([*join_keys, "choice_probs"])
        if self.original_panels is None:
            raise ValueError("Panel identifiers are required for aggregation.")
        panel_first_rows = onp.asarray(self.predict_data.panels) != onp.roll(
            onp.asarray(self.predict_data.panels), 1
        )
        panel_first_rows[0] = True
        panel_frame = pl.DataFrame(
            {
                "panels": self.original_panels[panel_first_rows],
                "_panel_weight": self.panel_weights,
            }
        )
        joined = elasticities.join(probability_columns, on=join_keys).join(
            panel_frame, on="panels"
        )
        demand_weight = pl.col("choice_probs") * pl.col("_panel_weight")
        aggregations = [demand_weight.sum().alias("_demand")]
        for variable in variables:
            column = f"elasticity_{variable}"
            aggregations.append(
                (demand_weight * pl.col(column)).sum().alias(f"_{column}_total")
            )
        result = joined.group_by(["alts", "target_alts"], maintain_order=True).agg(
            aggregations
        )
        return result.select(
            "alts",
            "target_alts",
            *(
                (pl.col(f"_elasticity_{variable}_total") / pl.col("_demand")).alias(
                    f"elasticity_{variable}"
                )
                for variable in variables
            ),
        ).sort(["alts", "target_alts"])


class LCLPrediction(_PredictionBase):
    """Latent-class prediction with partitioned WTP inference."""

    def compute_wtp(
        self,
        *wtp_requests: WTPRequest | Iterable[WTPRequest],
        partition_data: object | None = None,
        panel_col: str = "panels",
        num_decimals: int = 4,
        class_probabilities: Literal["stored", "prior", "posterior"] = "stored",
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
        show: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """Compute the Marginal Willingness-to-Pay (WTP) across demographic partitions.

        Evaluates the ratio of the target parameter to the negative cost parameter
        (marginal utility of income) for dynamically defined subsets of decision-makers.
        Outputs formatted LaTeX and terminal summary tables, including analytical
        standard errors derived via the Delta Method.

        Parameters
        ----------
        *wtp_requests : WTPRequest | Iterable[WTPRequest]
            One or more configuration objects specifying the target variable,
            the demographic partitioning variable, and the binning strategy (e.g.,
            quintiles, categorical, custom breaks, or a dummy-coded categorical
            factor).
        partition_data : object | None, optional
            Optional panel-level or long-format tabular data containing partitioning
            variables that were not included in the fitted class-membership
            specification. Values must be constant within each panel.
        panel_col : str, default="panels"
            Panel identifier column in ``partition_data``.
        num_decimals : int, default=4
            Number of decimal places used in printed WTP tables.
        class_probabilities : {"stored", "prior", "posterior"}, default="stored"
            Class-membership probabilities used for WTP/tradeoff point estimates.
            ``"stored"`` uses the probabilities already attached to this
            prediction object, including Bayesian posterior updates from
            ``past_choices``. ``"prior"`` recomputes demographics-only class
            probabilities. ``"posterior"`` requires that prediction was created
            with ``past_choices``.
        se : {"delta", "bootstrap", "none"}, default="delta"
            Standard-error method. Delta-method and asymptotic parametric-bootstrap
            standard errors are available for prior class probabilities.
            Posterior-updated WTP through ``past_choices`` requires differentiating
            through the Bayesian class update and is refused unless ``se="none"``.
        bootstrap_draws : int, default=500
            Number of asymptotic parameter draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible random seed for parametric-bootstrap draws.
        show : bool, default=True
            Emit LaTeX and terminal renderings for each request.

        Returns
        -------
        dict[str, pl.DataFrame]
            Summary tables keyed by their printed titles.  Each table preserves
            raw variable names in ``variable`` and ``partition_variable`` and
            includes presentation labels in ``label`` and ``partition_label``.

        Raises
        ------
        ValueError
            If the parent model was not estimated with a specified numeraire constraint.
        """
        if se not in {"delta", "bootstrap", "none"}:
            raise ValueError("se must be 'delta', 'bootstrap', or 'none'.")
        if class_probabilities not in {"stored", "prior", "posterior"}:
            raise ValueError(
                "class_probabilities must be 'stored', 'prior', or 'posterior'."
            )
        if (
            class_probabilities == "posterior"
            and self.class_probabilities_source != "posterior"
        ):
            raise ValueError(
                "class_probabilities='posterior' requires predict(..., past_choices=...)."
            )
        if (
            se in {"delta", "bootstrap"}
            and class_probabilities in {"stored", "posterior"}
            and self.class_probabilities_source == "posterior"
        ):
            raise NotImplementedError(
                "WTP uncertainty after past_choices requires differentiating "
                "through the posterior class update. Use se='none' or "
                "class_probabilities='prior'."
            )

        # We rely on the explicitly tracked numeraire index from _pre_fit
        if getattr(self.results.model, "numeraire_idx", None) is None:
            raise ValueError("A numeraire must be defined to compute WTP.")

        cost_idx = self.results.model.numeraire_idx
        if self.predict_data.panels is None or self.predict_data.num_panels is None:
            raise ValueError("Panel identifiers are required to compute WTP.")

        requests = _flatten_wtp_requests(wtp_requests)
        if not requests:
            return {}

        df_with_idx = self.wtp_alt_vars_by_panel.with_row_index("panel_idx")

        if (
            self.predict_data.dems is not None
            and self.results.model.dem_varnames is not None
        ):
            dems_df = pl.DataFrame(
                onp.array(self.predict_data.dems),
                schema=self.results.model.dem_varnames,
            ).with_row_index("panel_idx")

            df_with_idx = df_with_idx.join(dems_df, on="panel_idx")

        partition_cols = _partition_columns(requests)
        missing_partition_cols = [
            col for col in partition_cols if col not in df_with_idx.columns
        ]
        if missing_partition_cols:
            source_partition_data = partition_data
            source_panel_col = panel_col
            if source_partition_data is None and self.partition_data is not None:
                source_partition_data = self.partition_data
                source_panel_col = "panels"

            if source_partition_data is None:
                raise ValueError(
                    "WTP partition columns were not found in the fitted/prediction "
                    "demographics: "
                    f"{missing_partition_cols}. Pass partition_data=... for "
                    "panel-level grouping variables outside the model specification."
                )
            external_partitions = _coerce_partition_data(
                source_partition_data, source_panel_col, missing_partition_cols
            )
            df_with_idx = df_with_idx.join(external_partitions, on="panels", how="left")
            has_missing_partition = df_with_idx.select(
                pl.any_horizontal(pl.col(missing_partition_cols).is_null()).any()
            ).item()
            if has_missing_partition:
                raise ValueError(
                    "partition_data is missing partition values for one or more "
                    "prediction panels."
                )

        summary_tables: dict[str, pl.DataFrame] = {}

        for req in requests:
            partition_type = req.partition_type
            if not isinstance(partition_type, PartitionType):
                partition_type = PartitionType(partition_type)

            partitioned_df = _apply_wtp_partition(df_with_idx, req)
            if "_partition_order" in partitioned_df.columns:
                partitioned_df = partitioned_df.sort("_partition_order")
            try:
                target_idx = self.results.model.case_varnames.index(req.alt_var)
            except ValueError:
                raise ValueError(
                    f"Alternative-specific variable '{req.alt_var}' not found in "
                    "model specification."
                )
            target_label = self.results.model.variable_label(req.alt_var)
            partition_label = self.results.model.variable_label(req.demographic_var)
            selected_class_probs = None
            if se == "none":
                selected_class_probs = self._class_probs_for_wtp(class_probabilities)
            summary_rows = []

            for partition_name, subset_df in partitioned_df.group_by(
                "Partition", maintain_order=True
            ):
                subset_panel_indices = jnp.array(
                    subset_df["panel_idx"].to_numpy(), dtype=jnp.int32
                )
                subset_panel_weights = jnp.asarray(
                    self.panel_weights[onp.asarray(subset_panel_indices)]
                )

                if se == "delta":
                    mean_wtp, se_val = self.results._apply_delta_method(
                        self._compute_subset_mean_wtp,
                        self.results.flat_params,
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        subset_panel_weights=subset_panel_weights,
                        dems=self.predict_data.dems,
                        num_panels=self.predict_data.num_panels,
                    )
                    se_float = float(se_val)
                elif se == "bootstrap":
                    mean_wtp = self._compute_subset_mean_wtp(
                        self.results.flat_params,
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        subset_panel_weights=subset_panel_weights,
                        dems=self.predict_data.dems,
                        num_panels=self.predict_data.num_panels,
                    )
                    se_val = self.results._parametric_bootstrap_se(
                        self._compute_subset_mean_wtp,
                        self.results.flat_params,
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        subset_panel_weights=subset_panel_weights,
                        dems=self.predict_data.dems,
                        num_panels=self.predict_data.num_panels,
                        draws=bootstrap_draws,
                        seed=bootstrap_seed,
                    )
                    se_float = float(se_val)
                else:
                    if selected_class_probs is None:
                        raise ValueError("Class probabilities were not available.")
                    mean_wtp = self._compute_subset_mean_wtp_from_class_probs(
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        subset_panel_weights=subset_panel_weights,
                        class_probs=selected_class_probs,
                    )
                    se_float = float("nan")

                summary_rows.append(
                    {
                        "variable": req.alt_var,
                        "label": target_label,
                        "partition_variable": req.demographic_var,
                        "partition_label": partition_label,
                        req.demographic_var: str(_partition_label(partition_name)),
                        "Mean_Marginal_WTP": float(mean_wtp),
                        "Standard_Error": se_float,
                        "Class_Probabilities": class_probabilities,
                        "SE_Method": se,
                        "Panel_Count": subset_df.height,
                        "Effective_Panel_Weight": float(
                            onp.asarray(subset_panel_weights).sum()
                        ),
                    }
                )

            res_df = pl.DataFrame(summary_rows)
            partition_desc = (
                "dummy-coded categorical"
                if req.dummy_vars is not None
                else partition_type.value
            )
            title = (
                f"Marginal WTP for {target_label} by "
                f"{partition_label} ({partition_desc})"
            )
            summary_tables[title] = res_df
            if show:
                log_or_print(
                    logger,
                    "%s",
                    format_wtp_table(
                        title,
                        res_df,
                        req.demographic_var,
                        partition_label,
                        num_decimals,
                    ),
                )

        return summary_tables

    def tradeoff(
        self,
        *wtp_requests: WTPRequest | Iterable[WTPRequest],
        **kwargs: Any,
    ) -> dict[str, pl.DataFrame]:
        """Alias for :meth:`compute_wtp` with more neutral terminology."""
        return self.compute_wtp(*wtp_requests, **kwargs)

    def wtp_by_class(self, target: str | None = None) -> pl.DataFrame:
        """Return class-specific WTP/tradeoff ratios.

        Parameters
        ----------
        target : str | None, optional
            Optional target variable to filter.  By default, all non-numeraire
            alternative-specific variables are returned.

        Returns
        -------
        pl.DataFrame
            Class-specific ratios ``beta_target / -beta_numeraire`` with raw
            variable names, display labels, and denominator diagnostics.
        """
        numeraire_idx = getattr(self.results.model, "numeraire_idx", None)
        if numeraire_idx is None:
            raise ValueError("A numeraire must be defined to compute WTP.")
        structural_betas = self.results.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required.")

        denominator = -structural_betas[numeraire_idx, :]
        rows = []
        for var_idx, variable in enumerate(self.results.model.case_varnames):
            if var_idx == numeraire_idx:
                continue
            if target is not None and variable != target:
                continue
            ratios = structural_betas[var_idx, :] / denominator
            for class_idx in range(self.results.model.num_classes):
                rows.append(
                    {
                        "variable": variable,
                        "label": self.results.model.variable_label(variable),
                        "denominator": self.results.model.numeraire,
                        "denominator_label": self.results.model.variable_label(
                            str(self.results.model.numeraire)
                        ),
                        "class": class_idx,
                        "tradeoff": float(ratios[class_idx]),
                        "denominator_value": float(denominator[class_idx]),
                    }
                )
        return pl.DataFrame(rows)

    def denominator_diagnostics(self) -> pl.DataFrame:
        """Return denominator diagnostics for WTP/tradeoff ratios."""
        numeraire_idx = getattr(self.results.model, "numeraire_idx", None)
        if numeraire_idx is None:
            raise ValueError("A numeraire must be defined to compute diagnostics.")
        structural_betas = self.results.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required.")
        denominator = -structural_betas[numeraire_idx, :]
        return pl.DataFrame(
            {
                "class": list(range(self.results.model.num_classes)),
                "denominator": [self.results.model.numeraire]
                * self.results.model.num_classes,
                "denominator_label": [
                    self.results.model.variable_label(str(self.results.model.numeraire))
                ]
                * self.results.model.num_classes,
                "denominator_value": onp.asarray(denominator),
                "abs_denominator": onp.asarray(jnp.abs(denominator)),
                "min_abs_floor": [self.results._param_packing.numeraire_min_abs]
                * self.results.model.num_classes,
            }
        )

    def _class_probs_for_wtp(
        self,
        class_probabilities: Literal["stored", "prior", "posterior"],
    ) -> Float64[Array, "panels classes"]:
        """Return class probabilities for WTP point estimates."""
        if class_probabilities in {"stored", "posterior"}:
            if self.class_probs_by_panel is None:
                raise ValueError("Prediction does not contain class probabilities.")
            return self.class_probs_by_panel

        if self.predict_data.num_panels is None:
            raise ValueError("Panel identifiers are required to compute WTP.")
        if (
            self.results.em_res.thetas is not None
            and self.predict_data.dems is not None
        ):
            return self.results._get_class_probs(
                self.results.em_res.thetas,
                self.predict_data.dems,
                self.predict_data.num_panels,
            )
        shares = self.results.em_res.shares
        if shares is None:
            raise ValueError("Class shares are required.")
        return jnp.repeat(shares[None, :], self.predict_data.num_panels, axis=0)

    def _compute_subset_mean_wtp_from_class_probs(
        self,
        target_idx: int,
        cost_idx: int,
        subset_panel_indices: Int[Array, "subset_panels"],
        subset_panel_weights: Float64[Array, "subset_panels"],
        class_probs: Float64[Array, "panels classes"],
    ) -> Float64[Array, ""]:
        """Compute a subset mean WTP using fixed class probabilities."""
        structural_betas = self.results.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required.")
        weights = subset_panel_weights / jnp.sum(subset_panel_weights)
        subset_shares = jnp.sum(
            class_probs[subset_panel_indices] * weights[:, None], axis=0
        )
        wtp_by_class = structural_betas[target_idx, :] / (
            -structural_betas[cost_idx, :]
        )
        return jnp.sum(subset_shares * wtp_by_class)

    def _compute_subset_mean_wtp(
        self,
        flat_params: Float64[Array, "all_params"],
        target_idx: int,
        cost_idx: int,
        subset_panel_indices: Int[Array, "subset_panels"],
        subset_panel_weights: Float64[Array, "subset_panels"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, ""]:
        """Evaluate panel-weighted subset WTP for delta/bootstrap inference."""
        latent_betas, thetas = self.results._unpack_params(flat_params)
        class_probs = self.results._get_class_probs(thetas, dems, num_panels)
        weights = subset_panel_weights / jnp.sum(subset_panel_weights)
        subset_shares = jnp.sum(
            class_probs[subset_panel_indices] * weights[:, None], axis=0
        )
        structural_betas = self.results._param_packing.to_structural(latent_betas)
        wtp_by_class = structural_betas[target_idx, :] / (
            -structural_betas[cost_idx, :]
        )
        return jnp.sum(subset_shares * wtp_by_class)


class CLPrediction(_PredictionBase):
    """Conditional-logit prediction with WTP and elasticity diagnostics."""

    def wtp(
        self,
        target: str | None = None,
        *,
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
    ) -> pl.DataFrame:
        """Return homogeneous WTP ratios with delta or parametric-bootstrap SEs."""
        if se not in {"delta", "bootstrap", "none"}:
            raise ValueError("se must be 'delta', 'bootstrap', or 'none'.")
        cost_idx = getattr(self.results.model, "numeraire_idx", None)
        if cost_idx is None:
            raise ValueError("A numeraire must be defined to compute WTP.")
        target_indices = [
            idx
            for idx, variable in enumerate(self.results.model.case_varnames)
            if idx != cost_idx and (target is None or variable == target)
        ]
        if target is not None and not target_indices:
            raise ValueError(
                f"Variable {target!r} was not found in the utility design."
            )

        def ratio_function(params: Array) -> Array:
            return params[jnp.asarray(target_indices)] / (-params[cost_idx])

        coefficients = jnp.asarray(self.results.coeff_)
        ratios = ratio_function(coefficients)
        if se == "none":
            standard_errors = jnp.full_like(ratios, jnp.nan)
        elif se == "delta":
            jacobian = jax.jacrev(ratio_function)(coefficients)
            standard_errors = jnp.sqrt(
                jnp.maximum(
                    jnp.einsum(
                        "ip,pq,iq->i",
                        jacobian,
                        self.results.cov_matrix,
                        jacobian,
                    ),
                    0.0,
                )
            )
        else:
            if bootstrap_draws < 2:
                raise ValueError("bootstrap_draws must be at least 2.")
            covariance = onp.asarray(self.results.cov_matrix, dtype=onp.float64)
            if not onp.all(onp.isfinite(covariance)):
                raise ValueError(
                    "A finite covariance matrix is required for bootstrap SEs."
                )
            eigenvalues, eigenvectors = onp.linalg.eigh(
                0.5 * (covariance + covariance.T)
            )
            tolerance = (
                onp.finfo(onp.float64).eps
                * max(1.0, float(onp.max(onp.abs(eigenvalues))))
                * covariance.shape[0]
            )
            if float(eigenvalues.min()) < -tolerance:
                raise ValueError("The covariance matrix is not positive semidefinite.")
            root = eigenvectors * onp.sqrt(onp.maximum(eigenvalues, 0.0))[None, :]
            rng = onp.random.default_rng(bootstrap_seed)
            draws = (
                onp.asarray(coefficients)
                + rng.standard_normal((bootstrap_draws, coefficients.size)) @ root.T
            )
            draw_ratios = jax.vmap(ratio_function)(jnp.asarray(draws))
            standard_errors = jnp.std(draw_ratios, axis=0, ddof=1)

        rows = []
        for output_idx, variable_idx in enumerate(target_indices):
            variable = self.results.model.case_varnames[variable_idx]
            rows.append(
                {
                    "variable": variable,
                    "label": self.results.model.variable_label(variable),
                    "denominator": self.results.model.numeraire,
                    "tradeoff": float(ratios[output_idx]),
                    "std_error": float(standard_errors[output_idx]),
                    "se_method": se,
                }
            )
        return pl.DataFrame(rows)

    def compute_wtp(self, target: str | None = None, **kwargs: Any) -> pl.DataFrame:
        """Alias for :meth:`wtp`."""
        return self.wtp(target, **kwargs)

    def wtp_by_class(self, target: str | None = None) -> pl.DataFrame:
        """Return WTP with a single homogeneous class label."""
        return self.wtp(target, se="none").with_columns(pl.lit(0).alias("class"))

    def denominator_diagnostics(self) -> pl.DataFrame:
        """Report the homogeneous WTP denominator and configured floor."""
        cost_idx = getattr(self.results.model, "numeraire_idx", None)
        if cost_idx is None:
            raise ValueError("A numeraire must be defined to compute diagnostics.")
        denominator = float(-self.results.coeff_[cost_idx])
        return pl.DataFrame(
            {
                "class": [0],
                "denominator": [self.results.model.numeraire],
                "denominator_value": [denominator],
                "abs_denominator": [abs(denominator)],
                "min_abs_floor": [self.results.model.numeraire_min_abs],
            }
        )


def resolve_panel_weights(
    panel_weights: str | Mapping[object, float] | Sequence[float] | None,
    panel_ids: onp.ndarray,
    raw_data: pl.DataFrame | None,
    panel_col: str,
) -> onp.ndarray:
    """Resolve user panel weights into unique-panel order."""
    if panel_weights is None:
        return onp.ones(len(panel_ids), dtype=onp.float64)
    if isinstance(panel_weights, str):
        if raw_data is None or panel_weights not in raw_data.columns:
            raise ValueError(
                f"Panel-weight column {panel_weights!r} was not found in prediction data."
            )
        grouped = raw_data.group_by(panel_col, maintain_order=True).agg(
            pl.col(panel_weights).n_unique().alias("_n_unique"),
            pl.col(panel_weights).first().alias("_weight"),
        )
        if grouped["_n_unique"].max() != 1:
            raise ValueError("Panel weights must be constant within each panel.")
        weight_map = dict(zip(grouped[panel_col], grouped["_weight"]))
        values = onp.asarray([weight_map[panel] for panel in panel_ids], dtype=float)
    elif isinstance(panel_weights, Mapping):
        missing = [panel for panel in panel_ids if panel not in panel_weights]
        if missing:
            raise ValueError(f"panel_weights is missing panel IDs: {missing[:5]}.")
        values = onp.asarray([panel_weights[panel] for panel in panel_ids], dtype=float)
    else:
        values = onp.asarray(panel_weights, dtype=float)
    if values.shape != (len(panel_ids),):
        raise ValueError("panel_weights must contain one value per prediction panel.")
    if not onp.all(onp.isfinite(values)) or onp.any(values < 0.0):
        raise ValueError("panel_weights must be finite and nonnegative.")
    if not onp.any(values > 0.0):
        raise ValueError("At least one panel weight must be positive.")
    return values
