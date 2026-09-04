"""Out-of-sample prediction, elasticities, and willingness-to-pay (WTP) analysis."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jaxtyping import Array, Float64, Int

from lcl._case_utils import _to_structural_betas
from lcl._elasticities import compute_elasticities, elasticity_design_derivative
from lcl._em_alg_steps import _compute_conditional_class_probs
from lcl._logging import log_or_print
from lcl._prediction_inference import (
    aggregate_elasticities as _aggregate_elasticities_fn,
)
from lcl._prediction_inference import (
    build_within_case_pairs,
    market_shares as _market_shares_fn,
    mean_surplus as _mean_surplus_fn,
    mean_surplus_change as _mean_surplus_change_fn,
)
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
        past_diff_unchosen_chosen: Any | None = None,
        past_data: Data | None = None,
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
        # Retained so willingness to pay computed from Bayesian-updated class
        # membership can be differentiated through that update rather than
        # treating the posterior as a fixed constant.
        self.past_diff_unchosen_chosen = past_diff_unchosen_chosen
        self.past_data = past_data
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

    def _design_kwargs(self) -> dict[str, Any]:
        """Return the design arrays every differentiable counterfactual needs."""
        data = self.predict_data
        if data.panels is None or data.num_panels is None:
            raise ValueError("Panel identifiers are required for prediction inference.")
        return {
            "results": self.results,
            "X": data.X,
            "cases": data.cases,
            "panels": data.panels,
            "dems": data.dems,
            "num_cases": data.num_cases,
            "num_panels": data.num_panels,
        }

    def _row_panel_weights(self) -> jnp.ndarray:
        """Return each design row's panel weight."""
        data = self.predict_data
        if data.panels is None:
            raise ValueError("Panel identifiers are required for prediction inference.")
        return jnp.asarray(self.panel_weights)[data.panels]

    def _case_panel_weights(self) -> jnp.ndarray:
        """Return each choice situation's panel weight."""
        data = self.predict_data
        if data.panels_of_cases is None:
            raise ValueError("Panel identifiers are required for prediction inference.")
        return jnp.asarray(self.panel_weights)[data.panels_of_cases]

    def _quantity_se(
        self,
        func: Any,
        se: str,
        *,
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
        **kwargs: Any,
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """Evaluate a counterfactual quantity with the requested uncertainty.

        The delta method linearizes, which is exact for a mildly nonlinear
        aggregate and optimistic for a sharply curved one.  Market shares and
        consumer surplus can be sharply curved when a class's numeraire
        coefficient is weakly identified, so the parametric bootstrap -- drawing
        in the unconstrained parameterization, then transforming -- is offered
        alongside it.
        """
        if se not in {"delta", "bootstrap", "none"}:
            raise ValueError("se must be 'delta', 'bootstrap', or 'none'.")
        flat_params = self.results.flat_params
        if se == "delta":
            value, standard_error = self.results._apply_delta_method(
                func, flat_params, **kwargs
            )
            return onp.asarray(value), onp.asarray(standard_error)
        value = onp.asarray(func(flat_params, **kwargs))
        if se == "none":
            return value, onp.full(value.shape, onp.nan)
        standard_error = self.results._parametric_bootstrap_se(
            func,
            flat_params,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            **kwargs,
        )
        return value, onp.asarray(standard_error)

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

    def market_shares(
        self,
        *,
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
    ) -> pl.DataFrame:
        """Return panel-weighted predicted market shares by alternative.

        Parameters
        ----------
        se : {"delta", "bootstrap", "none"}, default="delta"
            Uncertainty method.  A market share is a smooth function of the
            coefficients, so the delta method costs one Jacobian; the bootstrap
            captures curvature that matters when a class's numeraire coefficient
            is weakly identified.
        bootstrap_draws : int, default=500
            Number of asymptotic draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible seed for those draws.

        Returns
        -------
        pl.DataFrame
            One row per alternative with its share and the standard error of that
            share, which is ``NaN`` when ``se="none"``.
        """
        data = self.predict_data
        if data.panels is None:
            raise ValueError("Panel identifiers are required for market shares.")
        row_weights = self.panel_weights[onp.asarray(data.panels)]
        first_case_rows = onp.asarray(data.cases) != onp.roll(
            onp.asarray(data.cases), 1
        )
        first_case_rows[0] = True
        denominator = float(row_weights[first_case_rows].sum())

        alt_codes = onp.asarray(data.alts)
        num_alts = int(alt_codes.max()) + 1 if alt_codes.size else 0
        # Alternative codes are contiguous and globally consistent, so the first
        # row carrying each code names it.
        labels: list[Any] = [None] * num_alts
        for code, label in zip(alt_codes, self.original_alts):
            if labels[code] is None:
                labels[code] = label

        shares, standard_errors = self._quantity_se(
            _market_shares_fn,
            se,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            alt_codes=data.alts,
            num_alts=num_alts,
            row_weights=jnp.asarray(row_weights),
            weight_total=denominator,
            **self._design_kwargs(),
        )

        frame = pl.DataFrame(
            {
                "alts": labels,
                "market_share": onp.asarray(shares, dtype=onp.float64),
                "std_error": onp.asarray(standard_errors, dtype=onp.float64),
            }
        ).sort("alts")
        return frame

    def mean_surplus(
        self,
        *,
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
    ) -> pl.DataFrame:
        """Return the panel-weighted mean consumer surplus per choice situation.

        Parameters
        ----------
        se : {"delta", "bootstrap", "none"}, default="delta"
            Uncertainty method.  Surplus in money units divides by the numeraire
            coefficient, so when that coefficient is weakly identified the ratio
            is sharply curved and the bootstrap is the more honest summary.
        bootstrap_draws : int, default=500
            Number of asymptotic draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible seed for those draws.

        Returns
        -------
        pl.DataFrame
            One row holding the mean surplus, its standard error, and the units
            (money when a numeraire is defined, otherwise utils).
        """
        data = self.predict_data
        if data.panels_of_cases is None:
            raise ValueError("Panel identifiers are required for surplus inference.")
        kwargs = self._design_kwargs()
        kwargs.pop("panels")
        kwargs["panels_of_cases"] = data.panels_of_cases
        value, standard_error = self._quantity_se(
            _mean_surplus_fn,
            se,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            case_weights=self._case_panel_weights(),
            **kwargs,
        )
        return pl.DataFrame(
            {
                "mean_surplus": [float(value)],
                "std_error": [float(standard_error)],
                "surplus_units": [self.surplus_units],
            }
        )

    def mean_surplus_change(
        self,
        counterfactual: "_PredictionBase",
        *,
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
    ) -> pl.DataFrame:
        """Return the mean counterfactual-minus-baseline surplus with inference.

        Both scenarios are evaluated at the same parameter vector, so the
        difference keeps the correlation between the two surplus levels instead
        of treating them as independent estimates.

        Parameters
        ----------
        counterfactual : :class:`_PredictionBase`
            Prediction under the counterfactual scenario.  It must cover the same
            choice situations, in the same order, as this baseline.
        se : {"delta", "bootstrap", "none"}, default="delta"
            Uncertainty method.
        bootstrap_draws : int, default=500
            Number of asymptotic draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible seed for those draws.

        Returns
        -------
        pl.DataFrame
            One row holding the mean change, its standard error, and the units.
        """
        if self.surplus_units != counterfactual.surplus_units:
            raise ValueError("Baseline and counterfactual surplus units do not match.")
        if self.results is not counterfactual.results:
            raise ValueError(
                "Baseline and counterfactual predictions must come from the same "
                "fitted model."
            )
        if self.predict_data.num_cases != counterfactual.predict_data.num_cases:
            raise ValueError(
                "Baseline and counterfactual predictions must contain identical "
                "choice situations."
            )

        def scenario(prediction: "_PredictionBase") -> dict[str, Any]:
            """Build the surplus keyword arguments for one scenario."""
            kwargs = prediction._design_kwargs()
            kwargs.pop("panels")
            panels_of_cases = prediction.predict_data.panels_of_cases
            if panels_of_cases is None:
                raise ValueError("Panel identifiers are required for surplus.")
            kwargs["panels_of_cases"] = panels_of_cases
            return kwargs

        value, standard_error = self._quantity_se(
            _mean_surplus_change_fn,
            se,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            baseline=scenario(self),
            counterfactual=scenario(counterfactual),
            case_weights=self._case_panel_weights(),
        )
        return pl.DataFrame(
            {
                "mean_surplus_change": [float(value)],
                "std_error": [float(standard_error)],
                "surplus_units": [self.surplus_units],
            }
        )

    def surplus_change(self, counterfactual: "_PredictionBase") -> pl.DataFrame:
        """Return identified counterfactual-minus-baseline surplus changes.

        Per-case changes are point estimates.  Use :meth:`mean_surplus_change`
        for the aggregate, which is the quantity a welfare analysis reports and
        the one that carries a standard error.
        """
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

    def aggregate_elasticities(
        self,
        vars: str | Iterable[str],
        *,
        se: Literal["delta", "bootstrap", "none"] = "delta",
        bootstrap_draws: int = 500,
        bootstrap_seed: int = 0,
    ) -> pl.DataFrame:
        """Aggregate elasticities over demand, with delta-method standard errors.

        Each alternative pair's elasticity is averaged over choice situations
        using predicted demand as the weight, exactly as in the point-estimate
        path, and the whole aggregate is differentiated with respect to the
        parameters.

        Parameters
        ----------
        vars : str | Iterable[str]
            Continuous variable(s) whose elasticities to aggregate.
        se : {"delta", "bootstrap", "none"}, default="delta"
            Uncertainty method.
        bootstrap_draws : int, default=500
            Number of asymptotic draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible seed for those draws.

        Returns
        -------
        pl.DataFrame
            One row per ``(alts, target_alts)`` pair with the aggregate elasticity
            of each requested variable and its standard error, which is ``NaN``
            when ``se="none"``.
        """
        variables = [vars] if isinstance(vars, str) else list(vars)
        if not variables:
            raise ValueError("At least one elasticity variable is required.")
        data = self.predict_data
        if data.panels is None:
            raise ValueError("Panel identifiers are required for aggregation.")

        alt_codes = onp.asarray(data.alts)
        num_alts = int(alt_codes.max()) + 1 if alt_codes.size else 0
        labels: list[Any] = [None] * num_alts
        for code, label in zip(alt_codes, self.original_alts):
            if labels[code] is None:
                labels[code] = label

        affected, target = build_within_case_pairs(onp.asarray(data.cases))
        group_codes = alt_codes[affected] * num_alts + alt_codes[target]
        num_groups = num_alts * num_alts
        present = onp.zeros(num_groups, dtype=bool)
        present[group_codes] = True

        design_kwargs = self._design_kwargs()
        row_weights = self._row_panel_weights()
        columns: dict[str, onp.ndarray] = {}
        for variable in variables:
            raw_values, design_derivative = self._elasticity_design_derivative(variable)
            call_kwargs = dict(
                design_derivative=design_derivative,
                raw_values=raw_values,
                affected=jnp.asarray(affected, dtype=jnp.int32),
                target=jnp.asarray(target, dtype=jnp.int32),
                group_codes=jnp.asarray(group_codes, dtype=jnp.int32),
                num_groups=num_groups,
                row_weights=row_weights,
                **design_kwargs,
            )
            value, standard_error = self._quantity_se(
                _aggregate_elasticities_fn,
                se,
                bootstrap_draws=bootstrap_draws,
                bootstrap_seed=bootstrap_seed,
                **call_kwargs,
            )
            columns[f"elasticity_{variable}"] = value
            columns[f"elasticity_{variable}_se"] = standard_error

        frame = pl.DataFrame(
            {
                "alts": [labels[code // num_alts] for code in range(num_groups)],
                "target_alts": [labels[code % num_alts] for code in range(num_groups)],
                "_present": present,
                **{
                    name: onp.asarray(values, dtype=onp.float64)
                    for name, values in columns.items()
                },
            }
        ).filter(pl.col("_present")).drop("_present")
        return frame.sort(["alts", "target_alts"])


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
            When the prediction used ``past_choices``, both methods
            differentiate through the Bayesian class update, so the reported
            uncertainty reflects the same posterior as the point estimate.  Note
            that partitions built from the data (quintiles, custom breaks) are
            treated as fixed, so standard errors are conditional on the realized
            partition and on the demographic design.
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
        use_posterior = (
            class_probabilities in {"stored", "posterior"}
            and self.class_probabilities_source == "posterior"
        )
        if (
            se in {"delta", "bootstrap"}
            and use_posterior
            and (self.past_data is None or self.past_diff_unchosen_chosen is None)
        ):
            raise ValueError(
                "Posterior-updated WTP inference needs the past-choice design "
                "that produced the posterior. Recreate the prediction with "
                "predict(..., past_choices=...)."
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
            # Differentiating through the Bayes update keeps the reported
            # uncertainty consistent with the point estimate: the posterior is a
            # smooth function of the same coefficients, not a fixed constant.
            posterior_kwargs = (
                {
                    "past_diff_unchosen_chosen": self.past_diff_unchosen_chosen,
                    "past_data": self.past_data,
                }
                if use_posterior
                else {"past_diff_unchosen_chosen": None, "past_data": None}
            )
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
                        **posterior_kwargs,
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
                        **posterior_kwargs,
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
                        **posterior_kwargs,
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
        past_diff_unchosen_chosen: Any | None = None,
        past_data: Data | None = None,
    ) -> Float64[Array, ""]:
        """Evaluate panel-weighted subset WTP for delta/bootstrap inference.

        When a past-choice design is supplied the class probabilities are the
        Bayesian posterior implied by ``flat_params``, so differentiating this
        function propagates uncertainty through the update itself.
        """
        latent_betas, thetas = self.results._unpack_params(flat_params)
        if past_data is None:
            class_probs = self.results._get_class_probs(thetas, dems, num_panels)
        else:
            structural = self.results._param_packing.to_structural(latent_betas)
            prior = self.results._get_class_probs(
                thetas, past_data.dems, past_data.num_panels
            )
            if past_diff_unchosen_chosen is None:
                raise ValueError(
                    "A past-choice design matrix is required alongside past_data."
                )
            class_probs, _ = _compute_conditional_class_probs(
                structural_betas=structural,
                thetas=thetas if past_data.dems is not None else None,
                shares=jnp.mean(prior, axis=0),
                diff_unchosen_chosen=past_diff_unchosen_chosen,
                data=past_data,
            )
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
        """Return homogeneous WTP ratios with delta or parametric-bootstrap SEs.

        Both methods work in the unconstrained parameterization and apply the
        softplus transform inside the target function.  Drawing structural
        coefficients directly would put mass on a positive numeraire coefficient
        -- a region the constraint excludes -- and the resulting ratios have no
        finite variance to summarize.

        Parameters
        ----------
        target : str | None, optional
            Restrict the table to one non-numeraire variable.
        se : {"delta", "bootstrap", "none"}, default="delta"
            Standard-error method.
        bootstrap_draws : int, default=500
            Number of asymptotic parameter draws for ``se="bootstrap"``.
        bootstrap_seed : int, default=0
            Reproducible seed for those draws.

        Returns
        -------
        pl.DataFrame
            One row per variable with its tradeoff ratio and standard error.
        """
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
        selector = jnp.asarray(target_indices)

        def ratio_function(latent: Array) -> Array:
            """Map latent coefficients to structural WTP ratios."""
            structural = _to_structural_betas(
                latent,
                self.results.model.numeraire_idx,
                self.results.model.numeraire_min_abs,
            )
            return structural[selector] / (-structural[cost_idx])

        latent = jnp.asarray(self.results.flat_params)
        ratios = ratio_function(latent)
        if se == "none":
            standard_errors = jnp.full_like(ratios, jnp.nan)
        elif se == "delta":
            _, standard_errors = self.results._apply_delta_method(
                ratio_function, latent
            )
        else:
            standard_errors = self.results._parametric_bootstrap_se(
                ratio_function, latent, draws=bootstrap_draws, seed=bootstrap_seed
            )

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
