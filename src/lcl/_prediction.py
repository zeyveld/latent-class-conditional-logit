"""Out-of-sample prediction, elasticities, and willingness-to-pay (WTP) analysis."""

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array
from jaxtyping import Float64, Int

from lcl._case_utils import _to_structural_betas
from lcl._encoding import _coerce_frame
from lcl._kernels import _choice_probabilities_and_logsum
from lcl._struct import Data, PartitionType, WTPRequest

logger = logging.getLogger(__name__)


def _flatten_wtp_requests(
    items: Iterable[WTPRequest | Iterable[WTPRequest]],
) -> list[WTPRequest]:
    """Return a flat list from a variadic mix of requests and request iterables."""
    requests: list[WTPRequest] = []
    for item in items:
        if isinstance(item, WTPRequest):
            requests.append(item)
        else:
            requests.extend(item)
    return requests


def _partition_columns(requests: Sequence[WTPRequest]) -> list[str]:
    """Return unique column names needed to build all WTP partitions."""
    columns: list[str] = []
    for req in requests:
        requested = (
            req.dummy_vars if req.dummy_vars is not None else [req.demographic_var]
        )
        for col in requested:
            if col not in columns:
                columns.append(col)
    return columns


def _coerce_partition_data(
    partition_data: object,
    panel_col: str,
    partition_cols: Sequence[str],
) -> pl.DataFrame:
    """Return one panel-level partition row per panel.

    Parameters
    ----------
    partition_data : object
        Tabular panel-level or long-format data containing the requested partition
        variables.
    panel_col : str
        Column in ``partition_data`` identifying decision-makers.
    partition_cols : Sequence[str]
        Partition columns to keep and validate.

    Returns
    -------
    pl.DataFrame
        A panel-level frame keyed by ``"panels"``.

    Raises
    ------
    ValueError
        If required columns are absent or a requested partition value is not constant
        within panel.
    """
    df = _coerce_frame(partition_data)
    required_cols = list(dict.fromkeys([panel_col, *partition_cols]))
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"partition_data is missing required columns: {missing}")

    unique_df = df.select(required_cols).unique(maintain_order=True)
    duplicate_panels = (
        unique_df.group_by(panel_col).len().filter(pl.col("len") > 1).select(panel_col)
    )
    if duplicate_panels.height:
        sample = duplicate_panels.head(5)[panel_col].to_list()
        raise ValueError(
            "partition_data must have one unique value per panel for each requested "
            f"partition column. Conflicting panels include: {sample}"
        )

    if panel_col != "panels":
        unique_df = unique_df.rename({panel_col: "panels"})
    return unique_df


def _apply_dummy_partition(df: pl.DataFrame, req: WTPRequest) -> pl.DataFrame:
    """Attach a categorical partition column from a one-hot dummy bundle."""
    dummy_vars = req.dummy_vars
    if dummy_vars is None:
        raise ValueError("Dummy-coded WTP partitions require dummy_vars.")

    missing = [col for col in dummy_vars if col not in df.columns]
    if missing:
        raise ValueError(f"WTP dummy partition columns were not found: {missing}")

    dummy_values = df.select(dummy_vars).to_numpy()
    valid_dummy_values = (dummy_values == 0) | (dummy_values == 1)
    if not onp.all(valid_dummy_values):
        raise ValueError("WTP dummy partition columns must contain only 0/1 values.")

    active = dummy_values.astype(bool)
    active_counts = active.sum(axis=1)
    if onp.any(active_counts > 1):
        raise ValueError(
            "WTP dummy partition columns must be mutually exclusive within panel."
        )

    dummy_labels = req.dummy_labels if req.dummy_labels is not None else dummy_vars
    partition = onp.full(df.height, req.base_category, dtype=object)
    partition_order = onp.zeros(df.height, dtype=onp.int64)
    for idx, label in enumerate(dummy_labels):
        mask = active[:, idx]
        partition[mask] = label
        partition_order[mask] = idx + 1

    return df.with_columns(
        pl.Series("Partition", partition),
        pl.Series("_partition_order", partition_order),
    )


def _apply_wtp_partition(df: pl.DataFrame, req: WTPRequest) -> pl.DataFrame:
    """Attach a ``Partition`` column according to a WTP request."""
    if req.dummy_vars is not None:
        return _apply_dummy_partition(df, req)

    if req.demographic_var not in df.columns:
        raise ValueError(
            f"WTP partition variable '{req.demographic_var}' was not found. "
            "Pass partition_data=... to compute_wtp for variables outside the "
            "fitted demographic specification."
        )

    partition_type = req.partition_type
    if not isinstance(partition_type, PartitionType):
        partition_type = PartitionType(partition_type)

    demo_col = pl.col(req.demographic_var)
    if partition_type == PartitionType.CATEGORICAL:
        group_expr = demo_col
    elif partition_type == PartitionType.QUINTILES:
        group_expr = demo_col.qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    elif partition_type == PartitionType.CUSTOM_BREAKS:
        if not isinstance(req.bins, list):
            raise ValueError(
                "Custom WTP partitions require bins as a list of breakpoints."
            )
        group_expr = demo_col.cut(req.bins)
    else:
        raise ValueError(f"Unsupported partition type: {partition_type}")

    return df.with_columns(group_expr.alias("Partition"))


def _partition_label(partition_name: object) -> object:
    """Extract a scalar label from Polars group-by keys."""
    if isinstance(partition_name, tuple):
        return partition_name[0]
    return partition_name


class LCLPrediction:
    """Container for counterfactual inference, consumer surplus, and willingness-to-pay (WTP).

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
    """

    def __init__(
        self,
        predicted_probs_df: pl.DataFrame,
        surplus_df: pl.DataFrame,
        wtp_alt_vars_by_panel_df: pl.DataFrame,
        predict_data: Data,
        results: Any,
        class_probs_by_panel: Float64[Array, "panels classes"] | None = None,
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
        """
        self.predicted_probs = predicted_probs_df
        self.surplus = surplus_df
        self.wtp_alt_vars_by_panel = wtp_alt_vars_by_panel_df
        self.predict_data = predict_data
        self.results = results
        self.class_probs_by_panel = class_probs_by_panel

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
        if isinstance(vars, str):
            vars = [vars]

        data = self.predict_data
        is_lc = hasattr(self.results, "em_res")

        if is_lc:
            betas = self.results.em_res.structural_betas  # (K, C)
            if self.class_probs_by_panel is None:
                raise ValueError(
                    "class_probs_by_panel must be available to compute LC elasticities."
                )
            if data.panels is None:
                raise ValueError("Panel identifiers are required for LC elasticities.")
            S_ic = self.class_probs_by_panel[data.panels]  # (N, C)
        else:
            betas = self.results.coeff_[:, None]  # (K, 1)
            S_ic = jnp.ones((data.X.shape[0], 1))  # (N, 1)

        num_classes = betas.shape[1]

        P_ij_c, _ = _choice_probabilities_and_logsum(
            data.X, betas, data.cases, data.num_cases
        )
        P_ij = jnp.sum(S_ic * P_ij_c, axis=1)  # (N,)

        base_dict_j = {
            "cases": onp.array(data.cases),
            "alts": onp.array(data.alts),
            "P_j": onp.array(jnp.maximum(P_ij, 1e-250)),
        }
        base_dict_k = {
            "cases": onp.array(data.cases),
            "target_alts": onp.array(data.alts),
        }

        for c in range(num_classes):
            base_dict_j[f"P_jc_{c}"] = onp.array(P_ij_c[:, c])
            base_dict_k[f"P_kc_{c}"] = onp.array(P_ij_c[:, c])

        df_j = pl.DataFrame(base_dict_j)
        df_k_base = pl.DataFrame(base_dict_k)

        res_dfs = []

        for var in vars:
            try:
                var_idx = self.results.model.case_varnames.index(var)
            except ValueError:
                raise ValueError(f"Variable '{var}' not found in model specification.")

            X_v = data.X[:, var_idx]
            beta_v = betas[var_idx, :]

            # A_{jc} = S_{ic} * beta_{vc} * P_{ij|c}
            A_jc = S_ic * beta_v[None, :] * P_ij_c
            U_j = jnp.sum(A_jc, axis=1)

            df_j_var = df_j.with_columns(pl.Series("U_j", onp.array(U_j)))
            for c in range(num_classes):
                df_j_var = df_j_var.with_columns(
                    pl.Series(f"A_jc_{c}", onp.array(A_jc[:, c]))
                )

            df_k_var = df_k_base.with_columns(pl.Series("X_k", onp.array(X_v)))
            cross_df = df_j_var.join(df_k_var, on="cases", how="inner")

            # V_{jk} = sum_c A_{jc} P_{kc}
            cross_df = cross_df.with_columns(
                V_jk=pl.sum_horizontal(
                    [
                        pl.col(f"A_jc_{c}") * pl.col(f"P_kc_{c}")
                        for c in range(num_classes)
                    ]
                )
            )

            # D_{jk} = U_j * (j == k) - V_{jk}
            cross_df = cross_df.with_columns(
                is_own=pl.col("alts") == pl.col("target_alts")
            ).with_columns(
                D_jk=pl.when(pl.col("is_own"))
                .then(pl.col("U_j") - pl.col("V_jk"))
                .otherwise(-pl.col("V_jk"))
            )

            # E_{jk} = D_{jk} * X_k / P_j
            elas_name = f"elasticity_{var}"
            cross_df = cross_df.with_columns(
                (pl.col("D_jk") * pl.col("X_k") / pl.col("P_j")).alias(elas_name)
            )

            res_dfs.append(cross_df.select(["cases", "alts", "target_alts", elas_name]))

        final_df = res_dfs[0]
        for i in range(1, len(res_dfs)):
            final_df = final_df.join(res_dfs[i], on=["cases", "alts", "target_alts"])

        if data.panels is not None:
            panel_df = pl.DataFrame(
                {"cases": onp.array(data.cases), "panels": onp.array(data.panels)}
            ).unique()
            final_df = panel_df.join(final_df, on="cases")
            final_cols = ["panels", "cases", "alts", "target_alts"] + [
                f"elasticity_{v}" for v in vars
            ]
        else:
            final_cols = ["cases", "alts", "target_alts"] + [
                f"elasticity_{v}" for v in vars
            ]

        return final_df.select(final_cols).sort(["cases", "alts", "target_alts"])

    def compute_wtp(
        self,
        *wtp_requests: WTPRequest | Iterable[WTPRequest],
        partition_data: object | None = None,
        panel_col: str = "panels",
    ) -> dict[str, pl.DataFrame]:
        """Compute the Marginal Willingness-to-Pay (WTP) across demographic partitions.

        Evaluates the ratio of the target parameter to the negative cost parameter
        (marginal utility of income) for dynamically defined subsets of decision-makers.
        Outputs formatted Markdown summary tables to the console, including analytical
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

        Returns
        -------
        dict[str, pl.DataFrame]
            Summary tables keyed by their printed titles.

        Raises
        ------
        ValueError
            If the parent model was not estimated with a specified numeraire constraint.
        """

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
            if partition_data is None:
                raise ValueError(
                    "WTP partition columns were not found in the fitted/prediction "
                    "demographics: "
                    f"{missing_partition_cols}. Pass partition_data=... for "
                    "panel-level grouping variables outside the model specification."
                )
            external_partitions = _coerce_partition_data(
                partition_data, panel_col, missing_partition_cols
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
            summary_rows = []

            for partition_name, subset_df in partitioned_df.group_by(
                "Partition", maintain_order=True
            ):
                subset_panel_indices = jnp.array(
                    subset_df["panel_idx"].to_numpy(), dtype=jnp.int32
                )

                mean_wtp, se_val = self.results._apply_delta_method(
                    self._compute_subset_mean_wtp,
                    self.results.flat_params,
                    target_idx=target_idx,
                    cost_idx=cost_idx,
                    subset_panel_indices=subset_panel_indices,
                    dems=self.predict_data.dems,
                    num_panels=self.predict_data.num_panels,
                )

                summary_rows.append(
                    {
                        req.demographic_var: str(_partition_label(partition_name)),
                        "Mean_Marginal_WTP": float(mean_wtp),
                        "Standard_Error": float(se_val),
                    }
                )

            res_df = pl.DataFrame(summary_rows)
            partition_desc = (
                "dummy-coded categorical"
                if req.dummy_vars is not None
                else partition_type.value
            )
            title = (
                f"Marginal WTP for {req.alt_var} by "
                f"{req.demographic_var} ({partition_desc})"
            )
            summary_tables[title] = res_df
            with pl.Config(tbl_rows=20, tbl_formatting="MARKDOWN", float_precision=4):
                logger.info("%s\n%s", title, res_df)

        return summary_tables

    def _compute_subset_mean_wtp(
        self,
        flat_params: Float64[Array, "all_params"],
        target_idx: int,
        cost_idx: int,
        subset_panel_indices: Int[Array, "subset_panels"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, ""]:
        """Internal objective function for Delta Method variance evaluation.

        Computes the expected WTP for a specific demographic subset by weighting the
        class-specific WTP ratios by the subset's average posterior class membership
        probabilities.

        Parameters
        ----------
        flat_params : Float64[Array, "all_params"]
            The flattened vector of unconstrained structural and demographic parameters.
        target_idx : int
            The column index of the target alternative-specific variable.
        cost_idx : int
            The column index of the numeraire (cost) variable.
        subset_panel_indices : Array
            The row indices corresponding to the decision-makers in the current partition.
        dems : Array | None
            The matrix of demographic variables for the full sample.
        num_panels : int
            The total number of unique decision-makers in the sample.

        Returns
        -------
        Float64[Array, ""]
            The scalar expected WTP for the defined demographic subset.
        """
        latent_betas, thetas = self.results._unpack_params(flat_params)
        class_probs = self.results._get_class_probs(thetas, dems, num_panels)

        subset_class_probs = class_probs[subset_panel_indices]
        subset_shares = jnp.mean(subset_class_probs, axis=0)

        structural_betas = _to_structural_betas(
            latent_betas, self.results.model.numeraire_idx
        )

        # WTP = Beta_target / (-Beta_cost) ensures positive estimates
        # given that Beta_cost is mathematically constrained to be negative.
        wtp_by_class = structural_betas[target_idx, :] / (
            -structural_betas[cost_idx, :]
        )
        return jnp.sum(subset_shares * wtp_by_class)
