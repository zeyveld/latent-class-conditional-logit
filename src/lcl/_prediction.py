"""Out-of-sample prediction, elasticities, and willingness-to-pay (WTP) analysis."""

import logging
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jaxtyping import Array, Float64, Int
from pylatexenc.latex2text import LatexNodes2Text
from tabulate import tabulate

from lcl._case_utils import _to_structural_betas
from lcl._encoding import _coerce_frame
from lcl._kernels import _choice_probabilities_and_logsum
from lcl._logging import log_or_print
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
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
            for req in item:
                if not isinstance(req, WTPRequest):
                    raise TypeError(
                        "compute_wtp expects WTPRequest objects or iterables of "
                        f"WTPRequest objects, not {type(req).__name__}."
                    )
                requests.append(req)
        else:
            hint = (
                " Did you pass the dictionary returned by an earlier compute_wtp call?"
                if isinstance(item, dict)
                else ""
            )
            raise TypeError(
                "compute_wtp expects WTPRequest objects or iterables of WTPRequest "
                f"objects, not {type(item).__name__}.{hint}"
            )
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


def _escape_latex(value: object) -> str:
    """Escape plain-text labels for insertion into a LaTeX table."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = str(value)
    return "".join(replacements.get(char, char) for char in text)


def _format_wtp_table(
    title: str,
    res_df: pl.DataFrame,
    demographic_var: str,
    num_decimals: int,
) -> str:
    """Format a WTP summary as LaTeX plus a terminal preview.

    Parameters
    ----------
    title : str
        Human-readable table title.
    res_df : pl.DataFrame
        WTP summary with a demographic column, ``Mean_Marginal_WTP``, and
        ``Standard_Error``.
    demographic_var : str
        Name of the partitioning variable column in ``res_df``.
    num_decimals : int
        Number of decimal places for estimates and standard errors.

    Returns
    -------
    str
        A formatted string containing the title, LaTeX table, and terminal preview.
    """
    converter = LatexNodes2Text(math_mode="text")
    header = (demographic_var, "Mean marginal WTP")
    header_clean = [converter.latex_to_text(col) for col in header]
    latex_header = [_escape_latex(col) for col in header]

    body_rows: list[str] = []
    data_clean: list[tuple[str, str]] = []
    for row in res_df.iter_rows(named=True):
        partition = str(row[demographic_var])
        mean_wtp = float(row["Mean_Marginal_WTP"])
        se_wtp = float(row["Standard_Error"])
        body_rows.append(
            f"{_escape_latex(partition)} & {mean_wtp:.{num_decimals}f} \\\\"
        )
        body_rows.append(f" & ({se_wtp:.{num_decimals}f}) \\\\")
        data_clean.append(
            (converter.latex_to_text(partition), f"{mean_wtp:.{num_decimals}f}")
        )
        data_clean.append(("", f"({se_wtp:.{num_decimals}f})"))

    latex_string = "\n".join(
        [r"\toprule", " & ".join(latex_header) + r" \\", r"\midrule", "%"]
        + body_rows
        + ["%", r"\bottomrule "]
    )
    table_preview = tabulate(
        data_clean,
        headers=header_clean,
        tablefmt="simple_outline",
        floatfmt=f".{num_decimals}f",
    )
    return (
        f"{title}\n\n"
        f"--- LaTeX Output ---\n\n{latex_string}\n\n"
        f"--- Table preview ---\n\n{table_preview}"
    )


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
        self.surplus = surplus_df
        self.wtp_alt_vars_by_panel = wtp_alt_vars_by_panel_df
        self.predict_data = predict_data
        self.results = results
        self.class_probs_by_panel = class_probs_by_panel
        self.class_probabilities_source = class_probabilities_source
        self.partition_data = partition_data_df

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
        num_decimals: int = 4,
        class_probabilities: Literal["stored", "prior", "posterior"] = "stored",
        se: Literal["delta", "none"] = "delta",
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
        se : {"delta", "none"}, default="delta"
            Standard-error method.  Delta-method standard errors are available for
            prior class probabilities.  Posterior-updated WTP through
            ``past_choices`` requires differentiating through the Bayesian class
            update and is therefore refused unless ``se="none"``.

        Returns
        -------
        dict[str, pl.DataFrame]
            Summary tables keyed by their printed titles.

        Raises
        ------
        ValueError
            If the parent model was not estimated with a specified numeraire constraint.
        """
        if se not in {"delta", "none"}:
            raise ValueError("se must be either 'delta' or 'none'.")
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
            se == "delta"
            and class_probabilities in {"stored", "posterior"}
            and self.class_probabilities_source == "posterior"
        ):
            raise NotImplementedError(
                "Delta-method WTP after past_choices requires differentiating "
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

                if se == "delta":
                    mean_wtp, se_val = self.results._apply_delta_method(
                        self._compute_subset_mean_wtp,
                        self.results.flat_params,
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        dems=self.predict_data.dems,
                        num_panels=self.predict_data.num_panels,
                    )
                    se_float = float(se_val)
                else:
                    if selected_class_probs is None:
                        raise ValueError("Class probabilities were not available.")
                    mean_wtp = self._compute_subset_mean_wtp_from_class_probs(
                        target_idx=target_idx,
                        cost_idx=cost_idx,
                        subset_panel_indices=subset_panel_indices,
                        class_probs=selected_class_probs,
                    )
                    se_float = float("nan")

                summary_rows.append(
                    {
                        req.demographic_var: str(_partition_label(partition_name)),
                        "Mean_Marginal_WTP": float(mean_wtp),
                        "Standard_Error": se_float,
                        "Class_Probabilities": class_probabilities,
                        "SE_Method": se,
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
            log_or_print(
                logger,
                "%s",
                _format_wtp_table(title, res_df, req.demographic_var, num_decimals),
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
            Class-specific ratios ``beta_target / -beta_numeraire`` and the
            denominator used for each class.
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
                        "denominator": self.results.model.numeraire,
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
                "denominator_value": onp.asarray(denominator),
                "abs_denominator": onp.asarray(jnp.abs(denominator)),
                "min_abs_floor": [
                    getattr(self.results.model, "numeraire_min_abs", 1e-5)
                ]
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
        class_probs: Float64[Array, "panels classes"],
    ) -> Float64[Array, ""]:
        """Compute a subset mean WTP using fixed class probabilities."""
        structural_betas = self.results.em_res.structural_betas
        if structural_betas is None:
            raise ValueError("Structural betas are required.")
        subset_shares = jnp.mean(class_probs[subset_panel_indices], axis=0)
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
            latent_betas,
            self.results.model.numeraire_idx,
            getattr(self.results.model, "numeraire_min_abs", 1e-5),
        )

        # WTP = Beta_target / (-Beta_cost) ensures positive estimates
        # given that Beta_cost is mathematically constrained to be negative.
        wtp_by_class = structural_betas[target_idx, :] / (
            -structural_betas[cost_idx, :]
        )
        return jnp.sum(subset_shares * wtp_by_class)
