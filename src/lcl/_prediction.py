"""Out-of-sample prediction, elasticities, and willingness-to-pay (WTP) analysis."""

from typing import Any, Generator, Iterable

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array
from jax.ops import segment_sum
from jaxtyping import Float64

from lcl._case_utils import _to_structural_betas
from lcl._struct import Data, PartitionType, WTPRequest


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
        results,
        class_probs_by_panel: Array | None = None,
    ) -> None:
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
            assert data.panels is not None
            S_ic = self.class_probs_by_panel[data.panels]  # (N, C)
        else:
            betas = self.results.coeff_[:, None]  # (K, 1)
            S_ic = jnp.ones((data.X.shape[0], 1))  # (N, 1)

        num_classes = betas.shape[1]

        eV = jnp.exp(jnp.clip(data.X @ betas, a_max=700.0))  # (N, C)
        sum_eV = segment_sum(eV, data.cases, num_segments=data.num_cases)
        P_ij_c = eV / sum_eV[data.cases]  # (N, C)
        P_ij = jnp.sum(S_ic * P_ij_c, axis=1)  # (N,)

        base_dict_j = {
            "cases": onp.array(data.cases),
            "alts": onp.array(data.alts),
            "P_j": onp.array(jnp.clip(P_ij, a_min=1e-250)),
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

    def compute_wtp(self, *wtp_requests: WTPRequest | Iterable[WTPRequest]) -> None:
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
            quintiles, categorical, or custom breaks).

        Raises
        ------
        ValueError
            If the parent model was not estimated with a specified numeraire constraint.
        """

        # We rely on the explicitly tracked numeraire index from _pre_fit
        if getattr(self.results.model, "numeraire_idx", None) is None:
            raise ValueError("A numeraire must be defined to compute WTP.")

        cost_idx = self.results.model.numeraire_idx

        def _flatten(items) -> Generator[Any, Any, None]:
            for item in items:
                if isinstance(item, Iterable):
                    yield from item
                else:
                    yield item

        panel_ids_in_order = onp.unique(self.predict_data.panels)
        panel_idx_map = pl.DataFrame(
            {
                "panels": panel_ids_in_order,
                "panel_idx": onp.array(jnp.arange(self.predict_data.num_panels)),
            }
        )

        df_with_idx = self.wtp_alt_vars_by_panel.join(panel_idx_map, on="panels")

        if (
            self.predict_data.dems is not None
            and self.results.model.dem_varnames is not None
        ):
            dems_df = pl.DataFrame(
                onp.array(self.predict_data.dems),
                schema=self.results.model.dem_varnames,
            ).with_columns(pl.Series("panels", panel_ids_in_order, dtype=pl.UInt32))

            df_with_idx = df_with_idx.join(dems_df, on="panels")

        for req in _flatten(wtp_requests):
            demo_col = pl.col(req.demographic_var)

            if req.partition_type == PartitionType.CATEGORICAL:
                group_expr = demo_col
            elif req.partition_type == PartitionType.QUINTILES:
                group_expr = demo_col.qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            elif req.partition_type == PartitionType.CUSTOM_BREAKS:
                group_expr = demo_col.cut(req.bins)

            partitioned_df = df_with_idx.with_columns(Partition=group_expr)
            target_idx = self.results.model.case_varnames.index(req.alt_var)
            summary_rows = []

            for partition_name, subset_df in partitioned_df.group_by(
                "Partition", maintain_order=True
            ):
                subset_panel_indices = jnp.array(subset_df["panel_idx"].to_numpy())

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
                        req.demographic_var: str(partition_name[0]),
                        "Mean_Marginal_WTP": float(mean_wtp),
                        "Standard_Error": float(se_val),
                    }
                )

            res_df = pl.DataFrame(summary_rows)
            title = f"Marginal WTP for {req.alt_var} by {req.demographic_var} ({req.partition_type.value})"
            print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
            with pl.Config(tbl_rows=20, tbl_formatting="MARKDOWN", float_precision=4):
                print(res_df)

    def _compute_subset_mean_wtp(
        self,
        flat_params: Array,
        target_idx: int,
        cost_idx: int,
        subset_panel_indices: Array,
        dems: Array | None,
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


# EOF
