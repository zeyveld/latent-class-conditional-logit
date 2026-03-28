"""Out-of-sample predictions and counterfactual inference."""

from typing import Any, Generator, Iterable

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array
from jaxtyping import Float64

from lcl._case_utils import _to_structural_betas
from lcl._struct import Data, PartitionType, WTPRequest


class LCLPrediction:
    """Container for counterfactual inference, consumer surplus, and willingness-to-pay (WTP).

    Provides methods to analyze decision-maker behavior under new choice sets or
    policy changes. Utilizes the Delta Method to compute rigorous standard errors
    for non-linear combinations of parameters (e.g., marginal WTP).

    Attributes
    ----------
    predicted_probs : pl.DataFrame
        DataFrame of out-of-sample choice probabilities for each alternative.
    surplus : pl.DataFrame
        DataFrame of expected consumer surplus (inclusive value) per choice situation.
    wtp_alt_vars_by_panel : pl.DataFrame
        DataFrame of expected marginal WTP for each alternative-specific characteristic,
        weighted by the decision-maker's posterior class membership probabilities.
    predict_data : :class:`~lcl._struct.Data`
        The parsed design matrices corresponding to the counterfactual scenarios.
    results : :class:`~lcl._results.LCLResults`
        Reference to the parent estimation results (required for Delta Method covariance).
    """

    def __init__(
        self,
        predicted_probs_df: pl.DataFrame,
        surplus_df: pl.DataFrame,
        wtp_alt_vars_by_panel_df: pl.DataFrame,
        predict_data: Data,
        results,
    ) -> None:
        self.predicted_probs = predicted_probs_df
        self.surplus = surplus_df
        self.wtp_alt_vars_by_panel = wtp_alt_vars_by_panel_df
        self.predict_data = predict_data
        self.results = results  # Link to the parent LCLResults instance

    def compute_wtp(self, *wtp_requests: WTPRequest | Iterable[WTPRequest]) -> None:
        """Compute the Marginal Willingness-to-Pay (WTP) across demographic partitions.

        Employs the Delta Method via JAX's forward/reverse-mode autodiff to calculate
        analytical standard errors for the ratio of structural parameters
        (:math:`\\beta_{target} / \\beta_{cost}`).

        Parameters
        ----------
        *wtp_requests : :class:`~lcl._struct.WTPRequest`
            One or more configuration objects specifying the target variable,
            the demographic partitioning variable, and the binning strategy.
        """
        if self.results.model.numeraire is None:
            raise ValueError("A numeraire must be defined to compute WTP.")

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

        # Reconstruct demographics DataFrame and join it
        if (
            self.predict_data.dems is not None
            and self.results.model.dem_varnames is not None
        ):
            dems_df = pl.DataFrame(
                onp.array(self.predict_data.dems),
                schema=self.results.model.dem_varnames,
            ).with_columns(pl.Series("panels", panel_ids_in_order, dtype=pl.UInt32))

            df_with_idx = df_with_idx.join(dems_df, on="panels")

        cost_idx = self.results.model.case_varnames.index(self.results.model.numeraire)

        for req in _flatten(wtp_requests):
            # Dynamic binning via Polars
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

                # Pull the delta method and flat parameters dynamically from the Results object
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
        """Internal objective function for Delta Method variance evaluation."""
        latent_betas, thetas = self.results._unpack_params(flat_params)
        class_probs = self.results._get_class_probs(thetas, dems, num_panels)

        subset_class_probs = class_probs[subset_panel_indices]
        subset_shares = jnp.mean(subset_class_probs, axis=0)

        # Get structural betas
        if self.results.model.numeraire:
            numeraire_idx = self.results.model.case_varnames.index(
                self.results.model.numeraire
            )
        else:
            numeraire_idx = None
        structural_betas = _to_structural_betas(latent_betas, numeraire_idx)

        wtp_by_class = structural_betas[target_idx, :] / structural_betas[cost_idx, :]
        return jnp.sum(subset_shares * wtp_by_class)


# EOF
