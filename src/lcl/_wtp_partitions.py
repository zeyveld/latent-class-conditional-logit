"""Partition construction for willingness-to-pay summaries."""

from collections.abc import Iterable, Sequence

import numpy as onp
import polars as pl

from lcl._encoding import _coerce_frame
from lcl.options import PartitionType, WTPRequest


def _flatten_wtp_requests(
    items: Iterable[WTPRequest | Iterable[WTPRequest]],
) -> list[WTPRequest]:
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
    dummy_vars = req.dummy_vars
    if dummy_vars is None:
        raise ValueError("Dummy-coded WTP partitions require dummy_vars.")
    missing = [col for col in dummy_vars if col not in df.columns]
    if missing:
        raise ValueError(f"WTP dummy partition columns were not found: {missing}")
    dummy_values = df.select(dummy_vars).to_numpy()
    if not onp.all((dummy_values == 0) | (dummy_values == 1)):
        raise ValueError("WTP dummy partition columns must contain only 0/1 values.")
    active = dummy_values.astype(bool)
    if onp.any(active.sum(axis=1) > 1):
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
        values = df[req.demographic_var].cast(pl.Float64).to_numpy()
        if not onp.all(onp.isfinite(values)):
            raise ValueError("WTP quintile variables must be finite and numeric.")
        raw_breaks = onp.quantile(values, [0.2, 0.4, 0.6, 0.8])
        breaks = onp.unique(
            raw_breaks[(raw_breaks > values.min()) & (raw_breaks < values.max())]
        )
        group_index = onp.digitize(values, breaks, right=True)
        num_groups = int(group_index.max()) + 1
        if num_groups == 5:
            labels = [f"Q{idx + 1}" for idx in range(num_groups)]
        elif num_groups == 1:
            labels = ["All values"]
        else:
            labels = [
                f"Quantile group {idx + 1} of {num_groups}" for idx in range(num_groups)
            ]
        return df.with_columns(
            pl.Series("Partition", [labels[idx] for idx in group_index]),
            pl.Series("_partition_order", group_index),
        )
    elif partition_type == PartitionType.CUSTOM_BREAKS:
        if not isinstance(req.bins, list):
            raise ValueError(
                "Custom WTP partitions require bins as a list of breakpoints."
            )
        group_expr = demo_col.cut(req.bins)
    else:
        raise ValueError(f"Unsupported partition type: {partition_type}")
    partitioned = df.with_columns(group_expr.alias("Partition"))
    if partition_type == PartitionType.CUSTOM_BREAKS:
        if not isinstance(req.bins, list):
            raise ValueError(
                "Custom WTP partitions require bins as a list of breakpoints."
            )
        bin_order = onp.digitize(
            df[req.demographic_var].to_numpy(), onp.asarray(req.bins), right=True
        )
        partitioned = partitioned.with_columns(pl.Series("_partition_order", bin_order))
    return partitioned


def _partition_label(partition_name: object) -> object:
    return partition_name[0] if isinstance(partition_name, tuple) else partition_name
