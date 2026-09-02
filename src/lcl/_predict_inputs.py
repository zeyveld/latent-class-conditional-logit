"""Prediction-input alignment and historical-choice parsing."""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax.typing import ArrayLike

from lcl._encoding import _coerce_frame
from lcl.options import PastChoicesData
from lcl._struct import ParsedData
from lcl._validation import validate_parsed_data


def _panel_constant_columns(data: object, panel_col: str) -> pl.DataFrame:
    df = _coerce_frame(data)
    candidate_cols = [col for col in df.columns if col != panel_col]
    if not candidate_cols:
        return (
            df.select(panel_col)
            .unique(maintain_order=True)
            .rename({panel_col: "panels"})
        )
    maxima = (
        df.group_by(panel_col)
        .agg([pl.col(col).n_unique().alias(col) for col in candidate_cols])
        .select(pl.exclude(panel_col).max())
        .row(0)
    )
    constant_cols = [
        col for col, max_unique in zip(candidate_cols, maxima) if max_unique <= 1
    ]
    return (
        df.select([panel_col, *constant_cols])
        .unique(subset=[panel_col], maintain_order=True)
        .rename({panel_col: "panels"})
    )


def _prediction_partition_data(
    data: object, dems_data: object | None, panel_col: str
) -> pl.DataFrame:
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
    case_varnames: list[str]
    dem_varnames: list[str] | None

    def _transform_data(
        self,
        data: object,
        dems_data: object | None = None,
        require_choice: bool = False,
    ) -> ParsedData: ...


def _parse_past_choices(
    model: _PastChoicesParser,
    past_choices: object,
    past_choices_dems_data: object | None,
) -> ParsedData:
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
            dem_panel_ids=past_choices.dem_panel_ids,
            y=past_choices.y,
            case_varnames=model.case_varnames,
            dem_varnames=model.dem_varnames,
        )
    return model._transform_data(
        past_choices,
        dems_data=past_choices_dems_data,
        require_choice=True,
    )


def _validate_past_choice_panels(
    parsed_past: ParsedData, parsed_predict: ParsedData
) -> None:
    past_panels = onp.unique(onp.asarray(parsed_past.original_panels))
    predict_panels = onp.unique(onp.asarray(parsed_predict.original_panels))
    if past_panels.shape == predict_panels.shape and onp.array_equal(
        past_panels, predict_panels
    ):
        return

    def sample(values: onp.ndarray) -> str:
        suffix = ", ..." if values.shape[0] > 5 else ""
        return f"{values[:5].tolist()}{suffix}"

    missing = onp.setdiff1d(predict_panels, past_panels)
    extra = onp.setdiff1d(past_panels, predict_panels)
    raise ValueError(
        "past_choices must contain exactly the panels present in the "
        "prediction data, because posterior class probabilities are matched "
        "to prediction panels by sorted panel ID. "
        f"Panels missing from past_choices: {sample(missing)}; "
        f"panels absent from the prediction data: {sample(extra)}."
    )


def _parsed_prediction_arrays(
    X: ArrayLike,
    dems: ArrayLike | None,
    alts: ArrayLike,
    cases: ArrayLike,
    panels: ArrayLike,
    dem_panel_ids: ArrayLike | None,
    case_varnames: list[str],
    dem_varnames: list[str] | None,
    y: ArrayLike | None = None,
) -> ParsedData:
    X_np = onp.asarray(X)
    alts_np, cases_np, panels_np = map(onp.asarray, (alts, cases, panels))
    if X_np.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    num_rows = X_np.shape[0]
    if any(arr.shape != (num_rows,) for arr in (alts_np, cases_np, panels_np)):
        raise ValueError(
            "alts, cases, and panels must align one-to-one with rows of X."
        )
    if X_np.shape[1] != len(case_varnames):
        raise ValueError(
            f"X has {X_np.shape[1]} columns; expected {len(case_varnames)}."
        )
    if y is not None and onp.asarray(y).shape != (num_rows,):
        raise ValueError("y must align one-to-one with rows of X.")
    order = onp.lexsort((alts_np, cases_np, panels_np))
    X_sorted = X_np[order]
    alts_sorted, cases_sorted, panels_sorted = (
        alts_np[order],
        cases_np[order],
        panels_np[order],
    )
    y_sorted = None if y is None else onp.asarray(y)[order]
    panel_ids, panel_seq = onp.unique(panels_sorted, return_inverse=True)
    _, alt_seq = onp.unique(alts_sorted, return_inverse=True)
    case_seq = onp.empty_like(cases_sorted, dtype=onp.uint32)
    case_lookup: dict[tuple[object, object], int] = {}
    for idx, key in enumerate(zip(panels_sorted.tolist(), cases_sorted.tolist())):
        if key not in case_lookup:
            case_lookup[key] = len(case_lookup)
        case_seq[idx] = case_lookup[key]
    dems_array = None
    if dems is not None:
        dems_np = onp.asarray(dems)
        if dems_np.ndim != 2:
            raise ValueError("dems must be a two-dimensional matrix.")
        if dems_np.shape[0] != panel_ids.shape[0]:
            raise ValueError("dems must have one row per unique panel.")
        expected_dem_vars = len(dem_varnames or [])
        if dems_np.shape[1] != expected_dem_vars:
            raise ValueError(
                f"dems has {dems_np.shape[1]} columns; expected {expected_dem_vars}."
            )
        if dem_panel_ids is not None:
            dem_panel_ids_np = onp.asarray(dem_panel_ids)
            if dem_panel_ids_np.shape != (dems_np.shape[0],):
                raise ValueError(
                    "dem_panel_ids must align one-to-one with rows of dems."
                )
            if onp.unique(dem_panel_ids_np).shape[0] != dem_panel_ids_np.shape[0]:
                raise ValueError("dem_panel_ids cannot contain duplicates.")
            lookup = {
                panel_id: idx for idx, panel_id in enumerate(dem_panel_ids_np.tolist())
            }
            panel_set = set(panel_ids.tolist())
            missing = [
                panel_id for panel_id in panel_ids.tolist() if panel_id not in lookup
            ]
            extra = [panel_id for panel_id in lookup if panel_id not in panel_set]
            if missing or extra:
                raise ValueError(
                    "dem_panel_ids must match the unique prediction panels exactly. "
                    f"Missing: {missing}; extra: {extra}."
                )
            dems_np = dems_np[[lookup[panel_id] for panel_id in panel_ids.tolist()]]
        dems_array = jnp.asarray(dems_np, dtype="float64")
    elif dem_panel_ids is not None:
        raise ValueError("dem_panel_ids can only be supplied with dems.")
    parsed = ParsedData(
        X=jnp.asarray(X_sorted, dtype="float64"),
        dems=dems_array,
        y=None if y_sorted is None else jnp.asarray(y_sorted, dtype="bool"),
        cases=jnp.asarray(case_seq, dtype="uint32"),
        panels=jnp.asarray(panel_seq, dtype="uint32"),
        alts=jnp.asarray(alt_seq, dtype="uint32"),
        case_varnames=case_varnames,
        dem_varnames=dem_varnames,
        original_alts=alts_sorted,
        original_cases=cases_sorted,
        original_panels=panels_sorted,
    )
    validate_parsed_data(parsed)
    return parsed
