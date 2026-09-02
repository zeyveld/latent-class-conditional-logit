"""Validation for aligned choice-model inputs."""

from collections.abc import Sequence

import numpy as onp
import polars as pl

from lcl._struct import ParsedData


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")


def validate_raw_choice_frame(
    df: pl.DataFrame, *, alts_col: str, cases_col: str, panels_col: str
) -> None:
    """Validate raw identifiers and reserved names."""
    reserved = {"_seq_alts", "_seq_cases", "_seq_panels"}
    collisions = sorted(reserved.intersection(df.columns))
    intercept_collisions = [col for col in df.columns if col.lower() == "intercept"]
    if collisions or intercept_collisions:
        names = [*collisions, *intercept_collisions]
        raise ValueError(
            "Input columns collide with names reserved by the encoder: "
            f"{sorted(set(names))}. Rename these columns before fitting."
        )
    id_cols = list(dict.fromkeys([panels_col, cases_col, alts_col]))
    null_counts = df.select([pl.col(col).is_null().sum() for col in id_cols]).row(0)
    null_ids = [col for col, count in zip(id_cols, null_counts) if count]
    if null_ids:
        raise ValueError(f"Identifier columns cannot contain null values: {null_ids}")
    duplicates = df.group_by(id_cols).len().filter(pl.col("len") > 1)
    if duplicates.height:
        sample = duplicates.select(id_cols).head(5).to_dicts()
        raise ValueError(
            "Each (panel, case, alternative) row must be unique. "
            f"Duplicate keys include: {sample}"
        )


def validate_external_demographics(
    choice_df: pl.DataFrame, dems_df: pl.DataFrame, *, panels_col: str
) -> None:
    """Validate a separate panel-level demographics table."""
    _require_columns(dems_df, [panels_col])
    if dems_df[panels_col].null_count():
        raise ValueError("dems_data panel identifiers cannot contain null values.")
    duplicate_panels = (
        dems_df.group_by(panels_col).len().filter(pl.col("len") > 1).select(panels_col)
    )
    if duplicate_panels.height:
        sample = duplicate_panels.head(5)[panels_col].to_list()
        raise ValueError(
            "dems_data must contain exactly one row per panel. "
            f"Duplicate panels include: {sample}"
        )
    collisions = sorted((set(choice_df.columns) & set(dems_df.columns)) - {panels_col})
    if collisions:
        raise ValueError(
            "dems_data columns cannot duplicate columns in the choice data because "
            "their precedence would be ambiguous. Conflicting columns: "
            f"{collisions}"
        )


def validate_parsed_data(parsed: ParsedData) -> None:
    """Validate aligned arrays at the ParsedData assembly seam."""
    X = onp.asarray(parsed.X, dtype=onp.float64)
    if X.ndim != 2 or X.shape[1] != len(parsed.case_varnames):
        raise ValueError("Encoded utility design has an invalid shape.")
    if not onp.all(onp.isfinite(X)):
        raise ValueError("Encoded utility design contains non-finite values.")
    num_rows = X.shape[0]
    ids = {
        "cases": onp.asarray(parsed.cases),
        "alternatives": onp.asarray(parsed.alts),
        "panels": onp.asarray(parsed.panels),
    }
    if any(values.shape != (num_rows,) for values in ids.values()):
        raise ValueError("Encoded identifiers must align one-to-one with utility rows.")
    if parsed.dems is not None:
        dems = onp.asarray(parsed.dems, dtype=onp.float64)
        if dems.ndim != 2 or dems.shape[1] != len(parsed.dem_varnames or []):
            raise ValueError("Encoded demographics have an invalid shape.")
        if not onp.all(onp.isfinite(dems)):
            raise ValueError("Encoded demographics contain non-finite values.")
    if parsed.y is None:
        return
    y = onp.asarray(parsed.y)
    if y.shape != (num_rows,):
        raise ValueError("Choice indicators must align one-to-one with utility rows.")
    y_bool = y.astype(bool)
    cases = ids["cases"].astype(onp.int64, copy=False)
    num_cases = int(cases.max()) + 1 if cases.size else 0
    choices_per_case = onp.bincount(cases, weights=y_bool, minlength=num_cases)
    if not onp.all(choices_per_case == 1):
        raise ValueError(
            "Every choice situation must have exactly one chosen alternative."
        )
    chosen_X = X[y_bool]
    unchosen = ~y_bool
    differenced = X[unchosen] - chosen_X[cases[unchosen]]
    rank = int(onp.linalg.matrix_rank(differenced)) if differenced.size else 0
    if rank < X.shape[1]:
        raise ValueError(
            "The chosen-differenced utility design is rank deficient "
            f"(rank {rank} for {X.shape[1]} columns). Conditional-logit "
            "coefficients—and utility levels such as consumer surplus—are not "
            "identified. Remove collinear columns or use K-1 alternative constants."
        )
