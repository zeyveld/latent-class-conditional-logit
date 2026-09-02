"""Pure formatting inputs shared by results facades."""

from typing import Any

import numpy as onp
import polars as pl


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


def _model_variable_label(model: Any, variable: str) -> str:
    """Return a fitted model's display label for a variable."""
    labeler = getattr(model, "variable_label", None)
    return str(labeler(variable)) if callable(labeler) else variable
