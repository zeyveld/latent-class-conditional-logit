"""Diagnostics containers for fitted latent-class models."""

from __future__ import annotations

import polars as pl
from tabulate import tabulate


class LCLDiagnostics:
    """Structured diagnostics for a fitted latent-class model.

    Parameters
    ----------
    frame : pl.DataFrame
        Diagnostic checks with at least ``section``, ``check``, ``value``,
        ``status``, and ``message`` columns.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        """Store diagnostic checks."""
        self._frame = frame

    def to_frame(self) -> pl.DataFrame:
        """Return diagnostics as a Polars DataFrame."""
        return self._frame

    def print(self) -> None:
        """Print a compact diagnostics table."""
        rows = self._frame.select(["section", "check", "status", "value", "message"])
        print(tabulate(rows.iter_rows(), headers=rows.columns, tablefmt="simple"))

    def __repr__(self) -> str:
        """Return a compact textual representation."""
        n_warn = self._frame.filter(pl.col("status") != "ok").height
        return f"LCLDiagnostics(checks={self._frame.height}, warnings={n_warn})"
