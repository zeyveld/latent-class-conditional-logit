"""Render tidy result frames for terminal and LaTeX presentation."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from pylatexenc.latex2text import LatexNodes2Text
from tabulate import tabulate


def format_cl_coefficients(
    table: pl.DataFrame,
    header: tuple[str, str, str],
    num_decimals: int,
) -> str:
    """Render a conditional-logit coefficient frame.

    Parameters
    ----------
    table : pl.DataFrame
        Tidy table containing ``label``, ``estimate``, and ``std_error``.
    header : tuple[str, str, str]
        Display headers.
    num_decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Combined LaTeX and terminal rendering.
    """
    converter = LatexNodes2Text(math_mode="text")
    body_rows: list[str] = []
    preview_rows: list[tuple[str, str, str]] = []
    for row in table.iter_rows(named=True):
        label = str(row["label"])
        estimate = float(row["estimate"])
        std_error = float(row["std_error"])
        body_rows.append(
            f"{label} & {estimate:.{num_decimals}f} & {std_error:.{num_decimals}f} \\\\"
        )
        preview_rows.append(
            (
                converter.latex_to_text(label),
                f"{estimate:.{num_decimals}f}",
                f"{std_error:.{num_decimals}f}",
            )
        )
    return _combine_renderings(body_rows, preview_rows, header, num_decimals)


def format_lcl_beta_summary(
    table: pl.DataFrame,
    header: tuple[str, str, str],
    num_decimals: int,
) -> str:
    """Render a latent-class population coefficient summary."""
    converter = LatexNodes2Text(math_mode="text")
    body_rows: list[str] = []
    preview_rows: list[tuple[str, str, str]] = []
    for row in table.iter_rows(named=True):
        label = str(row["label"])
        mean = float(row["mean"])
        mean_se = float(row["mean_se"])
        std = float(row["sd"])
        std_se = float(row["sd_se"])
        body_rows.extend(
            [
                f"{label} & {mean:.{num_decimals}f} & {std:.{num_decimals}f} \\\\",
                f" & ({mean_se:.{num_decimals}f}) & ({std_se:.{num_decimals}f}) \\\\",
            ]
        )
        preview_rows.extend(
            [
                (
                    converter.latex_to_text(label),
                    f"{mean:.{num_decimals}f}",
                    f"{std:.{num_decimals}f}",
                ),
                (
                    "",
                    f"({mean_se:.{num_decimals}f})",
                    f"({std_se:.{num_decimals}f})",
                ),
            ]
        )
    return _combine_renderings(body_rows, preview_rows, header, num_decimals)


def format_wtp_table(
    title: str,
    table: pl.DataFrame,
    partition_column: str,
    partition_label: str,
    num_decimals: int,
) -> str:
    """Render a tidy willingness-to-pay summary frame."""
    converter = LatexNodes2Text(math_mode="text")
    header = (partition_label, "Mean marginal WTP")
    body_rows: list[str] = []
    preview_rows: list[tuple[str, str]] = []
    for row in table.iter_rows(named=True):
        partition = str(row[partition_column])
        mean_wtp = float(row["Mean_Marginal_WTP"])
        std_error = float(row["Standard_Error"])
        body_rows.extend(
            [
                f"{_escape_latex(partition)} & {mean_wtp:.{num_decimals}f} \\\\",
                f" & ({std_error:.{num_decimals}f}) \\\\",
            ]
        )
        preview_rows.extend(
            [
                (
                    converter.latex_to_text(partition),
                    f"{mean_wtp:.{num_decimals}f}",
                ),
                ("", f"({std_error:.{num_decimals}f})"),
            ]
        )
    rendered = _combine_renderings(
        body_rows,
        preview_rows,
        header,
        num_decimals,
        escape_header=True,
    )
    return f"\n{title}\n{rendered}"


def _combine_renderings(
    body_rows: list[str],
    preview_rows: Sequence[tuple[str, ...]],
    header: tuple[str, ...],
    num_decimals: int,
    *,
    escape_header: bool = False,
) -> str:
    """Combine LaTeX rows and a terminal preview into one string."""
    converter = LatexNodes2Text(math_mode="text")
    latex_header = (
        [_escape_latex(value) for value in header] if escape_header else list(header)
    )
    latex = "\n".join(
        [r"\toprule", " & ".join(latex_header) + r" \\", r"\midrule", "%"]
        + body_rows
        + ["%", r"\bottomrule "]
    )
    preview = tabulate(
        preview_rows,
        headers=[converter.latex_to_text(value) for value in header],
        tablefmt="simple_outline",
        floatfmt=f".{num_decimals}f",
    )
    return f"\n--- LaTeX Output ---\n\n{latex}\n\n--- Table preview ---\n\n{preview}"


def _escape_latex(value: object) -> str:
    """Escape plain text for insertion into a LaTeX table."""
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
    return "".join(replacements.get(char, char) for char in str(value))
