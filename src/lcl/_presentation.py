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
    note: str | None = None,
    disable_numparse: bool = False,
) -> str:
    """Combine LaTeX rows and a terminal preview into one string.

    ``disable_numparse`` keeps every preview column left aligned as text.  A
    class table can hold a column whose cells all happen to parse as numbers --
    the reference class's zeros, for instance -- and letting ``tabulate`` right
    align just that column breaks the visual alignment of the rest.
    """
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
        disable_numparse=disable_numparse,
    )
    rendered = (
        f"\n--- LaTeX Output ---\n\n{latex}\n\n--- Table preview ---\n\n{preview}"
    )
    if note is not None:
        rendered = f"{rendered}\n\n{note}"
    return rendered


def _pair_rows(
    row_label: str,
    estimates: Sequence[float | None],
    standard_errors: Sequence[float | None],
    num_decimals: int,
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Build one estimate row and its parenthesised standard-error row.

    The two-line cell is the same shape :func:`format_lcl_beta_summary` uses for
    the population moments, so a class-by-class table reads as a refinement of
    the aggregate one rather than as a differently formatted object.
    """

    def cell(value: float | None) -> str:
        """Format one coefficient, blanking a value that is not estimated."""
        if value is None:
            return ""
        if value != value:  # NaN
            return "--"
        return f"{value:.{num_decimals}f}"

    def se_cell(value: float | None) -> str:
        """Format one standard error inside parentheses."""
        if value is None:
            return ""
        if value != value:
            return "(--)"
        return f"({value:.{num_decimals}f})"

    estimate_cells = [cell(value) for value in estimates]
    se_cells = [se_cell(value) for value in standard_errors]
    latex = [" & ".join([_escape_latex(row_label), *estimate_cells]) + r" \\"]
    preview: list[tuple[str, ...]] = [(row_label, *estimate_cells)]
    # A row whose standard errors are all missing -- an unidentified covariance,
    # or the reference class of the membership model -- would otherwise print a
    # line of empty parentheses and double the height of the table for nothing.
    if any(cell_text not in {"", "(--)"} for cell_text in se_cells):
        latex.append(" & ".join(["", *se_cells]) + r" \\")
        preview.append(("", *se_cells))
    return latex, preview


def _class_labels(num_classes: int) -> list[str]:
    """Return column headers for a class-per-column layout."""
    return [f"Class {index + 1}" for index in range(num_classes)]


DENSE_LAYOUT_MIN_CLASSES = 9
"""Class count at which the printed tables transpose to one row per class.

Eight classes already push a variable-per-row table to nine columns, which is
about where a terminal table and a ``\\textwidth`` LaTeX tabular both stop
fitting.  Beyond it the class axis becomes the row axis, which grows downward
instead of sideways and stays readable at the 64 classes an attribute
non-attendance design implies.
"""


def format_class_coefficients(
    table: pl.DataFrame,
    num_classes: int,
    num_decimals: int,
    *,
    layout: str = "auto",
    shares: Sequence[float] | None = None,
    share_standard_errors: Sequence[float] | None = None,
) -> str:
    """Render class-specific utility coefficients.

    Parameters
    ----------
    table : pl.DataFrame
        Long-format frame from :meth:`~lcl.LCLResults.class_coefficients` with
        ``label``, ``class``, ``coefficient``, and ``std_error``.
    num_classes : int
        Number of latent classes.
    num_decimals : int
        Decimal places for coefficients and standard errors.
    layout : {"auto", "wide", "dense"}, default="auto"
        ``"wide"`` puts variables on rows and classes on columns; ``"dense"``
        transposes so each class is one row.  ``"auto"`` picks ``"dense"`` from
        :data:`DENSE_LAYOUT_MIN_CLASSES` classes upward.
    shares : Sequence[float] | None, optional
        Aggregate class shares, appended to the table when supplied.
    share_standard_errors : Sequence[float] | None, optional
        Standard errors of those shares.

    Returns
    -------
    str
        Combined LaTeX and terminal rendering.
    """
    resolved = _resolve_layout(layout, num_classes)
    ordered_variables = list(dict.fromkeys(table["variable"].to_list()))
    labels = {
        row["variable"]: str(row["label"]) for row in table.iter_rows(named=True)
    }
    coefficients = {
        (row["variable"], int(row["class"])): (
            float(row["coefficient"]),
            float(row["std_error"]),
        )
        for row in table.iter_rows(named=True)
    }
    latex_rows: list[str] = []
    preview_rows: list[tuple[str, ...]] = []

    if resolved == "wide":
        header = ("Variable", *_class_labels(num_classes))
        for variable in ordered_variables:
            cells = [
                coefficients.get((variable, index), (float("nan"), float("nan")))
                for index in range(num_classes)
            ]
            latex, preview = _pair_rows(
                labels.get(variable, variable),
                [value for value, _ in cells],
                [error for _, error in cells],
                num_decimals,
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)
        if shares is not None:
            latex_rows.append(r"\midrule")
            latex, preview = _pair_rows(
                "Class share",
                list(shares),
                list(share_standard_errors)
                if share_standard_errors is not None
                else [None] * num_classes,
                num_decimals,
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)
    else:
        share_column = shares is not None
        header = (
            "Class",
            *(("Share",) if share_column else ()),
            *(labels.get(variable, variable) for variable in ordered_variables),
        )
        for index in range(num_classes):
            cells = [
                coefficients.get((variable, index), (float("nan"), float("nan")))
                for variable in ordered_variables
            ]
            estimates: list[float | None] = []
            errors: list[float | None] = []
            if share_column:
                assert shares is not None
                estimates.append(float(shares[index]))
                errors.append(
                    float(share_standard_errors[index])
                    if share_standard_errors is not None
                    else None
                )
            estimates.extend(value for value, _ in cells)
            errors.extend(error for _, error in cells)
            latex, preview = _pair_rows(
                f"{index + 1}", estimates, errors, num_decimals
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)

    note = _standard_error_note(
        preview_rows,
        "Standard errors in parentheses are Delta-method errors of the "
        "class-specific coefficients.",
        "Standard errors are unavailable: the observed information is "
        "singular at this fit, so no coefficient has an identified standard "
        "error. Fewer classes or a leaner specification usually restores it.",
    )
    return _combine_renderings(
        latex_rows,
        preview_rows,
        header,
        num_decimals,
        escape_header=True,
        note=note,
        disable_numparse=True,
    )


def format_membership_coefficients(
    table: pl.DataFrame,
    num_classes: int,
    num_decimals: int,
    *,
    layout: str = "auto",
    reference_class: int = 0,
    shares: Sequence[float] | None = None,
) -> str:
    """Render the class-membership (demographic) multinomial-logit coefficients.

    The reference class is printed as an explicit row of zeros rather than
    omitted, so the table shows every class and the reader can see which one the
    log-odds are measured against.

    Parameters
    ----------
    table : pl.DataFrame
        Long-format frame from
        :meth:`~lcl.LCLResults.membership_coefficients`.
    num_classes : int
        Number of latent classes.
    num_decimals : int
        Decimal places.
    layout : {"auto", "wide", "dense"}, default="auto"
        ``"dense"`` gives one row per class, which is the layout that scales to
        many classes; ``"wide"`` puts demographics on rows instead.
    reference_class : int, default=0
        Zero-indexed class normalised to zero.
    shares : Sequence[float] | None, optional
        Aggregate class shares, shown alongside the coefficients when supplied.

    Returns
    -------
    str
        Combined LaTeX and terminal rendering.
    """
    resolved = _resolve_layout(layout, num_classes)
    ordered_variables = list(dict.fromkeys(table["variable"].to_list()))
    labels = {
        row["variable"]: str(row["label"]) for row in table.iter_rows(named=True)
    }
    coefficients = {
        (row["variable"], int(row["class"])): (
            float(row["coefficient"]),
            float(row["std_error"]),
        )
        for row in table.iter_rows(named=True)
    }
    reference_label = f"Class {reference_class + 1}"
    latex_rows: list[str] = []
    preview_rows: list[tuple[str, ...]] = []

    def cells_for(variable: str, index: int) -> tuple[float | None, float | None]:
        """Return the coefficient and standard error, zeroed at the reference."""
        if index == reference_class:
            return 0.0, None
        return coefficients.get((variable, index), (float("nan"), float("nan")))

    if resolved == "wide":
        header = (
            "Variable",
            *(
                f"Class {index + 1}" + (" (ref.)" if index == reference_class else "")
                for index in range(num_classes)
            ),
        )
        for variable in ordered_variables:
            pairs = [cells_for(variable, index) for index in range(num_classes)]
            latex, preview = _pair_rows(
                labels.get(variable, variable),
                [value for value, _ in pairs],
                [error for _, error in pairs],
                num_decimals,
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)
    else:
        share_column = shares is not None
        header = (
            "Class",
            *(("Share",) if share_column else ()),
            *(labels.get(variable, variable) for variable in ordered_variables),
        )
        for index in range(num_classes):
            pairs = [cells_for(variable, index) for variable in ordered_variables]
            estimates: list[float | None] = []
            errors: list[float | None] = []
            if share_column:
                assert shares is not None
                estimates.append(float(shares[index]))
                errors.append(None)
            estimates.extend(value for value, _ in pairs)
            errors.extend(error for _, error in pairs)
            row_label = f"{index + 1}" + (
                " (ref.)" if index == reference_class else ""
            )
            latex, preview = _pair_rows(
                row_label, estimates, errors, num_decimals
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)

    normalisation = (
        f"Coefficients are log-odds of class membership relative to "
        f"{reference_label}, whose coefficients are normalised to zero. The "
        "choice of reference class is arbitrary and changes no fitted quantity."
    )
    note = f"{normalisation} " + _standard_error_note(
        preview_rows,
        "Standard errors in parentheses.",
        "Standard errors are unavailable: the observed information is singular "
        "at this fit. A membership coefficient diverges whenever some "
        "demographic cell is never assigned to a class.",
    )
    return _combine_renderings(
        latex_rows,
        preview_rows,
        header,
        num_decimals,
        escape_header=True,
        note=note,
        disable_numparse=True,
    )


def format_membership_marginal_effects(
    table: pl.DataFrame,
    num_classes: int,
    num_decimals: int,
    *,
    layout: str = "auto",
) -> str:
    """Render average marginal effects of demographics on class membership.

    Parameters
    ----------
    table : pl.DataFrame
        Long-format frame from
        :meth:`~lcl.LCLResults.membership_marginal_effects`.
    num_classes : int
        Number of latent classes.
    num_decimals : int
        Decimal places.
    layout : {"auto", "wide", "dense"}, default="auto"
        As in :func:`format_membership_coefficients`.

    Returns
    -------
    str
        Combined LaTeX and terminal rendering.
    """
    resolved = _resolve_layout(layout, num_classes)
    ordered_variables = list(dict.fromkeys(table["variable"].to_list()))
    labels = {
        row["variable"]: str(row["label"]) for row in table.iter_rows(named=True)
    }
    effects = {
        (row["variable"], int(row["class"])): (
            float(row["marginal_effect"]),
            float(row["std_error"]),
        )
        for row in table.iter_rows(named=True)
    }
    latex_rows: list[str] = []
    preview_rows: list[tuple[str, ...]] = []
    if resolved == "wide":
        header = ("Variable", *_class_labels(num_classes))
        for variable in ordered_variables:
            pairs = [
                effects.get((variable, index), (float("nan"), float("nan")))
                for index in range(num_classes)
            ]
            latex, preview = _pair_rows(
                labels.get(variable, variable),
                [value for value, _ in pairs],
                [error for _, error in pairs],
                num_decimals,
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)
    else:
        header = (
            "Class",
            *(labels.get(variable, variable) for variable in ordered_variables),
        )
        for index in range(num_classes):
            pairs = [
                effects.get((variable, index), (float("nan"), float("nan")))
                for variable in ordered_variables
            ]
            latex, preview = _pair_rows(
                f"{index + 1}",
                [value for value, _ in pairs],
                [error for _, error in pairs],
                num_decimals,
            )
            latex_rows.extend(latex)
            preview_rows.extend(preview)
    note = (
        "Average marginal effects: the mean over panels of the change in the "
        "probability of belonging to a class per one-unit change in the "
        "demographic. Each variable's effects sum to zero across classes, and "
        "unlike the log-odds above they do not depend on the reference class. "
    ) + _standard_error_note(
        preview_rows,
        "Standard errors in parentheses.",
        "Standard errors are unavailable at this fit.",
    )
    return "\nAverage marginal effects on class membership" + _combine_renderings(
        latex_rows,
        preview_rows,
        header,
        num_decimals,
        escape_header=True,
        note=note,
        disable_numparse=True,
    )


def _standard_error_note(
    preview_rows: Sequence[tuple[str, ...]],
    available: str,
    unavailable: str,
) -> str:
    """Describe standard errors only as the rendered table actually shows them."""
    rendered_any = any(
        cell.startswith("(") and cell != "(--)"
        for row in preview_rows
        for cell in row
    )
    return available if rendered_any else unavailable


def _resolve_layout(layout: str, num_classes: int) -> str:
    """Choose between the variable-per-row and class-per-row layouts."""
    if layout not in {"auto", "wide", "dense"}:
        raise ValueError("layout must be 'auto', 'wide', or 'dense'.")
    if layout != "auto":
        return layout
    return "dense" if num_classes >= DENSE_LAYOUT_MIN_CLASSES else "wide"


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
