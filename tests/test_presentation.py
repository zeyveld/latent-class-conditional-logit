"""Rendering of class-specific, membership, and marginal-effect result tables.

The population-moment table established the house style -- an estimate row over a
parenthesised standard-error row, wrapped in booktabs rules -- and these tables
are refinements of it, so the tests pin the shared shape rather than exact
numbers.  They also pin the two layouts: a table with one class per column reads
best for a handful of classes and stops fitting well before the sixty-odd an
attribute non-attendance design implies, at which point the class axis has to
become the row axis.
"""

import numpy as onp
import polars as pl
import pytest

from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
)
from lcl import fit as lcl_fit
from lcl._labels import numeraire_enters_linearly
from lcl._presentation import (
    DENSE_LAYOUT_MIN_CLASSES,
    format_class_coefficients,
    format_membership_coefficients,
)


def _header_row(rendered: str) -> str:
    r"""Return the LaTeX header line, which is the row just after ``\toprule``."""
    lines = rendered.split("--- Table preview ---")[0].splitlines()
    return lines[lines.index(r"\toprule") + 1]


def _panel(seed: int = 6, num_panels: int = 140) -> pl.DataFrame:
    """Two-class panel with a demographic that drives class membership."""
    rng = onp.random.default_rng(seed)
    income = rng.normal(size=num_panels)
    membership = 1.0 / (1.0 + onp.exp(-income))
    latent = (rng.uniform(size=num_panels) < membership).astype(int)
    price_by_class = onp.array([-1.9, -0.5])
    quality_by_class = onp.array([0.3, 1.5])
    rows = []
    for panel in range(num_panels):
        for case in range(4):
            price = rng.uniform(0.5, 3.0, size=3)
            quality = rng.uniform(0.0, 5.0, size=3)
            utility = (
                price_by_class[latent[panel]] * price
                + quality_by_class[latent[panel]] * quality
                + rng.gumbel(size=3)
            )
            chosen = int(onp.argmax(utility))
            rows.extend(
                {
                    "panel": panel,
                    "case": panel * 4 + case,
                    "alt": alt,
                    "choice": alt == chosen,
                    "price": float(price[alt]),
                    "quality": float(quality[alt]),
                    "income": float(income[panel]),
                }
                for alt in range(3)
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    """Return a converged two-class fit with labelled variables."""
    df = _panel()
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        membership_formula="~ income",
        classes=2,
        constraints={"price": NegativeCoefficient()},
        variable_labels={
            "price": "Price",
            "quality": "Product quality",
            "income": "Household income",
        },
    )
    return lcl_fit(
        df,
        spec,
        fit_options=FitOptions(seed=3, num_devices=1),
        optimization_options=OptimizationOptions(newton_decrement_tol=1e-8),
        inference=InferenceOptions(covariance="clustered"),
    )


def test_class_table_matches_the_population_summary_house_style(fitted) -> None:
    """Class coefficients stack estimate over standard error inside booktabs rules."""
    rendered = format_class_coefficients(
        fitted.class_coefficients(), fitted.model.num_classes, 3
    )
    assert r"\toprule" in rendered and r"\bottomrule" in rendered
    assert r"\midrule" in rendered
    assert "--- LaTeX Output ---" in rendered
    assert "--- Table preview ---" in rendered
    assert "Product quality" in rendered

    latex = rendered.split("--- Table preview ---")[0]
    body = [line for line in latex.splitlines() if line.endswith(r"\\")]
    # Every variable contributes a labelled estimate row and an unlabelled
    # standard-error row whose cells are parenthesised.
    standard_error_rows = [line for line in body if line.lstrip().startswith("&")]
    assert standard_error_rows
    assert all("(" in line and ")" in line for line in standard_error_rows)


def test_layout_switches_from_class_columns_to_class_rows(fitted) -> None:
    """The wide layout labels classes as columns; the dense layout as rows."""
    table = fitted.class_coefficients()
    wide = format_class_coefficients(table, 2, 3, layout="wide")
    dense = format_class_coefficients(table, 2, 3, layout="dense")

    wide_header = _header_row(wide)
    dense_header = _header_row(dense)
    assert wide_header.startswith("Variable")
    assert "Class 1" in wide_header and "Class 2" in wide_header
    assert dense_header.startswith("Class")
    assert "Product quality" in dense_header

    # The dense layout's width is set by the utility variables, not the classes,
    # which is what lets it survive many classes.
    assert len(dense_header.split("&")) == 1 + len(fitted.model.case_varnames)


def test_auto_layout_transposes_once_the_class_columns_stop_fitting(fitted) -> None:
    """``auto`` picks class-per-column below the threshold and class-per-row above."""
    table = fitted.class_coefficients()
    below = format_class_coefficients(
        table, DENSE_LAYOUT_MIN_CLASSES - 1, 3, layout="auto"
    )
    at_threshold = format_class_coefficients(
        table, DENSE_LAYOUT_MIN_CLASSES, 3, layout="auto"
    )
    assert _header_row(below).startswith("Variable")
    assert _header_row(at_threshold).startswith("Class")


def test_membership_table_shows_the_reference_class_and_its_normalisation(
    fitted,
) -> None:
    """The reference class is printed as zeros, not omitted, and is explained."""
    rendered = format_membership_coefficients(
        fitted.membership_coefficients(), fitted.model.num_classes, 3
    )
    assert "(ref.)" in rendered
    assert "0.000" in rendered
    assert "log-odds of class membership relative to Class 1" in rendered
    assert "arbitrary" in rendered
    # The reference column carries no standard errors, so nothing spurious is
    # printed where an estimated one would go.
    assert "(nan)" not in rendered


def test_membership_marginal_effects_sum_to_zero_across_classes(fitted) -> None:
    """Class probabilities sum to one, so their derivatives sum to zero."""
    effects = fitted.membership_marginal_effects()
    by_variable = effects.group_by("variable").agg(
        pl.col("marginal_effect").sum().alias("total")
    )
    assert onp.allclose(by_variable["total"].to_numpy(), 0.0, atol=1e-10)
    assert (effects["std_error"].to_numpy() > 0.0).all()


def test_summaries_return_the_same_frames_they_print(fitted) -> None:
    """Printing is a view over the tidy frames, not a second computation."""
    assert fitted.summarize_class_betas(show=False).equals(
        fitted.class_coefficients()
    )
    assert fitted.summarize_membership(show=False).equals(
        fitted.membership_coefficients()
    )


def test_summarize_membership_rejects_a_model_without_demographics() -> None:
    """A share-only mixture has no membership regression to summarise."""
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        classes=2,
        constraints={"price": NegativeCoefficient()},
    )
    results = lcl_fit(
        _panel(),
        spec,
        fit_options=FitOptions(seed=3, num_devices=1),
        inference=InferenceOptions(skip=True),
    )
    with pytest.raises(ValueError, match="no class-membership regression"):
        results.summarize_membership(show=False)
    with pytest.raises(ValueError, match="no class-membership regression"):
        results.membership_marginal_effects()


@pytest.mark.parametrize(
    ("numeraire", "linear"),
    [
        ("price", True),
        ("np.log(price)", False),
        ("I(price ** 2)", False),
        ("price:quality", False),
        ("C(band)[T.high]", False),
    ],
)
def test_numeraire_linearity_detects_transformed_terms(numeraire, linear) -> None:
    """Money-metric welfare needs a bare numeraire column, not a transform of one."""

    class _Model:
        pass

    model = _Model()
    model.numeraire = numeraire
    assert numeraire_enters_linearly(model) is linear


def test_transformed_numeraire_is_flagged_in_the_diagnostics() -> None:
    """Dividing by ``-beta`` is only a money metric when the numeraire is linear."""
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ np.log(price) + quality",
        membership_formula="~ income",
        classes=2,
        constraints={"np.log(price)": NegativeCoefficient()},
    )
    results = lcl_fit(
        _panel(),
        spec,
        fit_options=FitOptions(seed=3, num_devices=1),
        inference=InferenceOptions(skip=True),
    )
    checks = results.diagnostics().to_frame()
    flagged = checks.filter(pl.col("check") == "numeraire_enters_linearly")
    assert flagged["status"][0] == "warning"
