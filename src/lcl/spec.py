"""Public model specification objects."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    NegativeCoefficient,
    constraint_summary_rows,
    normalize_negative_constraints,
)
from lcl._labels import label_for_variable


@dataclass(frozen=True)
class ChoiceIds:
    """Column names identifying a long-format choice dataset.

    Parameters
    ----------
    alt : str
        Alternative identifier column.
    case : str
        Choice-situation identifier column.
    panel : str
        Decision-maker or panel identifier column.
    choice : str
        Boolean or binary chosen-alternative indicator column.
    """

    alt: str
    case: str
    panel: str
    choice: str


@dataclass(frozen=True)
class LCLSpec:
    """Declarative latent-class conditional-logit specification.

    Parameters
    ----------
    ids : ChoiceIds
        Identifier and choice columns for the long-format dataset.
    utility : Sequence[str] | None, default=None
        Alternative-specific variables in the utility specification.  Omit when
        ``utility_formula`` or legacy ``formula`` provides the utility design.
    membership : Sequence[str] | None, default=None
        Panel-level variables for class-membership probabilities.  Omit when
        ``membership_formula`` or legacy ``formula`` provides the demographic
        design.
    classes : int, default=2
        Number of latent classes.
    constraints : mapping or sequence, optional
        Coefficient constraints.  The current estimation engine supports one
        negative coefficient, typically a price, cost, or travel-time numeraire.
    formula : str | None, default=None
        Deprecated combined Formulaic string such as
        ``"choice ~ cost + time | income + C(segment)"``.  Prefer
        ``utility_formula`` and ``membership_formula`` in new code.
    utility_formula : str | None, default=None
        Formulaic string for the alternative-specific utility specification, such
        as ``"choice ~ cost + C(mode)"``.  A right-hand-side-only utility formula
        is permitted when the choice column is supplied by :class:`ChoiceIds`.
    membership_formula : str | None, default=None
        Right-hand-side Formulaic string for the class-membership demographic
        regression, such as ``"~ income + C(segment)"``.
    variable_labels : Mapping[str, str] | None, default=None
        Optional mapping from raw DataFrame/model variable names to human-readable
        labels used in printed coefficient and WTP/tradeoff tables.  Exact
        Formulaic-expanded names may also be labeled directly; otherwise labels
        for raw categorical columns are reused for terms such as
        ``"C(segment)[T.high]"``.
    """

    ids: ChoiceIds
    utility: Sequence[str] | None = None
    membership: Sequence[str] | None = None
    classes: int = 2
    constraints: (
        Mapping[str, NegativeCoefficient] | Sequence[NegativeCoefficient] | None
    ) = None
    formula: str | None = None
    utility_formula: str | None = None
    membership_formula: str | None = None
    variable_labels: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate internal consistency."""
        if self.classes < 2:
            raise ValueError("LCLSpec.classes must be at least 2.")
        if self.formula is not None and (
            self.utility_formula is not None or self.membership_formula is not None
        ):
            raise ValueError(
                "Use either legacy formula=... or utility_formula=.../"
                "membership_formula=..., not both."
            )
        if self.formula is None and self.utility_formula is None and not self.utility:
            raise ValueError("LCLSpec requires either utility variables or a formula.")
        if len(self.negative_constraints) > 1:
            raise NotImplementedError(
                "The current latent-class estimator supports one negative "
                "coefficient constraint. Multiple constraints can be added once "
                "the optimizer is generalized beyond a single numeraire row."
            )

    @property
    def negative_constraints(self) -> list[NegativeCoefficient]:
        """Return normalized negative-coefficient constraints."""
        return normalize_negative_constraints(self.constraints)

    @property
    def negative_constraint(self) -> NegativeCoefficient | None:
        """Return the single negative constraint, if present."""
        constraints = self.negative_constraints
        return constraints[0] if constraints else None

    @property
    def numeraire(self) -> str | None:
        """Return the constrained variable used as the numeraire."""
        constraint = self.negative_constraint
        return None if constraint is None else constraint.variable

    @property
    def numeraire_min_abs(self) -> float:
        """Return the numeraire floor implied by the specification."""
        constraint = self.negative_constraint
        if constraint is None:
            return DEFAULT_NEGATIVE_MIN_ABS
        return constraint.min_abs

    def summary_lines(self) -> list[str]:
        """Return a compact, human-readable specification summary."""
        lines = [
            "Latent-class conditional logit",
            f"Classes: {self.classes}",
            f"Panel id: {self.ids.panel}",
            f"Case id: {self.ids.case}",
            f"Alternative id: {self.ids.alt}",
            f"Choice column: {self.ids.choice}",
            "",
            "Utility variables:",
        ]
        if self.formula is not None:
            lines.append(f"  formula: {self.formula}")
        elif self.utility_formula is not None:
            lines.append(f"  formula: {self.utility_formula}")
        else:
            for variable in self.utility or []:
                suffix = ""
                for constraint in self.negative_constraints:
                    if constraint.variable == variable:
                        suffix = f" [negative, min_abs={constraint.min_abs:g}]"
                label = self._display_variable(variable)
                lines.append(f"  {label}{suffix}")
        lines.append("")
        lines.append("Class-membership variables:")
        if self.membership:
            lines.extend(
                f"  {self._display_variable(variable)}" for variable in self.membership
            )
        elif self.membership_formula is not None:
            lines.append(f"  formula: {self.membership_formula}")
        elif self.formula is not None:
            lines.append("  from formula")
        else:
            lines.append("  none")
        return lines

    def _display_variable(self, variable: str) -> str:
        """Return a specification variable with its label when available."""
        label = label_for_variable(variable, self.variable_labels or {})
        return label if label == variable else f"{label} ({variable})"

    def constraint_rows(self) -> list[dict[str, object]]:
        """Return serializable constraint metadata."""
        return constraint_summary_rows(self.negative_constraints)


def resolve_lcl_spec(
    *,
    spec: LCLSpec | None = None,
    alts_col: str | None = None,
    cases_col: str | None = None,
    panels_col: str | None = None,
    choice_col: str | None = None,
    case_varnames: Sequence[str] | None = None,
    dem_varnames: Sequence[str] | None = None,
    formula: str | None = None,
    utility_formula: str | None = None,
    membership_formula: str | None = None,
    classes: int | None = None,
    numeraire: str | None = None,
    numeraire_min_abs: float | None = None,
    variable_labels: Mapping[str, str] | None = None,
) -> LCLSpec:
    """Resolve all supported LCL inputs to one canonical specification.

    Explicit keyword values override fields on ``spec``. The legacy combined
    ``formula`` remains supported, but it cannot be mixed with the separate
    utility and membership formula fields.

    Parameters
    ----------
    spec : LCLSpec | None, optional
        Base specification whose fields are used when an explicit value is absent.
    alts_col, cases_col, panels_col, choice_col : str | None, optional
        Identifier and choice columns.
    case_varnames, dem_varnames : Sequence[str] | None, optional
        Explicit utility and class-membership variables.
    formula : str | None, optional
        Deprecated combined utility and membership formula.
    utility_formula, membership_formula : str | None, optional
        Separate Formulaic specifications.
    classes : int | None, optional
        Number of latent classes.
    numeraire : str | None, optional
        Variable constrained to be strictly negative.
    numeraire_min_abs : float | None, optional
        Minimum absolute value of the negative numeraire coefficient.
    variable_labels : Mapping[str, str] | None, optional
        Presentation labels that supplement labels stored on ``spec``.

    Returns
    -------
    LCLSpec
        Fully resolved immutable model specification.

    Raises
    ------
    ValueError
        If identifier columns are missing or incompatible formula interfaces are
        combined.
    """
    resolved_alts = (
        alts_col
        if alts_col is not None
        else (spec.ids.alt if spec is not None else None)
    )
    resolved_cases = (
        cases_col
        if cases_col is not None
        else (spec.ids.case if spec is not None else None)
    )
    resolved_panels = (
        panels_col
        if panels_col is not None
        else (spec.ids.panel if spec is not None else None)
    )
    resolved_choice = (
        choice_col
        if choice_col is not None
        else (spec.ids.choice if spec is not None else None)
    )
    formula_for_choice = formula
    if formula_for_choice is None and utility_formula is not None:
        formula_for_choice = utility_formula
    if formula_for_choice is None and spec is not None:
        formula_for_choice = spec.formula or spec.utility_formula
    if resolved_choice is None and formula_for_choice is not None:
        lhs, separator, _ = formula_for_choice.partition("~")
        if separator and lhs.strip():
            resolved_choice = lhs.strip()

    missing_ids = [
        name
        for name, value in (
            ("alts_col", resolved_alts),
            ("cases_col", resolved_cases),
            ("panels_col", resolved_panels),
            ("choice_col", resolved_choice),
        )
        if value is None
    ]
    if missing_ids:
        raise ValueError(
            "An LCL specification requires all identifier columns. Missing: "
            f"{missing_ids}"
        )

    has_separate_override = any(
        value is not None
        for value in (
            utility_formula,
            membership_formula,
            case_varnames,
            dem_varnames,
        )
    )
    if formula is not None and has_separate_override:
        raise ValueError(
            "Use either deprecated formula=... or utility_formula=.../"
            "membership_formula=/explicit variable lists, not both."
        )

    inherit_legacy_formula = (
        formula is None
        and not has_separate_override
        and spec is not None
        and spec.formula is not None
    )
    resolved_formula = (
        formula
        if formula is not None
        else (spec.formula if inherit_legacy_formula and spec is not None else None)
    )
    if resolved_formula is None:
        resolved_utility_formula = (
            None
            if case_varnames is not None
            else (
                utility_formula
                if utility_formula is not None
                else (spec.utility_formula if spec is not None else None)
            )
        )
        resolved_membership_formula = (
            None
            if dem_varnames is not None
            else (
                membership_formula
                if membership_formula is not None
                else (spec.membership_formula if spec is not None else None)
            )
        )
    else:
        resolved_utility_formula = None
        resolved_membership_formula = None

    if resolved_formula is not None:
        warnings.warn(
            "The combined formula= interface is deprecated; use utility_formula= "
            "and membership_formula= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    resolved_utility = (
        None
        if resolved_formula is not None or resolved_utility_formula is not None
        else (
            case_varnames
            if case_varnames is not None
            else (spec.utility if spec is not None else None)
        )
    )
    resolved_membership = (
        None
        if resolved_formula is not None or resolved_membership_formula is not None
        else (
            dem_varnames
            if dem_varnames is not None
            else (spec.membership if spec is not None else None)
        )
    )
    resolved_classes = (
        classes if classes is not None else (spec.classes if spec is not None else 2)
    )

    constraints = spec.constraints if spec is not None else None
    spec_numeraire = spec.numeraire if spec is not None else None
    if numeraire is not None and spec_numeraire not in {None, numeraire}:
        raise ValueError(
            "numeraire conflicts with the negative constraint in the base spec."
        )
    resolved_numeraire = numeraire if numeraire is not None else spec_numeraire
    if resolved_numeraire is not None:
        floor = (
            numeraire_min_abs
            if numeraire_min_abs is not None
            else (
                spec.numeraire_min_abs
                if spec_numeraire is not None and spec is not None
                else DEFAULT_NEGATIVE_MIN_ABS
            )
        )
        if constraints is None:
            constraints = {resolved_numeraire: NegativeCoefficient(min_abs=floor)}
        elif (
            spec is not None
            and spec_numeraire == resolved_numeraire
            and floor != spec.numeraire_min_abs
        ):
            constraint = spec.negative_constraint
            if constraint is None:
                raise ValueError("The base specification has no negative constraint.")
            constraints = [
                replace(constraint, min_abs=floor),
            ]

    labels = dict(spec.variable_labels or {}) if spec is not None else {}
    if variable_labels is not None:
        labels.update(variable_labels)

    return LCLSpec(
        ids=ChoiceIds(
            alt=str(resolved_alts),
            case=str(resolved_cases),
            panel=str(resolved_panels),
            choice=str(resolved_choice),
        ),
        utility=resolved_utility,
        membership=resolved_membership,
        classes=int(resolved_classes),
        constraints=constraints,
        formula=resolved_formula,
        utility_formula=resolved_utility_formula,
        membership_formula=resolved_membership_formula,
        variable_labels=labels or None,
    )
