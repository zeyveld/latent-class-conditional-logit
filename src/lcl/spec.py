"""Public model specification objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    NegativeCoefficient,
    constraint_summary_rows,
    normalize_negative_constraints,
)


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
        Backward-compatible combined Formulaic string such as
        ``"choice ~ cost + time | income + C(segment)"``.  Prefer
        ``utility_formula`` and ``membership_formula`` in new code.
    utility_formula : str | None, default=None
        Formulaic string for the alternative-specific utility specification, such
        as ``"choice ~ cost + C(mode)"``.  A right-hand-side-only utility formula
        is permitted when the choice column is supplied by :class:`ChoiceIds`.
    membership_formula : str | None, default=None
        Right-hand-side Formulaic string for the class-membership demographic
        regression, such as ``"~ income + C(segment)"``.
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
                lines.append(f"  {variable}{suffix}")
        lines.append("")
        lines.append("Class-membership variables:")
        if self.membership:
            lines.extend(f"  {variable}" for variable in self.membership)
        elif self.membership_formula is not None:
            lines.append(f"  formula: {self.membership_formula}")
        elif self.formula is not None:
            lines.append("  from formula")
        else:
            lines.append("  none")
        return lines

    def constraint_rows(self) -> list[dict[str, object]]:
        """Return serializable constraint metadata."""
        return constraint_summary_rows(self.negative_constraints)
