"""Negative coefficient specifications and optimizer bounds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float64


DEFAULT_NEGATIVE_MIN_ABS = 1e-5


@dataclass(frozen=True)
class NegativeCoefficient:
    """Constrain a coefficient to be strictly negative.

    Parameters
    ----------
    variable : str | None, default=None
        Name of the variable being constrained.  It may be omitted when the
        object is supplied in a mapping keyed by variable name.
    min_abs : float, default=1e-5
        Minimum absolute magnitude of the coefficient, enforced as
        ``coefficient <= -min_abs``.
    units : str | None, default=None
        Optional human-readable units for summaries and audit reports.
    warn_below : float | None, default=None
        Optional LCL diagnostic threshold overriding the general
        ``near_zero_numeraire_threshold``. The ``warn_near_zero_numeraire``
        switch still controls whether the diagnostic has warning status.
    """

    variable: str | None = None
    min_abs: float = DEFAULT_NEGATIVE_MIN_ABS
    units: str | None = None
    warn_below: float | None = None

    def __post_init__(self) -> None:
        """Validate constraint settings."""
        if not math.isfinite(self.min_abs) or self.min_abs <= 0:
            raise ValueError("NegativeCoefficient.min_abs must be finite and positive.")
        if self.warn_below is not None and (
            not math.isfinite(self.warn_below) or self.warn_below <= 0
        ):
            raise ValueError(
                "NegativeCoefficient.warn_below must be finite and positive."
            )

    def bind(self, variable: str) -> "NegativeCoefficient":
        """Return a copy tied to ``variable``.

        Parameters
        ----------
        variable : str
            Variable name from a specification mapping.

        Returns
        -------
        NegativeCoefficient
            A constraint with a concrete variable name.
        """
        if self.variable is not None and self.variable != variable:
            raise ValueError(
                "NegativeCoefficient variable mismatch: "
                f"{self.variable!r} != {variable!r}."
            )
        return replace(self, variable=variable)


@dataclass(frozen=True)
class NegativeCoefficientBound:
    """Resolved design-column constraint used by internal numerical kernels.

    Unlike the user-facing specification this holds a column index, not a
    variable name. ``index=None`` denotes an unconstrained fit. Frozen values
    are safe static JIT/cache keys.
    """

    index: int | None = None
    min_abs: float = DEFAULT_NEGATIVE_MIN_ABS

    def upper_bounds(
        self, params: Float64[Array, "params"], *, width: int = 1
    ) -> Float64[Array, "params"] | None:
        """Build structural bounds, repeating a column across packed classes.

        ``width=1`` is a CL or single-class M-step vector. For flat LCL vectors,
        ``width=num_classes`` selects the row-major beta block while leaving
        all membership parameters free.
        """
        if self.index is None:
            return None
        start = self.index * width
        return (
            jnp.full_like(params, jnp.inf).at[start : start + width].set(-self.min_abs)
        )


def normalize_negative_constraints(
    constraints: (
        Mapping[str, NegativeCoefficient] | Sequence[NegativeCoefficient] | None
    ),
) -> list[NegativeCoefficient]:
    """Normalize user-facing constraint containers.

    Parameters
    ----------
    constraints : mapping, sequence, or None
        Negative-coefficient constraints supplied either as
        ``{"price": NegativeCoefficient(min_abs=...)}`` or as a sequence of
        already-bound ``NegativeCoefficient("price", ...)`` objects.

    Returns
    -------
    list[NegativeCoefficient]
        Bound negative constraints.
    """
    if constraints is None:
        return []
    if isinstance(constraints, Mapping):
        return [
            constraint.bind(variable) for variable, constraint in constraints.items()
        ]
    normalized: list[NegativeCoefficient] = []
    for constraint in constraints:
        if constraint.variable is None:
            raise ValueError(
                "Sequence-style NegativeCoefficient constraints must set variable=...."
            )
        normalized.append(constraint)
    return normalized


def constraint_summary_rows(
    constraints: Sequence[NegativeCoefficient],
) -> list[dict[str, Any]]:
    """Return serializable rows for specification and audit summaries."""
    return [
        {
            "variable": constraint.variable,
            "constraint": "negative",
            "min_abs": constraint.min_abs,
            "units": constraint.units,
            "warn_below": constraint.warn_below,
        }
        for constraint in constraints
    ]
