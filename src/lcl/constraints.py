"""Parameter constraints and chain-rule helpers.

This module keeps transformations between unconstrained optimizer parameters and
structural econometric parameters in one place.  Optimizers can then share the
same forward map, gradient pullback, and Hessian pullback rather than carrying
parallel copies of derivative logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from jax.nn import sigmoid, softplus
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
        Minimum absolute magnitude of the structural coefficient.  The forward
        transform is ``-(softplus(raw) + min_abs)``.
    units : str | None, default=None
        Optional human-readable units for summaries and audit reports.
    warn_below : float | None, default=None
        Optional threshold used by diagnostics to flag weakly identified
        numeraires.
    """

    variable: str | None = None
    min_abs: float = DEFAULT_NEGATIVE_MIN_ABS
    units: str | None = None
    warn_below: float | None = None

    def __post_init__(self) -> None:
        """Validate constraint settings."""
        if self.min_abs <= 0:
            raise ValueError("NegativeCoefficient.min_abs must be positive.")
        if self.warn_below is not None and self.warn_below <= 0:
            raise ValueError("NegativeCoefficient.warn_below must be positive.")

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

    def forward(self, raw: Float64[Array, "..."]) -> Float64[Array, "..."]:
        """Map unconstrained parameters to negative structural coefficients."""
        return -(softplus(raw) + self.min_abs)

    def jacobian_diag(self, raw: Float64[Array, "..."]) -> Float64[Array, "..."]:
        """Return the diagonal Jacobian element of :meth:`forward`."""
        return -sigmoid(raw)

    def hessian_diag(self, raw: Float64[Array, "..."]) -> Float64[Array, "..."]:
        """Return the diagonal second derivative of :meth:`forward`."""
        d1 = self.jacobian_diag(raw)
        return d1 * (1.0 + d1)


def transform_negative_coefficient(
    latent_params: Float64[Array, "..."],
    index: int | None,
    min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
) -> Float64[Array, "..."]:
    """Apply a negative-coefficient transform at ``index``.

    Parameters
    ----------
    latent_params : Array
        Unconstrained optimizer parameters.  The constrained variable is expected
        on axis 0, matching the package's ``(variables, classes)`` convention.
    index : int | None
        Variable index to constrain.  If ``None``, ``latent_params`` is returned
        unchanged.
    min_abs : float, default=1e-5
        Minimum absolute magnitude for the transformed coefficient.

    Returns
    -------
    Array
        Structural parameters with the selected row constrained negative.
    """
    if index is None:
        return latent_params
    transformed = NegativeCoefficient(min_abs=min_abs).forward(latent_params[index])
    return latent_params.at[index].set(transformed)


def pullback_negative_gradient(
    raw_params: Float64[Array, "..."],
    index: int | None,
    grad_struct: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Pull a structural gradient back to unconstrained parameter space."""
    if index is None:
        return grad_struct
    derivative = NegativeCoefficient().jacobian_diag(raw_params[index])
    return grad_struct.at[index].multiply(derivative)


def pullback_negative_score_rows(
    raw_params: Float64[Array, "..."],
    index: int | None,
    score_rows: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Pull case-level score rows back to unconstrained parameter space."""
    if index is None:
        return score_rows
    derivative = NegativeCoefficient().jacobian_diag(raw_params[index])
    return score_rows.at[:, index].multiply(derivative)


def pullback_negative_hessian(
    raw_params: Float64[Array, "..."],
    index: int | None,
    grad_struct: Float64[Array, "..."],
    hessian_struct: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Pull a structural Hessian back to unconstrained parameter space.

    Parameters
    ----------
    raw_params : Array
        Unconstrained optimizer parameters.
    index : int | None
        Index of the constrained variable.  If ``None``, ``hessian_struct`` is
        returned unchanged.
    grad_struct : Array
        Structural gradient evaluated at the transformed parameters.
    hessian_struct : Array
        Structural Hessian evaluated at the transformed parameters.

    Returns
    -------
    Array
        Hessian with the constrained row and column transformed by the chain rule.
    """
    if index is None:
        return hessian_struct
    constraint = NegativeCoefficient()
    derivative = constraint.jacobian_diag(raw_params[index])
    hessian = hessian_struct.at[index, :].multiply(derivative)
    hessian = hessian.at[:, index].multiply(derivative)
    second_derivative = constraint.hessian_diag(raw_params[index])
    return hessian.at[index, index].add(grad_struct[index] * second_derivative)


def pullback_negative_derivatives(
    raw_params: Float64[Array, "..."],
    index: int | None,
    grad_struct: Float64[Array, "..."],
    score_rows_struct: Float64[Array, "..."],
    hessian_struct: Float64[Array, "..."],
) -> tuple[
    Float64[Array, "..."],
    Float64[Array, "..."],
    Float64[Array, "..."],
]:
    """Pull gradient, score rows, and Hessian through a negative constraint."""
    grad = pullback_negative_gradient(raw_params, index, grad_struct)
    score_rows = pullback_negative_score_rows(raw_params, index, score_rows_struct)
    hessian = pullback_negative_hessian(raw_params, index, grad_struct, hessian_struct)
    return grad, score_rows, hessian


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
