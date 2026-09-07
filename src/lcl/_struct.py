"""Internal numerical containers.

User-facing configuration and request types live in :mod:`lcl.options`.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as onp
from jaxtyping import Array, Bool, Float64, Shaped, UInt


@dataclass
class ParsedData:
    """Aligned arrays and original identifiers produced by the encoder."""

    X: Float64[Array, "rows alt_vars"]
    dems: Float64[Array, "panels dem_vars"] | None
    y: Bool[Array, "rows"] | None
    cases: UInt[Array, "rows"]
    alts: UInt[Array, "rows"]
    panels: UInt[Array, "rows"]
    case_varnames: list[str]
    dem_varnames: list[str] | None
    original_alts: Shaped[onp.ndarray, "rows"] | None = None
    original_cases: Shaped[onp.ndarray, "rows"] | None = None
    original_panels: Shaped[onp.ndarray, "rows"] | None = None


class Data(NamedTuple):
    """Immutable choice arrays consumed by likelihood kernels."""

    X: Float64[Array, "alts_by_case alt_vars"]
    dems: Float64[Array, "panels dem_vars"] | None
    y: Bool[Array, "alts_by_case"] | None
    alts: UInt[Array, "alts_by_case"]
    cases: UInt[Array, "alts_by_case"]
    panels: UInt[Array, "alts_by_case"] | None
    panels_of_cases: UInt[Array, "cases"] | None
    num_cases_per_panel: UInt[Array, "panels"] | None
    num_cases: int
    num_alt_vars: int
    num_panels: int | None
    num_dem_vars: int


class DiffUnchosenChosen(NamedTuple):
    """Chosen-differenced design and aligned identifiers."""

    X: Float64[Array, "unchosen_alts_by_case alt_vars"]
    alts: UInt[Array, "unchosen_alts_by_case"]
    cases: UInt[Array, "unchosen_alts_by_case"]
    panels: UInt[Array, "unchosen_alts_by_case"] | None
    num_cases: int


@dataclass
class OptimizeResult:
    """Internal optimizer output and information diagnostics."""

    success: bool
    params: Float64[Array, "params"]
    neg_loglik: float | Float64[Array, ""]
    message: str
    hess_inv: Float64[Array, "params params"]
    grad_n: Float64[Array, "cases params"]
    grad: Float64[Array, "params"]
    nit: int
    nfev: int
    njev: int
    information_diagnostics: Any = None


class EMStepDiagnostics(NamedTuple):
    """Device-resident convergence scalars produced by one EM recursion.

    Kept separate from :class:`EMVars` so the EM step can stay inside a single
    compiled region: materializing these would force a host synchronization on
    every iteration.
    """

    beta_newton_error: Float64[Array, "classes"]
    membership_newton_error: Float64[Array, ""]


class EMVars(NamedTuple):
    """Parameters, likelihood, and posterior evaluated at the same EM iterate."""

    latent_betas: Float64[Array, "alt_vars classes"] | None
    structural_betas: Float64[Array, "alt_vars classes"] | None
    thetas: Float64[Array, "dem_vars+1 classes-1"] | None
    shares: Float64[Array, "classes"] | None
    unconditional_loglik: Float64[Array, ""]
    class_probs_by_panel: Float64[Array, "panels classes"] | None


__all__ = [
    "Data",
    "DiffUnchosenChosen",
    "EMStepDiagnostics",
    "EMVars",
    "OptimizeResult",
    "ParsedData",
]
