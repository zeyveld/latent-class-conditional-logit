"""Internal numerical containers.

User-facing configuration and request types live in :mod:`lcl.options`.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

from jaxtyping import Array, Bool, Float64, UInt


@dataclass
class ParsedData:
    """Aligned arrays and original identifiers produced by the encoder."""

    X: Array
    dems: Array | None
    y: Array | None
    cases: Array
    alts: Array
    panels: Array
    case_varnames: list[str]
    dem_varnames: list[str] | None
    original_alts: Any | None = None
    original_cases: Any | None = None
    original_panels: Any | None = None


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
    params: Array
    neg_loglik: float | Array
    message: str
    hess_inv: Array
    grad_n: Array
    grad: Array
    nit: int
    nfev: int
    njev: int
    information_diagnostics: Any = None


class EMVars(NamedTuple):
    """Parameters and probabilities updated by the EM algorithm."""

    latent_betas: Float64[Array, "alt_vars classes"] | None
    structural_betas: Float64[Array, "alt_vars classes"] | None
    thetas: Float64[Array, "dem_vars+1 classes-1"] | None
    shares: Float64[Array, "classes"] | None
    unconditional_loglik: Float64[Array, ""]
    class_probs_by_panel: Float64[Array, "panels classes"] | None


__all__ = [
    "Data",
    "DiffUnchosenChosen",
    "EMVars",
    "OptimizeResult",
    "ParsedData",
]
