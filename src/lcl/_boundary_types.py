"""Typed, serializable records for boundary inference and coefficient moments."""

from typing import NamedTuple, TypedDict

import numpy as np
from jaxtyping import Float64, Integer


class BoundarySummaryInputs(TypedDict):
    """Structural derivatives retained for small parameter-space calculations."""

    information: Float64[np.ndarray, "all_params all_params"]
    meat: Float64[np.ndarray, "all_params all_params"]
    score: Float64[np.ndarray, "all_params"]
    groups: int
    price_indices: Integer[np.ndarray, "classes"]


class BoundarySummaryDiagnostics(TypedDict, total=False):
    """JSON-compatible audit fields; projection fields appear after summarizing."""

    method: str
    fallback_reason: str | None
    active_parameters: list[int]
    strict_parameters: list[int]
    weak_parameters: list[int]
    multiplier_z: list[float]
    selection_threshold: float
    directional_sd_variables: list[str]
    draws: int
    seed: int
    information: dict[str, int | float | bool]
    seconds: float


class CoefficientMoments(NamedTuple):
    """Class-weighted moments and their structural-parameter derivatives."""

    means: Float64[np.ndarray, "alt_vars"]
    variances: Float64[np.ndarray, "alt_vars"]
    shares: Float64[np.ndarray, "classes"]
    mean_jacobian: Float64[np.ndarray, "alt_vars all_params"]
    variance_jacobian: Float64[np.ndarray, "alt_vars all_params"]
