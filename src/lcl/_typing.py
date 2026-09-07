"""Shape-aware input aliases shared by public and internal entry points.

Inputs retain their original dtype until the ingestion layer validates and converts
values. Encoded arrays use Float64, Bool, and integer jaxtyping annotations.
"""

from collections.abc import Sequence
from typing import TypeAlias

from jaxtyping import ArrayLike, Shaped

DesignInput: TypeAlias = Shaped[ArrayLike, "rows alt_vars"] | Sequence[Sequence[float]]
DemographicsInput: TypeAlias = (
    Shaped[ArrayLike, "panels dem_vars"] | Sequence[Sequence[float]]
)
RowIdsInput: TypeAlias = Shaped[ArrayLike, "rows"] | Sequence[object]
PanelIdsInput: TypeAlias = Shaped[ArrayLike, "panels"] | Sequence[object]
ChoicesInput: TypeAlias = Shaped[ArrayLike, "rows"] | Sequence[float | bool]
CaseWeightsInput: TypeAlias = Shaped[ArrayLike, "cases"] | Sequence[float]
PanelWeightsInput: TypeAlias = Shaped[ArrayLike, "panels"] | Sequence[float]
InitialCoefficientsInput: TypeAlias = Shaped[ArrayLike, "alt_vars"] | Sequence[float]
