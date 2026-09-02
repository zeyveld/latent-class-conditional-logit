"""Public fitted-result and prediction types."""

from typing import Any, Protocol, runtime_checkable

from lcl._diagnostics import LCLDiagnostics
from lcl._prediction import CLPrediction, LCLPrediction
from lcl._results import LCLResults
from lcl.conditional_logit import CLResults


@runtime_checkable
class ResultsProtocol(Protocol):
    """Common surface implemented by conditional and latent-class results."""

    model: Any
    converged: bool
    cov_matrix: Any
    adjusted_bic: Any

    def parameter_names(self) -> list[str]:
        """Return labels aligned one-to-one with covariance rows and columns."""
        ...

    def predict(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Predict from new long-format data."""
        ...


__all__ = [
    "CLResults",
    "CLPrediction",
    "LCLDiagnostics",
    "LCLPrediction",
    "LCLResults",
    "ResultsProtocol",
]
