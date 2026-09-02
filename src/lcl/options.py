"""Public configuration and request types."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union

from jax import device_count
from jax.typing import ArrayLike


@dataclass
class OptimizationOptions:
    """Safeguarded exact-Newton settings."""

    maxiter: int = 75
    gradient_tol: float = 1e-5
    hessian_damping: float = 0.0
    max_step_norm: float = 1000.0
    line_search_maxiter: int = 40
    accept_any_decrease: bool = False
    method: str = "newton"
    line_search: str = "armijo"
    fallback: str = "gradient"

    def __post_init__(self) -> None:
        """Validate supported optimizer settings."""
        if self.maxiter < 0:
            raise ValueError("maxiter must be nonnegative.")
        if self.gradient_tol <= 0:
            raise ValueError("gradient_tol must be positive.")
        if self.hessian_damping < 0:
            raise ValueError("hessian_damping must be nonnegative.")
        if self.max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive.")
        if self.line_search_maxiter < 0:
            raise ValueError("line_search_maxiter must be nonnegative.")
        if self.method != "newton":
            raise ValueError("Only method='newton' is currently supported.")
        if self.line_search != "armijo":
            raise ValueError("Only line_search='armijo' is currently supported.")
        if self.fallback != "gradient":
            raise ValueError("Only fallback='gradient' is currently supported.")


@dataclass
class FitOptions:
    """Latent-class EM and multi-start settings."""

    seed: int = 0
    max_em_iter: int = 2000
    em_tol: float = 1e-6
    num_devices: int = field(default_factory=device_count)
    check_interval: int = 10
    starts: int = 1
    start_method: str = "panel_partition"

    def __post_init__(self) -> None:
        """Validate EM and multi-start settings."""
        if self.em_tol <= 0:
            raise ValueError("em_tol must be positive.")
        if self.max_em_iter < 0:
            raise ValueError("max_em_iter must be nonnegative.")
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive.")
        if self.starts < 1:
            raise ValueError("starts must be at least 1.")
        if self.start_method != "panel_partition":
            raise ValueError(
                "Only start_method='panel_partition' is currently supported."
            )
        available_devices = device_count()
        if not 1 <= self.num_devices <= available_devices:
            raise ValueError(
                "num_devices must be between 1 and the number of available JAX "
                f"devices ({available_devices})."
            )


@dataclass
class InferenceOptions:
    """Covariance and standard-error settings."""

    covariance: str = "clustered"
    cluster: str | None = "panel"
    finite_sample_correction: bool = True
    skip: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate covariance settings."""
        covariance = self.covariance.lower()
        if covariance in {"none", "unadjusted", "hessian"}:
            covariance = "unadjusted"
        elif covariance == "clustered":
            covariance = "clustered"
        elif covariance in {"robust", "sandwich", "huber-white"}:
            covariance = "robust"
        else:
            raise ValueError(
                "InferenceOptions.covariance must be one of 'clustered', "
                "'robust', 'sandwich', 'huber-white', 'unadjusted', or 'none'."
            )
        self.covariance = covariance


@dataclass
class DiagnosticsOptions:
    """Diagnostic switches and warning thresholds."""

    check_separation: bool = True
    check_collinearity: bool = True
    warn_near_zero_numeraire: bool = True
    warn_large_coefficients: bool = True
    near_zero_numeraire_threshold: float = 1e-3
    large_coefficient_threshold: float = 25.0


@dataclass(frozen=True)
class Options:
    """Complete configuration shared by all model-fitting entry points."""

    fit: FitOptions = field(default_factory=FitOptions)
    optimization: OptimizationOptions = field(default_factory=OptimizationOptions)
    inference: InferenceOptions = field(default_factory=InferenceOptions)
    diagnostics: DiagnosticsOptions = field(default_factory=DiagnosticsOptions)


def _resolve_options(
    options: Options | None,
    *,
    fit_options: FitOptions | None = None,
    optimization_options: OptimizationOptions | None = None,
    inference: InferenceOptions | None = None,
    diagnostics: DiagnosticsOptions | None = None,
) -> Options:
    """Resolve aggregate or legacy option arguments without implicit merging."""
    legacy = (fit_options, optimization_options, inference, diagnostics)
    if options is not None and any(value is not None for value in legacy):
        raise ValueError(
            "Pass either options=Options(...) or the legacy individual option "
            "arguments, not both."
        )
    if options is not None:
        return options
    return Options(
        fit=fit_options if fit_options is not None else FitOptions(),
        optimization=(
            optimization_options
            if optimization_options is not None
            else OptimizationOptions()
        ),
        inference=inference if inference is not None else InferenceOptions(),
        diagnostics=(diagnostics if diagnostics is not None else DiagnosticsOptions()),
    )


@dataclass
class PastChoicesData:
    """Array-style historical choices used to update class membership."""

    X: ArrayLike
    y: ArrayLike
    alts: ArrayLike
    cases: ArrayLike
    panels: ArrayLike
    dems: ArrayLike | None = None
    dem_panel_ids: ArrayLike | None = None


class PartitionType(StrEnum):
    """Supported binning strategies for WTP analysis."""

    CATEGORICAL = "categorical"
    QUINTILES = "quintiles"
    CUSTOM_BREAKS = "custom_breaks"


@dataclass
class WTPRequest:
    """Configuration for a marginal willingness-to-pay summary."""

    alt_var: str
    demographic_var: str
    partition_type: PartitionType | str
    bins: Optional[Union[int, list[float]]] = None
    dummy_vars: list[str] | None = None
    dummy_labels: list[str] | None = None
    base_category: str = "base"

    def __post_init__(self) -> None:
        """Normalize and validate the partition request."""
        if not isinstance(self.partition_type, PartitionType):
            try:
                self.partition_type = PartitionType(self.partition_type)
            except ValueError:
                valid_options = [item.value for item in PartitionType]
                raise ValueError(
                    f"Invalid partition type: {self.partition_type}\n"
                    f"Must be one of {valid_options}"
                ) from None
        if self.partition_type == PartitionType.CUSTOM_BREAKS and not isinstance(
            self.bins, list
        ):
            raise ValueError(
                "When partition_type is 'custom_breaks', bins must be breakpoints."
            )
        if isinstance(self.bins, list) and any(
            right <= left for left, right in zip(self.bins, self.bins[1:])
        ):
            raise ValueError("Custom WTP breakpoints must be strictly increasing.")
        if self.dummy_vars is not None:
            if not self.dummy_vars:
                raise ValueError("dummy_vars must contain at least one column name.")
            if len(set(self.dummy_vars)) != len(self.dummy_vars):
                raise ValueError("dummy_vars cannot contain duplicate column names.")
            if self.partition_type != PartitionType.CATEGORICAL:
                raise ValueError(
                    "Dummy-coded WTP partitions require partition_type='categorical'."
                )
            if self.dummy_labels is not None and len(self.dummy_labels) != len(
                self.dummy_vars
            ):
                raise ValueError("dummy_labels must have one label per dummy column.")


__all__ = [
    "DiagnosticsOptions",
    "FitOptions",
    "InferenceOptions",
    "Options",
    "OptimizationOptions",
    "PartitionType",
    "PastChoicesData",
    "WTPRequest",
]
