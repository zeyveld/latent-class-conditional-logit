"""Public configuration and request types."""

import math
import warnings
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral

from jax import device_count
from lcl._typing import (
    ChoicesInput,
    DemographicsInput,
    DesignInput,
    PanelIdsInput,
    RowIdsInput,
)


DEFAULT_NEWTON_DECREMENT_TOL = 1e-5
"""Default stopping tolerance for the safeguarded exact-Newton solver."""

WEIGHT_TYPES = ("probability", "frequency")
"""Supported interpretations of user-supplied case weights."""


@dataclass(frozen=True, init=False)
class OptimizationOptions:
    r"""Safeguarded exact-Newton settings.

    Parameters
    ----------
    maxiter : int, default=75
        Maximum number of Newton iterations.
    newton_decrement_tol : float, default=1e-5
        Stopping tolerance on the Newton decrement
        :math:`\lambda = \sqrt{g' H^{-1} g}`, not on a raw gradient norm.  The
        decrement is invariant to nonsingular diagonal rescaling of the
        parameters and approximates :math:`\sqrt{2(f - f^\star)}`, so a value
        of ``1e-5`` on the package's per-observation objective corresponds to
        roughly ``5e-11`` in objective units.
    hessian_damping : float, default=0.0
        Initial diagonal shift used only when the undamped Cholesky solve does
        not produce a finite descent direction.
    max_step_norm : float, default=1000.0
        Upper bound on the adaptive trust radius, in the local curvature metric.
    initial_trust_radius : float, default=1.0
        Starting trust radius, in the local curvature metric.  It expands or
        contracts with agreement between the quadratic model and the objective.
    line_search_maxiter : int, default=40
        Maximum number of Armijo backtracking steps per Newton iteration.
    accept_any_decrease : bool, default=False
        Accept a finite step that merely decreases the objective when the
        stricter Armijo sufficient-decrease rule is not met.
    gradient_tol : float, optional
        Deprecated alias for ``newton_decrement_tol``.  Reading it returns the
        resolved tolerance; passing it emits a :class:`DeprecationWarning`.
        If both spellings are supplied, ``newton_decrement_tol`` takes precedence.

    Notes
    -----
    The class is frozen so a configuration can be hashed and used as a static
    argument to a cached JIT-compiled M-step.  Use :func:`dataclasses.replace`
    to derive a modified configuration.
    """

    maxiter: int = 75
    newton_decrement_tol: float = DEFAULT_NEWTON_DECREMENT_TOL
    hessian_damping: float = 0.0
    max_step_norm: float = 1000.0
    initial_trust_radius: float = 1.0
    line_search_maxiter: int = 40
    accept_any_decrease: bool = False

    def __init__(
        self,
        maxiter: int = 75,
        newton_decrement_tol: float | None = None,
        hessian_damping: float = 0.0,
        max_step_norm: float = 1000.0,
        initial_trust_radius: float = 1.0,
        line_search_maxiter: int = 40,
        accept_any_decrease: bool = False,
        gradient_tol: float | None = None,
    ) -> None:
        """Resolve the compatibility alias without storing duplicate tolerances."""
        if gradient_tol is not None:
            warnings.warn(
                "OptimizationOptions.gradient_tol is deprecated; use "
                "newton_decrement_tol. The value is a Newton-decrement "
                "tolerance, not a gradient norm.",
                DeprecationWarning,
                stacklevel=2,
            )
        tolerance = newton_decrement_tol
        if tolerance is None:
            tolerance = (
                DEFAULT_NEWTON_DECREMENT_TOL if gradient_tol is None else gradient_tol
            )
        for name, value in (
            ("maxiter", maxiter),
            ("newton_decrement_tol", tolerance),
            ("hessian_damping", hessian_damping),
            ("max_step_norm", max_step_norm),
            ("initial_trust_radius", initial_trust_radius),
            ("line_search_maxiter", line_search_maxiter),
            ("accept_any_decrease", accept_any_decrease),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def gradient_tol(self) -> float:
        """Read the resolved Newton tolerance under its deprecated spelling."""
        return self.newton_decrement_tol

    def __post_init__(self) -> None:
        """Validate supported settings."""
        for name in ("maxiter", "line_search_maxiter"):
            _require_integer(getattr(self, name), name)
        for name in (
            "newton_decrement_tol",
            "hessian_damping",
            "max_step_norm",
            "initial_trust_radius",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.maxiter < 0:
            raise ValueError("maxiter must be nonnegative.")
        if self.newton_decrement_tol <= 0:
            raise ValueError("newton_decrement_tol must be positive.")
        if self.hessian_damping < 0:
            raise ValueError("hessian_damping must be nonnegative.")
        if self.max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive.")
        if self.initial_trust_radius <= 0:
            raise ValueError("initial_trust_radius must be positive.")
        if self.line_search_maxiter < 0:
            raise ValueError("line_search_maxiter must be nonnegative.")


@dataclass(frozen=True)
class FitOptions:
    """Latent-class EM and multi-start settings.

    Parameters
    ----------
    seed : int, default=0
        Base seed for the panel partition used to build starting values.
    max_em_iter : int, default=2000
        Maximum number of EM recursions.
    em_tol : float, default=1e-8
        Stopping tolerance on the Aitken-extrapolated log-likelihood change per
        panel.  Because EM converges linearly, the raw iteration-to-iteration
        change can understate the distance to the optimum. For an observed rate
        ``r``, the estimated remaining ascent after the latest iterate is
        ``change * r / (1 - r)``. Normalizing by the panel
        count keeps the tolerance's meaning fixed as the sample grows.
    score_tol : float, default=1e-4
        Stopping tolerance on the maximum absolute component of the observed-data
        score, per panel, used for the final public ``converged`` flag. It is
        checked after EM and optional polishing; it does not stop EM early.
    polish : bool, default=True
        Run safeguarded Newton steps on the observed-data log likelihood after
        EM, using the exact analytic score and Hessian. EM may stop before
        reaching a stationary point; polishing attempts to close this gap
        before covariance estimation. A polish step is kept
        only if it does not decrease the log likelihood.
    polish_maxiter : int, default=25
        Maximum number of observed-data Newton iterations.
    num_devices : int
        Number of JAX devices across which class-specific M-steps are sharded.
    check_interval : int, default=1
        Number of EM recursions between Aitken stopping checks. History and
        progress callbacks are still updated on every recursion.
    starts : int, default=1
        Number of independent EM starts.  The start with the highest final log
        likelihood is kept.
    start_method : str, default="panel_partition"
        Compatibility setting; only ``"panel_partition"`` is supported.

    Notes
    -----
    The class is frozen so a configuration can be hashed and used as a static
    argument to a cached JIT-compiled M-step.

    Polishing uses ``polish_maxiter`` and a fixed Newton decrement tolerance of
    ``1e-10``. The M-step iteration budget and decrement tolerance come from
    :class:`OptimizationOptions`; its remaining solver settings also apply to
    polishing. The final score criterion depends on the scale of the predictors
    and certifies approximate stationarity, not a global maximum.
    """

    seed: int = 0
    max_em_iter: int = 2000
    em_tol: float = 1e-8
    score_tol: float = 1e-4
    polish: bool = True
    polish_maxiter: int = 25
    num_devices: int = field(default_factory=device_count)
    check_interval: int = 1
    starts: int = 1
    start_method: str = "panel_partition"

    def __post_init__(self) -> None:
        """Validate EM and multi-start settings."""
        for name in (
            "seed",
            "max_em_iter",
            "polish_maxiter",
            "num_devices",
            "check_interval",
            "starts",
        ):
            _require_integer(getattr(self, name), name)
        for name in ("em_tol", "score_tol"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        if self.em_tol <= 0:
            raise ValueError("em_tol must be positive.")
        if self.score_tol <= 0:
            raise ValueError("score_tol must be positive.")
        if self.max_em_iter < 1:
            raise ValueError("max_em_iter must be at least one.")
        if self.polish_maxiter < 0:
            raise ValueError("polish_maxiter must be nonnegative.")
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
    """Covariance and standard-error settings.

    Parameters
    ----------
    covariance : str, default="clustered"
        One of ``"clustered"``, ``"robust"``, or ``"unadjusted"``.  Latent-class
        models accept only ``"clustered"`` and ``"unadjusted"``, because the
        latent class is shared within a panel.
    cluster : str | None, default="panel"
        Grouping used when ``covariance="clustered"``.  ``"panel"`` clusters at
        the decision-maker.  Any other string names a column of the estimation
        data holding a coarser grouping, which must be constant within each
        panel — a household, market, or region identifier, for instance.  The
        value is ignored when ``covariance`` is not ``"clustered"``.
    finite_sample_correction : bool, default=True
        Apply the ``G / (G - 1)`` cluster multiplier, or ``n / (n - 1)`` for the
        unclustered sandwich, matching Stata's maximum-likelihood convention.
    skip : bool, default=False
        Skip covariance estimation entirely and return a matrix of ``NaN``.
    """

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
        if covariance == "clustered" and self.cluster is None:
            raise ValueError(
                "covariance='clustered' requires a cluster grouping. Pass "
                "cluster='panel' for decision-maker clustering, cluster='<column>' "
                "for a coarser grouping, or covariance='robust'/'unadjusted'."
            )
        if self.cluster is not None and not isinstance(self.cluster, str):
            raise ValueError("InferenceOptions.cluster must be a string or None.")

    @property
    def clusters_at_panel(self) -> bool:
        """Report whether clustering uses the panel identifier itself."""
        return self.covariance == "clustered" and self.cluster == "panel"

    @property
    def cluster_column(self) -> str | None:
        """Return the data column naming a coarser cluster, if any."""
        if self.covariance != "clustered" or self.cluster in (None, "panel"):
            return None
        return self.cluster


@dataclass
class DiagnosticsOptions:
    """Diagnostic switches and warning thresholds.

    Parameters
    ----------
    check_separation : bool, default=True
        Report whether any demographic cell has a class-membership probability
        pinned at zero.  When one does, the membership coefficients for that cell
        are unbounded and the observed information is singular in that direction,
        which is the usual cause of an otherwise puzzling rank deficiency.
    check_collinearity : bool, default=True
        Report the rank and conditioning of the observed information.
    warn_near_zero_numeraire : bool, default=True
        Warn when a class's numeraire coefficient sits near its floor, which makes
        every willingness-to-pay ratio for that class unstable.
    warn_large_coefficients : bool, default=True
        Warn on implausibly large utility coefficients.
    separation_threshold : float, default=1e-8
        Membership probability at or below which a cell counts as separated.
    near_zero_numeraire_threshold : float, default=1e-3
        Numeraire magnitude below which the near-zero warning fires, unless the
        specification supplies a constraint-specific ``warn_below`` threshold.
    large_coefficient_threshold : float, default=25.0
        Absolute coefficient magnitude above which the large-coefficient warning
        fires.
    """

    check_separation: bool = True
    check_collinearity: bool = True
    warn_near_zero_numeraire: bool = True
    warn_large_coefficients: bool = True
    separation_threshold: float = 1e-8
    near_zero_numeraire_threshold: float = 1e-3
    large_coefficient_threshold: float = 25.0

    def __post_init__(self) -> None:
        """Reject non-finite or out-of-range warning thresholds."""
        for name in (
            "separation_threshold",
            "near_zero_numeraire_threshold",
            "large_coefficient_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.separation_threshold > 1:
            raise ValueError("separation_threshold must be at most one.")


@dataclass(frozen=True)
class Options:
    """Configuration bundle accepted by all model-fitting entry points.

    Parameters
    ----------
    fit : FitOptions
        EM and multi-start settings, used by LCL and cross-validation.
    optimization : OptimizationOptions
        Solver settings used by both estimators.
    inference : InferenceOptions
        Covariance settings used by both estimators.
    diagnostics : DiagnosticsOptions
        LCL diagnostic switches and warning thresholds. Conditional logit uses
        ``check_collinearity``; membership and class-warning settings apply to LCL.

    Notes
    -----
    Conditional logit has no EM stage and does not use ``fit``. Do not combine
    ``options=`` with individual option arguments. Mutable inference and diagnostic
    settings are copied on fit, so later edits cannot change a fitted result.
    """

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
        return replace(
            options,
            inference=replace(options.inference),
            diagnostics=replace(options.diagnostics),
        )
    return Options(
        fit=fit_options if fit_options is not None else FitOptions(),
        optimization=(
            optimization_options
            if optimization_options is not None
            else OptimizationOptions()
        ),
        inference=replace(inference) if inference is not None else InferenceOptions(),
        diagnostics=(
            replace(diagnostics) if diagnostics is not None else DiagnosticsOptions()
        ),
    )


def _require_integer(value: object, name: str) -> None:
    """Reject fractional counts and booleans before iteration or seed coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")


def _resolve_weight_type(weight_type: str) -> str:
    """Validate a user-supplied weight interpretation."""
    normalized = str(weight_type).lower()
    aliases = {
        "pweight": "probability",
        "pweights": "probability",
        "survey": "probability",
        "importance": "probability",
        "fweight": "frequency",
        "fweights": "frequency",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in WEIGHT_TYPES:
        raise ValueError(
            "weight_type must be 'probability' (survey or sampling weights) or "
            "'frequency' (replication counts for collapsed data)."
        )
    return normalized


@dataclass
class PastChoicesData:
    """Array-style historical choices used to update class membership.

    Parameters
    ----------
    X : array-like, shape (rows, alt_vars)
        Historical utility design in fitted expanded-column order.
    y : array-like, shape (rows,)
        Binary historical choices, exactly one per (panel, case).
    alts, cases, panels : array-like, shape (rows,)
        Original identifiers aligned with ``X``. History may cover a subset of
        prediction panels; case IDs need only be unique within a panel.
    dems : array-like, shape (panels, dem_vars), optional
        Historical demographics retained for input compatibility. Membership
        priors use prediction demographics. Utility interactions must already be
        included in ``X``.
    dem_panel_ids : array-like, shape (panels,), optional
        IDs identifying rows of ``dems``. Without these, demographic rows follow
        sorted unique historical panel-ID order.
    """

    X: DesignInput
    y: ChoicesInput
    alts: RowIdsInput
    cases: RowIdsInput
    panels: RowIdsInput
    dems: DemographicsInput | None = None
    dem_panel_ids: PanelIdsInput | None = None


class PartitionType(str, Enum):
    """Supported binning strategies for WTP analysis."""

    CATEGORICAL = "categorical"
    QUINTILES = "quintiles"
    CUSTOM_BREAKS = "custom_breaks"

    def __str__(self) -> str:
        """Return the public value, including on Python 3.10."""
        return self.value


@dataclass
class WTPRequest:
    """Configuration for a partitioned marginal willingness-to-pay summary.

    Parameters
    ----------
    alt_var : str
        Raw utility attribute or expanded design-column target.
    demographic_var : str
        Panel-level grouping column, or a descriptive name for a dummy-coded factor.
    partition_type : PartitionType or str
        ``"categorical"``, ``"quintiles"``, or ``"custom_breaks"``.
    bins : list[float] | None, optional
        Finite, strictly increasing cutpoints for ``"custom_breaks"`` only.
        An empty list creates one group; integer bin counts are not supported.
    dummy_vars : list[str] | None, optional
        Mutually exclusive binary columns for a categorical partition.
    dummy_labels : list[str] | None, optional
        Distinct labels for ``dummy_vars``, defaulting to their column names.
    base_category : str, default="base"
        Label for panels with all dummy columns zero. It must differ from the
        other dummy labels. Used only for dummy-coded partitions.
    """

    alt_var: str
    demographic_var: str
    partition_type: PartitionType | str
    bins: list[float] | None = None
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
        if self.partition_type != PartitionType.CUSTOM_BREAKS and self.bins is not None:
            raise ValueError("bins is only used with partition_type='custom_breaks'.")
        if self.bins is not None and not all(math.isfinite(x) for x in self.bins):
            raise ValueError("Custom WTP breakpoints must be finite.")
        if self.dummy_labels is not None and self.dummy_vars is None:
            raise ValueError("dummy_labels requires dummy_vars.")
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
            labels = (
                self.dummy_labels if self.dummy_labels is not None else self.dummy_vars
            )
            if len(set([self.base_category, *labels])) != len(labels) + 1:
                raise ValueError(
                    "Dummy partition labels and base_category must be distinct."
                )
            if self.dummy_labels is not None and len(self.dummy_labels) != len(
                self.dummy_vars
            ):
                raise ValueError("dummy_labels must have one label per dummy column.")


__all__ = [
    "DEFAULT_NEWTON_DECREMENT_TOL",
    "DiagnosticsOptions",
    "FitOptions",
    "InferenceOptions",
    "Options",
    "OptimizationOptions",
    "PartitionType",
    "PastChoicesData",
    "WEIGHT_TYPES",
    "WTPRequest",
]
