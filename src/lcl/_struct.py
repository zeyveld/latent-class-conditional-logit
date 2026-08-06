"""Containers for data."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple, Optional, Union

from jax import device_count
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float64, UInt


@dataclass
class ParsedData:
    """Intermediate container for aligned, JAX-ready arrays and metadata.

    Attributes
    ----------
    X : Array
        The design matrix of alternative-specific characteristics.
    dems : Array | None
        The matrix of decision-maker demographic characteristics, tightly aligned.
    y : Array | None
        Boolean array indicating the chosen alternatives.
    cases : Array
        Sequential, zero-indexed choice situation IDs.
    alts : Array
        Sequential, zero-indexed alternative IDs.
    panels : Array
        Sequential, zero-indexed decision-maker IDs.
    case_varnames : list[str]
        Extracted names of the alternative-specific features.
    dem_varnames : list[str] | None
        Extracted names of the demographic features.
    """

    X: Array
    dems: Array | None
    y: Array | None
    cases: Array
    alts: Array
    panels: Array
    case_varnames: list[str]
    dem_varnames: Optional[list[str]]
    original_alts: Any | None = None
    original_cases: Any | None = None
    original_panels: Any | None = None


class Data(NamedTuple):
    """Container for choice data used by the estimation engine.

    This immutable struct houses the primary design matrices and metadata required
    to evaluate conditional logit likelihoods across choice situations and decision-makers.

    Attributes
    ----------
    X : Float64[Array, "alts_by_case alt_vars"]
        The design matrix of alternative-specific characteristics (features) in long format.
    dems : Float64[Array, "panels dem_vars"] | None
        The matrix of decision-maker (panel) demographic characteristics.
    y : Bool[Array, "alts_by_case"] | None
        Boolean array indicating the chosen alternatives.
    alts : UInt[Array, "alts_by_case"]
        Vector of alternative IDs corresponding to each row in the design matrix.
    cases : UInt[Array, "alts_by_case"]
        Vector grouping rows into distinct choice situations (cases).
    panels : UInt[Array, "alts_by_case"] | None
        Vector mapping rows to specific decision-makers (panels).
    panels_of_cases : UInt[Array, "cases"] | None
        Vector mapping each choice situation (case) to its respective decision-maker.
    num_cases_per_panel : UInt[Array, "panels"] | None
        Vector containing the count of choice situations observed per decision-maker.
    num_cases : int
        Total number of choice situations observed.
    num_alt_vars : int
        Number of alternative-specific characteristics (taste parameters).
    num_panels : int | None
        Total number of unique decision-makers.
    num_dem_vars : int
        Number of demographic variables.
    """

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
    """Container for the differenced design matrix.

    To efficiently compute the log-likelihood, the characteristics of the chosen
    alternative are subtracted from the characteristics of the unchosen alternatives
    within each choice situation.

    Attributes
    ----------
    X : Float64[Array, "unchosen_alts_by_case alt_vars"]
        Differenced design matrix: :math:`X_{ij} - X_{iy_i}`.
    alts : UInt[Array, "unchosen_alts_by_case"]
        Alternative IDs for the unchosen alternatives.
    cases : UInt[Array, "unchosen_alts_by_case"]
        Choice situation IDs for the unchosen alternatives.
    panels : UInt[Array, "unchosen_alts_by_case"] | None
        Decision-maker IDs for the unchosen alternatives.
    num_cases : int
        Total number of choice situations.
    """

    X: Float64[Array, "unchosen_alts_by_case alt_vars"]
    alts: UInt[Array, "unchosen_alts_by_case"]
    cases: UInt[Array, "unchosen_alts_by_case"]
    panels: UInt[Array, "unchosen_alts_by_case"] | None
    num_cases: int


@dataclass
class OptimizationOptions:
    """User-facing optimizer settings.

    Parameters
    ----------
    maxiter : int, default=75
        Maximum Newton iterations used inside each M-step.
    method : str, default="newton"
        Optimizer family requested by the user.  The latent-class M-step currently
        uses exact Newton updates.
    gradient_tol : float, default=1e-5
        Infinity-norm tolerance on the mean negative-log-likelihood gradient.
    hessian_damping : float, default=1e-6
        Initial relative diagonal shift used for modified-Cholesky regularization.
    max_step_norm : float, default=25.0
        Maximum Euclidean norm of a trial search direction.
    line_search : str, default="armijo"
        Name of the line-search strategy.
    fallback : str, default="gradient"
        Fallback direction when the Newton direction is not a descent direction.
    """

    maxiter: int = 75
    gradient_tol: float = 1e-5
    hessian_damping: float = 1e-6
    max_step_norm: float = 25.0
    line_search_maxiter: int = 25
    accept_any_decrease: bool = False
    method: str = "newton"
    line_search: str = "armijo"
    fallback: str = "gradient"

    def __post_init__(self) -> None:
        """Validate optimizer settings."""
        if self.maxiter < 0:
            raise ValueError("maxiter must be nonnegative.")
        if self.gradient_tol <= 0:
            raise ValueError("gradient_tol must be positive.")
        if self.hessian_damping <= 0:
            raise ValueError("hessian_damping must be positive.")
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
    """User-facing EM fit options.

    Parameters
    ----------
    seed : int, default=0
        Random seed used for panel-partition starts.
    max_em_iter : int, default=2000
        Maximum number of EM recursions.
    em_tol : float, default=1e-6
        Relative log-likelihood tolerance checked over the EM history.
    num_devices : int, default=device_count()
        Number of local JAX devices used for class-wise beta updates.
    check_interval : int, default=10
        Frequency of convergence checks.
    starts : int, default=1
        Number of independent panel-partition starts.
    start_method : str, default="panel_partition"
        Initialization method label.
    """

    seed: int = 0
    max_em_iter: int = 2000
    em_tol: float = 1e-6
    num_devices: int = field(default_factory=device_count)
    check_interval: int = 10
    starts: int = 1
    start_method: str = "panel_partition"

    def __post_init__(self) -> None:
        """Validate user-facing EM settings."""
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
    """User-facing inference and covariance settings.

    Parameters
    ----------
    covariance : str, default="clustered"
        Covariance estimator label. ``"none"``/``"unadjusted"`` disable the
        sandwich correction; ``"clustered"`` and ``"robust"`` enable it.
    cluster : str | None, default="panel"
        Cluster level label for reports.  The current latent-class estimator
        clusters at panel level.
    finite_sample_correction : bool, default=True
        Whether reports should describe the finite-sample correction.
    skip : bool, default=False
        Skip covariance and standard-error calculations.
    """

    covariance: str = "clustered"
    cluster: str | None = "panel"
    finite_sample_correction: bool = True
    skip: bool = False

    def __post_init__(self) -> None:
        """Normalize user-facing covariance labels."""
        covariance = self.covariance.lower()
        if covariance in {"none", "unadjusted", "hessian"}:
            covariance = "unadjusted"
        elif covariance in {"clustered", "robust", "sandwich"}:
            covariance = "clustered"
        else:
            raise ValueError(
                "InferenceOptions.covariance must be one of 'clustered', "
                "'robust', 'sandwich', 'unadjusted', or 'none'."
            )
        self.covariance = covariance


@dataclass
class DiagnosticsOptions:
    """Options controlling public diagnostic summaries."""

    check_separation: bool = True
    check_collinearity: bool = True
    warn_near_zero_numeraire: bool = True
    warn_large_coefficients: bool = True
    near_zero_numeraire_threshold: float = 1e-3
    large_coefficient_threshold: float = 25.0


@dataclass
class OptimizeResult:
    """Container for optimization results.

    Patterned after Virtanen, Pauli, Ralf Gommers, Travis E. Oliphant,
        Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski et al.
        "SciPy 1.0: fundamental algorithms for scientific computing in Python."
        Nature methods 17, no. 3 (2020): 261-272.
    """

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


class EMVars(NamedTuple):
    """Container for parameters iteratively updated by the EM algorithm.

    Attributes
    ----------
    latent_betas : Float64[Array, "alt_vars classes"] | None
        Unconstrained taste parameters optimized by the MLE solver.
    structural_betas : Float64[Array, "alt_vars classes"] | None
        Structural taste parameters (e.g., enforcing sign constraints on the numeraire).
    thetas : Float64[Array, "dem_vars+1 classes-1"] | None
        Coefficients relating demographic variables to latent class membership.
    shares : Float64[Array, "classes"] | None
        Aggregate unconditional class shares (used when demographics are absent).
    unconditional_loglik : Float64[Array, ""]
        Scalar unconditional log-likelihood for the current EM step.
    class_probs_by_panel : Float64[Array, "panels classes"] | None
        Posterior probabilities of class membership for each decision-maker, conditional
        on their observed choice sequence.
    """

    latent_betas: Float64[Array, "alt_vars classes"] | None
    structural_betas: Float64[Array, "alt_vars classes"] | None
    thetas: (
        Float64[Array, "dem_vars+1 classes-1"] | None
    )  # (num_dem_vars + 1, num_classes - 1)  # (D + 1, C - 1)
    shares: Float64[Array, "classes"] | None
    unconditional_loglik: Float64[Array, ""]
    class_probs_by_panel: Float64[Array, "panels classes"] | None


@dataclass
class PastChoicesData:
    """Array-style container for historical choices used during prediction.

    Pass to ``LCLResults.predict(past_choices=...)`` to update latent-class
    membership probabilities with observed choices before scoring counterfactual
    choice sets. Users with tabular historical-choice data can pass that DataFrame
    directly to ``past_choices``; this wrapper is intended for callers that already
    manage design matrices and ID arrays.

    Attributes
    ----------
    X : ArrayLike
        Alternative-specific design matrix in long format.
    y : ArrayLike
        Boolean or binary choice indicators aligned to rows of ``X``.
    alts : ArrayLike
        Alternative identifiers aligned to rows of ``X``.
    cases : ArrayLike
        Choice-situation identifiers aligned to rows of ``X``.
    panels : ArrayLike
        Decision-maker identifiers aligned to rows of ``X``.
    dems : ArrayLike | None, optional
        Panel-level demographic matrix, one row per unique panel. When
        ``dem_panel_ids`` is omitted, rows must follow sorted unique panel-ID order.
    dem_panel_ids : ArrayLike | None, optional
        Panel identifiers aligned to rows of ``dems``. Supplying these IDs allows
        the parser to validate and reorder demographic rows safely.
    """

    X: ArrayLike
    y: ArrayLike
    alts: ArrayLike
    cases: ArrayLike
    panels: ArrayLike
    dems: ArrayLike | None = None
    dem_panel_ids: ArrayLike | None = None


class PartitionType(StrEnum):
    """Supported binning strategies for marginal Willingness-To-Pay (WTP) analysis."""

    CATEGORICAL = "categorical"
    QUINTILES = "quintiles"
    CUSTOM_BREAKS = "custom_breaks"


@dataclass
class WTPRequest:
    """Configuration object for calculating Marginal Willingness-to-Pay (WTP).

    Parameters
    ----------
    alt_var : str
        The target alternative-specific variable for the WTP numerator.
    demographic_var : str
        The demographic variable used to partition the decision-makers. When
        ``dummy_vars`` is supplied, this is the semantic factor name used in the
        output table.
    partition_type : PartitionType | str
        The strategy for grouping ``demographic_var``. Dummy-coded factors must use
        ``PartitionType.CATEGORICAL``.
    bins : list[float] | None, optional
        Custom breakpoints if ``partition_type`` is ``"custom_breaks"``.
    dummy_vars : list[str] | None, optional
        One-hot dummy columns that jointly represent a categorical variable. The
        all-zero row is treated as the base category.
    dummy_labels : list[str] | None, optional
        Display labels for ``dummy_vars`` in the same order. Defaults to the dummy
        column names.
    base_category : str, default="base"
        Display label for the all-zero base category when ``dummy_vars`` is supplied.
    """

    alt_var: str
    demographic_var: str
    partition_type: PartitionType | str
    bins: Optional[Union[int, list[float]]] = None
    dummy_vars: list[str] | None = None
    dummy_labels: list[str] | None = None
    base_category: str = "base"

    def __post_init__(self) -> None:
        """Validate and normalize WTP request options."""
        # Attempt to coerce partition into PartitionType
        if not isinstance(self.partition_type, PartitionType):
            try:
                self.partition_type = PartitionType(self.partition_type)
            except ValueError:
                valid_options = [e.value for e in PartitionType]
                raise ValueError(
                    "\n".join(
                        [
                            f"Invalid partition type: {self.partition_type}",
                            f"Must be one of {valid_options}",
                        ]
                    )
                )
        # When using custom breaks, ensure bins are a list of breakpoints
        if self.partition_type == PartitionType.CUSTOM_BREAKS and not isinstance(
            self.bins, list
        ):
            raise ValueError(
                "When partition_type is 'custom_breaks', 'bins' must be a list of breakpoints."
            )
        if isinstance(self.bins, list) and any(
            right <= left for left, right in zip(self.bins, self.bins[1:])
        ):
            raise ValueError("Custom WTP breakpoints must be strictly increasing.")
        if self.dummy_vars is not None:
            if not self.dummy_vars:
                raise ValueError("'dummy_vars' must contain at least one column name.")
            if len(set(self.dummy_vars)) != len(self.dummy_vars):
                raise ValueError("'dummy_vars' cannot contain duplicate column names.")
            if self.partition_type != PartitionType.CATEGORICAL:
                raise ValueError(
                    "Dummy-coded WTP partitions require partition_type='categorical'."
                )
            if self.dummy_labels is not None and len(self.dummy_labels) != len(
                self.dummy_vars
            ):
                raise ValueError(
                    "'dummy_labels' must have one label for each dummy column."
                )
