"""Containers for data."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, Optional, Union

from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Bool, Float64, UInt


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
class MleConfig:
    """Container for Maximum Likelihood Estimation (MLE) optimization options."""

    maxiter: int = 75
    ftol: float = 1e-5


@dataclass  # (frozen=True)
class EMAlgConfig:
    """Container for Expectation-Maximization (EM) algorithm options."""

    jax_prng_seed: int = 0
    loglik_tol: float = 1e-6
    maxiter: int = 2000


@dataclass  # (frozen=True)
class ErrorConfig:
    """Container for standard error and covariance matrix options."""

    robust: bool = True
    skip_std_errs: bool = False


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
    """Container for a decision-maker's historical choices.

    Pass to predict() to update conditional class membership probabilities
    using Bayesian updating for out-of-sample counterfactual prediction.
    """

    X: ArrayLike
    y: ArrayLike
    alts: ArrayLike
    cases: ArrayLike
    panels: ArrayLike
    dems: ArrayLike | None = None


class PartitionType(StrEnum):
    CATEGORICAL = "categorical"
    QUINTILES = "quintiles"
    CUSTOM_BREAKS = "custom_breaks"


@dataclass
class WTPRequest:
    alt_var: str
    demographic_var: str
    partition_type: PartitionType | str
    bins: Optional[Union[int, list[float]]] = None

    def __post_init__(self) -> None:
        """Catch runtime errors if the user overlooks type hints"""
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


# EOF
