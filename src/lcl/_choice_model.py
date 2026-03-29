from abc import ABC
from functools import partial
from time import time
from typing import Sequence

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import Array, tree
from jax.typing import ArrayLike
from jaxtyping import Bool, Float64, UInt

from lcl._struct import Data
from lcl.utils import _as_array_or_none, _ensure_sequential


class ChoiceModel(ABC):
    """Base class for the specification and estimation of discrete choice models.

    Provides core data validation, dimensionality reduction, and structural
    preparation routines shared across cross-sectional and panel choice models.
    """

    case_varnames: list[str]
    numeraire: str | None
    dem_varnames: list[str] | None
    convergence: bool
    _fit_start_time: float

    def __init__(self) -> None:
        self.case_varnames, self.dem_varnames = [], []
        self.numeraire = None
        self.convergence = False
        self._fit_start_time = 0.0

    def _setup_data(
        self,
        X: ArrayLike,
        dems: ArrayLike | None,
        cases: ArrayLike,
        alts: ArrayLike,
        y: ArrayLike | None,
        panels: ArrayLike | None,
        weights: ArrayLike | None = None,
        init_beta: ArrayLike | None = None,
    ) -> tuple[Data, Float64[Array, "cases"], Float64[Array, "alt_vars"]]:
        """Validate and package raw arrays into the core estimation struct.

        Ensures that cases and panels are contiguous and sequential, masks first
        observations for differencing logic, and extracts observation counts.

        Parameters
        ----------
        X : ArrayLike
            ``(N, K)`` design matrix of alternative-specific characteristics in long format.
        dems : ArrayLike | None
            ``(Np, D)`` or ``(N, D)`` matrix of decision-maker demographic characteristics.
        cases : ArrayLike
            ``(N,)`` vector grouping rows into distinct choice situations.
        alts : ArrayLike
            ``(N,)`` vector of alternative identifiers.
        y : ArrayLike | None
            ``(N,)`` boolean vector indicating the chosen alternatives.
        panels : ArrayLike | None
            ``(N,)`` vector mapping observations to specific decision-makers.
        weights : ArrayLike | None, optional
            ``(Nc,)`` vector of choice situation importance weights.
        init_beta : ArrayLike | None, optional
            ``(K,)`` vector of initial taste parameters for the optimization routine.

        Returns
        -------
        data : :class:`~lcl._struct.Data`
            Immutable container holding the validated design matrices and metadata.
        weights : Float64[Array, "cases"]
            Vector of importance weights per choice situation.
        init_beta : Float64[Array, "alt_vars"]
            Vector of starting values for the structural parameters.
        """
        # Convert args to arrays (when provided)

        X, dems, weights = tree.map(
            partial(_as_array_or_none, dtype="float64"), (X, dems, weights)
        )
        assert isinstance(X, Array), "Unable to convert `X` to Jax array"
        if X.ndim != 2:
            raise ValueError("`X` must be an array of two dimensions in long format")
        if dems is not None:
            assert isinstance(dems, Array), "Unable to convert `dems` to Jax array"

        cases, alts, panels = tree.map(
            partial(_as_array_or_none, dtype="uint32"), (cases, alts, panels)
        )
        y = _as_array_or_none(y, "bool")

        # Ensure case cases and panels are sequential
        cases = _ensure_sequential(cases)
        if panels is not None:
            panels = _ensure_sequential(panels)

        # Mask first observation for each choice ID
        mask_first_obs_by_id: Bool[Array, "alts_by_case"] = cases != jnp.roll(
            cases, shift=1
        )

        # Get one panel per choice situation
        if panels is not None:
            # Old version: panels_of_cases = panels[y]
            panels_of_cases = panels[mask_first_obs_by_id]
        else:
            panels_of_cases = None

        num_cases_per_panel, num_panels = self._count_choices_per_panel(panels, cases)
        num_cases = jnp.unique(cases).shape[0]

        dems, num_dem_vars = self._squeeze_dems(
            X=X,
            dems=dems,
            panels=panels,
            panels_of_cases=panels_of_cases,
            num_panels=num_panels,
            num_cases=num_cases,
        )

        # Weights equal unity if not explicitly provided
        if weights is None:
            weights = jnp.ones(shape=num_cases, dtype="float64")
        else:
            weights = jnp.asarray(weights, dtype="float64")

        # Get array representation of case-data explanatory variables
        case_varnames = onp.array(self.case_varnames.copy())

        # Initialize or inspect initial coefficients guess
        num_alt_vars = len(self.case_varnames)
        if init_beta is None:
            init_beta = jnp.zeros(shape=num_alt_vars, dtype="float64")
        else:
            init_beta = jnp.asarray(init_beta, dtype="float64")
            if jnp.size(init_beta) != num_alt_vars:
                raise ValueError(f"The size of `initial_coeff` must be: {num_alt_vars}")

        # Manually check individual args so that linter isn't confused

        assert isinstance(dems, Array) or (dems is None)
        if y is not None and y.ndim > 1:
            raise ValueError("`y` must be an array of one dimension in long format")
        assert isinstance(alts, Array), "Unable to convert `alts` to Jax array"
        if case_varnames.size != X.shape[1]:
            raise ValueError(
                "The length of `case_varnames` must match the number of columns in `X`"
            )

        return (
            Data(
                X=X,
                y=y,
                dems=dems,
                alts=alts,
                cases=cases,
                panels=panels,
                panels_of_cases=panels_of_cases,
                num_cases=num_cases,
                num_alt_vars=num_alt_vars,
                num_dem_vars=num_dem_vars,
                num_cases_per_panel=num_cases_per_panel,
                num_panels=num_panels,
            ),
            weights,
            init_beta,
        )

    @staticmethod
    def _squeeze_dems(
        X: Float64[Array, "alts_by_case alt_vars"],
        dems: Float64[Array, "panels dem_vars"]
        | Float64[Array, "alts_by_case dem_vars"]
        | None,
        panels: UInt[Array, "alts_by_case"] | None,
        panels_of_cases: UInt[Array, "cases"] | None,
        num_panels: int | None,
        num_cases: int,
    ) -> tuple[Float64[Array, "panels dem_vars"] | None, int]:
        """Reduce demographic design matrix to a strict panel-level dimensionality.

        Essential for the fractional response regression step, which models latent class
        membership at the decision-maker level rather than the choice-situation level.
        """
        if dems is None:
            return None, 0

        # If `dems` is passed, check that `panels` and related vars seem usable
        if panels is None:
            raise ValueError(
                "The parameter `panels` needs to be array-valued if `dems` is not None"
            )
        assert panels_of_cases is not None and num_panels is not None, (
            "Something is wrong with the `panels` argument"
        )
        num_dem_vars = dems.shape[1]
        # Squeeze `dems` to have one observation per panel for the fractional logit model
        if dems.shape[0] == num_panels:
            dems = dems
        elif dems.shape[0] == num_cases:
            dems = dems[panels_of_cases != jnp.roll(panels_of_cases, shift=1)]
        elif dems.shape[0] == X.shape[0]:
            dems = dems[panels != jnp.roll(panels, shift=1)]
        else:
            raise Exception("Could not squeeze `dems` into a usable shape.")
        return dems, num_dem_vars

    @staticmethod
    def _count_choices_per_panel(
        panels: UInt[Array, "alts_by_case"] | None, cases: UInt[Array, "alts_by_case"]
    ) -> tuple[UInt[Array, "panels"] | None, int | None]:
        """Determine the sequence length (number of choice situations) per decision-maker."""
        if panels is None:
            return None, None

        panels_cases_df = pl.DataFrame(
            {"panels": onp.array(panels), "cases": onp.array(cases)}
        ).unique()
        num_cases_per_panel = (
            panels_cases_df.group_by("panels")
            .agg(num_choices=pl.len())
            .sort("panels")["num_choices"]
            .cast(pl.UInt32)
            .to_jax()
        )
        return num_cases_per_panel, num_cases_per_panel.shape[0]

    def _pre_fit(
        self,
        case_varnames: Sequence[str],
        dem_varnames: Sequence[str] | None = None,
        numeraire: str | None = None,
    ) -> None:
        self._fit_start_time = time()
        self.case_varnames = list(case_varnames)  # Easier to handle with lists
        self.dem_varnames = list(dem_varnames) if dem_varnames is not None else None
        self.numeraire = numeraire


# EOF
