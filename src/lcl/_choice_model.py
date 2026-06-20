from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import time
from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from lcl._encoding import ChoiceDataEncoder
from lcl._struct import Data, ParsedData


class ChoiceModel(ABC):
    """Abstract base class for discrete choice models.

    Attributes
    ----------
    case_varnames : list[str]
        Names of the alternative-specific variables.
    numeraire : str | None
        The name of the numeraire variable, if specified.
    dem_varnames : list[str] | None
        Names of the demographic variables, if any.
    convergence : bool
        Indicates whether the optimization routine converged successfully.
    """

    case_varnames: list[str]
    numeraire: str | None
    dem_varnames: list[str] | None
    convergence: bool
    _fit_start_time: float

    def __init__(self) -> None:
        """Initialize shared model metadata before fitting."""
        self.case_varnames, self.dem_varnames = [], []
        self.numeraire = None
        self.convergence = False
        self._fit_start_time = 0.0
        self._encoder: ChoiceDataEncoder | None = None

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the model to the data. Must be implemented by subclasses."""
        pass

    def _ingest_data(
        self,
        data: Any,
        alts_col: str,
        cases_col: str,
        panels_col: str,
        formula: str | None,
        choice_col: str | None,
        case_varnames: Sequence[str] | None,
        dem_varnames: Sequence[str] | None,
        dems_data: Any | None,
        utility_formula: str | None = None,
        membership_formula: str | None = None,
    ) -> ParsedData:
        """The core ingestion layer: Converts flexible user inputs into strict matrices.

        This method centralizes the data wrangling process. It handles sorting,
        merging separate demographic datasets, generating zero-indexed sequential IDs
        (essential for JAX operations), and parsing R-style formulas.

        Parameters
        ----------
        data : Any
            The main dataset (pandas/Polars DataFrame) containing choice situations.
        alts_col : str
            Name of the column containing alternative identifiers.
        cases_col : str
            Name of the column grouping observations into distinct choice situations.
        panels_col : str
            Name of the column mapping observations to specific decision-makers.
        formula : str | None
            Backward-compatible combined Formulaic string such as
            ``"choice ~ price + C(brand) | income"``.
        utility_formula : str | None, optional
            Formulaic string for the alternative-specific utility specification,
            such as ``"choice ~ price + C(brand)"``.  A right-hand-side-only
            formula may be used when ``choice_col`` is supplied.
        membership_formula : str | None, optional
            Right-hand-side Formulaic string for class membership demographics,
            such as ``"~ income + C(segment)"``.
        choice_col : str | None
            Name of the boolean/binary column indicating chosen alternatives.
        case_varnames : Sequence[str] | None
            List of alternative-specific variables.
        dem_varnames : Sequence[str] | None
            List of demographic variables.
        dems_data : Any | None
            A separate panel-level dataset containing demographics.

        Returns
        -------
        :class:`~lcl._struct.ParsedData`
            A container holding strictly aligned and sorted JAX arrays and their metadata.
        """
        self._encoder = ChoiceDataEncoder(
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            formula=formula,
            utility_formula=utility_formula,
            membership_formula=membership_formula,
            choice_col=choice_col,
            explicit_case_varnames=case_varnames,
            explicit_dem_varnames=dem_varnames,
        )
        return self._encoder.fit_transform(data, dems_data=dems_data)

    def _transform_data(
        self,
        data: Any,
        dems_data: Any | None = None,
        require_choice: bool = False,
    ) -> ParsedData:
        """Transform new data with the encoder learned during fitting.

        Parameters
        ----------
        data : Any
            New long-format data to encode.
        dems_data : Any | None, optional
            Optional panel-level demographics for prediction data.
        require_choice : bool, default=False
            Whether a valid choice indicator must be present.

        Returns
        -------
        :class:`~lcl._struct.ParsedData`
            Encoded arrays and metadata aligned to the fitted model specification.
        """
        if self._encoder is None:
            raise ValueError("Model must be fitted before transforming new data.")
        return self._encoder.transform(
            data, dems_data=dems_data, require_choice=require_choice
        )

    def _setup_data(
        self,
        parsed: ParsedData,
        weights: ArrayLike | None = None,
        init_beta: ArrayLike | None = None,
    ) -> tuple[Data, Array, Array]:
        """Construct the immutable Data struct for the JAX estimation engine.

        Extracts panel counts and bounds, relying on the guarantee from `_ingest_data`
        that panel and case IDs are strictly contiguous and zero-indexed.

        Parameters
        ----------
        parsed : :class:`~lcl._struct.ParsedData`
            The sanitized, aligned dataset generated by the ingestion layer.
        weights : Array | None, optional
            Case-level importance weights. If None, defaults to ones.
        init_beta : Array | None, optional
            Starting values for taste parameters. If None, defaults to zeros.

        Returns
        -------
        tuple
            A tuple containing `(data_struct, weights, init_beta)`.
        """
        # Mask first observation for differencing logic
        mask_first_obs_by_id = parsed.cases != jnp.roll(parsed.cases, shift=1)
        mask_first_obs_by_id = mask_first_obs_by_id.at[0].set(True)
        panels_of_cases = parsed.panels[mask_first_obs_by_id]

        # Because `_ingest_data` guarantees zero-indexed, sequential panels,
        # we can leverage lightning-fast native JAX `bincount` instead of Polars group-bys.
        num_cases_per_panel = jnp.bincount(panels_of_cases)
        num_panels = num_cases_per_panel.shape[0]
        num_cases = jnp.unique(parsed.cases).shape[0]

        num_dem_vars = parsed.dems.shape[1] if parsed.dems is not None else 0

        weights = (
            jnp.ones(shape=num_cases, dtype="float64")
            if weights is None
            else jnp.array(weights, dtype="float64")
        )
        if weights.shape != (num_cases,):
            raise ValueError("weights must have one entry per choice situation.")

        init_beta = (
            jnp.zeros(shape=len(parsed.case_varnames), dtype="float64")
            if init_beta is None
            else jnp.array(init_beta, dtype="float64")
        )
        if init_beta.shape != (len(parsed.case_varnames),):
            raise ValueError(
                "init_beta must have one entry per alternative-specific variable."
            )

        data_struct = Data(
            X=parsed.X,
            y=parsed.y,
            dems=parsed.dems,
            alts=parsed.alts,
            cases=parsed.cases,
            panels=parsed.panels,
            panels_of_cases=panels_of_cases,
            num_cases=num_cases,
            num_alt_vars=len(parsed.case_varnames),
            num_dem_vars=num_dem_vars,
            num_cases_per_panel=num_cases_per_panel,
            num_panels=num_panels,
        )
        return data_struct, weights, init_beta

    def _pre_fit(
        self,
        case_varnames: Sequence[str],
        dem_varnames: Sequence[str] | None,
        numeraire: str | None,
    ) -> None:
        """Initialize metadata tracking prior to estimation loop."""
        self._fit_start_time = time()
        self.case_varnames = list(case_varnames)
        self.dem_varnames = list(dem_varnames) if dem_varnames is not None else None
        self.numeraire = numeraire
        self.numeraire_idx = case_varnames.index(numeraire) if numeraire else None
