from abc import ABC, abstractmethod
from time import time
from typing import Any, Sequence

import jax.numpy as jnp
import polars as pl
from formulaic import Formula  # type: ignore
from jax import Array

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
        self.case_varnames, self.dem_varnames = [], []
        self.numeraire = None
        self.convergence = False
        self._fit_start_time = 0.0

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
            R-style formula string (e.g., "choice ~ price + C(brand) | income").
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
        # 1. Coerce to Polars DataFrame
        if isinstance(data, pl.DataFrame):
            df = data
        elif hasattr(data, "columns"):
            df = pl.from_pandas(data)
        else:
            df = pl.DataFrame(data)

        # CRITICAL: Sort values contiguously. JAX's `jnp.roll` masking relies entirely on
        # observations being grouped contiguously by panel and case.
        df = df.sort([panels_col, cases_col, alts_col])

        # 2. Extract Sequential Zero-Indexed IDs safely
        df = df.with_columns(
            [
                pl.col(panels_col).rank(method="dense").sub(1).alias("_seq_panels"),
                pl.col(cases_col).rank(method="dense").sub(1).alias("_seq_cases"),
                pl.col(alts_col).rank(method="dense").sub(1).alias("_seq_alts"),
            ]
        )

        # 3. Parse Features (Formula vs Explicit Lists)
        if formula:
            f = Formula(formula)
            y_df = f.lhs.get_model_matrix(df.to_pandas())
            y_array = jnp.array(y_df.to_numpy().ravel(), dtype="bool")

            # Handle multiple RHS components (e.g., choice ~ price | income)
            if isinstance(f.rhs, tuple):
                X_df = f.rhs[0].get_model_matrix(df.to_pandas())
                case_vars = list(X_df.columns)
                dems_df_raw = f.rhs[1].get_model_matrix(df.to_pandas())
                dem_vars = list(dems_df_raw.columns)

                df = df.with_columns(pl.from_pandas(X_df))
                df = df.with_columns(pl.from_pandas(dems_df_raw))
            else:
                X_df = f.rhs.get_model_matrix(df.to_pandas())
                case_vars = list(X_df.columns)
                dem_vars = None

                df = df.with_columns(pl.from_pandas(X_df))

        else:
            if not choice_col or not case_varnames:
                raise ValueError(
                    "Must provide either a 'formula' OR 'choice_col' and 'case_varnames'."
                )
            if choice_col in df.columns:
                y_array = jnp.array(df[choice_col].to_numpy(), dtype="bool")
            else:
                y_array = None

            case_vars = list(case_varnames)
            dem_vars = list(dem_varnames) if dem_varnames else None

        # Single-outcome check
        if y_array is not None:
            import numpy as onp

            cases_array = df["_seq_cases"].to_numpy()
            y_onp = onp.asarray(y_array, dtype=onp.int32)

            # Count the number of True values per case
            choices_per_case = onp.bincount(cases_array, weights=y_onp)

            if not onp.all(choices_per_case == 1):
                raise ValueError(
                    "Data inconsistency: Every choice situation must have exactly one "
                    "chosen alternative (True/1). Check your data or formula."
                )

        X_array = jnp.array(df.select(case_vars).to_numpy(), dtype="float64")

        # 4. Handle Demographics (Inline vs Separate Dataset)
        dems_array = None
        if dem_vars:
            if dems_data is not None:
                if isinstance(dems_data, pl.DataFrame):
                    dems_df_ext = dems_data
                elif hasattr(dems_data, "columns"):
                    dems_df_ext = pl.from_pandas(dems_data)
                else:
                    dems_df_ext = pl.DataFrame(dems_data)

                unique_panels = df.select([panels_col, "_seq_panels"]).unique(
                    subset=[panels_col], maintain_order=True
                )
                aligned_dems = unique_panels.join(
                    dems_df_ext, on=panels_col, how="left"
                )
                dems_array = jnp.array(
                    aligned_dems.select(dem_vars).to_numpy(), dtype="float64"
                )
            else:
                unique_dems = df.select(["_seq_panels"] + dem_vars).unique(
                    subset=["_seq_panels"], maintain_order=True
                )
                dems_array = jnp.array(
                    unique_dems.select(dem_vars).to_numpy(), dtype="float64"
                )

        return ParsedData(
            X=X_array,
            dems=dems_array,
            y=y_array,
            cases=jnp.array(df["_seq_cases"].to_numpy(), dtype="uint32"),
            panels=jnp.array(df["_seq_panels"].to_numpy(), dtype="uint32"),
            alts=jnp.array(df["_seq_alts"].to_numpy(), dtype="uint32"),
            case_varnames=case_vars,
            dem_varnames=dem_vars,
        )

    def _setup_data(
        self,
        parsed: ParsedData,
        weights: Array | None = None,
        init_beta: Array | None = None,
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
        panels_of_cases = parsed.panels[mask_first_obs_by_id]

        # Because `_ingest_data` guarantees zero-indexed, sequential panels,
        # we can leverage lightning-fast native JAX `bincount` instead of Polars group-bys.
        num_cases_per_panel = jnp.bincount(panels_of_cases)
        num_panels = num_cases_per_panel.shape[0]
        num_cases = jnp.unique(parsed.cases).shape[0]

        num_dem_vars = parsed.dems.shape[1] if parsed.dems is not None else 0

        weights = (
            jnp.ones(shape=num_cases, dtype="float64") if weights is None else weights
        )
        init_beta = (
            jnp.zeros(shape=len(parsed.case_varnames), dtype="float64")
            if init_beta is None
            else init_beta
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


# EOF
