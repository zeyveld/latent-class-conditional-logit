from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from time import time
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax.typing import ArrayLike
from jaxtyping import Array

from lcl._encoding import ChoiceDataEncoder, _coerce_frame
from lcl._labels import label_for_variable, normalize_variable_labels
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
    variable_labels : dict[str, str]
        Optional mapping from raw variable names to presentation labels.
    convergence : bool
        Indicates whether the optimization routine converged successfully.
    """

    case_varnames: list[str]
    numeraire: str | None
    dem_varnames: list[str] | None
    variable_labels: dict[str, str]
    convergence: bool
    _fit_start_time: float

    def __init__(self) -> None:
        """Initialize shared model metadata before fitting."""
        self.case_varnames, self.dem_varnames = [], []
        self.numeraire = None
        self.variable_labels = {}
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
        choice_col: str | None,
        case_varnames: Sequence[str] | None,
        dem_varnames: Sequence[str] | None,
        dems_data: Any | None,
        utility_formula: str | None = None,
        membership_formula: str | None = None,
    ) -> ParsedData:
        """Convert flexible user inputs into strictly aligned model matrices.

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
        if self._encoder is not None:
            raise RuntimeError(
                "This model already has a fitted encoder. Create a new model "
                "instance for another fit."
            )
        encoder = ChoiceDataEncoder(
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            utility_formula=utility_formula,
            membership_formula=membership_formula,
            choice_col=choice_col,
            explicit_case_varnames=case_varnames,
            explicit_dem_varnames=dem_varnames,
        )
        parsed = encoder.fit_transform(data, dems_data=dems_data)
        self._encoder = encoder
        return parsed

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
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("weights must contain only finite values.")
        if bool(jnp.any(weights < 0)):
            raise ValueError("weights must be nonnegative.")
        if not bool(jnp.any(weights > 0)):
            raise ValueError("At least one case weight must be positive.")

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

    @staticmethod
    def _resolve_case_weights(
        data: Any,
        parsed: ParsedData,
        weights: (
            str
            | Mapping[object, float | int]
            | Sequence[float | int]
            | ArrayLike
            | None
        ),
        *,
        cases_col: str,
        panels_col: str,
    ) -> ArrayLike | None:
        """Align case weights from user order to encoder-sorted case order.

        Parameters
        ----------
        data : Any
            Original long-format data before encoder sorting.
        parsed : ParsedData
            Encoded data whose case order is consumed by the likelihood.
        weights : str, mapping, ArrayLike, or None
            A constant-per-case column name, case-keyed mapping, or vector ordered
            by first case appearance in ``data``.
        cases_col : str
            Choice-situation identifier column.
        panels_col : str
            Panel identifier column. Cases are keyed jointly by panel and case
            whenever the two columns differ.

        Returns
        -------
        ArrayLike | None
            Weights aligned to sequential encoded case IDs.
        """
        if weights is None:
            return None

        df = _coerce_frame(data)
        case_key_cols = (
            [cases_col] if panels_col == cases_col else [panels_col, cases_col]
        )
        case_keys_frame = df.select(case_key_cols).unique(maintain_order=True)
        appearance_keys = _case_keys(case_keys_frame, case_key_cols)

        first_case_rows = parsed.cases != jnp.roll(parsed.cases, shift=1)
        first_case_rows = first_case_rows.at[0].set(True)
        if parsed.original_panels is None or parsed.original_cases is None:
            raise ValueError(
                "Encoded data does not preserve original case identifiers."
            )
        sorted_panels = onp.asarray(parsed.original_panels[first_case_rows]).tolist()
        sorted_cases = onp.asarray(parsed.original_cases[first_case_rows]).tolist()
        sorted_keys: list[object]
        if panels_col == cases_col:
            sorted_keys = sorted_cases
        else:
            sorted_keys = list(zip(sorted_panels, sorted_cases))

        if isinstance(weights, str):
            if weights not in df.columns:
                raise ValueError(f"Weight column {weights!r} was not found in data.")
            grouped = df.group_by(case_key_cols, maintain_order=True).agg(
                pl.col(weights).n_unique().alias("_weight_n_unique"),
                pl.col(weights).first().alias("_case_weight"),
            )
            nonconstant = grouped.filter(pl.col("_weight_n_unique") != 1)
            if nonconstant.height:
                sample = nonconstant.select(case_key_cols).head(5).to_dicts()
                raise ValueError(
                    f"Weight column {weights!r} must be constant within each "
                    f"choice situation. Conflicting cases include: {sample}"
                )
            value_by_key = dict(
                zip(
                    _case_keys(grouped, case_key_cols),
                    grouped["_case_weight"].to_list(),
                )
            )
        elif isinstance(weights, Mapping):
            mapping_keys = list(weights)
            uses_joint_keys = bool(mapping_keys) and all(
                isinstance(key, tuple) and len(key) == 2 for key in mapping_keys
            )
            case_ids_are_unique = len(set(sorted_cases)) == len(sorted_cases)
            if (
                panels_col != cases_col
                and not case_ids_are_unique
                and not uses_joint_keys
            ):
                raise ValueError(
                    "Case IDs repeat across panels. Key the weights mapping by "
                    "(panel_id, case_id) tuples."
                )
            lookup_keys = (
                sorted_keys
                if uses_joint_keys or not case_ids_are_unique
                else sorted_cases
            )
            missing = [key for key in lookup_keys if key not in weights]
            extra = [key for key in weights if key not in set(lookup_keys)]
            if missing or extra:
                raise ValueError(
                    "Weights mapping keys must match the fitted choice situations "
                    f"exactly. Missing: {missing[:5]}; extra: {extra[:5]}."
                )
            return onp.asarray([weights[key] for key in lookup_keys], dtype=float)
        else:
            values = onp.asarray(weights, dtype=float)
            if values.shape != (len(appearance_keys),):
                raise ValueError(
                    "A weights vector must have one entry per choice situation "
                    "in first-appearance order."
                )
            value_by_key = dict(zip(appearance_keys, values.tolist()))

        return onp.asarray([value_by_key[key] for key in sorted_keys], dtype=float)

    @staticmethod
    def _resolve_panel_cluster_ids(
        data: Any,
        parsed: ParsedData,
        cluster_col: str,
        *,
        panels_col: str,
    ) -> tuple[onp.ndarray, int]:
        """Map a coarser cluster column onto encoded panel order.

        Standard errors cluster at the decision-maker by default.  A study whose
        sampling or treatment varies at a coarser level -- household, market,
        school, region -- needs the scores summed at that level instead, so the
        grouping variable must be constant within each panel and is resolved to
        contiguous zero-indexed identifiers in the encoder's panel order.

        Parameters
        ----------
        data : Any
            Original long-format data, before encoder sorting.
        parsed : :class:`~lcl._struct.ParsedData`
            Encoded arrays, used for the canonical panel order.
        cluster_col : str
            Column naming the coarser grouping.
        panels_col : str
            Panel identifier column.

        Returns
        -------
        cluster_ids : numpy.ndarray
            One contiguous zero-indexed cluster identifier per encoded panel.
        num_clusters : int
            Number of distinct clusters.
        """
        df = _coerce_frame(data)
        if cluster_col not in df.columns:
            raise ValueError(
                f"Cluster column {cluster_col!r} was not found in the estimation "
                "data. Pass inference=InferenceOptions(cluster='panel') to cluster "
                "at the decision-maker instead."
            )
        grouped = df.group_by(panels_col, maintain_order=True).agg(
            pl.col(cluster_col).n_unique().alias("_cluster_n_unique"),
            pl.col(cluster_col).first().alias("_cluster_value"),
        )
        nonconstant = grouped.filter(pl.col("_cluster_n_unique") != 1)
        if nonconstant.height:
            sample = nonconstant.select(panels_col).head(5).to_dicts()
            raise ValueError(
                f"Cluster column {cluster_col!r} must be constant within each "
                f"panel. Conflicting panels include: {sample}"
            )
        value_by_panel = dict(
            zip(grouped[panels_col].to_list(), grouped["_cluster_value"].to_list())
        )

        if parsed.original_panels is None:
            raise ValueError(
                "Encoded data does not preserve original panel identifiers."
            )
        first_panel_rows = parsed.panels != jnp.roll(parsed.panels, shift=1)
        first_panel_rows = first_panel_rows.at[0].set(True)
        panel_ids = onp.asarray(parsed.original_panels[first_panel_rows]).tolist()
        missing = [panel for panel in panel_ids if panel not in value_by_panel]
        if missing:
            raise ValueError(
                f"Cluster column {cluster_col!r} has no value for panels "
                f"{missing[:5]}."
            )

        codes: list[int] = []
        seen: dict[object, int] = {}
        for panel in panel_ids:
            value = value_by_panel[panel]
            if value is None:
                raise ValueError(
                    f"Cluster column {cluster_col!r} contains a null value for "
                    f"panel {panel!r}."
                )
            if value not in seen:
                seen[value] = len(seen)
            codes.append(seen[value])
        return onp.asarray(codes, dtype=onp.int32), len(seen)

    def _pre_fit(
        self,
        case_varnames: Sequence[str],
        dem_varnames: Sequence[str] | None,
        numeraire: str | None,
        variable_labels: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize metadata tracking prior to estimation loop."""
        self._fit_start_time = time()
        self.case_varnames = list(case_varnames)
        self.dem_varnames = list(dem_varnames) if dem_varnames is not None else None
        self.numeraire = numeraire
        self.variable_labels = normalize_variable_labels(variable_labels)
        self.numeraire_idx = case_varnames.index(numeraire) if numeraire else None

    def variable_label(self, variable: str) -> str:
        """Return the human-readable label for a fitted model variable.

        Parameters
        ----------
        variable : str
            Raw model variable name or Formulaic-expanded coefficient name.

        Returns
        -------
        str
            Display label when available; otherwise ``variable`` unchanged.
        """
        return label_for_variable(variable, self.variable_labels)


def _case_keys(df: pl.DataFrame, columns: Sequence[str]) -> list[object]:
    """Return scalar or joint case keys from a case-level frame."""
    if len(columns) == 1:
        return df[columns[0]].to_list()
    return [tuple(row) for row in df.select(columns).iter_rows()]
