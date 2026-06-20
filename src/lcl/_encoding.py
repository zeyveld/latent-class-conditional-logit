"""Data encoding layer shared by fit and predict APIs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl
from formulaic import Formula  # type: ignore

from lcl._struct import ParsedData


@dataclass
class ChoiceDataEncoder:
    """Store ID mappings, formula transforms, and aligned variable metadata."""

    alts_col: str
    cases_col: str
    panels_col: str
    formula: str | None = None
    utility_formula: str | None = None
    membership_formula: str | None = None
    choice_col: str | None = None
    explicit_case_varnames: Sequence[str] | None = None
    explicit_dem_varnames: Sequence[str] | None = None
    case_varnames: list[str] | None = None
    dem_varnames: list[str] | None = None
    y_model_spec: Any | None = None
    x_model_spec: Any | None = None
    dem_model_spec: Any | None = None

    def __post_init__(self) -> None:
        """Validate mutually exclusive formula interfaces."""
        if self.formula is not None and (
            self.utility_formula is not None or self.membership_formula is not None
        ):
            raise ValueError(
                "Use either legacy formula=... or separate utility_formula=... "
                "and membership_formula=..., not both."
            )
        if self.membership_formula is not None and _formula_has_lhs(
            self.membership_formula
        ):
            raise ValueError(
                "membership_formula must be a right-hand-side formula such as "
                "'~ income + C(segment)'. Latent classes are unobserved, so a "
                "left-hand side is not meaningful."
            )

    def fit_transform(self, data: Any, dems_data: Any | None = None) -> ParsedData:
        """Fit the encoder metadata and return aligned JAX-ready arrays.

        Parameters
        ----------
        data : Any
            Long-format choice data accepted by :func:`_coerce_frame`.
        dems_data : Any | None, optional
            Optional panel-level demographics to join by ``panels_col``.

        Returns
        -------
        :class:`~lcl._struct.ParsedData`
            Encoded arrays, sequential IDs, original labels, and variable names.
        """
        return self._transform(data, dems_data=dems_data, fit=True, require_choice=True)

    def transform(
        self,
        data: Any,
        dems_data: Any | None = None,
        require_choice: bool = False,
    ) -> ParsedData:
        """Transform new data with an already-fitted encoder.

        Parameters
        ----------
        data : Any
            Long-format prediction or validation data.
        dems_data : Any | None, optional
            Optional panel-level demographics to join by ``panels_col``.
        require_choice : bool, default=False
            Whether the transformed data must include a valid choice indicator.

        Returns
        -------
        :class:`~lcl._struct.ParsedData`
            Encoded arrays using the model specification learned during fitting.
        """
        return self._transform(
            data, dems_data=dems_data, fit=False, require_choice=require_choice
        )

    def _transform(
        self,
        data: Any,
        dems_data: Any | None,
        fit: bool,
        require_choice: bool,
    ) -> ParsedData:
        """Encode raw data into sorted arrays and zero-indexed IDs.

        Parameters
        ----------
        data : Any
            Long-format choice data.
        dems_data : Any | None
            Optional panel-level demographic data.
        fit : bool
            If True, learn formula model specs and variable names. If False, reuse
            specs stored on the encoder.
        require_choice : bool
            Whether to encode and validate one chosen row per choice situation.

        Returns
        -------
        :class:`~lcl._struct.ParsedData`
            Strictly sorted, aligned arrays and metadata.
        """
        df = _coerce_frame(data)
        _require_columns(df, [self.alts_col, self.cases_col, self.panels_col])
        sort_cols = list(
            dict.fromkeys([self.panels_col, self.cases_col, self.alts_col])
        )
        df = df.sort(sort_cols)
        df = self._attach_sequential_ids(df)
        if dems_data is not None and self._has_demographic_formula():
            df = self._attach_external_demographics(df, dems_data)

        y_array, case_vars, dem_vars, df = self._encode_features(
            df, fit=fit, require_choice=require_choice
        )
        X_array = jnp.array(df.select(case_vars).to_numpy(), dtype="float64")
        demographic_data_source = None if self.dem_model_spec is not None else dems_data
        dems_array = self._encode_demographics(df, dem_vars, demographic_data_source)

        if y_array is not None:
            self._validate_one_choice_per_case(df, y_array)

        return ParsedData(
            X=X_array,
            dems=dems_array,
            y=y_array,
            cases=jnp.array(df["_seq_cases"].to_numpy(), dtype="uint32"),
            panels=jnp.array(df["_seq_panels"].to_numpy(), dtype="uint32"),
            alts=jnp.array(df["_seq_alts"].to_numpy(), dtype="uint32"),
            case_varnames=case_vars,
            dem_varnames=dem_vars,
            original_alts=df[self.alts_col].to_numpy(),
            original_cases=df[self.cases_col].to_numpy(),
            original_panels=df[self.panels_col].to_numpy(),
        )

    def _attach_sequential_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """Attach contiguous panel, case, and alternative IDs.

        Parameters
        ----------
        df : pl.DataFrame
            Sorted choice data containing the configured identifier columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``_seq_panels``, ``_seq_cases``, and ``_seq_alts`` columns.
        """
        panel_keys = df.select(self.panels_col).unique(maintain_order=True)
        panel_keys = panel_keys.with_row_index("_seq_panels")

        case_key_cols = (
            [self.cases_col]
            if self.panels_col == self.cases_col
            else [self.panels_col, self.cases_col]
        )
        case_keys = df.select(case_key_cols).unique(maintain_order=True)
        case_keys = case_keys.with_row_index("_seq_cases")

        alt_keys = df.select(self.alts_col).unique(maintain_order=True)
        alt_keys = alt_keys.with_row_index("_seq_alts")

        return (
            df.join(panel_keys, on=self.panels_col, how="left")
            .join(case_keys, on=case_key_cols, how="left")
            .join(alt_keys, on=self.alts_col, how="left")
            .sort(["_seq_panels", "_seq_cases", "_seq_alts"])
        )

    def _attach_external_demographics(
        self, df: pl.DataFrame, dems_data: Any
    ) -> pl.DataFrame:
        """Join external panel-level columns needed by demographic formulas.

        Parameters
        ----------
        df : pl.DataFrame
            Choice data after sequential identifiers have been attached.
        dems_data : Any
            Panel-level demographic data.

        Returns
        -------
        pl.DataFrame
            Choice data augmented with raw external demographic columns that were
            not already present in ``df``.
        """
        dems_df = _coerce_frame(dems_data)
        _require_columns(dems_df, [self.panels_col])
        cols_to_join = [
            col
            for col in dems_df.columns
            if col != self.panels_col and col not in df.columns
        ]
        if not cols_to_join:
            return df

        unique_dems = dems_df.select([self.panels_col, *cols_to_join]).unique(
            maintain_order=True
        )
        duplicate_panels = (
            unique_dems.group_by(self.panels_col)
            .len()
            .filter(pl.col("len") > 1)
            .select(self.panels_col)
        )
        if duplicate_panels.height:
            sample = duplicate_panels.head(5)[self.panels_col].to_list()
            raise ValueError(
                "dems_data must contain one unique value per panel for formula "
                f"demographics. Conflicting panels include: {sample}"
            )

        return df.join(unique_dems, on=self.panels_col, how="left")

    def _encode_features(
        self, df: pl.DataFrame, fit: bool, require_choice: bool
    ) -> tuple[jnp.ndarray | None, list[str], list[str] | None, pl.DataFrame]:
        """Encode choice indicators and alternative-specific features.

        Parameters
        ----------
        df : pl.DataFrame
            Choice data with sequential ID columns attached.
        fit : bool
            Whether to fit formulaic model specifications and store column metadata.
        require_choice : bool
            Whether a choice outcome is required in the returned arrays.

        Returns
        -------
        y_array : jnp.ndarray | None
            Boolean choice indicators, or None when choices are not required.
        case_vars : list[str]
            Encoded alternative-specific feature names.
        dem_vars : list[str] | None
            Encoded demographic feature names, if present.
        df : pl.DataFrame
            DataFrame with encoded formula columns appended.
        """
        y_array = None

        if self._has_formula_spec():
            pandas_df = _to_pandas_frame(df)
            X_df: Any | None = None
            y_df: Any | None = None

            if self.formula is not None:
                X_df, y_df, df = self._encode_legacy_formula(
                    pandas_df, df, fit=fit, require_choice=require_choice
                )
            else:
                if self.utility_formula is not None:
                    X_df, y_df = self._encode_utility_formula(
                        pandas_df, df, fit=fit, require_choice=require_choice
                    )
                    _validate_matrix_height(X_df, df.height, "utility formula")
                    df = df.with_columns(pl.from_pandas(X_df))
                    case_vars = list(self.case_varnames or X_df.columns)
                else:
                    case_vars = self._encode_explicit_case_variables(
                        df, require_choice=require_choice, fit=fit
                    )
                    y_array = self._choice_array_from_column(df, fit, require_choice)

                if self.membership_formula is not None:
                    df = self._encode_membership_formula(pandas_df, df, fit=fit)
                else:
                    self.dem_varnames = (
                        list(self.explicit_dem_varnames)
                        if self.explicit_dem_varnames is not None
                        else None
                    )

                dem_vars = self.dem_varnames
                if self.utility_formula is not None and (fit or require_choice):
                    y_array = self._choice_array_from_formula_or_column(
                        df, y_df, fit=fit, require_choice=require_choice
                    )

            if self.formula is not None:
                if X_df is None:
                    raise ValueError(
                        "Formula encoder did not produce a utility matrix."
                    )
                _validate_matrix_height(X_df, df.height, "case formula")
                df = df.with_columns(pl.from_pandas(X_df))
                case_vars = list(self.case_varnames or X_df.columns)
                dem_vars = self.dem_varnames
                if fit or require_choice:
                    y_array = self._choice_array_from_formula_or_column(
                        df, y_df, fit=fit, require_choice=require_choice
                    )

        else:
            case_vars = self._encode_explicit_case_variables(
                df, require_choice=require_choice, fit=fit
            )
            dem_vars = (
                list(self.explicit_dem_varnames)
                if self.explicit_dem_varnames is not None
                else None
            )
            y_array = self._choice_array_from_column(df, fit, require_choice)
            self.case_varnames = case_vars
            self.dem_varnames = dem_vars

        return y_array, case_vars, dem_vars, df

    def _has_formula_spec(self) -> bool:
        """Return whether any formula-based interface is active."""
        return any(
            item is not None
            for item in (self.formula, self.utility_formula, self.membership_formula)
        )

    def _has_demographic_formula(self) -> bool:
        """Return whether raw external demographics may be needed for formulas."""
        return self.formula is not None or self.membership_formula is not None

    def _encode_legacy_formula(
        self,
        pandas_df: Any,
        df: pl.DataFrame,
        *,
        fit: bool,
        require_choice: bool,
    ) -> tuple[Any, Any | None, pl.DataFrame]:
        """Encode the backward-compatible ``choice ~ utility | membership`` formula."""
        if fit:
            f = Formula(self.formula)
            y_df = f.lhs.get_model_matrix(pandas_df)
            self.y_model_spec = y_df.model_spec

            if isinstance(f.rhs, tuple):
                raw_X_df = f.rhs[0].get_model_matrix(pandas_df)
                raw_dems_df = f.rhs[1].get_model_matrix(pandas_df)
                dems_df = _drop_formula_intercepts(raw_dems_df)
                self.dem_model_spec = raw_dems_df.model_spec
                self.dem_varnames = list(dems_df.columns) or None
                if self.dem_varnames:
                    df = df.with_columns(pl.from_pandas(dems_df))
            else:
                raw_X_df = f.rhs.get_model_matrix(pandas_df)
                self.dem_model_spec = None
                self.dem_varnames = None

            X_df = _drop_formula_intercepts(raw_X_df)
            self.x_model_spec = raw_X_df.model_spec
            self.case_varnames = list(X_df.columns)
            self._validate_case_formula_columns()
            return X_df, y_df, df

        if self.x_model_spec is None:
            raise ValueError("Formula encoder has not been fitted.")
        X_df = _drop_formula_intercepts(self.x_model_spec.get_model_matrix(pandas_df))
        if self.dem_model_spec is not None:
            dems_df = _drop_formula_intercepts(
                self.dem_model_spec.get_model_matrix(pandas_df)
            )
            if self.dem_varnames:
                df = df.with_columns(pl.from_pandas(dems_df))
        if require_choice and self.y_model_spec is not None:
            y_df = self.y_model_spec.get_model_matrix(pandas_df)
        else:
            y_df = None
        return X_df, y_df, df

    def _encode_utility_formula(
        self,
        pandas_df: Any,
        df: pl.DataFrame,
        *,
        fit: bool,
        require_choice: bool,
    ) -> tuple[Any, Any | None]:
        """Encode the alternative-specific utility formula."""
        if fit:
            f = Formula(self.utility_formula)
            if _formula_object_has_lhs(f):
                y_df = f.lhs.get_model_matrix(pandas_df)
                self.y_model_spec = y_df.model_spec
                if isinstance(f.rhs, tuple):
                    raise ValueError(
                        "utility_formula must describe only the utility design. "
                        "Move class-membership terms after '|' into "
                        "membership_formula."
                    )
                raw_X_df = f.rhs.get_model_matrix(pandas_df)
            else:
                y_df = None
                raw_X_df = f.get_model_matrix(pandas_df)
                if not self.choice_col:
                    raise ValueError(
                        "choice_col is required when utility_formula has no "
                        "left-hand side."
                    )
                _require_columns(df, [self.choice_col])

            X_df = _drop_formula_intercepts(raw_X_df)
            self.x_model_spec = raw_X_df.model_spec
            self.case_varnames = list(X_df.columns)
            self._validate_case_formula_columns()
            return X_df, y_df

        if self.x_model_spec is None:
            raise ValueError("Utility formula encoder has not been fitted.")
        X_df = _drop_formula_intercepts(self.x_model_spec.get_model_matrix(pandas_df))
        if require_choice and self.y_model_spec is not None:
            y_df = self.y_model_spec.get_model_matrix(pandas_df)
        else:
            y_df = None
        return X_df, y_df

    def _encode_membership_formula(
        self,
        pandas_df: Any,
        df: pl.DataFrame,
        *,
        fit: bool,
    ) -> pl.DataFrame:
        """Encode the class-membership demographic formula."""
        if fit:
            f = Formula(self.membership_formula)
            if not hasattr(f, "get_model_matrix"):
                raise ValueError(
                    "membership_formula must be a right-hand-side formula such "
                    "as '~ income + C(segment)'."
                )
            raw_dems_df = f.get_model_matrix(pandas_df)
            dems_df = _drop_formula_intercepts(raw_dems_df)
            self.dem_model_spec = raw_dems_df.model_spec
            self.dem_varnames = list(dems_df.columns) or None
        else:
            if self.dem_model_spec is None:
                raise ValueError("Membership formula encoder has not been fitted.")
            dems_df = _drop_formula_intercepts(
                self.dem_model_spec.get_model_matrix(pandas_df)
            )

        _validate_matrix_height(dems_df, df.height, "membership formula")
        if self.dem_varnames:
            df = df.with_columns(pl.from_pandas(dems_df))
        return df

    def _encode_explicit_case_variables(
        self, df: pl.DataFrame, *, require_choice: bool, fit: bool
    ) -> list[str]:
        """Validate and store explicit alternative-specific variables."""
        del require_choice, fit
        if self.explicit_case_varnames is None:
            raise ValueError(
                "Must provide either utility_formula, formula, or explicit case_varnames."
            )
        case_vars = list(self.explicit_case_varnames)
        _require_columns(df, case_vars)
        self.case_varnames = case_vars
        return case_vars

    def _choice_array_from_column(
        self, df: pl.DataFrame, fit: bool, require_choice: bool
    ) -> jnp.ndarray | None:
        """Return choice indicators from ``choice_col`` when required."""
        if not (fit or require_choice):
            return None
        if not self.choice_col:
            raise ValueError(
                "choice_col is required when fitting without a formula LHS."
            )
        _require_columns(df, [self.choice_col])
        return jnp.array(df[self.choice_col].to_numpy(), dtype="bool")

    def _choice_array_from_formula_or_column(
        self,
        df: pl.DataFrame,
        y_df: Any | None,
        *,
        fit: bool,
        require_choice: bool,
    ) -> jnp.ndarray | None:
        """Return choice indicators from a formula LHS or ``choice_col``."""
        if not (fit or require_choice):
            return None
        if y_df is not None:
            _validate_matrix_height(y_df, df.height, "choice formula")
            return jnp.array(y_df.to_numpy().ravel(), dtype="bool")
        return self._choice_array_from_column(df, fit, require_choice)

    def _validate_case_formula_columns(self) -> None:
        """Validate that a utility formula produced identified columns."""
        if not self.case_varnames:
            raise ValueError(
                "Alternative-specific formulas must include at least one "
                "identified variable after dropping the intercept."
            )

    def _encode_demographics(
        self, df: pl.DataFrame, dem_vars: list[str] | None, dems_data: Any | None
    ) -> jnp.ndarray | None:
        """Return panel-level demographic arrays aligned to sequential panel IDs.

        Parameters
        ----------
        df : pl.DataFrame
            Choice data with sequential panel IDs.
        dem_vars : list[str] | None
            Demographic variables to encode.
        dems_data : Any | None
            Optional separate panel-level demographics source.

        Returns
        -------
        jnp.ndarray | None
            ``(panels, dem_vars)`` matrix sorted by ``_seq_panels``, or None when
            no demographic variables are specified.
        """
        if not dem_vars:
            return None

        if dems_data is not None:
            dems_df_ext = _coerce_frame(dems_data)
            _require_columns(dems_df_ext, [self.panels_col] + dem_vars)
            unique_panels = df.select([self.panels_col, "_seq_panels"]).unique(
                subset=[self.panels_col], maintain_order=True
            )
            aligned = unique_panels.join(dems_df_ext, on=self.panels_col, how="left")
        else:
            _require_columns(df, dem_vars)
            aligned = df.select(["_seq_panels"] + dem_vars).unique(
                subset=["_seq_panels"] + dem_vars, maintain_order=True
            )
            num_panels = df["_seq_panels"].n_unique()
            if aligned.height != num_panels:
                raise ValueError(
                    "Demographic variables must be constant within each panel."
                )

        has_missing = aligned.select(
            pl.any_horizontal(pl.col(dem_vars).is_null()).any()
        ).item()
        if has_missing:
            raise ValueError(
                "Missing demographic rows after panel alignment. Check dems_data joins."
            )

        return jnp.array(
            aligned.sort("_seq_panels").select(dem_vars).to_numpy(), dtype="float64"
        )

    @staticmethod
    def _validate_one_choice_per_case(df: pl.DataFrame, y_array: jnp.ndarray) -> None:
        """Validate that every choice situation has exactly one selected alternative.

        Parameters
        ----------
        df : pl.DataFrame
            Encoded choice data containing ``_seq_cases``.
        y_array : jnp.ndarray
            Boolean choice indicator aligned to ``df``.

        Raises
        ------
        ValueError
            If any choice situation has zero or multiple chosen alternatives.
        """
        cases = df["_seq_cases"].to_numpy()
        y = onp.asarray(y_array, dtype=onp.int32)
        choices_per_case = onp.bincount(cases, weights=y)
        if not onp.all(choices_per_case == 1):
            raise ValueError(
                "Every choice situation must have exactly one chosen alternative."
            )


def _coerce_frame(data: Any) -> pl.DataFrame:
    """Coerce supported tabular inputs to a Polars DataFrame.

    Parameters
    ----------
    data : Any
        Polars DataFrame, pandas-like DataFrame, dict-like data, or other object
        accepted by ``pl.DataFrame``.

    Returns
    -------
    pl.DataFrame
        Polars representation of the input.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if hasattr(data, "columns"):
        return pl.from_pandas(data)
    return pl.DataFrame(data)


def _to_pandas_frame(df: pl.DataFrame) -> Any:
    """Convert Polars data to pandas for formulaic.

    Parameters
    ----------
    df : pl.DataFrame
        Data to convert.

    Returns
    -------
    Any
        A pandas DataFrame. The return type is kept broad to avoid importing pandas
        unless Polars requires the fallback path.
    """
    try:
        return df.to_pandas()
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        import pandas as pd  # type: ignore[import-untyped]

        return pd.DataFrame(df.to_dicts())


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    """Raise when required columns are absent.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.
    columns : Sequence[str]
        Required column names.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")


def _validate_matrix_height(matrix: Any, expected_height: int, label: str) -> None:
    """Validate that a formula matrix preserves the input row count.

    Parameters
    ----------
    matrix : Any
        Formulaic model matrix, or None when no matrix was produced.
    expected_height : int
        Required number of rows.
    label : str
        Human-readable source label used in error messages.

    Raises
    ------
    ValueError
        If the encoded matrix has a different number of rows than the source data.
    """
    if matrix is None:
        return
    if len(matrix) != expected_height:
        raise ValueError(
            f"{label} produced {len(matrix)} rows for {expected_height} input rows. "
            "Check missing values in formula variables."
        )


def _drop_formula_intercepts(matrix: Any) -> Any:
    """Remove formulaic intercept columns from encoded design matrices."""
    columns = [
        col for col in matrix.columns if str(col).lower() not in {"intercept", "1"}
    ]
    return matrix.loc[:, columns]


def _formula_has_lhs(formula: str) -> bool:
    """Return whether a formula string contains a left-hand side."""
    return _formula_object_has_lhs(Formula(formula))


def _formula_object_has_lhs(formula: Any) -> bool:
    """Return whether a Formulaic object exposes a left-hand side."""
    return hasattr(formula, "lhs")
