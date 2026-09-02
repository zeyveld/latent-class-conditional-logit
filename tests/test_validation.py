"""Encoder round-trip and malformed-input properties."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from lcl._encoding import ChoiceDataEncoder


def _valid_frame() -> pl.DataFrame:
    rng = np.random.default_rng(41)
    rows = []
    for panel in [10, 20, 30, 40]:
        segment = "high" if panel >= 30 else "low"
        for case_within_panel in range(3):
            case = panel * 10 + case_within_panel
            for alt, mode in enumerate(["bus", "rail", "air"]):
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": mode,
                        "choice": alt == case_within_panel % 3,
                        "x": float(rng.normal() + 0.3 * alt),
                        "z": float(rng.normal() - 0.2 * alt),
                        "segment": segment,
                    }
                )
    return pl.DataFrame(rows)


def _encoder() -> ChoiceDataEncoder:
    return ChoiceDataEncoder(
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        utility_formula="choice ~ x + z + C(alt)",
        membership_formula="~ C(segment)",
    )


def test_encoder_training_round_trip_is_exact() -> None:
    data = _valid_frame().sample(fraction=1.0, shuffle=True, seed=7)
    encoder = _encoder()
    fitted = encoder.fit_transform(data)
    transformed = encoder.transform(data, require_choice=True)

    assert fitted.case_varnames == transformed.case_varnames
    assert fitted.dem_varnames == transformed.dem_varnames
    assert jnp.array_equal(fitted.X, transformed.X)
    assert jnp.array_equal(fitted.dems, transformed.dems)
    assert jnp.array_equal(fitted.y, transformed.y)
    assert np.array_equal(fitted.original_alts, transformed.original_alts)
    assert np.array_equal(fitted.original_cases, transformed.original_cases)
    assert np.array_equal(fitted.original_panels, transformed.original_panels)


def _duplicate_row(df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([df, df.head(1)])


def _null_identifier(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("alt"))
            .alias("alt")
        )
        .drop("row")
    )


def _nonfinite_design(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("x"))
            .alias("x")
        )
        .drop("row")
    )


def _multiple_choices(df: pl.DataFrame) -> pl.DataFrame:
    first_case = df["case"][0]
    return df.with_columns(
        pl.when(pl.col("case") == first_case)
        .then(pl.lit(True))
        .otherwise(pl.col("choice"))
        .alias("choice")
    )


def _reserved_intercept(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.lit(1.0).alias("Intercept"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_duplicate_row, "must be unique"),
        (_null_identifier, "cannot contain null"),
        (_nonfinite_design, "non-finite"),
        (_multiple_choices, "exactly one chosen"),
        (_reserved_intercept, "reserved by the encoder"),
    ],
)
def test_malformed_choice_frames_fail_before_likelihood(
    mutation: Callable[[pl.DataFrame], pl.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _encoder().fit_transform(mutation(_valid_frame()))


def test_duplicate_external_demographic_panels_are_rejected() -> None:
    data = _valid_frame().drop("segment")
    demographics = pl.DataFrame({"panel": [10, 10, 20, 30, 40], "segment": ["low"] * 5})
    with pytest.raises(ValueError, match="exactly one row per panel"):
        _encoder().fit_transform(data, dems_data=demographics)


def test_overlapping_external_demographic_columns_are_rejected() -> None:
    data = _valid_frame()
    demographics = data.select(["panel", "segment"]).unique()
    with pytest.raises(ValueError, match="precedence would be ambiguous"):
        _encoder().fit_transform(data, dems_data=demographics)


def test_unseen_prediction_category_is_a_hard_error() -> None:
    data = _valid_frame()
    encoder = _encoder()
    encoder.fit_transform(data)
    unseen = data.with_columns(
        pl.when(pl.col("alt") == "air")
        .then(pl.lit("ferry"))
        .otherwise(pl.col("alt"))
        .alias("alt")
    )
    with pytest.raises(ValueError, match="categories outside"):
        encoder.transform(unseen)
