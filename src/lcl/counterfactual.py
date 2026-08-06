"""Choice-set resampling for counterfactual future-profit simulations."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from lcl._encoding import _coerce_frame


_METADATA_COLUMNS = {
    "simulation_stage",
    "simulation_round",
    "simulation_trip",
    "source_case",
    "source_time",
    "days_after_intervention",
}


@dataclass(frozen=True)
class FutureSimulationConfig:
    """Configuration for modest Monte Carlo integration over future choice sets.

    Parameters
    ----------
    num_draws : int, default=20
        Number of independently resampled future worlds.  Twenty is intentionally
        modest and matches the scale of the motivating implementation.
    horizon_days : float, default=365
        Length of the future-profit horizon in days.
    max_trips_per_panel : int, default=104
        Upper bound on simulated trips per panel and draw (twice weekly over a year
        under the default horizon).
    seed : int, default=0
        Seed for reproducible sampling.  Anticipated and realized pools receive
        independent child random streams.
    trip_timing : {"poisson", "fixed"}, default="poisson"
        ``"poisson"`` samples each draw's trip count and dates from a zero-truncated,
        capped homogeneous Poisson process with the panel's estimated trip rate.
        ``"fixed"`` uses the rounded expected count and evenly spaced trips.
    """

    num_draws: int = 20
    horizon_days: float | int = 365.0
    max_trips_per_panel: int = 104
    seed: int = 0
    trip_timing: Literal["poisson", "fixed"] = "poisson"

    def __post_init__(self) -> None:
        """Validate simulation settings."""
        if self.num_draws < 1:
            raise ValueError("num_draws must be at least 1.")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")
        if self.max_trips_per_panel < 1:
            raise ValueError("max_trips_per_panel must be at least 1.")
        if self.trip_timing not in {"poisson", "fixed"}:
            raise ValueError("trip_timing must be either 'poisson' or 'fixed'.")


@dataclass(frozen=True)
class CounterfactualWorlds:
    """Synthetic future choice sets for policy design and realized evaluation.

    Attributes
    ----------
    anticipated : pl.DataFrame
        Future choice sets sampled only from each panel's history through the
        intervention choice set.  These worlds represent information available to
        the store when selecting a policy.
    realized : pl.DataFrame
        Future choice sets sampled from each panel's complete observed history,
        including observations before and after the intervention.  These worlds are
        for ex post policy evaluation, never for policy selection.
    trip_summary : pl.DataFrame
        Panel-by-stage diagnostics containing sampling-pool size, estimated trip
        interval, and simulated trip count per draw.
    config : FutureSimulationConfig
        Configuration used to generate the worlds.
    """

    anticipated: pl.DataFrame
    realized: pl.DataFrame
    trip_summary: pl.DataFrame
    config: FutureSimulationConfig


def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")


def _time_as_days(series: pl.Series) -> np.ndarray:
    """Convert supported chronological values to floating-point days."""
    dtype = series.dtype
    if dtype == pl.Date:
        return series.cast(pl.Int32).to_numpy().astype(np.float64)
    if isinstance(dtype, pl.Datetime):
        units_per_day = {
            "ms": 86_400_000.0,
            "us": 86_400_000_000.0,
            "ns": 86_400_000_000_000.0,
        }[dtype.time_unit]
        return series.cast(pl.Int64).to_numpy().astype(np.float64) / units_per_day
    if dtype.is_numeric():
        return series.cast(pl.Float64).to_numpy()
    raise TypeError(
        "time_col must be numeric, pl.Date, or pl.Datetime; "
        f"received {dtype}."
    )


def _future_times(
    cutoffs: list[date | datetime | int | float],
    offsets: np.ndarray,
    dtype: Any,
    name: str,
) -> pl.Series:
    """Construct future times in the input time column's dtype."""
    if dtype == pl.Date:
        date_cutoffs = cast(list[date], cutoffs)
        values = [
            cutoff + timedelta(days=int(round(float(offset))))
            for cutoff, offset in zip(date_cutoffs, offsets)
        ]
        return pl.Series(name, values, dtype=pl.Date)
    if isinstance(dtype, pl.Datetime):
        datetime_cutoffs = cast(list[datetime], cutoffs)
        values = [
            cutoff + timedelta(days=float(offset))
            for cutoff, offset in zip(datetime_cutoffs, offsets)
        ]
        return pl.Series(name, values, dtype=dtype)

    numeric = np.asarray(cutoffs, dtype=np.float64) + offsets
    if dtype.is_integer():
        numeric = np.rint(numeric)
    return pl.Series(name, numeric).cast(dtype)


def _choice_set_index(
    df: pl.DataFrame,
    *,
    panel_col: str,
    case_col: str,
    time_col: str,
    intervention_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one validated row per choice set and one cutoff per panel."""
    invalid_times = (
        df.group_by([panel_col, case_col])
        .agg(pl.col(time_col).n_unique().alias("_n_times"))
        .filter(pl.col("_n_times") != 1)
    )
    if invalid_times.height:
        sample = invalid_times.select([panel_col, case_col]).head(5).to_dicts()
        raise ValueError(
            "time_col must be constant within each choice set. "
            f"Conflicting choice sets include: {sample}"
        )

    case_index = (
        df.group_by([panel_col, case_col], maintain_order=True)
        .agg(
            pl.col(time_col).first(),
            pl.col(intervention_col).fill_null(False).any().alias("_intervention"),
        )
        .sort([panel_col, time_col, case_col])
    )
    cutoffs = case_index.filter(pl.col("_intervention")).select(
        panel_col, pl.col(time_col).alias("_cutoff")
    )
    intervention_counts = cutoffs.group_by(panel_col).len()
    all_panels = case_index.select(panel_col).unique()
    invalid_panels = (
        all_panels.join(intervention_counts, on=panel_col, how="left")
        .with_columns(pl.col("len").fill_null(0))
        .filter(pl.col("len") != 1)
    )
    if invalid_panels.height:
        sample = invalid_panels.select(panel_col).head(5)[panel_col].to_list()
        raise ValueError(
            "intervention_col must identify exactly one choice set per panel. "
            f"Invalid panels include: {sample}"
        )
    return case_index, cutoffs


def _pool_summary(
    pool: pl.DataFrame,
    *,
    panel_col: str,
    time_col: str,
    config: FutureSimulationConfig,
    stage: str,
) -> pl.DataFrame:
    """Estimate trip frequency from a choice-set sampling pool."""
    panels: list[Any] = []
    observed: list[int] = []
    raw_gaps: list[float] = []
    for panel_key, panel_pool in pool.partition_by(panel_col, as_dict=True).items():
        panel = panel_key[0] if isinstance(panel_key, tuple) else panel_key
        days = _time_as_days(panel_pool[time_col])
        panels.append(panel)
        observed.append(panel_pool.height)
        if days.size >= 2 and np.max(days) > np.min(days):
            raw_gaps.append(
                float((np.max(days) - np.min(days)) / (days.size - 1))
            )
        else:
            raw_gaps.append(np.nan)

    finite_gaps = np.asarray(raw_gaps)[np.isfinite(raw_gaps)]
    fallback_gap = (
        float(np.median(finite_gaps))
        if finite_gaps.size
        else float(config.horizon_days)
    )
    minimum_gap = config.horizon_days / config.max_trips_per_panel
    gaps = np.clip(
        np.where(np.isfinite(raw_gaps), raw_gaps, fallback_gap),
        minimum_gap,
        config.horizon_days,
    )
    simulated_trips = np.clip(
        np.rint(config.horizon_days / gaps),
        1,
        config.max_trips_per_panel,
    ).astype(np.int64)
    return pl.DataFrame(
        {
            panel_col: panels,
            "simulation_stage": [stage] * len(panels),
            "observed_choice_sets": observed,
            "mean_days_between_trips": gaps,
            "expected_trips_per_draw": config.horizon_days / gaps,
            "simulated_trips_per_draw": simulated_trips,
        }
    ).sort(panel_col)


def _sample_stage(
    df: pl.DataFrame,
    pool: pl.DataFrame,
    cutoffs: pl.DataFrame,
    summary: pl.DataFrame,
    *,
    panel_col: str,
    case_col: str,
    time_col: str,
    intervention_col: str,
    choice_col: str | None,
    stage: str,
    rng: np.random.Generator,
    config: FutureSimulationConfig,
    case_offset: int,
) -> pl.DataFrame:
    """Sample complete choice sets using a vectorized case-level draw map."""
    pool = pool.join(cutoffs, on=panel_col, how="left").join(
        summary.select(
            panel_col,
            "mean_days_between_trips",
            "expected_trips_per_draw",
            "simulated_trips_per_draw",
        ),
        on=panel_col,
        how="left",
    )
    panel_groups = pool.partition_by(panel_col, as_dict=True, maintain_order=True)

    panel_values: list[Any] = []
    source_cases: list[Any] = []
    source_times: list[Any] = []
    cutoff_values: list[Any] = []
    rounds: list[np.ndarray] = []
    trips: list[np.ndarray] = []
    offsets: list[np.ndarray] = []

    for panel_key, panel_pool in panel_groups.items():
        panel = panel_key[0] if isinstance(panel_key, tuple) else panel_key
        gap = float(panel_pool["mean_days_between_trips"][0])
        expected_trips = float(panel_pool["expected_trips_per_draw"][0])
        if config.trip_timing == "poisson":
            counts = np.clip(
                rng.poisson(expected_trips, size=config.num_draws),
                1,
                config.max_trips_per_panel,
            ).astype(np.int64)
            panel_offsets = [
                np.sort(rng.uniform(0.0, config.horizon_days, size=int(count)))
                for count in counts
            ]
        else:
            fixed_count = int(panel_pool["simulated_trips_per_draw"][0])
            counts = np.repeat(fixed_count, config.num_draws)
            fixed_offsets = np.arange(1, fixed_count + 1, dtype=np.float64) * gap
            panel_offsets = [fixed_offsets] * config.num_draws

        count = int(np.sum(counts))
        sampled = rng.integers(0, panel_pool.height, size=count)

        panel_values.extend([panel] * count)
        source_cases.extend(panel_pool[case_col].gather(sampled).to_list())
        source_times.extend(panel_pool[time_col].gather(sampled).to_list())
        cutoff_values.extend([panel_pool["_cutoff"][0]] * count)
        rounds.append(np.repeat(np.arange(config.num_draws), counts))
        trip_numbers = np.concatenate(
            [np.arange(1, int(draw_count) + 1) for draw_count in counts]
        )
        trips.append(trip_numbers)
        offsets.append(np.concatenate(panel_offsets))

    round_array = np.concatenate(rounds)
    trip_array = np.concatenate(trips)
    offset_array = np.concatenate(offsets)
    num_sampled_cases = len(panel_values)
    simulated_times = _future_times(
        cutoff_values, offset_array, df.schema[time_col], "_simulated_time"
    )
    sample_map = pl.DataFrame(
        {
            panel_col: panel_values,
            "source_case": source_cases,
            "source_time": source_times,
            case_col: np.arange(
                case_offset, case_offset + num_sampled_cases, dtype=np.int64
            ),
            "simulation_stage": [stage] * num_sampled_cases,
            "simulation_round": round_array,
            "simulation_trip": trip_array,
            "days_after_intervention": offset_array,
        }
    ).with_columns(simulated_times)

    source = df.rename({case_col: "source_case"}).drop(time_col)
    sampled_rows = sample_map.join(
        source,
        on=[panel_col, "source_case"],
        how="left",
        validate="m:m",
    ).rename({"_simulated_time": time_col})
    sampled_rows = sampled_rows.with_columns(
        pl.lit(False).cast(df.schema[intervention_col]).alias(intervention_col)
    )
    if choice_col is not None and choice_col in sampled_rows.columns:
        sampled_rows = sampled_rows.drop(choice_col)
    return sampled_rows


def simulate_future_choice_sets(
    data: object,
    *,
    panel_col: str,
    case_col: str,
    time_col: str,
    intervention_col: str,
    choice_col: str | None = None,
    config: FutureSimulationConfig | None = None,
) -> CounterfactualWorlds:
    """Simulate anticipated and realized future choice-set worlds.

    Complete choice sets are sampled with replacement, preserving the joint
    distribution of product availability, prices, costs, and any other row-level
    attributes on a shopping trip.  The store's anticipated worlds draw only from
    choice sets dated through the intervention/original-order choice set.  Realized
    evaluation worlds draw from the complete panel, both before and after the
    intervention.

    Trip frequency is estimated separately for each panel and information set using
    the mean interval between observed choice sets.  By default, each simulated world
    draws trip counts and dates from a zero-truncated, capped Poisson process at that
    estimated rate, thereby integrating over trip timing as well as future product
    states while retaining at least one future trip per panel.  A pooled median
    interval handles panels with only one observed date.  The returned long-format
    frames are designed to be generated once and reused across all candidate policies,
    providing common simulated worlds for efficient, low-noise comparisons.

    Parameters
    ----------
    data : object
        Long-format panel choice data.
    panel_col : str
        Decision-maker identifier.
    case_col : str
        Choice-set or shopping-trip identifier.  It may repeat across panels.
    time_col : str
        Numeric, Date, or Datetime column identifying trip timing.
    intervention_col : str
        Boolean marker identifying exactly one intervention/original-order choice set
        per panel.  It may be true on one or all rows in that choice set.
    choice_col : str | None, optional
        Observed-choice column to remove from synthetic future data.  Choice outcomes
        should be integrated using predicted probabilities, not copied from the
        sampled historical trip.
    config : FutureSimulationConfig | None, optional
        Simulation settings.  Defaults to :class:`FutureSimulationConfig`.

    Returns
    -------
    CounterfactualWorlds
        Separate policy-design and realized-evaluation worlds plus trip diagnostics.

    Notes
    -----
    Full-panel draws use post-intervention data and are therefore valid only for ex
    post evaluation.  They must not be supplied to a policy-selection rule that is
    intended to represent the store's information at the intervention date.
    """
    config = FutureSimulationConfig() if config is None else config
    df = _coerce_frame(data)
    required = [panel_col, case_col, time_col, intervention_col]
    if choice_col is not None:
        required.append(choice_col)
    _require_columns(df, required)
    collisions = sorted(_METADATA_COLUMNS.intersection(df.columns))
    if collisions:
        raise ValueError(
            "Input data uses reserved simulation metadata columns: "
            f"{collisions}"
        )
    if df.select(pl.any_horizontal(pl.col(required).is_null()).any()).item():
        raise ValueError("Required simulation columns must not contain null values.")

    case_index, cutoffs = _choice_set_index(
        df,
        panel_col=panel_col,
        case_col=case_col,
        time_col=time_col,
        intervention_col=intervention_col,
    )
    indexed = case_index.join(cutoffs, on=panel_col, how="left")
    anticipated_pool = indexed.filter(pl.col(time_col) <= pl.col("_cutoff")).drop(
        "_cutoff"
    )
    realized_pool = case_index

    anticipated_summary = _pool_summary(
        anticipated_pool,
        panel_col=panel_col,
        time_col=time_col,
        config=config,
        stage="anticipated",
    )
    realized_summary = _pool_summary(
        realized_pool,
        panel_col=panel_col,
        time_col=time_col,
        config=config,
        stage="realized",
    )
    anticipated_rng, realized_rng = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(config.seed).spawn(2)
    ]
    anticipated = _sample_stage(
        df,
        anticipated_pool,
        cutoffs,
        anticipated_summary,
        panel_col=panel_col,
        case_col=case_col,
        time_col=time_col,
        intervention_col=intervention_col,
        choice_col=choice_col,
        stage="anticipated",
        rng=anticipated_rng,
        config=config,
        case_offset=0,
    )
    realized = _sample_stage(
        df,
        realized_pool,
        cutoffs,
        realized_summary,
        panel_col=panel_col,
        case_col=case_col,
        time_col=time_col,
        intervention_col=intervention_col,
        choice_col=choice_col,
        stage="realized",
        rng=realized_rng,
        config=config,
        case_offset=anticipated.select(case_col).n_unique(),
    )
    return CounterfactualWorlds(
        anticipated=anticipated,
        realized=realized,
        trip_summary=pl.concat([anticipated_summary, realized_summary]),
        config=config,
    )
