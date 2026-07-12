"""Re-run the examples whose outputs appear in the documentation.

Run from the repository root with ``uv run --group docs python
tools/rebuild_docs_examples.py``. The script prints section-delimited transcripts for
the homepage, estimation tutorial, and cross-validation tutorial.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

import lcl
from lcl import (
    ChoiceIds,
    FitOptions,
    InferenceOptions,
    LCLSpec,
    NegativeCoefficient,
    OptimizationOptions,
    PartitionType,
    WTPRequest,
)


DATA_URL = (
    "https://www.apollochoicemodelling.com/files/examples/data/"
    "apollo_modeChoiceData.csv"
)


def _section(title: str) -> None:
    """Print a transcript section delimiter."""
    print(f"\n{'=' * 24} {title} {'=' * 24}\n")


def _apollo_long_data() -> pl.DataFrame:
    """Download and reshape Apollo mode-choice data to long format."""
    df_wide = pl.read_csv(DATA_URL).with_row_index("qID")
    alts_map = {1: "car", 2: "bus", 3: "air", 4: "rail"}
    frames = []
    for number, name in alts_map.items():
        frames.append(
            df_wide.select(
                [
                    pl.col("ID"),
                    pl.col("qID"),
                    pl.col("income"),
                    pl.col("female"),
                    pl.col(f"time_{name}").alias("time"),
                    pl.col(f"cost_{name}").alias("cost"),
                    pl.col(f"av_{name}").alias("av"),
                    (pl.col("choice") == number).alias("choice"),
                ]
            ).with_columns(pl.lit(name).alias("alt"))
        )
    return (
        pl.concat(frames)
        .filter(pl.col("av") == 1)
        .sort(["ID", "qID", "alt"])
        .with_columns(
            pl.col("income")
            .qcut(3, labels=["low", "mid", "high"])
            .cast(pl.String)
            .alias("income_band")
        )
    )


def _spec(classes: int = 3) -> LCLSpec:
    """Return the shared labeled Apollo model specification."""
    return LCLSpec(
        ids=ChoiceIds(alt="alt", case="qID", panel="ID", choice="choice"),
        utility_formula="choice ~ cost + time + C(alt)",
        membership_formula="~ C(income_band) + female",
        classes=classes,
        constraints={"cost": NegativeCoefficient(units="dollars")},
        variable_labels={
            "cost": "Fare",
            "time": "Travel time",
            "alt": "Travel mode",
            "income_band": "Household income band",
            "female": "Female traveler",
        },
    )


def _homepage_example() -> None:
    """Run the synthetic homepage quickstart."""
    rng = np.random.default_rng(7)
    n_panels, n_choices, n_alts = 200, 4, 3
    true_class = rng.choice(2, size=n_panels, p=[0.55, 0.45])
    beta_price = np.array([-1.8, -0.3])
    beta_quality = np.array([0.4, 1.6])
    rows = []
    for panel in range(n_panels):
        income = rng.normal()
        for choice_index in range(n_choices):
            prices = rng.uniform(0.5, 3.0, size=n_alts)
            quality = rng.uniform(0.0, 5.0, size=n_alts)
            utility = (
                beta_price[true_class[panel]] * prices
                + beta_quality[true_class[panel]] * quality
                + rng.gumbel(size=n_alts)
            )
            chosen = int(np.argmax(utility))
            for alternative in range(n_alts):
                rows.append(
                    {
                        "panel": panel,
                        "case": panel * n_choices + choice_index,
                        "alt": alternative,
                        "choice": alternative == chosen,
                        "price": float(prices[alternative]),
                        "quality": float(quality[alternative]),
                        "income": float(income),
                    }
                )
    data = pl.DataFrame(rows)
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ price + quality",
        membership_formula="~ income",
        classes=2,
        constraints={"price": NegativeCoefficient()},
        variable_labels={
            "price": "Price",
            "quality": "Product quality",
            "income": "Household income",
        },
    )
    results = lcl.fit(
        data,
        spec,
        fit_options=FitOptions(max_em_iter=50, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=40),
    )
    results.summarize_betas()
    print(results)


def _estimation_example(data: pl.DataFrame) -> None:
    """Run all printed estimation and prediction examples."""
    print(data.drop("income_band").head(8))
    results = lcl.fit(
        data,
        _spec(),
        fit_options=FitOptions(max_em_iter=25, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=40),
        inference=InferenceOptions(covariance="clustered"),
    )
    results.summarize_betas()
    print(results)
    results.diagnostics().print()
    print(results.convergence_report())
    print(results.class_shares())
    print(results.class_coefficients())

    counterfactual = data.with_columns(
        pl.when(pl.col("alt").is_in(["bus", "rail"]))
        .then(pl.col("cost") * 1.25)
        .otherwise(pl.col("cost"))
        .alias("cost")
    )
    prediction = results.predict(data=counterfactual, past_choices=data)
    print(prediction.predicted_probs.head(8))
    print(prediction.elasticities(["cost", "time"]).head(8))
    prediction.compute_wtp(
        WTPRequest(
            alt_var="time",
            demographic_var="income_band",
            partition_type=PartitionType.CATEGORICAL,
        ),
        WTPRequest(
            alt_var="time",
            demographic_var="female",
            partition_type=PartitionType.CATEGORICAL,
        ),
        class_probabilities="prior",
    )
    print(prediction.wtp_by_class("time"))
    print(prediction.denominator_diagnostics())
    prediction.compute_wtp(
        WTPRequest(
            alt_var="time",
            demographic_var="income",
            partition_type=PartitionType.QUINTILES,
        ),
        class_probabilities="prior",
    )


def _cross_validation_example(data: pl.DataFrame) -> None:
    """Run the model-selection sweep and render its documented chart."""
    cv_results = lcl.cv_optimal_classes(
        data,
        _spec(),
        num_classes_list=[2, 3, 4, 5],
        folds=3,
        seed=42,
        fit_options=FitOptions(max_em_iter=60, num_devices=1),
        optimization_options=OptimizationOptions(maxiter=30),
    )
    print(cv_results)
    chart = (
        alt.Chart(cv_results)
        .mark_line(point=True, color="#3F2B47")
        .encode(
            x=alt.X("Num_Classes:O", title="Number of latent classes"),
            y=alt.Y(
                "Avg_OOS_LL:Q",
                title="Average out-of-sample log likelihood",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(width=600, height=380)
    )
    chart.save(Path("site") / "cv_plot.html")


def main() -> None:
    """Run every documentation example and print one combined transcript."""
    Path("site").mkdir(exist_ok=True)
    transcript = io.StringIO()
    with contextlib.redirect_stdout(transcript):
        _section("HOMEPAGE")
        _homepage_example()
        data = _apollo_long_data()
        _section("ESTIMATION")
        _estimation_example(data)
        _section("CROSS VALIDATION")
        _cross_validation_example(data)
    output = transcript.getvalue()
    Path("site/example-transcript.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
