"""Run a reproducible parity benchmark against an isolated LCL installation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import numpy as np
import polars as pl


def synthetic_data() -> pl.DataFrame:
    """Generate a stable two-class panel with demographic membership."""
    rng = np.random.default_rng(20260902)
    rows: list[dict[str, object]] = []
    num_panels, cases_per_panel, num_alts = 180, 7, 3
    betas = np.array([[-1.5, -0.35], [0.35, 1.25]])
    for panel in range(num_panels):
        income = rng.normal()
        class_probability = 1.0 / (1.0 + np.exp(-(-0.2 + 1.0 * income)))
        class_index = int(rng.random() < class_probability)
        for occasion in range(cases_per_panel):
            case = panel * cases_per_panel + occasion
            price = rng.uniform(0.5, 4.0, num_alts)
            quality = rng.normal(1.5, 0.8, num_alts)
            utility = (
                betas[0, class_index] * price
                + betas[1, class_index] * quality
                + rng.gumbel(size=num_alts)
            )
            chosen = int(np.argmax(utility))
            for alt in range(num_alts):
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == chosen,
                        "price": float(price[alt]),
                        "quality": float(quality[alt]),
                        "income": float(income),
                    }
                )
    return pl.DataFrame(rows)


def apollo_data(path: Path) -> pl.DataFrame:
    """Convert Apollo's official wide modeChoice data to an identified long design."""
    wide = pl.read_csv(path).with_row_index("qID")
    alternatives = {1: "car", 2: "bus", 3: "air", 4: "rail"}
    frames = []
    for number, name in alternatives.items():
        frames.append(
            wide.select(
                "ID",
                "qID",
                "income",
                "female",
                pl.col(f"time_{name}").alias("time"),
                pl.col(f"cost_{name}").alias("cost"),
                pl.col(f"av_{name}").alias("av"),
                (pl.col("choice") == number).alias("choice"),
            ).with_columns(pl.lit(name).alias("alt"))
        )
    long = pl.concat(frames).filter(pl.col("av") == 1)
    income_mean = float(long["income"].mean())
    income_std = float(long["income"].std())
    return (
        long.with_columns(
            ((pl.col("income") - income_mean) / income_std).alias("income_z"),
            (pl.col("alt") == "bus").cast(pl.Float64).alias("asc_bus"),
            (pl.col("alt") == "car").cast(pl.Float64).alias("asc_car"),
            (pl.col("alt") == "rail").cast(pl.Float64).alias("asc_rail"),
        )
        .sort(["ID", "qID", "alt"])
        .select(
            "ID",
            "qID",
            "alt",
            "choice",
            "cost",
            "time",
            "asc_bus",
            "asc_car",
            "asc_rail",
            "income_z",
            "female",
        )
    )


def fit_once(
    implementation: str,
    dataset: str,
    model_family: str,
    data: pl.DataFrame,
) -> tuple[Any, float]:
    """Fit one implementation with equivalent settings."""
    if dataset == "synthetic":
        ids = ("alt", "case", "panel", "choice")
        utility = ["price", "quality"]
        demographics = ["income"]
        classes = 2
        max_em_iter = 120
    else:
        ids = ("alt", "qID", "ID", "choice")
        utility = ["cost", "time", "asc_bus", "asc_car", "asc_rail"]
        demographics = ["income_z", "female"]
        classes = 3
        max_em_iter = 80

    start = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if model_family == "conditional":
            from lcl.conditional_logit import ConditionalLogit

            model = ConditionalLogit(numeraire=utility[0])
            common_arguments = {
                "alts_col": ids[0],
                "cases_col": ids[1],
                "panels_col": ids[2],
                "choice_col": ids[3],
                "case_varnames": utility,
            }
            if implementation == "old":
                from lcl._struct import MleConfig

                result = model.fit(
                    data,
                    **common_arguments,
                    mle_config=MleConfig(maxiter=75, ftol=1e-5),
                )
            else:
                from lcl.options import InferenceOptions, OptimizationOptions

                result = model.fit(
                    data,
                    **common_arguments,
                    optimization_options=OptimizationOptions(
                        maxiter=75, gradient_tol=1e-5
                    ),
                    inference=InferenceOptions(covariance="clustered"),
                )
        elif implementation == "old":
            from lcl.latent_class_conditional_logit import (
                LatentClassConditionalLogit,
            )
            from lcl._struct import EMAlgConfig, MleConfig

            model = LatentClassConditionalLogit(
                num_classes=classes, numeraire=utility[0]
            )
            result = model.fit(
                data,
                alts_col=ids[0],
                cases_col=ids[1],
                panels_col=ids[2],
                choice_col=ids[3],
                case_varnames=utility,
                dem_varnames=demographics,
                em_alg_config=EMAlgConfig(
                    jax_prng_seed=7,
                    loglik_tol=1e-6,
                    maxiter=max_em_iter,
                    num_devices=1,
                    check_interval=5,
                ),
                mle_config=MleConfig(maxiter=75, ftol=1e-5),
            )
        else:
            from lcl.latent_class_conditional_logit import (
                LatentClassConditionalLogit,
            )
            from lcl.options import FitOptions, InferenceOptions, OptimizationOptions

            model = LatentClassConditionalLogit(
                num_classes=classes, numeraire=utility[0]
            )
            result = model.fit(
                data,
                alts_col=ids[0],
                cases_col=ids[1],
                panels_col=ids[2],
                choice_col=ids[3],
                case_varnames=utility,
                dem_varnames=demographics,
                fit_options=FitOptions(
                    seed=7,
                    max_em_iter=max_em_iter,
                    em_tol=1e-6,
                    num_devices=1,
                    check_interval=5,
                ),
                optimization_options=OptimizationOptions(maxiter=75, gradient_tol=1e-5),
                inference=InferenceOptions(covariance="clustered"),
            )
    parameters = (
        result.coeff_
        if model_family == "conditional"
        else result.em_res.structural_betas
    )
    jax.block_until_ready(parameters)
    elapsed = perf_counter() - start
    return result, elapsed


def main() -> None:
    """Run cold and warm fits and emit machine-readable estimates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("old", "new"))
    parser.add_argument("dataset", choices=("synthetic", "apollo"))
    parser.add_argument(
        "--model", choices=("latent-class", "conditional"), default="latent-class"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--apollo-path",
        type=Path,
        default=Path("/private/tmp/apollo_modeChoiceData.csv"),
    )
    args = parser.parse_args()
    data = (
        synthetic_data()
        if args.dataset == "synthetic"
        else apollo_data(args.apollo_path)
    )

    timings = []
    final_result = None
    for _ in range(args.repeats):
        final_result, elapsed = fit_once(
            args.implementation, args.dataset, args.model, data
        )
        timings.append(elapsed)
    if final_result is None:
        raise RuntimeError("No benchmark fit was run.")
    is_conditional = args.model == "conditional"
    theta = None if is_conditional else final_result.em_res.thetas
    payload = {
        "implementation": args.implementation,
        "dataset": args.dataset,
        "model": args.model,
        "rows": data.height,
        "panels": data["panel" if args.dataset == "synthetic" else "ID"].n_unique(),
        "timings_seconds": timings,
        "loglik": float(
            final_result.loglikelihood
            if is_conditional
            else final_result.em_res.unconditional_loglik
        ),
        "betas": np.asarray(
            final_result.coeff_
            if is_conditional
            else final_result.em_res.structural_betas
        ).tolist(),
        "shares": (
            None if is_conditional else np.asarray(final_result.em_res.shares).tolist()
        ),
        "thetas": None if theta is None else np.asarray(theta).tolist(),
        "iterations": int(
            final_result.total_iter if is_conditional else final_result.total_recursions
        ),
        "converged": bool(
            final_result.convergence
            if is_conditional and args.implementation == "old"
            else final_result.converged
        ),
    }
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
