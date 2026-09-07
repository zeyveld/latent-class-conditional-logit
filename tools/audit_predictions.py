"""Reproduce prediction and welfare checks on synthetic and public Apollo data.

No network access is used. Supply Apollo's official apollo_modeChoiceData.csv.
Outputs are compact JSON; temporary logs and data belong in a local sandbox.
"""

from __future__ import annotations

import argparse
from copy import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy.special import logsumexp, softmax

from lcl import (
    ConditionalLogit,
    FitOptions,
    InferenceOptions,
    LatentClassConditionalLogit,
)
from lcl import OptimizationOptions, PartitionType, WTPRequest


TRUE_BETA = np.array([[-1.8, -0.5], [0.4, 1.6]])


def synthetic(seed=20260907, panels=800, occasions=12):
    """Generate choices from a two-class DGP with demographic membership."""
    rng = np.random.default_rng(seed)
    income = rng.normal(size=panels)
    prior = softmax(np.c_[np.zeros(panels), -0.2 + 1.1 * income], axis=1)
    classes = (rng.random(panels) < prior[:, 1]).astype(int)
    x = np.stack(
        [
            rng.uniform(0.5, 4.0, (panels, occasions, 3)),
            rng.uniform(0.0, 5.0, (panels, occasions, 3)),
        ],
        axis=-1,
    )
    utilities = np.einsum("ntjk,kn->ntj", x, TRUE_BETA[:, classes])
    choices = (utilities + rng.gumbel(size=utilities.shape)).argmax(axis=-1)
    return pl.DataFrame(
        {
            "panel": np.repeat(np.arange(panels), occasions * 3),
            "case": np.tile(np.repeat(np.arange(occasions), 3), panels),
            "alt": np.tile(np.arange(3), panels * occasions),
            "choice": (np.arange(3)[None, None, :] == choices[:, :, None]).ravel(),
            "price": x[..., 0].ravel(),
            "quality": x[..., 1].ravel(),
            "income": np.repeat(income, occasions * 3),
            "segment": np.repeat(np.where(income < 0, "low", "high"), occasions * 3),
        }
    )


def apollo(path):
    """Use SP observations only, avoiding an unmodelled RP/SP scale restriction."""
    wide = (
        pl.read_csv(path)
        .with_row_index("case")
        .filter(pl.col("SP") == 1)
        .with_columns(pl.col("SP_task").cast(pl.Int64))
    )
    frames = []
    for number, name in enumerate(["car", "bus", "air", "rail"], 1):
        frames.append(
            wide.select(
                pl.col("ID").alias("panel"),
                "case",
                "SP_task",
                "income",
                "female",
                pl.col(f"cost_{name}").alias("price"),
                pl.col(f"time_{name}").alias("time"),
                pl.col(f"av_{name}").alias("available"),
                (pl.col("choice") == number).alias("choice"),
            ).with_columns(pl.lit(name).alias("alt"))
        )
    return (
        pl.concat(frames)
        .filter(pl.col("available") == 1)
        .with_columns(
            (pl.col("income") / 10000).round(4).alias("income_scaled"),
            *[
                (pl.col("alt") == name).cast(pl.Float64).alias(f"asc_{name}")
                for name in ["bus", "car", "rail"]
            ],
        )
        .sort("panel", "case", "alt")
    )


def fit_model(data, utility, demographics, classes=2, seed=7, starts=2):
    """Fit with multiple starts and panel-clustered covariance."""
    with contextlib.redirect_stdout(io.StringIO()):
        return LatentClassConditionalLogit(num_classes=classes, numeraire="price").fit(
            data=data,
            alts_col="alt",
            cases_col="case",
            panels_col="panel",
            choice_col="choice",
            case_varnames=utility,
            dem_varnames=demographics,
            fit_options=FitOptions(
                seed=seed, starts=starts, max_em_iter=150, num_devices=1
            ),
            optimization_options=OptimizationOptions(maxiter=50),
            inference=InferenceOptions(covariance="clustered"),
        )


def scores(prediction, observed):
    """Score held-out outcomes using original identity keys."""
    joined = prediction.predicted_probs.join(
        observed.select(
            pl.col("panel").alias("panels"),
            pl.col("case").alias("cases"),
            pl.col("alt").alias("alts"),
            "choice",
        ),
        on=["panels", "cases", "alts"],
    )
    chosen = joined.filter(pl.col("choice"))["choice_probs"].to_numpy()
    brier = joined.with_columns(
        (pl.col("choice_probs") - pl.col("choice").cast(pl.Float64)).pow(2).alias("sq")
    )
    accuracy = (
        joined.sort("choice_probs", descending=True)
        .unique(subset=["panels", "cases"], maintain_order=True)["choice"]
        .mean()
    )
    return dict(
        log_loss=float(-np.log(np.maximum(chosen, 1e-300)).mean()),
        brier=float(brier["sq"].sum() / chosen.size),
        accuracy=float(accuracy),
        cases=chosen.size,
    )


def independent(prediction):
    """Evaluate class logit probabilities and logsum welfare in NumPy/SciPy."""
    data = prediction.predict_data
    beta = np.asarray(prediction.results.em_res.structural_betas)
    cases = np.asarray(data.cases)
    utilities = np.asarray(data.X) @ beta
    logsum = np.stack(
        [logsumexp(utilities[cases == case], axis=0) for case in range(data.num_cases)]
    )
    class_probs = np.exp(utilities - logsum[cases])
    weights = np.asarray(prediction.class_probs_by_panel)
    probs = np.sum(weights[np.asarray(data.panels)] * class_probs, axis=1)
    surplus = np.sum(
        weights[np.asarray(data.panels_of_cases)] * logsum / -beta[0], axis=1
    )
    return probs, surplus


def holdout(name, data, train, history, future, utility, demographics):
    """Fit on other consumers; compare priors and history on untouched future choices."""
    start = perf_counter()
    result = fit_model(
        train, utility, demographics, classes=3 if name == "apollo_sp" else 2
    )
    prior = result.predict(data=future)
    pooled = copy(prior)
    pooled.class_probs_by_panel = jnp.repeat(
        result.em_res.shares[None, :], prior.predict_data.num_panels, axis=0
    )
    pooled.predicted_probs = prior.predicted_probs.with_columns(
        pl.Series("choice_probs", independent(pooled)[0])
    )
    posterior = result.predict(data=future, past_choices=history)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = ConditionalLogit(numeraire="price").fit(
            train,
            alts_col="alt",
            cases_col="case",
            panels_col="panel",
            choice_col="choice",
            case_varnames=utility,
            inference=InferenceOptions(covariance="clustered"),
        )
    altered = future.with_columns((pl.col("price") * 1.1).alias("price"))
    cf = result.predict(data=altered, past_choices=history)
    welfare = posterior.mean_surplus_change(cf)
    probs, surplus = independent(posterior)
    output = dict(
        name=name,
        train_panels=train["panel"].n_unique(),
        test_panels=future["panel"].n_unique(),
        train_cases=train.select("panel", "case").n_unique(),
        history_cases=history.select("panel", "case").n_unique(),
        converged=bool(result.converged),
        score=float(result.observed_score_max),
        cl=scores(cl.predict(future), future),
        population_shares=scores(pooled, future),
        prior=scores(prior, future),
        posterior=scores(posterior, future),
        probability_max_error=float(
            abs(probs - posterior.predicted_probs["choice_probs"].to_numpy()).max()
        ),
        surplus_max_error=float(
            abs(surplus - posterior.surplus["surplus"].to_numpy()).max()
        ),
        welfare_10pct_price_increase=welfare.to_dicts()[0],
        covariance_finite=bool(np.isfinite(result.latent_cov_matrix).all()),
        elapsed_seconds=perf_counter() - start,
    )
    # Independent finite-difference parameter gradient for posterior welfare.
    from lcl._prediction_inference import mean_surplus_change

    def kwargs(pred):
        values = pred._design_kwargs()
        values.pop("panels")
        values["panels_of_cases"] = pred.predict_data.panels_of_cases
        return values

    params = np.asarray(result.flat_params)

    def target(p):
        return float(
            mean_surplus_change(
                jnp.asarray(p),
                baseline=kwargs(posterior),
                counterfactual=kwargs(cf),
                case_weights=posterior._case_panel_weights(),
            )
        )

    step = 1e-5
    gradient = np.array(
        [
            (
                target(params + np.eye(params.size)[i] * step)
                - target(params - np.eye(params.size)[i] * step)
            )
            / (2 * step)
            for i in range(params.size)
        ]
    )
    se = float(np.sqrt(gradient @ np.asarray(result.latent_cov_matrix) @ gradient))
    output["welfare_se_finite_difference"] = se
    output["welfare_se_error"] = abs(se - welfare["std_error"][0])
    return output


def coverage(repetitions):
    """Small Monte Carlo diagnostic for WTP SE calibration, conditional on demographics."""
    estimates = []
    truths = []
    ses = []
    convergence = []
    request = WTPRequest("quality", "segment", PartitionType.CATEGORICAL)
    for run in range(repetitions):
        data = synthetic(seed=1000 + run, panels=220, occasions=6)
        result = fit_model(data, ["price", "quality"], ["income"], starts=2)
        prediction = result.predict(data=data)
        table = next(iter(prediction.compute_wtp(request, show=False).values()))
        row = table.filter(pl.col("segment") == "high").row(0, named=True)
        income = (
            data.select("panel", "income")
            .unique()
            .filter(pl.col("income") >= 0)["income"]
            .to_numpy()
        )
        prior = softmax(np.c_[np.zeros(len(income)), -0.2 + 1.1 * income], axis=1)
        truth = float((prior @ (TRUE_BETA[1] / -TRUE_BETA[0])).mean())
        estimates.append(row["Mean_Marginal_WTP"])
        truths.append(truth)
        ses.append(row["Standard_Error"])
        convergence.append(bool(result.converged))
        print(f"Coverage replication {run + 1}/{repetitions}", flush=True)
    errors = np.asarray(estimates) - np.asarray(truths)
    se = np.asarray(ses)
    valid = np.isfinite(se) & np.asarray(convergence)
    return dict(
        repetitions=repetitions,
        usable=int(valid.sum()),
        mean_error=float(errors[valid].mean()),
        rmse=float(np.sqrt((errors[valid] ** 2).mean())),
        mean_se=float(se[valid].mean()),
        coverage_95=float((abs(errors[valid]) <= 1.96 * se[valid]).mean()),
        estimates=estimates,
        truths=truths,
        standard_errors=ses,
        converged=convergence,
        interpretation="Exploratory small Monte Carlo; not a precision guarantee or exhaustive coverage study.",
    )


def main():
    """Run reproducible local audit experiments and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apollo", type=Path, default=Path("/private/tmp/apollo_modeChoiceData.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=100)
    args = parser.parse_args()
    output = {"seed": 20260907, "jax_version": jax.__version__, "holdouts": []}
    synth = synthetic()
    output["holdouts"].append(
        holdout(
            "synthetic",
            synth,
            synth.filter((pl.col("panel") < 500) & (pl.col("case") < 8)),
            synth.filter((pl.col("panel") >= 500) & (pl.col("case") < 8)),
            synth.filter((pl.col("panel") >= 500) & (pl.col("case") >= 8)),
            ["price", "quality"],
            ["income"],
        )
    )
    print("Synthetic holdout complete", flush=True)
    if args.apollo.exists():
        real = apollo(args.apollo)
        ids = real["panel"].unique().sort().to_numpy()
        training_ids = np.random.default_rng(8).choice(ids, size=350, replace=False)
        is_train = pl.col("panel").is_in(training_ids)
        test = real.filter(~is_train)
        # Within-consumer order, using the supplied SP task number.
        output["apollo_sha256"] = hashlib.sha256(args.apollo.read_bytes()).hexdigest()
        output["holdouts"].append(
            holdout(
                "apollo_sp",
                real,
                real.filter(is_train),
                test.filter(pl.col("SP_task") <= 8),
                test.filter(pl.col("SP_task") > 8),
                ["price", "time", "asc_bus", "asc_car", "asc_rail"],
                ["income_scaled", "female"],
            )
        )
        print("Apollo holdout complete", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.replications:
        output["wtp_coverage"] = coverage(args.replications)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
