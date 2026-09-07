"""Measure fit parity, independent likelihoods, timing, and EM executable memory.

Run against another checkout by setting PYTHONPATH to its src directory. Apollo
input is the official public apollo_modeChoiceData.csv; no network is used here.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from compare_release import apollo_data, synthetic_data
from lcl import FitOptions, InferenceOptions
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def independent_likelihood(result):
    """Score the undifferenced long design with NumPy/SciPy only."""
    data, state = result.data, result.em_res
    utility = np.asarray(data.X) @ np.asarray(state.structural_betas)
    cases = np.asarray(data.cases)
    starts = np.r_[0, np.flatnonzero(np.diff(cases)) + 1]
    maxima = np.maximum.reduceat(utility, starts, axis=0)
    denominator = np.add.reduceat(np.exp(utility - maxima[cases]), starts, axis=0)
    chosen = utility[np.asarray(data.y)]
    log_choice = chosen - maxima - np.log(denominator)
    panel_cases = np.asarray(data.panels_of_cases)
    panel_starts = np.r_[0, np.flatnonzero(np.diff(panel_cases)) + 1]
    kernels = np.add.reduceat(log_choice, panel_starts, axis=0)
    if data.dems is None:
        log_prior = np.log(np.asarray(state.shares))[None, :]
    else:
        theta = np.asarray(state.thetas)
        tail = theta[0] + np.asarray(data.dems) @ theta[1:]
        logits = np.c_[np.zeros(data.num_panels), tail]
        log_prior = logits - logsumexp(logits, axis=1, keepdims=True)
    weighted = kernels + log_prior
    panel_ll = logsumexp(weighted, axis=1)
    posterior = np.exp(weighted - panel_ll[:, None])
    return float(panel_ll.sum()), float(
        np.max(np.abs(posterior - state.class_probs_by_panel))
    )


def fit_benchmark(args):
    """Fit synthetic or Apollo data with identical options on each repeat."""
    synthetic = args.dataset == "synthetic"
    frame = synthetic_data() if synthetic else apollo_data(args.apollo_path)
    utility = (
        ["price", "quality"]
        if synthetic
        else ["cost", "time", "asc_bus", "asc_car", "asc_rail"]
    )
    kwargs = dict(
        alts_col="alt",
        cases_col="case" if synthetic else "qID",
        panels_col="panel" if synthetic else "ID",
        choice_col="choice",
        case_varnames=utility,
        dem_varnames=["income"] if synthetic else ["income_z", "female"],
        fit_options=FitOptions(
            seed=args.seed,
            em_tol=args.em_tol,
            max_em_iter=args.max_em_iter,
            check_interval=args.check_interval,
            starts=args.starts,
            num_devices=args.devices,
            polish=not args.no_polish,
        ),
        inference=InferenceOptions(covariance="clustered"),
    )
    times = []
    result = None
    for _ in range(args.repeats):
        start = perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = LatentClassConditionalLogit(
                num_classes=2 if synthetic else 3, numeraire=utility[0]
            ).fit(frame, **kwargs)
        jax.block_until_ready(result.em_res)
        times.append(perf_counter() - start)
    assert result is not None
    independent, posterior_error = independent_likelihood(result)
    history = result.em_history_["loglik"].to_numpy()
    return dict(
        dataset=args.dataset,
        timings_seconds=times,
        em_tol=args.em_tol,
        polish=not args.no_polish,
        seed=args.seed,
        starts=args.starts,
        iterations=result.total_recursions,
        loglik=float(result.em_res.unconditional_loglik),
        independent_loglik=independent,
        posterior_max_error=posterior_error,
        min_em_increment=float(np.min(np.diff(history))) if len(history) > 1 else None,
        score=float(result.observed_score_max),
        converged=bool(result.converged),
        betas=np.asarray(result.em_res.structural_betas).tolist(),
        thetas=np.asarray(result.em_res.thetas).tolist(),
        shares=np.asarray(result.em_res.shares).tolist(),
        covariance=np.asarray(result.cov_matrix).tolist(),
        polish_iterations=result.polish_report.iterations
        if result.polish_report
        else 0,
    )


def kernel_benchmark(args):
    """Time a compiled recursion and report XLA's buffer allocation sizes."""
    from lcl._case_utils import _diff_unchosen_chosen
    from lcl._em_alg_startup import _get_starting_vals
    from lcl._em_alg_steps import _compiled_em_step, place_em_vars
    from lcl._struct import Data
    from lcl.options import OptimizationOptions

    n, k, d, c = args.shape
    rng = np.random.default_rng(751)
    cases, rows = n * 7, n * 21
    data = Data(
        X=jnp.asarray(rng.normal(size=(rows, k))),
        dems=jnp.asarray(rng.normal(size=(n, d))) if d else None,
        y=jnp.asarray(np.tile([True, False, False], cases)),
        alts=jnp.asarray(np.tile(np.arange(3), cases), dtype=jnp.uint32),
        cases=jnp.asarray(np.repeat(np.arange(cases), 3), dtype=jnp.uint32),
        panels=jnp.asarray(np.repeat(np.arange(n), 21), dtype=jnp.uint32),
        panels_of_cases=jnp.asarray(np.repeat(np.arange(n), 7), dtype=jnp.uint32),
        num_cases_per_panel=jnp.full(n, 7, dtype=jnp.uint32),
        num_cases=cases,
        num_alt_vars=k,
        num_panels=n,
        num_dem_vars=d,
    )
    diff = _diff_unchosen_chosen(data)
    opt, fit = OptimizationOptions(), FitOptions(num_devices=args.devices)
    start = perf_counter()
    state = _get_starting_vals(diff, data, c, fit, opt)
    jax.block_until_ready(state)
    startup = perf_counter() - start
    state = place_em_vars(state, args.devices)
    step = _compiled_em_step(c, opt, args.devices, None, 1e-5)
    start = perf_counter()
    compiled = step.lower(state, diff, data).compile()
    compile_time = perf_counter() - start
    mem = compiled.compiled.memory_analysis()
    for _ in range(5):
        state, _ = compiled(state, diff, data)
    jax.block_until_ready(state)
    # Repeatedly run the same state to keep the Newton workload fixed.
    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        for _ in range(100):
            out = compiled(state, diff, data)
            jax.block_until_ready(out)
        times.append((perf_counter() - start) * 10)
    return dict(
        shape=args.shape,
        startup_seconds=startup,
        compile_seconds=compile_time,
        step_ms=times,
        argument_bytes=mem.argument_size_in_bytes,
        output_bytes=mem.output_size_in_bytes,
        temporary_bytes=mem.temp_size_in_bytes,
        alias_bytes=mem.alias_size_in_bytes,
        executable_total_bytes=mem.argument_size_in_bytes
        + mem.output_size_in_bytes
        + mem.temp_size_in_bytes
        - mem.alias_size_in_bytes,
    )


def main():
    """Emit benchmark results as one JSON object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["synthetic", "apollo", "kernel"])
    parser.add_argument(
        "--apollo-path", type=Path, default=Path("/tmp/apollo_modeChoiceData.csv")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--em-tol", type=float, default=1e-8)
    parser.add_argument("--max-em-iter", type=int, default=2000)
    parser.add_argument("--check-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[10000, 8, 3, 3],
        metavar=("PANELS", "VARS", "DEMS", "CLASSES"),
    )
    args = parser.parse_args()
    payload = (
        kernel_benchmark(args) if args.dataset == "kernel" else fit_benchmark(args)
    )
    payload.update(
        jax_version=jax.__version__,
        devices=str(jax.devices()),
        max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
