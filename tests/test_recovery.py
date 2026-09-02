"""Simulation checks for parameter recovery and standard-error calibration."""

import numpy as np
import polars as pl

from lcl import FitOptions, InferenceOptions, OptimizationOptions
from lcl.conditional_logit import ConditionalLogit
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def test_two_class_parameter_recovery_up_to_canonical_order() -> None:
    rng = np.random.default_rng(82)
    true_betas = np.array([[-1.6, -0.35], [0.35, 1.4]])
    rows: list[dict[str, object]] = []
    realized_classes = []
    for panel in range(100):
        latent_class = int(rng.random() > 0.55)
        realized_classes.append(latent_class)
        for case_within_panel in range(7):
            design = rng.normal(size=(3, 2))
            utility = design @ true_betas[:, latent_class]
            probs = np.exp(utility - utility.max())
            probs /= probs.sum()
            chosen = rng.choice(3, p=probs)
            for alt in range(3):
                rows.append(
                    {
                        "panel": panel,
                        "case": panel * 10 + case_within_panel,
                        "alt": alt,
                        "choice": alt == chosen,
                        "x1": float(design[alt, 0]),
                        "x2": float(design[alt, 1]),
                    }
                )

    result = LatentClassConditionalLogit(num_classes=2).fit(
        pl.DataFrame(rows),
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x1", "x2"],
        fit_options=FitOptions(max_em_iter=150, em_tol=1e-7, check_interval=5),
        optimization_options=OptimizationOptions(gradient_tol=1e-7),
        inference=InferenceOptions(skip=True),
    )

    assert result.converged
    np.testing.assert_allclose(
        np.asarray(result.em_res.structural_betas), true_betas, atol=0.25
    )
    realized_shares = np.bincount(realized_classes, minlength=2) / len(realized_classes)
    np.testing.assert_allclose(result.em_res.shares, realized_shares, atol=0.12)


def test_hessian_standard_errors_have_reasonable_simulated_coverage() -> None:
    true_beta = 0.8
    covered = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        rows = []
        for case in range(250):
            x = rng.normal(size=3)
            probs = np.exp(true_beta * x - np.max(true_beta * x))
            probs /= probs.sum()
            chosen = rng.choice(3, p=probs)
            for alt in range(3):
                rows.append(
                    {
                        "case": case,
                        "alt": alt,
                        "choice": alt == chosen,
                        "x": float(x[alt]),
                    }
                )
        result = ConditionalLogit().fit(
            pl.DataFrame(rows),
            alts_col="alt",
            cases_col="case",
            choice_col="choice",
            case_varnames=["x"],
            inference=InferenceOptions(covariance="unadjusted"),
        )
        estimate = float(result.coeff_[0])
        standard_error = float(result.stderr[0])
        covered.append(abs(estimate - true_beta) <= 1.96 * standard_error)

    # A deliberately broad non-flaky guard: grossly mis-scaled SEs fail, while
    # ordinary Monte Carlo variation in a 20-replication test does not.
    assert 15 <= sum(covered) <= 20
