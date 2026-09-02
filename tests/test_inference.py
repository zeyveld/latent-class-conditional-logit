"""Regression tests for scale-equivariant fitting and inference conventions."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from lcl import InferenceOptions, OptimizationOptions
from lcl._case_utils import _to_structural_betas
from lcl._encoding import ChoiceDataEncoder
from lcl._inference import _robust_covariance
from lcl._prediction import LCLPrediction
from lcl.conditional_logit import ConditionalLogit


def _synthetic_cl(seed: int = 19, num_cases: int = 700) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    beta = np.array([-1.1, 0.75])
    for case in range(num_cases):
        design = rng.normal(size=(3, 2))
        probs = np.exp(design @ beta - np.max(design @ beta))
        probs /= probs.sum()
        chosen = rng.choice(3, p=probs)
        for alt in range(3):
            rows.append(
                {
                    "case": 1000 + 7 * case,
                    "alt": f"a{alt}",
                    "choice": alt == chosen,
                    "price": float(design[alt, 0]),
                    "quality": float(design[alt, 1]),
                }
            )
    return pl.DataFrame(rows)


def test_conditional_logit_is_affine_equivariant() -> None:
    """Rescaling a utility column rescales only its coefficient."""
    data = _synthetic_cl()
    scaled = data.with_columns((pl.col("quality") / 1e5).alias("quality"))
    kwargs = {
        "alts_col": "alt",
        "cases_col": "case",
        "choice_col": "choice",
        "case_varnames": ["price", "quality"],
        "optimization_options": OptimizationOptions(gradient_tol=1e-8),
        "inference": InferenceOptions(skip=True),
    }
    reference = ConditionalLogit(numeraire="price").fit(data, **kwargs)
    transformed = ConditionalLogit(numeraire="price").fit(scaled, **kwargs)

    np.testing.assert_allclose(
        np.asarray(transformed.coeff_) / np.array([1.0, 1e5]),
        np.asarray(reference.coeff_),
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(transformed.loglikelihood) == pytest.approx(
        float(reference.loglikelihood), abs=1e-8
    )


def test_frequency_weighted_meat_matches_literal_replication() -> None:
    """Integer frequency weights reproduce duplicated score rows exactly."""
    scores = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]])
    weights = np.array([1, 3, 2])
    bread = np.array([[0.8, 0.1], [0.1, 0.6]])
    weighted = _robust_covariance(
        bread,
        scores,
        finite_sample_correction=False,
        weights=weights,
    )
    replicated = _robust_covariance(
        bread,
        np.repeat(scores, weights, axis=0),
        finite_sample_correction=False,
    )
    np.testing.assert_allclose(weighted, replicated, rtol=1e-12, atol=1e-12)


def test_constrained_cl_exposes_structural_covariance() -> None:
    """The public covariance is aligned with the reported structural coefficients."""
    result = ConditionalLogit(numeraire="price").fit(
        _synthetic_cl(num_cases=250),
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
        inference=InferenceOptions(covariance="unadjusted"),
    )

    jacobian = jax.jacrev(
        lambda raw: _to_structural_betas(
            raw,
            result.model.numeraire_idx,
            result.model.numeraire_min_abs,
        )
    )(result.latent_coeff_)
    expected = jacobian @ result.hess_inv @ jacobian.T
    np.testing.assert_allclose(result.cov_matrix, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.stderr**2, jnp.diag(result.cov_matrix))


def test_elasticity_uses_original_ids_and_full_formula_chain_rule() -> None:
    """Polynomial utility terms contribute to derivatives and labels stay external."""
    rows = []
    for case_idx, case in enumerate([101, 205, 309, 412]):
        for alt_idx, alt in enumerate(["bus", "rail", "air"]):
            price = 1.0 + 0.7 * alt_idx + 0.2 * case_idx
            rows.append(
                {
                    "panel": f"p{case_idx // 2}",
                    "case": case,
                    "alt": alt,
                    "choice": alt_idx == case_idx % 3,
                    "price": price,
                    "quality": float((case_idx + 2 * alt_idx) % 4),
                }
            )
    raw = pl.DataFrame(rows).sort(["panel", "case", "alt"])
    encoder = ChoiceDataEncoder(
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        utility_formula="choice ~ price + I(price**2) + quality",
    )
    parsed = encoder.fit_transform(raw)
    model = ConditionalLogit()
    data, _, _ = model._setup_data(parsed)
    fitted_model = SimpleNamespace(
        case_varnames=parsed.case_varnames,
        _encoder=encoder,
    )
    coefficients = jnp.array([-0.8, 0.12, 0.35])
    results = SimpleNamespace(model=fitted_model, coeff_=coefficients)
    prediction = LCLPrediction(
        predicted_probs_df=pl.DataFrame(),
        surplus_df=pl.DataFrame(),
        wtp_alt_vars_by_panel_df=pl.DataFrame(),
        predict_data=data,
        results=results,
        original_alts=parsed.original_alts,
        original_cases=parsed.original_cases,
        original_panels=parsed.original_panels,
        raw_prediction_data=raw,
    )

    elasticities = prediction.elasticities("price")
    assert set(elasticities["alts"]) == {"air", "bus", "rail"}
    assert set(elasticities["cases"]) == {101, 205, 309, 412}
    own = elasticities.filter(
        (pl.col("cases") == 101)
        & (pl.col("alts") == "air")
        & (pl.col("target_alts") == "air")
    )["elasticity_price"][0]

    case = raw.filter(pl.col("case") == 101)
    price = case["price"].to_numpy()
    quality = case["quality"].to_numpy()
    utility = (
        coefficients[0] * price + coefficients[1] * price**2 + coefficients[2] * quality
    )
    probs = np.exp(np.asarray(utility) - float(jnp.max(utility)))
    probs /= probs.sum()
    air_index = case["alt"].to_list().index("air")
    slope = float(coefficients[0] + 2 * coefficients[1] * price[air_index])
    expected = slope * price[air_index] * (1.0 - probs[air_index])
    assert own == pytest.approx(expected, rel=2e-6, abs=2e-7)
