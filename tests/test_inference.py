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
        "optimization_options": OptimizationOptions(newton_decrement_tol=1e-8),
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
        weight_type="frequency",
    )
    replicated = _robust_covariance(
        bread,
        np.repeat(scores, weights, axis=0),
        finite_sample_correction=False,
    )
    np.testing.assert_allclose(weighted, replicated, rtol=1e-12, atol=1e-12)


def test_probability_weighted_meat_squares_the_weights() -> None:
    """Survey weights enter the meat squared, matching Stata's pweight."""
    scores = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]])
    weights = np.array([1.0, 3.0, 2.0])
    bread = np.array([[0.8, 0.1], [0.1, 0.6]])
    actual = _robust_covariance(
        bread,
        scores,
        finite_sample_correction=False,
        weights=weights,
        weight_type="probability",
    )
    weighted_scores = scores * weights[:, None]
    expected = bread @ (weighted_scores.T @ weighted_scores) @ bread
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_weight_types_agree_when_every_weight_is_one() -> None:
    """The unweighted default is untouched by the weight-type distinction."""
    scores = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]])
    bread = np.array([[0.8, 0.1], [0.1, 0.6]])
    unweighted = _robust_covariance(bread, scores)
    for weight_type in ("probability", "frequency"):
        np.testing.assert_allclose(
            _robust_covariance(
                bread, scores, weights=np.ones(3), weight_type=weight_type
            ),
            unweighted,
            rtol=1e-12,
            atol=1e-12,
        )


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


def test_conditional_logit_weight_types_share_estimates_and_differ_in_variance() -> None:
    """Survey and frequency weights change only the sandwich, and only when they vary."""
    data = _synthetic_cl(num_cases=400)
    rng = np.random.default_rng(2)
    weights = 1.0 + rng.integers(0, 4, size=data["case"].n_unique()).astype(float)
    weight_map = dict(zip(sorted(set(data["case"].to_list())), weights))
    data = data.with_columns(
        pl.col("case").replace_strict(weight_map).alias("sampling_weight")
    )
    common = dict(
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
        weights="sampling_weight",
        inference=InferenceOptions(covariance="robust"),
    )
    probability = ConditionalLogit().fit(data, weight_type="probability", **common)
    frequency = ConditionalLogit().fit(data, weight_type="frequency", **common)

    np.testing.assert_allclose(
        np.asarray(probability.coeff_), np.asarray(frequency.coeff_)
    )
    assert float(probability.loglikelihood) == pytest.approx(
        float(frequency.loglikelihood)
    )
    assert not np.allclose(
        np.asarray(probability.stderr), np.asarray(frequency.stderr)
    )

    # The scores of the weighted objective are w_i s_i, so the probability meat
    # squares the weights.
    scores = np.asarray(probability.grad_n) * np.asarray(probability.case_weights)[
        :, None
    ]
    bread = np.asarray(probability.hess_inv)
    n = scores.shape[0]
    expected = bread @ (scores.T @ scores) @ bread * (n / (n - 1))
    np.testing.assert_allclose(
        np.asarray(probability.latent_cov_matrix), expected, rtol=1e-10, atol=1e-14
    )


def test_unweighted_fits_are_identical_under_either_weight_type() -> None:
    """The default path is untouched by the weight-type distinction."""
    data = _synthetic_cl(num_cases=200)
    common = dict(
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
        inference=InferenceOptions(covariance="robust"),
    )
    probability = ConditionalLogit().fit(data, weight_type="probability", **common)
    frequency = ConditionalLogit().fit(data, weight_type="frequency", **common)
    np.testing.assert_allclose(
        np.asarray(probability.cov_matrix), np.asarray(frequency.cov_matrix)
    )


def test_conditional_logit_bootstrap_wtp_respects_the_sign_constraint() -> None:
    """Ratio draws are taken where the numeraire coefficient cannot change sign.

    Drawing structural coefficients directly puts mass on a positive numeraire,
    which the softplus parameterization excludes; the ratios that follow are
    heavy tailed and inflate the bootstrap standard error several fold.
    """
    data = _synthetic_cl(seed=31, num_cases=150)
    result = ConditionalLogit(numeraire="price").fit(
        data,
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["price", "quality"],
        inference=InferenceOptions(covariance="robust"),
    )
    prediction = result.predict(data)
    delta = float(prediction.wtp("quality", se="delta")["std_error"][0])
    bootstrap = float(
        prediction.wtp(
            "quality", se="bootstrap", bootstrap_draws=8000, bootstrap_seed=1
        )["std_error"][0]
    )
    # Both estimate the same asymptotic standard error; the bootstrap only picks
    # up curvature, so it should be close rather than a multiple.
    assert bootstrap == pytest.approx(delta, rel=0.25)

    # The draws themselves stay inside the constrained region.
    from lcl._case_utils import _to_structural_betas

    covariance = np.asarray(result.latent_cov_matrix)
    rng = np.random.default_rng(1)
    draws = np.asarray(result.latent_coeff_) + rng.multivariate_normal(
        np.zeros(covariance.shape[0]), covariance, size=4000
    )
    structural = np.asarray(
        jax.vmap(
            lambda p: _to_structural_betas(
                p, result.model.numeraire_idx, result.model.numeraire_min_abs
            )
        )(jnp.asarray(draws))
    )
    assert np.all(structural[:, result.model.numeraire_idx] < 0.0)
