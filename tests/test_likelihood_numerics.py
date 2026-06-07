import jax
import jax.numpy as jnp
import numpy as onp
import polars as pl
import pytest
from jax.nn import softmax

from lcl._case_utils import _diff_unchosen_chosen, _loglik_gradient
from lcl._demographics import _compute_grouped_data_loglik_grad_hess
from lcl._em_alg_steps import _compute_panel_logliks
from lcl._jax_compat import device_put_array_leaves
from lcl._optimize import exact_newton_minimize
from lcl._struct import (
    Data,
    EMAlgConfig,
    ErrorConfig,
    MleConfig,
    PartitionType,
    PastChoicesData,
    WTPRequest,
)
from lcl.conditional_logit import ConditionalLogit
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def _tiny_panel_data() -> Data:
    X = jnp.array(
        [
            [0.0, 0.0],
            [800.0, 1.0],
            [-800.0, -1.0],
            [0.0, 0.0],
            [-1.0, 900.0],
            [1.0, -900.0],
        ]
    )
    y = jnp.array([True, False, False, False, True, False])
    cases = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.uint32)
    alts = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.uint32)
    panels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.uint32)
    panels_of_cases = jnp.array([0, 1], dtype=jnp.uint32)
    num_cases_per_panel = jnp.array([1, 1], dtype=jnp.uint32)
    return Data(
        X=X,
        dems=None,
        y=y,
        alts=alts,
        cases=cases,
        panels=panels,
        panels_of_cases=panels_of_cases,
        num_cases_per_panel=num_cases_per_panel,
        num_cases=2,
        num_alt_vars=2,
        num_panels=2,
        num_dem_vars=0,
    )


def _device_platform(array: jax.Array) -> str:
    return next(iter(array.devices())).platform


def test_device_put_array_leaves_preserves_static_metadata() -> None:
    data = _tiny_panel_data()

    moved = device_put_array_leaves(data, jax.devices("cpu")[0])

    assert _device_platform(moved.X) == "cpu"
    assert isinstance(moved.num_cases, int)
    assert isinstance(moved.num_panels, int)


def test_loglik_gradient_and_hessian_match_autodiff() -> None:
    data = _tiny_panel_data()
    diff = _diff_unchosen_chosen(data)
    weights = jnp.ones(data.num_cases)
    betas = jnp.array([0.002, -0.003])

    def objective(params):
        return _loglik_gradient(params, diff, weights)[0][0]

    _, grad, hess = _loglik_gradient(betas, diff, weights)

    assert jnp.allclose(grad, jax.grad(objective)(betas))
    assert jnp.allclose(hess, jax.hessian(objective)(betas))


def _demographic_data_from_dems(dems) -> Data:
    panels = jnp.arange(dems.shape[0], dtype=jnp.uint32)
    return Data(
        X=jnp.zeros((dems.shape[0], 1)),
        dems=dems,
        y=None,
        alts=jnp.zeros(dems.shape[0], dtype=jnp.uint32),
        cases=panels,
        panels=panels,
        panels_of_cases=panels,
        num_cases_per_panel=jnp.ones(dems.shape[0], dtype=jnp.uint32),
        num_cases=dems.shape[0],
        num_alt_vars=1,
        num_panels=dems.shape[0],
        num_dem_vars=dems.shape[1],
    )


def test_fractional_response_gradient_and_hessian_match_autodiff() -> None:
    dems = jnp.array(
        [
            [-1.0, 0.25],
            [0.0, -0.5],
            [0.75, 1.0],
            [1.5, -1.25],
        ]
    )
    data = _demographic_data_from_dems(dems)
    class_probs_by_panel = jnp.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.65, 0.20],
            [0.25, 0.25, 0.50],
            [0.55, 0.05, 0.40],
        ]
    )
    thetas = jnp.array(
        [
            [0.10, -0.20],
            [0.30, 0.40],
            [-0.50, 0.20],
        ]
    ).ravel()

    def objective(params):
        loss, _, _ = _compute_grouped_data_loglik_grad_hess(
            params, class_probs_by_panel, data, 3
        )
        return loss

    _, grad, hess = _compute_grouped_data_loglik_grad_hess(
        thetas, class_probs_by_panel, data, 3
    )

    assert jnp.all(jnp.isfinite(hess))
    assert jnp.allclose(grad, jax.grad(objective)(thetas), rtol=1e-8, atol=1e-8)
    assert jnp.allclose(hess, jax.hessian(objective)(thetas), rtol=1e-8, atol=1e-8)


def test_fractional_response_hessian_matches_random_autodiff_simulations() -> None:
    rng = onp.random.default_rng(0)
    num_panels = 6
    num_dem_vars = 2
    num_classes = 3

    for _ in range(5):
        data = _demographic_data_from_dems(
            jnp.array(rng.normal(size=(num_panels, num_dem_vars)))
        )
        class_probs_by_panel = softmax(
            jnp.array(rng.normal(size=(num_panels, num_classes))),
            axis=1,
        )
        thetas = jnp.array(
            rng.normal(scale=0.3, size=(num_dem_vars + 1, num_classes - 1))
        ).ravel()

        def objective(params):
            loss, _, _ = _compute_grouped_data_loglik_grad_hess(
                params, class_probs_by_panel, data, num_classes
            )
            return loss

        _, grad, hess = _compute_grouped_data_loglik_grad_hess(
            thetas, class_probs_by_panel, data, num_classes
        )

        autodiff_grad = jax.grad(objective)(thetas)
        autodiff_hess = jax.hessian(objective)(thetas)
        assert jnp.max(jnp.abs(grad - autodiff_grad)) < 1e-8
        assert jnp.max(jnp.abs(hess - autodiff_hess)) < 1e-8


def test_panel_loglik_hessian_stays_finite_with_extreme_utilities() -> None:
    data = _tiny_panel_data()
    diff = _diff_unchosen_chosen(data)
    class_probs = jnp.array([[0.65, 0.35], [0.25, 0.75]])
    flat_betas = jnp.array([0.9, -0.8, -0.7, 0.6])

    def objective(params):
        betas = params.reshape(2, 2)
        return jnp.sum(_compute_panel_logliks(betas, class_probs, diff, data))

    assert jnp.all(jnp.isfinite(jax.hessian(objective)(flat_betas)))


def test_newton_backtracking_only_uses_scalar_value_function() -> None:
    counts = {"value": 0, "heavy": 0}

    def bump(kind):
        counts[kind] += 1

    def value_fn(params):
        jax.debug.callback(lambda _: bump("value"), params[0])
        base_loss = (params[0] - 1.0) ** 2
        return jnp.where(params[0] > 0.75, 100.0, base_loss)

    def value_grad_hess_fn(params):
        jax.debug.callback(lambda _: bump("heavy"), params[0])
        loss = (params[0] - 1.0) ** 2
        grad = jnp.array([2.0 * (params[0] - 1.0)])
        hess = jnp.array([[2.0]])
        return loss, grad, hess

    res = exact_newton_minimize(
        value_fn,
        value_grad_hess_fn,
        jnp.array([0.0]),
        maxiter=1,
        tol=1e-12,
    )
    jax.block_until_ready(res.params)

    assert counts == {"value": 2, "heavy": 2}


def _small_lcl_df() -> pl.DataFrame:
    rows = []
    for panel in [101, 205]:
        for case in [1, 2]:
            for alt in [0, 1]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == ((panel + case) % 2),
                        "x": float(alt),
                        "dem": float(panel == 205),
                    }
                )
    return pl.DataFrame(rows)


def _small_wtp_df() -> pl.DataFrame:
    rows = []
    panel_ids = [101, 205, 309, 415, 588]
    for panel_idx, panel in enumerate(panel_ids):
        quintile = panel_idx + 1
        for case in [1, 2]:
            for alt in [0, 1]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == ((panel_idx + case) % 2),
                        "cost": float(2.0 + alt + case + 0.25 * quintile),
                        "time": float(8.0 - alt + 0.5 * case + quintile),
                        "income_quintile": f"Q{quintile}",
                        "income_q2": float(quintile == 2),
                        "income_q3": float(quintile == 3),
                        "income_q4": float(quintile == 4),
                        "income_q5": float(quintile == 5),
                    }
                )
    return pl.DataFrame(rows)


def test_lcl_no_dem_share_pack_unpack_roundtrip() -> None:
    shares = jnp.array([0.2, 0.3, 0.5])
    theta = jnp.log(shares[1:] / shares[0])
    reconstructed = softmax(jnp.r_[0.0, theta])
    assert jnp.allclose(reconstructed, shares)


def test_prediction_uses_demographics_when_no_past_choices() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=["dem"],
        em_alg_config=EMAlgConfig(maxiter=1, num_devices=1),
        mle_config=MleConfig(maxiter=2),
        error_config=ErrorConfig(skip_std_errs=True),
    )
    results.em_res = results.em_res._replace(
        structural_betas=jnp.array([[0.0, 0.0]]),
        latent_betas=jnp.array([[0.0, 0.0]]),
        thetas=jnp.array([[0.0], [2.0]]),
        shares=jnp.array([0.5, 0.5]),
    )

    prediction = results.predict(data=df)

    assert prediction.class_probs_by_panel is not None
    assert not jnp.allclose(
        prediction.class_probs_by_panel[0], prediction.class_probs_by_panel[1]
    )


def test_lcl_robust_covariance_and_delta_method_outputs_are_on_cpu() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)

    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=["dem"],
        em_alg_config=EMAlgConfig(maxiter=1, num_devices=1),
        mle_config=MleConfig(maxiter=2),
        error_config=ErrorConfig(robust=True),
    )

    means, se_means = results._apply_delta_method(
        results._calc_population_mean_betas,
        results.flat_params,
        dems=results.data.dems,
        num_panels=results.data.num_panels,
    )

    assert _device_platform(results.cov_matrix) == "cpu"
    assert _device_platform(means) == "cpu"
    assert _device_platform(se_means) == "cpu"
    assert jnp.all(jnp.isfinite(results.cov_matrix))
    assert jnp.all(jnp.isfinite(se_means))


def test_prediction_accepts_tabular_past_choices() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=["dem"],
        em_alg_config=EMAlgConfig(maxiter=1, num_devices=1),
        mle_config=MleConfig(maxiter=2),
        error_config=ErrorConfig(skip_std_errs=True),
    )
    results.em_res = results.em_res._replace(
        structural_betas=jnp.array([[1.0, -1.0]]),
        latent_betas=jnp.array([[1.0, -1.0]]),
        thetas=jnp.array([[0.0], [0.5]]),
        shares=jnp.array([0.5, 0.5]),
    )
    dem_matrix = (
        df.select(["panel", "dem"])
        .unique(subset=["panel"], maintain_order=True)
        .sort("panel")
        .select(["dem"])
        .to_numpy()
    )
    wrapped_past_choices = PastChoicesData(
        X=df.select(["x"]).to_numpy(),
        y=df["choice"].to_numpy(),
        alts=df["alt"].to_numpy(),
        cases=df["case"].to_numpy(),
        panels=df["panel"].to_numpy(),
        dems=dem_matrix,
    )

    from_tabular = results.predict(data=df, past_choices=df)
    from_tabular_with_separate_dems = results.predict(
        data=df,
        past_choices=df.drop("dem"),
        past_choices_dems_data=df.select(["panel", "dem"]).unique(
            subset=["panel"], maintain_order=True
        ),
    )
    from_wrapper = results.predict(data=df, past_choices=wrapped_past_choices)

    assert from_tabular.class_probs_by_panel is not None
    assert from_tabular_with_separate_dems.class_probs_by_panel is not None
    assert from_wrapper.class_probs_by_panel is not None
    assert jnp.allclose(
        from_tabular.class_probs_by_panel,
        from_wrapper.class_probs_by_panel,
    )
    assert jnp.allclose(
        from_tabular_with_separate_dems.class_probs_by_panel,
        from_wrapper.class_probs_by_panel,
    )
    assert onp.allclose(
        from_tabular.predicted_probs["choice_probs"].to_numpy(),
        from_wrapper.predicted_probs["choice_probs"].to_numpy(),
    )


def test_wtp_accepts_raw_prediction_dummy_bundle_and_external_partition_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = _small_wtp_df()
    dummy_vars = ["income_q2", "income_q3", "income_q4", "income_q5"]
    model = LatentClassConditionalLogit(num_classes=2, numeraire="cost")
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["cost", "time"],
        em_alg_config=EMAlgConfig(maxiter=1, num_devices=1),
        mle_config=MleConfig(maxiter=2),
        error_config=ErrorConfig(robust=True),
    )
    prediction = results.predict(data=df)

    dummy_tables = prediction.compute_wtp(
        WTPRequest(
            alt_var="time",
            demographic_var="income_quintile",
            partition_type=PartitionType.CATEGORICAL,
            dummy_vars=dummy_vars,
            dummy_labels=["Q2", "Q3", "Q4", "Q5"],
            base_category="Q1",
        )
    )
    partition_data = df.select(["panel", "income_quintile"]).unique(
        subset=["panel"], maintain_order=True
    )
    raw_tables = prediction.compute_wtp(
        WTPRequest(
            alt_var="time",
            demographic_var="income_quintile",
            partition_type=PartitionType.CATEGORICAL,
        ),
        partition_data=partition_data,
        panel_col="panel",
    )

    dummy_df = next(iter(dummy_tables.values())).sort("income_quintile")
    raw_df = next(iter(raw_tables.values())).sort("income_quintile")

    assert dummy_df["income_quintile"].to_list() == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert dummy_df["income_quintile"].to_list() == raw_df["income_quintile"].to_list()
    assert onp.allclose(
        dummy_df["Mean_Marginal_WTP"].to_numpy(),
        raw_df["Mean_Marginal_WTP"].to_numpy(),
    )
    assert onp.allclose(
        dummy_df["Standard_Error"].to_numpy(),
        raw_df["Standard_Error"].to_numpy(),
    )
    captured = capsys.readouterr()
    assert "Marginal WTP for time by income_quintile" in captured.out
    assert "--- LaTeX Output ---" in captured.out
    assert "--- Table preview ---" in captured.out
    assert "shape:" not in captured.out


def test_formula_encoder_drops_unidentified_intercepts() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)

    parsed = model._ingest_data(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        formula="choice ~ x | dem",
        choice_col=None,
        case_varnames=None,
        dem_varnames=None,
        dems_data=None,
    )

    assert parsed.case_varnames == ["x"]
    assert parsed.dem_varnames == ["dem"]
    assert parsed.X.shape == (df.height, 1)
    assert parsed.dems is not None
    assert parsed.dems.shape == (2, 1)


def test_demographic_formula_intercept_only_means_no_demographics() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)

    parsed = model._ingest_data(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        formula="choice ~ x | 1",
        choice_col=None,
        case_varnames=None,
        dem_varnames=None,
        dems_data=None,
    )

    assert parsed.case_varnames == ["x"]
    assert parsed.dem_varnames is None
    assert parsed.dems is None


def test_prediction_noncontiguous_panel_ids() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=["dem"],
        em_alg_config=EMAlgConfig(maxiter=1, num_devices=1),
        mle_config=MleConfig(maxiter=2),
        error_config=ErrorConfig(skip_std_errs=True),
    )

    prediction = results.predict(data=df)

    assert set(prediction.predicted_probs["panels"].to_list()) == {101, 205}


def test_case_ids_repeated_across_panels_are_not_merged() -> None:
    df = _small_lcl_df()
    model = LatentClassConditionalLogit(num_classes=2)
    parsed = model._ingest_data(
        data=df,
        alts_col="alt",
        cases_col="case",
        panels_col="panel",
        formula=None,
        choice_col="choice",
        case_varnames=["x"],
        dem_varnames=["dem"],
        dems_data=None,
    )
    data, *_ = model._setup_data(parsed)

    assert data.num_cases == 4


def test_cl_hessian_covariance_uses_exact_hessian_not_opg_inverse() -> None:
    rows = []
    for case in range(6):
        for alt in range(3):
            rows.append(
                {
                    "case": case,
                    "alt": alt,
                    "choice": alt == (case % 3),
                    "x": float(alt - 1),
                    "z": float((case + alt) % 3),
                }
            )
    df = pl.DataFrame(rows)
    model = ConditionalLogit()
    results = model.fit(
        data=df,
        alts_col="alt",
        cases_col="case",
        choice_col="choice",
        case_varnames=["x", "z"],
        mle_config=MleConfig(maxiter=20),
        error_config=ErrorConfig(robust=False),
    )
    diff = _diff_unchosen_chosen(results.data)
    weights = jnp.ones(results.data.num_cases)

    def objective(params):
        return _loglik_gradient(params, diff, weights)[0][0]

    expected_cov = jnp.linalg.pinv(jax.hessian(objective)(results.latent_coeff_))

    assert onp.allclose(onp.array(results.covariance), onp.array(expected_cov))
