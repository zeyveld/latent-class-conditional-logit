"""Own, cross, and aggregate elasticity calculations."""

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import numpy as onp
import polars as pl

from lcl._encoding import _drop_formula_intercepts, _get_model_matrix, _to_pandas_frame
from lcl._kernels import _choice_probabilities_and_logsum


def elasticity_design_derivative(
    prediction: Any, variable: str
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return raw values and row-wise derivatives of all design columns."""
    model = prediction.results.model
    data = prediction.predict_data
    raw = prediction.raw_prediction_data
    encoder = getattr(model, "_encoder", None)
    if raw is not None and variable in raw.columns:
        try:
            raw_values = raw[variable].cast(pl.Float64).to_numpy()
        except Exception as exc:
            raise ValueError(
                f"Elasticities require a numeric raw variable; {variable!r} "
                "cannot be converted to floating point."
            ) from exc
        if not onp.all(onp.isfinite(raw_values)):
            raise ValueError(
                f"Elasticity variable {variable!r} contains non-finite values."
            )
        step = onp.cbrt(onp.finfo(onp.float64).eps) * onp.maximum(
            1.0, onp.abs(raw_values)
        )
        plus = raw.with_columns(
            pl.Series(variable, raw_values + step, dtype=pl.Float64)
        )
        minus = raw.with_columns(
            pl.Series(variable, raw_values - step, dtype=pl.Float64)
        )

        def utility_matrix(frame: pl.DataFrame) -> onp.ndarray:
            if encoder is not None and encoder.x_model_spec is not None:
                matrix = _drop_formula_intercepts(
                    _get_model_matrix(
                        encoder.x_model_spec,
                        _to_pandas_frame(frame),
                        formula=encoder.utility_formula,
                        label="utility formula for elasticity differentiation",
                    )
                )
                return onp.asarray(matrix, dtype=onp.float64)
            return (
                frame.select(model.case_varnames)
                .to_numpy()
                .astype(onp.float64, copy=False)
            )

        derivative = (utility_matrix(plus) - utility_matrix(minus)) / (
            2.0 * step[:, None]
        )
        if not onp.all(onp.isfinite(derivative)):
            raise ValueError(
                f"Could not construct a finite utility derivative for {variable!r}."
            )
        if not onp.any(onp.abs(derivative) > 1e-12):
            raise ValueError(
                f"Raw variable {variable!r} has no differentiable effect in the "
                "fitted utility specification."
            )
        return jnp.asarray(raw_values), jnp.asarray(derivative)
    try:
        variable_index = model.case_varnames.index(variable)
    except ValueError:
        raise ValueError(
            f"Variable {variable!r} is neither a raw prediction column nor an "
            "expanded utility-design column."
        )
    derivative = jnp.zeros_like(data.X).at[:, variable_index].set(1.0)
    return data.X[:, variable_index], derivative


def compute_elasticities(
    prediction: Any, variables: str | Iterable[str]
) -> pl.DataFrame:
    """Compute full own- and cross-elasticity matrices."""
    variables = [variables] if isinstance(variables, str) else list(variables)
    if not variables:
        raise ValueError("At least one elasticity variable is required.")
    data = prediction.predict_data
    if hasattr(prediction.results, "em_res"):
        betas = prediction.results.em_res.structural_betas
        if prediction.class_probs_by_panel is None:
            raise ValueError(
                "class_probs_by_panel must be available to compute LC elasticities."
            )
        if data.panels is None:
            raise ValueError("Panel identifiers are required for LC elasticities.")
        class_weights = prediction.class_probs_by_panel[data.panels]
    else:
        betas = prediction.results.coeff_[:, None]
        class_weights = jnp.ones((data.X.shape[0], 1))
    probabilities_by_class, _ = _choice_probabilities_and_logsum(
        data.X, betas, data.cases, data.num_cases
    )
    probabilities = jnp.sum(class_weights * probabilities_by_class, axis=1)
    j_values: dict[str, Any] = {
        "_cases": onp.asarray(data.cases),
        "cases": prediction.original_cases,
        "alts": prediction.original_alts,
        "P_j": onp.asarray(jnp.maximum(probabilities, 1e-250)),
    }
    if prediction.original_panels is not None:
        j_values["panels"] = prediction.original_panels
    k_values: dict[str, Any] = {
        "_cases": onp.asarray(data.cases),
        "target_alts": prediction.original_alts,
    }
    num_classes = betas.shape[1]
    for class_idx in range(num_classes):
        j_values[f"SP_jc_{class_idx}"] = onp.asarray(
            class_weights[:, class_idx] * probabilities_by_class[:, class_idx]
        )
        k_values[f"P_kc_{class_idx}"] = onp.asarray(
            probabilities_by_class[:, class_idx]
        )
    frame_j = pl.DataFrame(j_values)
    frame_k_base = pl.DataFrame(k_values)
    outputs = []
    for variable in variables:
        raw_values, design_derivative = elasticity_design_derivative(
            prediction, variable
        )
        slope = design_derivative @ betas
        direct = jnp.sum(class_weights * probabilities_by_class * slope, axis=1)
        frame_j_variable = frame_j.with_columns(pl.Series("U_j", onp.asarray(direct)))
        frame_k = frame_k_base
        for class_idx in range(num_classes):
            frame_k = frame_k.with_columns(
                pl.Series(f"slope_kc_{class_idx}", onp.asarray(slope[:, class_idx]))
            )
        frame_k = frame_k.with_columns(pl.Series("X_k", onp.asarray(raw_values)))
        cross = frame_j_variable.join(frame_k, on="_cases", how="inner")
        cross = cross.with_columns(
            V_jk=pl.sum_horizontal(
                [
                    pl.col(f"SP_jc_{class_idx}")
                    * pl.col(f"P_kc_{class_idx}")
                    * pl.col(f"slope_kc_{class_idx}")
                    for class_idx in range(num_classes)
                ]
            )
        )
        cross = cross.with_columns(
            is_own=pl.col("alts") == pl.col("target_alts")
        ).with_columns(
            D_jk=pl.when(pl.col("is_own"))
            .then(pl.col("U_j") - pl.col("V_jk"))
            .otherwise(-pl.col("V_jk"))
        )
        name = f"elasticity_{variable}"
        cross = cross.with_columns(
            (pl.col("D_jk") * pl.col("X_k") / pl.col("P_j")).alias(name)
        )
        ids = ["_cases"]
        if "panels" in cross.columns:
            ids.append("panels")
        ids.extend(["cases", "alts", "target_alts"])
        outputs.append(cross.select([*ids, name]))
    final = outputs[0]
    for output in outputs[1:]:
        keys = [
            column
            for column in ["_cases", "panels", "cases", "alts", "target_alts"]
            if column in final.columns
        ]
        final = final.join(output, on=keys)
    ids = ["cases", "alts", "target_alts"]
    if "panels" in final.columns:
        ids.insert(0, "panels")
    return final.select(
        [*ids, *(f"elasticity_{variable}" for variable in variables)]
    ).sort(["cases", "alts", "target_alts"])
