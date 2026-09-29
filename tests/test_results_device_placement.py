"""Regress the LCL 0.1.44 boundary-score placement failure.

On CPU, start a fresh process with
XLA_FLAGS=--xla_force_host_platform_device_count=5 JAX_PLATFORMS=cpu.
This puts fitted arrays on devices separate from the CPU inference device.
On a GPU node, run without JAX_PLATFORMS=cpu to test actual GPU/CPU transfers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import polars as pl
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from lcl import ChoiceIds, FitOptions, InferenceOptions, LCLResults, LCLSpec
from lcl import NegativeCoefficient, fit
from lcl._case_utils import _diff_unchosen_chosen
from lcl._jax_compat import device_put_array_leaves
from lcl._polish import em_vars_from_flat


def synthetic_choices(
    *,
    price_effect: float | tuple[float, float] = -1.0,
    alias_noise: float | None = None,
):
    """Cross attributes independently; classes persist across a panel's choices."""
    rng = np.random.default_rng(20260922)
    rows = []
    for panel in range(300):
        demographic = rng.normal()
        group = int(rng.random() < 1 / (1 + np.exp(-demographic)))
        for occasion in range(8):
            price = rng.uniform(1, 5, 4)
            brand = np.array([0.0, 0.0, 1.0, 1.0])
            flavor = np.array([0.0, 1.0, 0.0, 1.0])
            if alias_noise is not None:
                flavor = brand + alias_noise * rng.normal(size=4)
            effect = (
                price_effect[group] if isinstance(price_effect, tuple) else price_effect
            )
            utility = (
                effect * price + (-1.8 if group == 0 else 1.8) * brand + 0.5 * flavor
            )
            chosen = np.argmax(utility + rng.gumbel(size=4))
            for alt in range(4):
                rows.append(
                    dict(
                        panel=panel,
                        case=panel * 8 + occasion,
                        alt=alt,
                        choice=alt == chosen,
                        price=price[alt],
                        brand=brand[alt],
                        flavor=flavor[alt],
                        demographic=demographic,
                    )
                )
    return pl.DataFrame(rows)


def _spec(classes=2):
    return LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility=("price", "brand", "flavor"),
        membership=("demographic",),
        classes=classes,
        constraints={"price": NegativeCoefficient()},
    )


@pytest.fixture(scope="module")
def mixed_fit():
    # Projected covariance already moves all inputs to CPU, allowing the
    # unpatched package to supply a real fitted state for the regression.
    result = fit(
        synthetic_choices(price_effect=(0.7, -1.0)),
        _spec(),
        fit_options=FitOptions(seed=82, max_em_iter=500, num_devices=1),
        inference=InferenceOptions(boundary="projected"),
    )
    assert result.converged and len(result.boundary_parameter_indices) == 1
    return result


def _source_sharding(device_count):
    cpu = jax.devices("cpu")[0]
    devices = [device for device in jax.devices() if device != cpu]
    if len(devices) < device_count:
        pytest.skip(f"Need {device_count} devices separate from CPU inference")
    return NamedSharding(
        Mesh(np.array(devices[:device_count]), ("fit",)), PartitionSpec()
    )


def _place(tree, sharding):
    return jax.tree_util.tree_map(
        lambda leaf: (
            jax.device_put(leaf, sharding) if isinstance(leaf, jax.Array) else leaf
        ),
        tree,
    )


@pytest.mark.parametrize("device_count", [1, 4])
@pytest.mark.parametrize(
    "skip,boundary", [(True, "projected"), (True, "strict"), (False, "strict")]
)
@pytest.mark.parametrize("invalid_boundary", [False, True])
def test_boundary_result_matches_cpu(
    mixed_fit, device_count, skip, boundary, invalid_boundary
):
    source = _source_sharding(device_count)
    cpu = jax.devices("cpu")[0]
    r = mixed_fit
    with jax.default_device(cpu):
        data = device_put_array_leaves(r.data, cpu)
        state = device_put_array_leaves(r.em_res, cpu)
        if invalid_boundary:
            # Force the negative-price class onto its bound while claiming
            # convergence. The inward score must still invalidate that claim.
            flat = device_put_array_leaves(r.flat_params, cpu)
            beta, theta = r._param_packing.unpack(flat)
            beta = beta.at[0, :].set(-r.model.numeraire_min_abs)
            flat = jnp.concatenate((beta.ravel(), theta.ravel()))
            state = em_vars_from_flat(
                flat, _diff_unchosen_chosen(data), data, r._param_packing
            )

        def construct(em_state, encoded_data):
            return LCLResults(
                r.model,
                em_state,
                encoded_data,
                r.total_recursions,
                converged=True,
                inference=InferenceOptions(skip=skip, boundary=boundary),
                estim_time_sec=0.0,
                observed_score_max=0.0,
                param_packing=r._param_packing,
            )

        reference = construct(state, data)

    relocated = construct(_place(state, source), _place(data, source))
    assert relocated.flat_params.sharding.device_set == source.device_set
    np.testing.assert_allclose(
        relocated.flat_params, reference.flat_params, atol=0, rtol=0
    )
    assert relocated.boundary_parameter_indices == reference.boundary_parameter_indices
    assert relocated.observed_score_max == pytest.approx(
        reference.observed_score_max, abs=1e-12
    )
    assert relocated.boundary_kkt_violation == pytest.approx(
        reference.boundary_kkt_violation, abs=1e-12
    )
    assert relocated.converged == reference.converged == (not invalid_boundary)
    assert relocated.covariance_available == reference.covariance_available
    if invalid_boundary:
        assert relocated.boundary_kkt_violation > relocated.score_tol


@pytest.mark.parametrize("device_count", [1, 4])
def test_real_skipped_inference_fit(device_count):
    if len(jax.devices()) < device_count:
        pytest.skip(f"Need {device_count} fitting devices")
    result = fit(
        synthetic_choices(price_effect=0.7),
        _spec(classes=max(2, device_count)),
        fit_options=FitOptions(seed=82, max_em_iter=500, num_devices=device_count),
        inference=InferenceOptions(skip=True, boundary="projected"),
    )
    assert result.boundary_parameter_indices
    assert np.isfinite(result.observed_score_max)
    assert result.converged == (result.observed_score_max <= result.score_tol)
    assert result.inference_status == "skipped"
    inference_device = jax.local_devices(backend="cpu")[0]
    for array in jax.tree.leaves((result.data, result.em_res, result.flat_params)):
        if isinstance(array, jax.Array):
            assert array.devices() == {inference_device}
