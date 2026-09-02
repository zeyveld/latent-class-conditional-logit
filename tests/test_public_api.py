"""Tests for the documented package surface."""

import inspect

import jax.numpy as jnp
import pytest

import lcl
from lcl import options, results


def test_all_matches_public_package_namespace() -> None:
    """Every non-private package name is intentional and documented."""
    public_names = {name for name in dir(lcl) if not name.startswith("_")}
    assert public_names == set(lcl.__all__)


def test_public_modules_own_user_facing_types() -> None:
    """Public imports and module imports resolve to identical class objects."""
    assert lcl.FitOptions is options.FitOptions
    assert lcl.InferenceOptions is options.InferenceOptions
    assert lcl.LCLResults is results.LCLResults
    assert lcl.CLResults is results.CLResults


def test_all_fit_entry_points_accept_the_shared_options_bundle() -> None:
    """High- and low-level fit APIs use the same aggregate option name."""
    for fit_callable in (
        lcl.fit,
        lcl.LatentClassConditionalLogit.fit,
        lcl.ConditionalLogit.fit,
    ):
        assert "options" in inspect.signature(fit_callable).parameters


def test_both_predict_methods_take_data_first() -> None:
    """Both result families accept raw tabular data as the first argument."""
    for result_type in (results.LCLResults, results.CLResults):
        parameters = list(inspect.signature(result_type.predict).parameters)
        assert parameters[:2] == ["self", "data"]


def test_divergent_result_names_remain_deprecated_aliases() -> None:
    """Legacy names expose the canonical values during the migration window."""
    conditional = object.__new__(results.CLResults)
    conditional.converged = True
    conditional.cov_matrix = jnp.eye(1)
    conditional.adjusted_bic = jnp.asarray(3.0)
    latent = object.__new__(results.LCLResults)
    latent.converged = False
    latent.cov_matrix = jnp.eye(2)
    latent.adjusted_bic = jnp.asarray(4.0)

    for result in (conditional, latent):
        with pytest.warns(DeprecationWarning):
            assert result.convergence is result.converged
        with pytest.warns(DeprecationWarning):
            assert jnp.array_equal(result.covariance, result.cov_matrix)
        with pytest.warns(DeprecationWarning):
            assert result.abic == result.adjusted_bic


def test_options_bundle_cannot_be_implicitly_merged() -> None:
    """An aggregate configuration has one unambiguous source of truth."""
    from lcl.options import _resolve_options

    with pytest.raises(ValueError, match="either options"):
        _resolve_options(lcl.Options(), inference=lcl.InferenceOptions())
