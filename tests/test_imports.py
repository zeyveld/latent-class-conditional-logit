import jax
import jax.numpy as jnp
import polars as pl

import lcl


def test_jax_installation() -> None:
    x = jnp.array([1.0, 2.0, 3.0])
    assert x.sum() == 6.0


def test_polars_installation() -> None:
    df = pl.DataFrame({"choice": [0, 1], "price": [10.5, 12.0]})
    assert df.shape == (2, 2)


def test_lcl_import() -> None:
    assert lcl is not None
