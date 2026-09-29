"""Use float64 for tests that call internal numerical kernels directly."""

import jax

jax.config.update("jax_enable_x64", True)
