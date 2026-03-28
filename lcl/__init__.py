import jax.numpy as jnp
from jax import config
from jaxtyping import install_import_hook

# Adopt 64-bit precision before any JAX arrays are created
config.update("jax_enable_x64", True)

import numpy as onp
import polars as pl

# Ensure array args have mutually compatible shapes
with install_import_hook("lcl", "beartype.beartype"):
    from lcl._prediction import LCLPrediction
    from lcl._results import LCLResults
    from lcl._struct import PartitionType, PastChoicesData, WTPRequest
    from lcl.conditional_logit import CLResults, ConditionalLogit
    from lcl.latent_class_conditional_logit import LatentClassConditionalLogit

# Expose classes at the top level
__all__ = [
    "LatentClassConditionalLogit",
    "ConditionalLogit",
    "CLResults",
    "LCLResults",
    "LCLPrediction",
    "WTPRequest",
    "PartitionType",
    "PastChoicesData",
]

# EOF
