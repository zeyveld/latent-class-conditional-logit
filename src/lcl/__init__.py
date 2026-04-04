"""
Latent Class Conditional Logit (LCL)
====================================

A JAX-accelerated Python library for the estimation, inference, and prediction
of standard and latent-class conditional logit models. LCL utilizes an
Expectation-Maximization (EM) algorithm combined with hardware-accelerated
L-BFGS optimization to dramatically reduce estimation times for complex
discrete choice paradigms.

Provides seamless support for R-style formulaic data ingestion, robust sandwich
covariance estimators, fractional-response demographic regressions, and delta-method
willingness-to-pay (WTP) distributions.
"""

import jax.numpy as jnp
from jax import config
from jaxtyping import install_import_hook

# Adopt 64-bit precision before any JAX arrays are created.
# Discrete choice models are highly sensitive to vanishing gradients
# in the denominator of the logit probability.
config.update("jax_enable_x64", True)

import numpy as onp
import polars as pl

# Ensure array args have mutually compatible shapes throughout the package
with install_import_hook("lcl", "beartype.beartype"):
    from lcl._prediction import LCLPrediction
    from lcl._results import LCLResults
    from lcl._struct import PartitionType, PastChoicesData, WTPRequest
    from lcl._wip_cross_validation import cv_optimal_classes
    from lcl.conditional_logit import CLResults, ConditionalLogit
    from lcl.latent_class_conditional_logit import LatentClassConditionalLogit

# Expose core classes and functions at the top level for clean user imports
__all__ = [
    "LatentClassConditionalLogit",
    "ConditionalLogit",
    "CLResults",
    "LCLResults",
    "LCLPrediction",
    "WTPRequest",
    "PartitionType",
    "PastChoicesData",
    "cv_optimal_classes",
]

# EOF
