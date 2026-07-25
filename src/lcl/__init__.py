"""Latent-class conditional logit estimation and inference.

A JAX-accelerated Python library for the estimation, inference, and prediction
of standard and latent-class conditional logit models. LCL combines an
expectation-maximization algorithm with safeguarded, hardware-accelerated exact
Newton updates.

Provides seamless support for R-style formulaic data ingestion, robust sandwich
covariance estimators, fractional-response demographic regressions, and delta-method
willingness-to-pay (WTP) distributions.
"""

from collections.abc import Mapping

from jax import config
from jaxtyping import install_import_hook

# Adopt 64-bit precision before any JAX arrays are created.
# Discrete choice models are highly sensitive to vanishing gradients
# in the denominator of the logit probability.
config.update("jax_enable_x64", True)

# Ensure array arguments have mutually compatible shapes throughout the package.
with install_import_hook("lcl", "beartype.beartype"):
    from lcl.constraints import NegativeCoefficient
    from lcl._cross_validation import cv_optimal_classes
    from lcl._prediction import LCLPrediction
    from lcl._results import LCLResults
    from lcl._struct import (
        DiagnosticsOptions,
        EMAlgConfig,
        ErrorConfig,
        FitOptions,
        InferenceOptions,
        MleConfig,
        OptimizationOptions,
        PartitionType,
        PastChoicesData,
        WTPRequest,
    )
    from lcl.conditional_logit import CLResults, ConditionalLogit
    from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
    from lcl.spec import ChoiceIds, LCLSpec


def fit(
    data: object,
    spec: LCLSpec,
    *,
    fit_options: FitOptions | None = None,
    optimization_options: OptimizationOptions | None = None,
    inference: InferenceOptions | None = None,
    diagnostics: DiagnosticsOptions | None = None,
    variable_labels: Mapping[str, str] | None = None,
) -> LCLResults:
    """Fit a latent-class conditional-logit model from an :class:`LCLSpec`.

    Parameters
    ----------
    data : object
        Long-format choice data.
    spec : LCLSpec
        Declarative model specification.
    fit_options : FitOptions | None, optional
        EM algorithm options.
    optimization_options : OptimizationOptions | None, optional
        M-step optimizer options.
    inference : InferenceOptions | None, optional
        Covariance and standard-error options.
    diagnostics : DiagnosticsOptions | None, optional
        Diagnostic thresholds and switches.
    variable_labels : Mapping[str, str] | None, optional
        Optional display labels for raw model/DataFrame variable names.  Labels
        supplement any labels stored on ``spec`` and are used only in
        presentation tables.

    Returns
    -------
    LCLResults
        Fitted latent-class results.
    """
    model = LatentClassConditionalLogit(spec=spec)
    return model.fit(
        data=data,
        fit_options=fit_options,
        optimization_options=optimization_options,
        inference=inference,
        diagnostics=diagnostics,
        variable_labels=variable_labels,
    )


# Expose core classes and functions at the top level for clean user imports
__all__ = [
    "LatentClassConditionalLogit",
    "ConditionalLogit",
    "CLResults",
    "ChoiceIds",
    "LCLSpec",
    "NegativeCoefficient",
    "FitOptions",
    "OptimizationOptions",
    "InferenceOptions",
    "DiagnosticsOptions",
    "EMAlgConfig",
    "ErrorConfig",
    "MleConfig",
    "LCLResults",
    "LCLPrediction",
    "WTPRequest",
    "PartitionType",
    "PastChoicesData",
    "fit",
    "cv_optimal_classes",
]

# EOF
