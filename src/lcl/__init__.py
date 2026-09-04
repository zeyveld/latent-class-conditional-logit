"""Latent-class conditional logit estimation and inference.

A JAX-accelerated Python library for the estimation, inference, and prediction
of standard and latent-class conditional logit models. LCL combines an
expectation-maximization algorithm with safeguarded, hardware-accelerated exact
Newton updates.

Provides seamless support for R-style formulaic data ingestion, robust sandwich
covariance estimators, fractional-response demographic regressions, and delta-method
willingness-to-pay (WTP) distributions.
"""

from __future__ import annotations

from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping

from jax import config as _jax_config
from jaxtyping import install_import_hook as _install_import_hook

# Adopt 64-bit precision before any JAX arrays are created.
# Discrete choice models are highly sensitive to vanishing gradients
# in the denominator of the logit probability.
_jax_config.update("jax_enable_x64", True)

# Ensure array arguments have mutually compatible shapes throughout the package.
with _install_import_hook("lcl", "beartype.beartype"):
    from lcl.constraints import NegativeCoefficient
    from lcl._cross_validation import cv_optimal_classes
    from lcl.options import (
        DiagnosticsOptions,
        FitOptions,
        InferenceOptions,
        Options,
        OptimizationOptions,
        PartitionType,
        PastChoicesData,
        WTPRequest,
    )
    from lcl.results import (
        CLResults,
        CLPrediction,
        LCLDiagnostics,
        LCLPrediction,
        LCLResults,
        ResultsProtocol,
    )
    from lcl.conditional_logit import ConditionalLogit
    from lcl.latent_class_conditional_logit import LatentClassConditionalLogit
    from lcl.spec import ChoiceIds, LCLSpec


def fit(
    data: object,
    spec: LCLSpec,
    *,
    options: Options | None = None,
    fit_options: FitOptions | None = None,
    optimization_options: OptimizationOptions | None = None,
    inference: InferenceOptions | None = None,
    diagnostics: DiagnosticsOptions | None = None,
    variable_labels: _Mapping[str, str] | None = None,
    dems_data: object | None = None,
    progress_callback: _Callable[[dict[str, object]], None] | None = None,
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

    Notes
    -----
    Latent-class estimation does not accept case or survey weights.  Use
    :class:`~lcl.conditional_logit.ConditionalLogit` when weighting is required.
    """
    model = LatentClassConditionalLogit(spec=spec)
    return model.fit(
        data=data,
        options=options,
        fit_options=fit_options,
        optimization_options=optimization_options,
        inference=inference,
        diagnostics=diagnostics,
        variable_labels=variable_labels,
        dems_data=dems_data,
        progress_callback=progress_callback,
    )


# Expose core classes and functions at the top level for clean user imports
__all__ = [
    "LatentClassConditionalLogit",
    "ConditionalLogit",
    "CLResults",
    "CLPrediction",
    "ChoiceIds",
    "LCLSpec",
    "NegativeCoefficient",
    "FitOptions",
    "OptimizationOptions",
    "InferenceOptions",
    "Options",
    "DiagnosticsOptions",
    "LCLResults",
    "LCLPrediction",
    "LCLDiagnostics",
    "ResultsProtocol",
    "WTPRequest",
    "PartitionType",
    "PastChoicesData",
    "fit",
    "cv_optimal_classes",
]

# Imported submodules are implementation details of package initialization.  Keep
# ``dir(lcl)`` honest: every public, non-underscore name is declared in ``__all__``.
for _imported_name in tuple(globals()):
    if not _imported_name.startswith("_") and _imported_name not in __all__:
        globals().pop(_imported_name)
del _imported_name

# EOF
