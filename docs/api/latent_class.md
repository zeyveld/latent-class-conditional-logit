# Latent-class conditional logit

The latent-class estimator fits a finite mixture of conditional logits by
expectation-maximization. Class membership may be represented by aggregate shares or
modeled as a function of panel characteristics with fractional multinomial logit.

Most users should start with [`LCLSpec` and `lcl.fit`](specification.md). The lower-level
class remains available for direct use.

## Model

::: lcl.latent_class_conditional_logit.LatentClassConditionalLogit

## Results

::: lcl._results.LCLResults

## Diagnostics

::: lcl._diagnostics.LCLDiagnostics

## Prediction and counterfactuals

::: lcl._prediction.LCLPrediction

::: lcl._struct.WTPRequest

::: lcl._struct.PartitionType

::: lcl._struct.PastChoicesData
