# Latent-class conditional logit

The headline estimator: a finite mixture of conditional logits, fit by expectation-maximization, with optional class-membership demographic regression. Class-specific taste vectors are recovered via maximum likelihood at each M-step; class probabilities are updated either as aggregate shares or, when demographics are present, using a fractional-response multinomial logit model.

Most users reach this estimator through the declarative [`LCLSpec` + `lcl.fit`](specification.md) workflow; the class below is what `lcl.fit` constructs and fits under the hood, and it can also be driven directly.

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
