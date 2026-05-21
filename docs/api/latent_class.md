# Latent-class conditional logit

The headline estimator. A finite mixture of conditional logits, fit by expectation-maximization, with optional class-membership demographic regression. Class-specific taste vectors are recovered via maximum likelihood at each M-step; class probabilities are updated either as aggregate shares or, when demographics are present, through a fractional-response multinomial logit.

## Model

::: lcl.latent_class_conditional_logit.LatentClassConditionalLogit

## Results

::: lcl._results.LCLResults

## Prediction and counterfactuals

::: lcl._prediction.LCLPrediction
