# Cross-validation

Blocked K-fold cross-validation for choosing the number of latent classes. Folds are drawn at the decision-maker level so that an individual's entire choice history sits in exactly one fold.

!!! warning "Experimental"
    The cross-validation utility is functional but still under active refinement. See the [model-selection tutorial](../tutorials/cross_validation.md) for a worked example.

::: lcl._cross_validation.cv_optimal_classes
