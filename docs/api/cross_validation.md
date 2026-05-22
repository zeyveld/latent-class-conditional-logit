# Cross-validation

Blocked K-fold cross-validation for choosing the number of latent classes. Folds are drawn at the decision-maker level so that an individual's entire choice history sits in exactly one fold.

!!! warning "Experimental"
    The cross-validation utility is functional but still under active refinement. See the [model-selection tutorial](../tutorials/cross_validation.md) for a worked example. And please feel free to suggest improvements or extensions; I've never done any scholarship on hyperparameter tuning, so I'd be excited to hear about recent developments in this area!

::: lcl._cross_validation.cv_optimal_classes
