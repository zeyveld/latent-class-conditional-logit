# Cross-validation

Blocked K-fold cross-validation for choosing a class count. Each decision-maker's
complete choice history remains within one fold. Use the same immutable `LCLSpec`
as estimation and pass it with `spec=`.

!!! warning "Experimental"
    The API remains experimental and may change between minor releases. See the
    [model-selection tutorial](../tutorials/cross_validation.md) for a complete example.

```python
import lcl
from lcl import FitOptions

cv_results = lcl.cv_optimal_classes(
    data,
    spec=spec,
    num_classes_list=[2, 3, 4],
    fit_options=FitOptions(seed=42, starts=3),
)
```

`folds=` may be an integer, an explicit sequence of test-panel groups, or a
mapping from panel ID to a user fold label. Explicit folds must cover every panel
exactly once. This makes externally defined geographic, temporal, or grouped
splits reproducible without row leakage.

Inference is skipped by default because covariance estimation does not affect
held-out likelihood. Every validation fold is transformed with its training
fold's fitted encoder.

`Avg_OOS_LL` is the pooled mean held-out log likelihood per panel. If any fold
fails, `Avg_OOS_LL` and `Total_OOS_LL` are `NaN`; use
`Avg_Successful_OOS_LL` and `Fold_Errors` for diagnosis, not model ranking.
Convergence is reported independently in `Converged_Folds`,
`Nonconverged_Folds`, and `Fold_Converged`.
`Fold_SE_Panel_LL` contains within-fold standard errors and `SE_OOS_LL` is the
pooled panel-level standard error. `Selected_Best` marks the largest mean score;
`Selected_One_SE` marks the smallest class count within one best-model standard
error of it.

::: lcl.cv_optimal_classes
