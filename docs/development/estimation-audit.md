# Latent-class estimation audit — September 2026

The retained changes speed up estimation, reduce measured peak memory, and preserve fitted coefficients, likelihoods, and covariance matrices. The default EM tolerance remains unchanged. This audit also corrects convergence reporting and several ineffective settings.

Baseline: `f93f41cdcb524f5fa4a3ed8aa08f4412e169b0c9`. Measurements use JAX 0.9.2, double precision, and one Apple CPU device. Two logical CPU devices exercise sharding; physical GPU performance and memory were not measured. Timings describe these workloads, not a guarantee across hardware or all model dimensions.

## Matrix calculations and memory

- The posterior already stored in `EMVars` now describes that state's coefficients and priors. Evaluate it together with the completed iteration's likelihood, then reuse it in the next M-step. This removes a duplicate likelihood pass without retaining an additional matrix. Startup and post-polish state reconstruction also share their likelihood calculation.
- Expand posterior weights from panels to cases inside each class solve, instead of constructing a cases-by-classes matrix. Sharding transfers the smaller panel weight matrix.
- For wider demographic designs whose dense outer product would exceed one MiB, build the membership Hessian from small batches of weighted Gram matrices. This avoids the panels-by-demographics-squared intermediate while retaining the original stable `p_k (delta_kl - p_l)` coefficients. Small designs retain the original dense contraction.
- Evaluate the final EM likelihood in bounded class blocks: one class for small models, two for four or more classes. This keeps utility/probability temporaries bounded while preserving matrix throughput. Prediction and inference keep their existing vectorized likelihood kernel.
- Generate initialization subsets lazily. The initializer no longer retains a full extra copy of the differenced design split into every class, and compiled executables no longer capture subset arrays as constants.
- Transfer shares alongside iteration diagnostics and retain host floats in history. Previously each history row dispatched and retained an individual JAX scalar for each class.

The taste Hessian remains `Xdiff.T @ diag(p * w) @ Xdiff - xbar.T @ diag(w) @ xbar`. Probabilities change during Newton and posterior weights change during EM, so a cached unweighted `X.T @ X` cannot replace these products. Precomputing all row outer products would raise memory. A trial reusing the probability-weighted design improved timing but raised compiled temporary storage; it was discarded. A naive posterior cache also raised peak storage; the bounded likelihood blocks and class-local weight expansion remove that increase.

| Data | Cold fit seconds, before → after | Warm fit seconds, before → after | Process peak MiB, before → after | EM iterations |
| --- | ---: | ---: | ---: | ---: |
| Synthetic | 4.844 → 3.892 | 0.479 → 0.032 | 973.2 → 776.7 | 20 |
| Apollo | 6.165 → 5.155 | 0.903 → 0.192 | 1134.4 → 901.1 | 45 |

Full fits include encoding, initialization, EM, polishing, covariance, and diagnostics. Warm values are medians of three subsequent fits in the same process. Process peaks are macOS `ru_maxrss` across all four fits, including compilation and allocator/cache retention. The warm improvements are about **15× on synthetic data and 4.7× on Apollo**; process peaks fall about **20%**.

| Kernel shape [panels, utility variables, demographics, classes] | Recursion ms, before → after | XLA temporary MiB, before → after | XLA arguments + outputs + temporaries MiB, before → after |
| --- | ---: | ---: | ---: |
| [1000, 2, 0, 2] | 1.514 → 1.354 | 1.22 → 0.95 | 1.64 → 1.38 |
| [10000, 8, 3, 3] | 16.331 → 16.222 | 29.11 → 26.29 | 55.44 → 52.85 |
| [5000, 6, 8, 5] | 11.861 → 12.279 | 13.87 → 11.52 | 24.64 → 22.47 |
| [3000, 12, 5, 8] | 15.847 → 15.256 | 13.62 → 12.22 | 24.88 → 23.67 |

Kernel timings block on completion and repeat the same state to keep Newton workload fixed. Memory figures come from compiled executable buffer analysis; they are distinct from process peak RSS. All tested shapes reduce both reported temporary storage and total executable buffer storage. The wider-demographics five-class microbenchmark has a small runtime tradeoff (about 3.5%, or 0.42 ms per recursion) for a 17% reduction in temporary memory; this is not a blanket claim that every individual kernel is faster. Its initializer drops from 2.22 to 0.65 seconds. The tested complete synthetic and Apollo fits remain substantially faster. Single-class scanning for the eight-class example was about 4% slower; blocks of two recovered that loss with the same temporary allocation. No larger cache of utilities, Hessians, or design cross-products was added.

## JIT and sharding

Initialization now has a shared `filter_jit` entry point with subset arrays as dynamic arguments. Shape, dtype, device placement, and static solver settings legitimately specialize a computation; array contents and random seeds do not. This follows [JAX's caching guidance](https://docs.jax.dev/en/latest/201/jit.html) and avoids its documented [closed-over-array constant costs](https://docs.jax.dev/en/latest/internals/constants.html).

Compilation logging for **two complete Apollo fits, each with three starts**, recorded:

| Executable | Compilations | Reason |
| --- | ---: | --- |
| EM recursion | 1 | Shared across every iteration and seed |
| Subset initializer | 9 | Nine distinct ragged subset shapes across three seeds; reused on the second fit |
| Observed-data polish | 1 | Same parameter layout and solver settings |
| Observed score | 1 | Shared before/after polish and final ordering |
| Total observed likelihood | 1 | Shared throughout fit checking |
| Covariance derivative kernel | 1 | Shared across repeated fits |

The existing EM placement normalization is necessary: it makes first-iteration inputs match the named sharding of subsequent outputs. The inner Newton loops and class maps execute inside the outer compiled recursion; their Python closures are traced there, not recreated as independent compiled solvers every iteration. Class-specific solves remain sequential within each device to bound Hessian working storage, with class parallelism across devices. Odd class counts and dummy zero-weight classes are checked against an independently recomputed E-step on two logical devices. Padding all initialization subsets merely to force one shape was avoided because that trades extra memory for compilation savings.

The demographic memory threshold was imported by value, so changing the documented `_scheduling.ITERATION_THRESHOLD_BYTES` had no effect. It now reads the module setting at trace time. Existing schedule-equivalence tests also reused a cached executable for both branches; they now force fresh traces and actually compare both schedules. As documented, changing thresholds after a fit does not alter an existing executable.

## EM mathematics and stopping

For each panel and class, the E-step multiplies the membership prior by the likelihood of the panel's entire choice sequence and normalizes across classes. These posterior weights are fixed while updating all class tastes and the membership model. Taste updates maximize weighted conditional logits; membership updates maximize fractional multinomial-logit likelihood, or average the panel posteriors when there are no demographics. This matches the latent-class construction in [Train (2008), sections 2–4](https://eml.berkeley.edu/~train/papers/EMtrain.pdf). Safeguarded M-steps that improve their objectives give a generalized EM ascent step even when an inner iteration budget is reached.

For increments `d_t` and contraction rate `r = d_t / d_(t-1)`, the estimated ascent remaining **after the current iterate** is `d_t * r / (1-r)`. That geometric-tail calculation is correct and consistent with the Aitken treatment in [Böhning et al. (1994), appendix](https://www.ism.ac.jp/editsec/aism/pdf/046_2_0373.pdf). It is a local extrapolation, not proof of stationarity. Using an observed score as an additional diagnostic is consistent with the caution about slow EM in Train and [Mplus's mixture-model convergence guidance](https://www.statmodel.com/HTML_UG/chapter14V8.htm).

Corrections:

- Material likelihood decreases and non-finite likelihoods no longer satisfy Aitken stopping. The fit rejects such a recursion; a multi-start fit can proceed to another start. Negative differences within 64 floating-point ULPs remain eligible as numerical plateaus.
- `score_tol` is now applied to the **mean panel score**, as documented. Both the fit kernel and covariance reporting had used the total score. Final class ordering is accounted for because changing the reference membership class changes score coordinates; inference reporting keeps the convergence flag consistent with the reported score.
- `score_tol` controls the final convergence flag. The previous claim that it also stopped the EM loop early was incorrect and has been removed. No expensive score pass was added to every EM iteration.
- `hessian_damping`, `initial_trust_radius`, and `accept_any_decrease` now reach the final polish as well as the M-steps. Polish intentionally retains its separate `polish_maxiter` budget and `1e-10` Newton-decrement tolerance; those distinctions are documented.
- `check_interval` spaces Aitken stopping checks. History and callbacks still run each iteration. `seed`, `starts`, `num_devices`, iteration limits, `polish`, and the supported start method are active; unsupported start methods raise an error. Zero iteration budgets explicitly disable the corresponding solver stage.

Tightening `em_tol` from **1e-8 to 1e-10** increased synthetic EM iterations **20 → 25** and Apollo iterations **45 → 55**, with **no improvement in the final polished log likelihood**. In the synthetic run it also used all 25 polish iterations at a numerically indistinguishable likelihood. The default was therefore retained. Score magnitudes depend on predictor units; convergence here means approximate stationarity, not a guarantee of a global optimum. Apollo's three-start run reached the same final likelihood as the single-start benchmark.

## Verification and reproduction

The [official Apollo mode-choice data](https://www.apollochoicemodelling.com/examples.html) contain 500 panels and 8,000 choice situations; retaining available alternatives produces 26,448 long rows. The benchmark estimates three classes, five utility coefficients per class, income and gender membership effects, and a negative cost coefficient. Synthetic data use known two-class preferences and a demographic membership model.

- Final log likelihoods: synthetic **-1019.8267441581306**, Apollo **-6413.798988549108**.
- Before/after maximum absolute differences: coefficients and membership parameters below `1e-15`; covariance entries below `3.4e-15`.
- Independent NumPy/SciPy evaluation of the **undifferenced** design reproduces the final likelihoods; posterior differences are below `2.5e-15`, including the real-data multi-start fit.
- Tests compare the reused recursion with an E-step recomputed at every iteration, including ragged panels, demographic/no-demographic models, negative-coefficient transforms, uneven sharding, and incomplete final class blocks. Existing parameter-recovery, analytic/autodiff Hessian, covariance, prediction, and inference tests remain in the full suite.

Final validation: **214 passed, 8 skipped** on one physical CPU device; the eight skips require two devices. All **32 targeted tests passed with two logical CPU devices**, including the wide-demographic Hessian checks. Ruff and mypy pass. Raw measurements are in [estimation-audit-results.json](estimation-audit-results.json); the repeatable entry point is [tools/audit_estimation.py](https://github.com/zeyveld/latent-class-conditional-logit/blob/main/tools/audit_estimation.py).

```sh
# Use the same environment and run the two checkouts in separate processes.
PYTHONPATH=/path/to/baseline/src .venv/bin/python tools/audit_estimation.py synthetic --repeats 4
PYTHONPATH=src .venv/bin/python tools/audit_estimation.py synthetic --repeats 4
PYTHONPATH=src .venv/bin/python tools/audit_estimation.py apollo --apollo-path /path/to/apollo_modeChoiceData.csv --repeats 4
PYTHONPATH=src .venv/bin/python tools/audit_estimation.py kernel --shape 10000 8 3 3
JAX_LOG_COMPILES=1 PYTHONPATH=src .venv/bin/python tools/audit_estimation.py apollo --starts 3 --repeats 2
.venv/bin/python -m pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv/bin/python -m pytest tests/test_em_efficiency.py -q
.venv/bin/ruff check src tests tools/audit_estimation.py
.venv/bin/mypy src
```
