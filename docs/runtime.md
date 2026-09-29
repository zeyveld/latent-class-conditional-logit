# Runtime, devices, and precision

## Compilation reuse

LCL reuses its compiled initializer and EM step across starts with the same data
shape, class count, solver settings, precision, and device placement. Startup
fits each random partition with full-data 0/1 panel weights, so unbalanced panel
lengths do not create a different executable for every class.

Different CV folds and class counts can still change array shapes and parameter
layouts. They can require new compilations. To reuse executables across script
runs, enable JAX's persistent cache before fitting:

```python
from pathlib import Path
import jax

jax.config.update(
    "jax_compilation_cache_dir",
    str(Path.home() / ".cache" / "lcl-jax"),
)
```

By default, JAX stores compilations taking at least one second. For repeated
short kernels, also set
`jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)`.
See the [JAX cache guide](https://docs.jax.dev/en/latest/501/compilation-cache.html)
for cache thresholds and platform compatibility.

## Devices and memory

`FitOptions.num_devices` defaults to the number of local JAX devices. The beta
M-step distributes classes across those devices; the EM state and data are
replicated on that mesh. The E-step and membership M-step remain replicated.

After EM, LCL moves data and parameters to CPU before polishing, checking the
score, and computing covariance. Results from fitting retain their data and
posteriors there, which avoids accumulating GPU replicas in a notebook sweep.
Delta-method and Gaussian-simulation inference also prefer CPU. If the CPU
backend is disabled, LCL logs a warning and uses a local default device instead.
For CUDA fitting with CPU inference, keep both backends enabled, for example
with `JAX_PLATFORMS=cuda,cpu` configured before importing JAX.

Observed derivatives accumulate blocks of complete panels. Gaussian simulation
evaluates at most 32 draws at a time. Neither needs to retain the full
panels-by-parameters or draws-by-output matrix. A large parameter count still
requires a parameter-squared Hessian and covariance, and each complete panel
must fit in the derivative workspace.

## Precision and repeatability

Importing LCL leaves the caller's JAX precision setting unchanged. Public fit,
prediction, and numerical reporting methods use a scoped float64 context and
restore the prior setting when they return. Returned JAX arrays retain float64,
but JAX operations you perform on those arrays outside LCL use your own setting.
Use `jax.enable_x64(True)` as a context on current JAX, or enable x64 explicitly
for a script that works with LCL's numerical arrays or private kernels.

The seed fixes random starting partitions and Gaussian coefficient draws.
Floating-point reductions on GPUs are not guaranteed bitwise reproducible, so
nearly tied starts, iteration counts, and convergence at the tolerance boundary
can vary. LCL retains a 64-ULP allowance for round-off in EM likelihood ascent.

See the [runtime audit](development/runtime-audit.md) for before/after compilation
counts, numerical comparisons, and the limits of the hardware verification.
