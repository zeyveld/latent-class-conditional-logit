"""Measure bounded price optimization, memory, and actual solver compilations.

Run in separate processes, without other compute-heavy jobs:
    .venv/bin/python benchmarks/price_constraint.py --source src

Use --maxiter 1 to isolate per-iteration overhead, or --price-effect 0.8 for a
binding optimum. Memory reports distinguish XLA buffers from process peak RSS
(which also includes imports, compilation, and host data). No timing assertions.
"""

import argparse
import json
import logging
from pathlib import Path
import platform
import resource
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="src")
parser.add_argument("--cases", type=int, default=10000)
parser.add_argument("--variables", type=int, default=8)
parser.add_argument("--classes", type=int, default=4)
parser.add_argument("--repeats", type=int, default=30)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--price-effect", type=float, default=-1.2)
parser.add_argument("--start", type=float, default=0.0)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.source).resolve()))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lcl._case_utils import _loglik_value  # noqa: E402
from lcl.constraints import NegativeCoefficientBound  # noqa: E402
from lcl._em_alg_steps import _distributed_update  # noqa: E402
from lcl._struct import DiffUnchosenChosen  # noqa: E402
from lcl.options import OptimizationOptions  # noqa: E402

rng = np.random.default_rng(728)
x = rng.normal(size=(args.cases, 4, args.variables))
truth = rng.normal(scale=0.3, size=args.variables)
truth[0] = args.price_effect
chosen = np.argmax(x @ truth + rng.gumbel(size=(args.cases, 4)), axis=1)
contrasts = x - x[np.arange(args.cases), chosen, None, :]
contrasts = contrasts[np.arange(4)[None, :] != chosen[:, None]]
diff = DiffUnchosenChosen(
    X=jnp.asarray(contrasts),
    alts=jnp.zeros(args.cases * 3, dtype=jnp.uint32),
    cases=jnp.repeat(jnp.arange(args.cases, dtype=jnp.uint32), 3),
    panels=None,
    num_cases=args.cases,
)
weights = jnp.asarray(rng.uniform(0.2, 1.0, size=(args.classes, args.cases)))
starts = jnp.zeros((args.classes, args.variables)).at[:, 0].set(args.start)
options = OptimizationOptions(maxiter=args.maxiter, newton_decrement_tol=1e-6)


@eqx.filter_jit
def benchmark_update(beta, w, d):
    """Run all class-specific M-steps in one compiled call."""
    return _distributed_update(beta, w, d, NegativeCoefficientBound(0), options)


class CompilationCounter(logging.Handler):
    """Count actual XLA compile announcements, not Python trace side effects."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        """Record solver compilation announcements."""
        message = record.getMessage()
        if (
            "Compiling jit(benchmark_update)" in message
            or "Compiling benchmark_update " in message
        ):
            self.count += 1


counter = CompilationCounter()
logger = logging.getLogger("jax._src.interpreters.pxla")
logger.addHandler(counter)
logger.propagate = False
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
timings = []
with jax.log_compiles(True):
    before = time.perf_counter()
    result = benchmark_update(starts, weights, diff)
    jax.block_until_ready(result)
    cold_seconds = time.perf_counter() - before
    for _ in range(args.repeats):
        before = time.perf_counter()
        result = benchmark_update(starts, weights, diff)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - before)
    # Changing starts, weights, and active sets must reuse the executable.
    for start in (-1e-5, -1.0, 1.0):
        changed = benchmark_update(starts.at[:, 0].set(start), weights * 0.9, diff)
        jax.block_until_ready(changed)

compiled = benchmark_update.lower(starts, weights, diff).compile().compiled
memory = compiled.memory_analysis()
memory_fields = {
    name: getattr(memory, name)
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
}
memory_fields["total_buffer_bytes"] = (
    memory.argument_size_in_bytes
    + memory.output_size_in_bytes
    + memory.temp_size_in_bytes
    - memory.alias_size_in_bytes
)
betas, error = result
losses = [float(_loglik_value(b, diff, w) / w.sum()) for b, w in zip(betas, weights)]
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
print(
    json.dumps(
        {
            "source": str(Path(args.source).resolve()),
            "jax": jax.__version__,
            "platform": platform.platform(),
            "config": vars(args),
            "cold_seconds": cold_seconds,
            "warm_median_ms": np.median(timings) * 1000,
            "warm_p10_ms": np.quantile(timings, 0.1) * 1000,
            "warm_p90_ms": np.quantile(timings, 0.9) * 1000,
            "solver_compilations": counter.count,
            "memory": memory_fields,
            "process_peak_rss_bytes": rss_bytes,
            "prices": np.asarray(betas[:, 0]).tolist(),
            "mean_losses": losses,
            "errors": np.asarray(error).tolist(),
        },
        indent=2,
    )
)
