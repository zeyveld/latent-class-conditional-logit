"""Float64 contexts for LCL's public numerical operations."""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import jax

P = ParamSpec("P")
R = TypeVar("R")

# The public spelling appeared after the supported JAX 0.5.3 floor.
if hasattr(jax, "enable_x64"):
    from jax import enable_x64
else:
    from jax.experimental import enable_x64 as experimental_enable_x64

    enable_x64 = experimental_enable_x64


def use_float64(function: Callable[P, R]) -> Callable[P, R]:
    """Run a numerical entry point in float64 and restore the caller's setting."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Keep precision local to this operation, including exceptional exits."""
        with enable_x64(True):
            return function(*args, **kwargs)

    return wrapped
