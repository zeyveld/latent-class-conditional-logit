"""Size-based selection between batched and sequential contraction schedules.

Several derivative blocks in this package can be assembled either as a single
batched contraction or as a sequential loop over the batched axis.  The two are
mathematically identical -- they differ only in summation order -- but they
trade wall-clock against peak memory in opposite directions:

* The **batched** form issues one large fused operation.  It is faster, but XLA
  materializes the full outer-product intermediate, so peak temporary memory
  grows with the product of every axis.
* The **sequential** form (:func:`jax.lax.scan` over the batched axis) bounds
  the live set to a single slice.  It costs a modest amount of extra time, but
  peak memory no longer carries the looped axis.

Measurements on this package's kernels show the relative penalty of the
sequential schedule shrinking as the problem grows -- roughly +50% when the
intermediate is a few megabytes (where the absolute cost is well under a
millisecond anyway), falling into single digits in the hundreds of megabytes,
and turning *negative* in the gigabyte range where the batched form starts
thrashing.  The memory saving moves the other way, reaching 5-13x at the top
end.

The switchovers below therefore keep the batched schedule -- bit-for-bit the
historical code path -- for every problem where memory is not a concern, and
engage the sequential schedule only where the penalty has become small and the
saving large.

Two thresholds are provided because the package's two call sites have different
cost profiles.  Covariance assembly runs once per fit, so a modest penalty
there is invisible and memory safety dominates.  The class-membership M-step
runs on every EM iteration, so its penalty is multiplied by the iteration count
and it waits for a larger problem before switching.

The decision is made from static shapes at trace time, so it costs nothing at
runtime and produces exactly one compiled variant per problem shape.

Notes
-----
Either threshold may be reassigned before fitting, e.g. lowered on a
memory-constrained device or raised to force the batched schedule::

    import lcl._scheduling
    lcl._scheduling.INFERENCE_THRESHOLD_BYTES = 32 * 1024**2
"""

from __future__ import annotations

from math import prod

FLOAT64_BYTES = 8

INFERENCE_THRESHOLD_BYTES: int = 128 * 1024**2
"""Switchover for work performed once per fit, such as covariance assembly."""

ITERATION_THRESHOLD_BYTES: int = 1024**3
"""Switchover for work repeated every EM iteration, such as the M-steps.

Higher than :data:`INFERENCE_THRESHOLD_BYTES` because the penalty is paid on
every iteration; at this size the sequential schedule is at worst break-even.
"""


def use_sequential(
    *dims: int,
    threshold: int | None = None,
    itemsize: int = FLOAT64_BYTES,
) -> bool:
    """Report whether an intermediate of shape ``dims`` warrants a scan.

    Parameters
    ----------
    *dims : int
        Extents of the dense intermediate the batched schedule would
        materialize.  Their product times ``itemsize`` is the size compared
        against the threshold.
    threshold : int | None, optional
        Byte threshold to compare against.  Defaults to
        :data:`INFERENCE_THRESHOLD_BYTES`; pass
        :data:`ITERATION_THRESHOLD_BYTES` from code that runs every EM
        iteration.  Read at call time so either constant may be reassigned.
    itemsize : int, default=8
        Bytes per element; the package computes in float64 throughout.

    Returns
    -------
    bool
        ``True`` when the batched intermediate reaches the threshold and the
        sequential schedule should be used instead.

    Examples
    --------
    >>> use_sequential(72_000, 20, 6)  # 69 MB of float64
    False
    >>> use_sequential(288_000, 30, 8)  # 553 MB of float64
    True
    """
    limit = INFERENCE_THRESHOLD_BYTES if threshold is None else threshold
    return prod(dims) * itemsize >= limit
