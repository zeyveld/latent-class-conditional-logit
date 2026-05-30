"""Compatibility helpers for JAX APIs used by LCL.

JAX has kept the same sharding concepts across recent releases, but the public
spelling of ``shard_map`` moved between versions.  This module gives the rest of
the package one stable import site for those sharding primitives.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, cast

import jax
from jax import Device
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# A JAX spec is a PyTree whose leaves are PartitionSpec values.  JAX does not
# expose a precise public type for that shape, so callers keep their exact data
# structure while this boundary accepts Any.
ShardSpecTree: TypeAlias = Any


def _load_shard_map() -> Callable[..., Any]:
    """Return the best available ``shard_map`` implementation for this JAX."""
    public_shard_map = getattr(jax, "shard_map", None)
    if public_shard_map is not None:
        return cast(Callable[..., Any], public_shard_map)

    # JAX 0.5.x exposes shard_map only through the experimental namespace.
    from jax.experimental.shard_map import shard_map as experimental_shard_map

    return cast(Callable[..., Any], experimental_shard_map)


def _resolve_shard_map_check_kwarg(raw_shard_map: Callable[..., Any]) -> str:
    """Return the replication-check keyword accepted by ``raw_shard_map``.

    JAX 0.5.x names this flag ``check_rep``; modern JAX names it ``check_vma``.
    The behavior we need is the same: disable the extra output-replication check
    for the class-wise beta update.
    """
    parameters = inspect.signature(raw_shard_map).parameters
    if "check_vma" in parameters:
        return "check_vma"
    if "check_rep" in parameters:
        return "check_rep"
    raise RuntimeError(
        "Unsupported JAX shard_map signature: expected a check_vma or check_rep "
        "keyword argument."
    )


_RAW_SHARD_MAP = _load_shard_map()
_SHARD_MAP_CHECK_KWARG = _resolve_shard_map_check_kwarg(_RAW_SHARD_MAP)


def shard_map(
    fun: F,
    *,
    mesh: Mesh,
    in_specs: ShardSpecTree,
    out_specs: ShardSpecTree,
    check_vma: bool = True,
) -> F:
    """Wrap JAX ``shard_map`` with a version-stable replication-check keyword."""
    kwargs: dict[str, Any] = {
        "mesh": mesh,
        "in_specs": in_specs,
        "out_specs": out_specs,
        _SHARD_MAP_CHECK_KWARG: check_vma,
    }
    return cast(F, _RAW_SHARD_MAP(fun, **kwargs))


def cpu_device() -> Device:
    """Return the first available CPU device for host-side inference work."""
    try:
        return jax.devices("cpu")[0]
    except IndexError as exc:
        raise RuntimeError("JAX did not report an addressable CPU device.") from exc


def device_put_array_leaves(tree: T, device: Device) -> T:
    """Move JAX array leaves in a PyTree to ``device`` while preserving metadata.

    ``jax.device_put`` accepts whole Python containers, but it also converts
    ordinary scalar metadata in :class:`~lcl._struct.Data` into device arrays.  The
    likelihood kernels expect those counts to remain Python integers, so inference
    uses this leaf-wise helper instead.
    """

    def put_leaf(leaf: object) -> object:
        if isinstance(leaf, jax.Array):
            return jax.device_put(leaf, device)
        return leaf

    return cast(T, jax.tree_util.tree_map(put_leaf, tree))


__all__ = [
    "Mesh",
    "NamedSharding",
    "P",
    "cpu_device",
    "device_put_array_leaves",
    "shard_map",
]
