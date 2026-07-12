"""Variable-label helpers for presentation tables."""

from __future__ import annotations

import re
from collections.abc import Mapping


_FORMULAIC_CATEGORICAL_RE = re.compile(
    r"^C\((?P<variable>[^)]+)\)\[T\.(?P<level>[^\]]+)\]$"
)
_BARE_CATEGORICAL_RE = re.compile(
    r"^(?P<variable>[A-Za-z_]\w*)\[T\.(?P<level>[^\]]+)\]$"
)


def normalize_variable_labels(labels: Mapping[str, str] | None) -> dict[str, str]:
    """Return a plain string dictionary of variable labels.

    Parameters
    ----------
    labels : Mapping[str, str] | None
        Optional mapping from raw model or DataFrame column names to
        human-readable presentation labels.

    Returns
    -------
    dict[str, str]
        Normalized label mapping.  A copy is always returned so callers can store
        metadata without retaining mutable user-owned mappings.
    """
    if labels is None:
        return {}
    return {str(variable): str(label) for variable, label in labels.items()}


def merge_variable_labels(
    *label_maps: Mapping[str, str] | None,
) -> dict[str, str]:
    """Merge label dictionaries from lowest to highest precedence.

    Parameters
    ----------
    *label_maps : Mapping[str, str] | None
        Optional mappings.  Later mappings override earlier ones.

    Returns
    -------
    dict[str, str]
        Merged label mapping.
    """
    merged: dict[str, str] = {}
    for labels in label_maps:
        merged.update(normalize_variable_labels(labels))
    return merged


def label_for_variable(variable: str, labels: Mapping[str, str]) -> str:
    """Return the display label for a raw or Formulaic-expanded variable.

    Parameters
    ----------
    variable : str
        Raw model variable name, Formulaic-expanded categorical coefficient name,
        or interaction name.
    labels : Mapping[str, str]
        Mapping from raw DataFrame/model names to display labels.  Exact matches
        take precedence over derived Formulaic labels.

    Returns
    -------
    str
        Human-readable label when available; otherwise ``variable`` unchanged.
    """
    if variable in labels:
        return labels[variable]
    if ":" in variable:
        parts = variable.split(":")
        labelled_parts = [label_for_variable(part, labels) for part in parts]
        if labelled_parts != parts:
            return " x ".join(labelled_parts)
        return variable
    return _label_formulaic_categorical(variable, labels)


def _label_formulaic_categorical(variable: str, labels: Mapping[str, str]) -> str:
    """Return a label for Formulaic dummy columns derived from a raw column."""
    match = _FORMULAIC_CATEGORICAL_RE.match(variable)
    if match is None:
        match = _BARE_CATEGORICAL_RE.match(variable)
    if match is None:
        return variable

    raw_variable = match.group("variable")
    if raw_variable not in labels:
        return variable
    return f"{labels[raw_variable]}: {match.group('level')}"
