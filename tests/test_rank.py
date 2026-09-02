"""Identification and observed-information gate tests."""

import numpy as np
import polars as pl
import pytest

from lcl import ChoiceIds, LCLSpec
from lcl.conditional_logit import ConditionalLogit
from lcl._inference import _invert_information
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def _asc_frame() -> pl.DataFrame:
    rows = []
    for panel in range(6):
        for case_within_panel in range(3):
            case = 10 * panel + case_within_panel
            for alt in ["bus", "rail", "air"]:
                rows.append(
                    {
                        "panel": panel,
                        "case": case,
                        "alt": alt,
                        "choice": alt == ["bus", "rail", "air"][case % 3],
                    }
                )
    return pl.DataFrame(rows)


def test_cl_rejects_unidentified_full_set_of_ascs() -> None:
    with pytest.raises(ValueError, match="chosen-differenced utility design"):
        ConditionalLogit().fit(
            _asc_frame(),
            alts_col="alt",
            cases_col="case",
            utility_formula="choice ~ C(alt) - 1",
        )


def test_lcl_rejects_unidentified_full_set_of_ascs() -> None:
    spec = LCLSpec(
        ids=ChoiceIds(alt="alt", case="case", panel="panel", choice="choice"),
        utility_formula="choice ~ C(alt) - 1",
        classes=2,
    )
    with pytest.raises(ValueError, match="K-1 alternative constants"):
        LatentClassConditionalLogit(spec=spec).fit(_asc_frame())


def test_reference_coded_ascs_are_identified() -> None:
    model = ConditionalLogit()
    parsed = model._ingest_data(
        _asc_frame(),
        alts_col="alt",
        cases_col="case",
        panels_col="case",
        utility_formula="choice ~ C(alt)",
        membership_formula=None,
        choice_col=None,
        case_varnames=None,
        dem_varnames=None,
        dems_data=None,
    )
    assert parsed.X.shape[1] == 2


def test_information_uses_one_spectrum_and_gates_non_pd(monkeypatch) -> None:
    calls = 0
    original = np.linalg.eigvalsh

    def counted(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", counted)
    inverse, diagnostics = _invert_information(np.diag([2.0, -0.5]))
    assert calls == 1
    assert not diagnostics.positive_definite
    assert np.isnan(np.asarray(inverse)).all()
