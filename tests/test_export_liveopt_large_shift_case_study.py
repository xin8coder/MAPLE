from __future__ import annotations

import pytest

from scripts.analysis.export_liveopt_large_shift_case_study import apply_formal_metric_gate


def test_hidden_infeasible_trace_is_zeroed_for_paper_plot() -> None:
    curve = [(0, 0.12), (20, 0.47), (80, 0.73)]

    gated = apply_formal_metric_gate(
        curve,
        {"true_pass": "False", "normalized_hv": "0.0"},
        context="P010 seed 0 t11",
    )

    assert gated == [(0, 0.0), (20, 0.0), (80, 0.0)]


def test_formally_feasible_trace_must_match_table_metric() -> None:
    curve = [(0, 0.0), (200, 0.42)]

    assert apply_formal_metric_gate(
        curve,
        {"true_pass": "True", "normalized_hv": "0.42"},
        context="P010 seed 0 t11",
    ) == curve

    with pytest.raises(ValueError, match="trace/table mismatch"):
        apply_formal_metric_gate(
            curve,
            {"true_pass": "True", "normalized_hv": "0.41"},
            context="P010 seed 0 t11",
        )
