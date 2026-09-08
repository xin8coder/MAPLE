"""Unit tests for the churn-discriminator branch suite.

Covers the deterministic structural table-churn rule wrapper and the
programmatic branch table transforms in
``scripts/analysis/run_liveopt_churn_discriminator.py``.  All fixtures are
synthetic; no benchmark artifacts, optimizer runs, or provider calls.
"""

from __future__ import annotations

import copy

from scripts.analysis.run_liveopt_churn_discriminator import (
    EASY_INJECT_COUNT,
    UNIT_SCALE_FIELDS,
    churn_decision,
    easy_inject_context,
    relabel_context,
    relabel_id,
    unit_scale_context,
)


def p010_context() -> dict:
    return {
        "tables": {
            "depot": [{"id": "DEPOT", "x": 0, "y": 0}],
            "vehicles": [
                {
                    "id": "V1",
                    "capacity": 25,
                    "emission_rate": 0.8,
                    "available": True,
                    "shift_end": 270,
                }
            ],
            "orders": [
                {
                    "id": "O001",
                    "x": 10,
                    "y": 10,
                    "demand": 2,
                    "ready": 5,
                    "due": 50,
                    "service": 5,
                    "priority": 1,
                    "active": True,
                },
                {
                    "id": "O002",
                    "x": 20,
                    "y": 20,
                    "demand": 3,
                    "ready": 10,
                    "due": 60,
                    "service": 5,
                    "priority": 2,
                    "active": True,
                },
                {
                    "id": "O003",
                    "x": 30,
                    "y": 30,
                    "demand": 1,
                    "ready": 0,
                    "due": 70,
                    "service": 5,
                    "priority": 1,
                    "active": False,
                },
            ],
            "policy": [{"carbon_multiplier": 1.0}],
        }
    }


def p015_context() -> dict:
    return {
        "tables": {
            "machines": [
                {
                    "id": "M01",
                    "cpu": 40,
                    "mem": 80,
                    "energy_idle": 8.0,
                    "energy_per_cpu": 0.2,
                    "available": True,
                    "gpu": False,
                }
            ],
            "jobs": [
                {
                    "id": "J001",
                    "burst": "",
                    "cpu": 5,
                    "mem": 10,
                    "deadline": 90,
                    "latency_sensitivity": 1.2,
                    "priority": 3,
                    "gpu_required": False,
                    "active": True,
                },
                {
                    "id": "J002",
                    "burst": "",
                    "cpu": 6,
                    "mem": 8,
                    "deadline": 95,
                    "latency_sensitivity": 1.0,
                    "priority": 1,
                    "gpu_required": False,
                    "active": True,
                },
            ],
            "global_params": [{"energy_price": 1.0, "carbon_intensity": 1.0}],
        }
    }


def test_unchanged_context_selects_warm():
    context = p010_context()
    decision = churn_decision(context, copy.deepcopy(context))
    assert decision["churn_ratio"] == 0.0
    assert decision["selected_action"] == "Warm"
    assert decision["selected_restart_skill"] == "warm_restart_v1"


def test_new_active_row_counts_fully_changed():
    previous = p010_context()
    current = copy.deepcopy(previous)
    new_row = copy.deepcopy(current["tables"]["orders"][0])
    new_row["id"] = "O900"
    current["tables"]["orders"].append(new_row)
    decision = churn_decision(previous, current)
    # 2 old active rows unchanged, 1 newly active row counts 100%: 7/(3*7).
    assert decision["decision_table"] == "orders"
    assert decision["changed_cells"] == 7
    assert decision["compared_cells"] == 21
    assert abs(decision["churn_ratio"] - 1.0 / 3.0) < 1e-9
    assert decision["selected_action"] == "Warm"


def test_threshold_boundary_selects_full_at_exact_half():
    previous = p010_context()
    current = copy.deepcopy(previous)
    new_row = copy.deepcopy(current["tables"]["orders"][0])
    new_row["id"] = "O900"
    current["tables"]["orders"].append(new_row)
    # 1 new active row among 2 active rows -> exactly 0.5.
    current["tables"]["orders"] = current["tables"]["orders"][:1] + [new_row]
    decision = churn_decision(previous, current)
    assert abs(decision["churn_ratio"] - 0.5) < 1e-9
    assert decision["selected_action"] == "Full"


def test_tables_without_active_field_are_ignored():
    previous = p010_context()
    current = copy.deepcopy(previous)
    current["tables"]["vehicles"][0]["capacity"] = 250
    current["tables"]["policy"][0]["carbon_multiplier"] = 9.0
    decision = churn_decision(previous, current)
    assert decision["churn_ratio"] == 0.0
    assert decision["selected_action"] == "Warm"


def test_unit_scale_p010_fires_full_and_preserves_ids():
    previous = p010_context()
    current = unit_scale_context("NLDO-P010", previous)
    orders = current["tables"]["orders"]
    assert [row["id"] for row in orders] == ["O001", "O002", "O003"]
    assert orders[0]["x"] == 100 and orders[0]["due"] == 500
    assert orders[0]["priority"] == 1  # ordinal, not scaled
    assert current["tables"]["vehicles"][0]["capacity"] == 250
    assert current["tables"]["vehicles"][0]["emission_rate"] == 0.8
    assert current["tables"]["policy"][0]["carbon_multiplier"] == 1.0
    decision = churn_decision(previous, current)
    # 6 of 7 non-key order fields scaled (priority kept).
    assert abs(decision["churn_ratio"] - 6.0 / 7.0) < 1e-9
    assert decision["churn_ratio"] >= 0.5
    assert decision["selected_action"] == "Full"


def test_unit_scale_p015_fires_full():
    previous = p015_context()
    current = unit_scale_context("NLDO-P015", previous)
    jobs = current["tables"]["jobs"]
    assert jobs[0]["cpu"] == 50 and jobs[0]["mem"] == 100
    assert jobs[0]["gpu_required"] is False and jobs[0]["active"] is True
    machines = current["tables"]["machines"]
    assert machines[0]["cpu"] == 400 and machines[0]["energy_per_cpu"] == 0.2
    assert current["tables"]["global_params"][0]["energy_price"] == 1.0
    decision = churn_decision(previous, current)
    # cpu, mem, deadline, latency_sensitivity, priority = 5 of 7 job fields.
    assert abs(decision["churn_ratio"] - 5.0 / 7.0) < 1e-9
    assert decision["selected_action"] == "Full"


def test_unit_scale_field_whitelist_matches_implementation():
    assert set(UNIT_SCALE_FIELDS["NLDO-P010"]["orders"]) == {
        "x",
        "y",
        "demand",
        "ready",
        "due",
        "service",
    }
    assert set(UNIT_SCALE_FIELDS["NLDO-P015"]["jobs"]) == {
        "cpu",
        "mem",
        "deadline",
        "latency_sensitivity",
        "priority",
    }


def test_relabel_fires_full_with_injective_mapping():
    previous = p010_context()
    current = relabel_context("NLDO-P010", previous)
    mapping = current["relabel_id_mapping"]
    assert mapping["O001"] == "C001" and mapping["V1"] == "W1"
    assert mapping["DEPOT"] == "HUB"
    assert len(set(mapping.values())) == len(mapping)
    orders = current["tables"]["orders"]
    assert [row["id"] for row in orders] == ["C001", "C002", "C003"]
    # Non-id fields untouched.
    assert orders[0]["x"] == 10 and orders[0]["due"] == 50
    assert orders[0]["active"] is True and orders[2]["active"] is False
    decision = churn_decision(previous, current)
    assert decision["churn_ratio"] == 1.0
    assert decision["selected_action"] == "Full"


def test_relabel_id_prefix_rules():
    assert relabel_id("O017") == "C017"
    assert relabel_id("J024") == "T024"
    assert relabel_id("M07") == "N07"
    assert relabel_id("V12") == "W12"
    assert relabel_id("DEPOT") == "HUB"
    assert relabel_id("burst-2") == "burst-2"


def test_easy_inject_p010_appends_trivial_rows_and_fires_full():
    previous = p010_context()
    current = easy_inject_context("NLDO-P010", previous)
    orders = current["tables"]["orders"]
    added = orders[len(previous["tables"]["orders"]):]
    assert len(added) == EASY_INJECT_COUNT["NLDO-P010"]
    assert all(row["active"] is True for row in added)
    assert all(row["demand"] == 1 and row["priority"] == 1 for row in added)
    assert all(abs(row["x"]) <= 5 and abs(row["y"]) <= 5 for row in added)
    existing_ids = {row["id"] for row in previous["tables"]["orders"]}
    assert not ({row["id"] for row in added} & existing_ids)
    # Previously active rows are untouched.
    assert orders[0] == previous["tables"]["orders"][0]
    decision = churn_decision(previous, current)
    # 2 old active unchanged; 20 new active count 100%.
    assert abs(decision["churn_ratio"] - 20.0 / 22.0) < 1e-9
    assert decision["selected_action"] == "Full"


def test_easy_inject_p015_appends_small_jobs_and_fires_full():
    previous = p015_context()
    current = easy_inject_context("NLDO-P015", previous)
    jobs = current["tables"]["jobs"]
    added = jobs[len(previous["tables"]["jobs"]):]
    assert len(added) == EASY_INJECT_COUNT["NLDO-P015"]
    assert all(row["active"] is True and row["gpu_required"] is False for row in added)
    assert all(row["cpu"] == 1 and row["mem"] == 1 for row in added)
    decision = churn_decision(previous, current)
    assert abs(decision["churn_ratio"] - 26.0 / 28.0) < 1e-9
    assert decision["selected_action"] == "Full"
