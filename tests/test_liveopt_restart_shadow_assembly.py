from scripts.analysis.assemble_liveopt_restart_shadow_comparison import (
    audit_shadow_runtime_lineage,
    expected_slot_action,
    route_slot,
    selected_stages_for_slot,
)


def test_shadow_routes_use_liveopt_backbone_at_every_stage() -> None:
    assert route_slot("Fixed Warm", "DS", 1) == "liveopt_ds"
    assert route_slot("Fixed Warm", "DM", 10) == "liveopt_dm"
    assert route_slot("Fixed Warm", "DM", 11) == "warm_t11_dm"
    assert route_slot("Fixed Warm", "DM", 12) == "warm_t12_dm"

    assert route_slot("Fixed Full", "DS", 1) == "full_t01_t10_ds"
    assert route_slot("Fixed Full", "DM", 10) == "full_t01_t10_dm"
    assert route_slot("Fixed Full", "DM", 11) == "liveopt_dm"
    assert route_slot("Fixed Full", "DM", 12) == "full_t12_dm"


def test_liveopt_action_follows_frozen_semantic_decision() -> None:
    decisions = {
        ("NLDO-P007", 1): False,
        ("NLDO-P007", 11): True,
    }
    assert expected_slot_action("liveopt_ds", "NLDO-P007", 1, decisions) == "Warm"
    assert expected_slot_action("liveopt_ds", "NLDO-P007", 11, decisions) == "Full"
    assert expected_slot_action("warm_t11_ds", "NLDO-P007", 11, decisions) == "Warm"
    assert expected_slot_action("full_t12_ds", "NLDO-P007", 12, decisions) == "Full"


def test_early_full_shadow_covers_all_first_ten_transitions() -> None:
    assert selected_stages_for_slot("full_t01_t10_ds") == tuple(range(1, 11))
    assert selected_stages_for_slot("full_t01_t10_dm") == tuple(range(1, 11))


def test_shadow_lineage_requires_exact_seed_matched_population() -> None:
    row = {
        "episode_id": "NLDO-P007",
        "stage_index": 12,
        "run_seed": 0,
        "use_source_history_population": True,
        "previous_stage_index": 11,
        "source_population_kind": "final_population",
        "source_population_count": 200,
    }
    report = audit_shadow_runtime_lineage(
        {"warm_t12_ds": {("NLDO-P007", 12, 0): row}}
    )
    assert report["warm_t12_ds"] == {"audited_cells": 1, "bad_cells": 0}
