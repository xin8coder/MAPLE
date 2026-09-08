from __future__ import annotations

import copy

from evo2.evaluation.reference_solvers.moea_reference import (
    MOEAConfig,
    _constraint_feasible_candidates,
    _constraint_penalty,
    _rank_map,
    _select_next_generation,
    cloud_stage_reference,
    green_vrp_stage_reference,
)
from scripts import materialize_cloud_scheduling_mo as cloud
from scripts import materialize_green_vrp_mo as green


def test_green_vrp_moea_reference_smoke():
    ep = green.build_episodes()[0]
    config = MOEAConfig(population_size=12, generations=2, seeds=1, archive_limit=8, hv_samples=200)
    ref = green_vrp_stage_reference(copy.deepcopy(ep["hidden_initial_state"]), None, green._score_solution, config)
    assert ref["archive"]
    assert ref["reference_hv"] >= 0.0
    assert ref["self_igd"] == 0.0


def test_cloud_moea_reference_smoke():
    ep = cloud.build_episodes()[0]
    config = MOEAConfig(population_size=12, generations=2, seeds=1, archive_limit=8, hv_samples=200)
    ref = cloud_stage_reference(copy.deepcopy(ep["hidden_initial_state"]), None, cloud._score_solution, config)
    assert ref["archive"]
    assert ref["reference_hv"] >= 0.0
    assert ref["self_igd"] == 0.0


def test_cloud_reference_feasibility_is_resource_based_not_sla_or_migration_path_dependent():
    ep = cloud.build_episodes()[0]
    evaluation = ep["evaluation"]
    initial_state = copy.deepcopy(ep["hidden_initial_state"])
    representative = evaluation["reference_trajectory"][0]["representative_solution"]

    assert "migration_budget" not in initial_state
    assert evaluation["constraints"] == ["resource_feasibility"]
    assert "migration_budget_violation" not in evaluation["metrics"]
    assert "migration_budget_violation" not in evaluation["auxiliary_metrics"]

    shifted_previous = {
        "assignments": {
            job["id"]: "M01"
            for job in initial_state["jobs"]
            if job.get("active", True) and not job.get("gpu_required", False)
        }
    }
    score_without_previous = cloud._score_solution(initial_state, representative, None)
    score_with_previous = cloud._score_solution(initial_state, representative, shifted_previous)

    assert "migration_budget_violation" not in score_without_previous
    assert "migration_budget_violation" not in score_with_previous
    assert score_without_previous["resource_violations"] == 0.0
    assert score_with_previous["resource_violations"] == 0.0
    assert score_without_previous["sla_violations"] == score_with_previous["sla_violations"]


def test_moea_constraint_filter_uses_resource_violations_and_ignores_diagnostics():
    items = [
        {
            "solution": "kept",
            "objectives": {"resource_violations": 0.0, "sla_violations": 99.0, "migration_budget_violation": 99.0, "energy": 1.0},
            "scalar_score": 1.0,
        },
        {
            "solution": "blocked",
            "objectives": {"resource_violations": 2.0, "sla_violations": 0.0, "migration_budget_violation": 0.0, "energy": 0.1},
            "scalar_score": 0.1,
        },
    ]

    assert _constraint_penalty(items[0]["objectives"]) == 0.0
    assert _constraint_feasible_candidates(items) == [items[0]]


def test_nsga_internal_ranking_and_selection_keep_tradeoff_front():
    items = [
        {"solution": "a", "objectives": {"x": 1.0, "y": 4.0}, "scalar_score": 5.0},
        {"solution": "b", "objectives": {"x": 2.0, "y": 2.0}, "scalar_score": 4.0},
        {"solution": "c", "objectives": {"x": 4.0, "y": 1.0}, "scalar_score": 5.0},
        {"solution": "dominated", "objectives": {"x": 3.0, "y": 3.0}, "scalar_score": 6.0},
    ]

    ranks = _rank_map(items, ["x", "y"])
    selected = _select_next_generation(items, ["x", "y"], 3)

    assert [ranks[idx] for idx in range(3)] == [0, 0, 0]
    assert ranks[3] > 0
    assert {item["solution"] for item in selected} == {"a", "b", "c"}
