from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import materialize_cloud_scheduling_mo as cloud
from scripts import materialize_green_vrp_mo as green
from scripts import materialize_inrc_realistic_dynamic as inrc
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import replay_data_patch


RESTART_LABEL_LEAKAGE = (
    "high-penalty basin",
    "high-cost local basin",
    "preferred basin",
    "substantially different allocation",
    "previous Pareto region",
    "re-optimize",
    "reoptimize",
    "reliable search region",
)


@pytest.mark.parametrize(
    ("module", "delta_type", "minimum_patch_operations"),
    [
        (inrc, "roster_regime_set", 6),
        (green, "dispatch_regime_set", 24),
        (cloud, "cloud_regime_set", 74),
    ],
)
def test_late_regimes_are_public_large_state_replacements(module, delta_type, minimum_patch_operations):
    episode = module.build_episodes()[0]
    assert len(episode["update_stream"]) == 12
    state = copy.deepcopy(episode["hidden_initial_state"])
    for stage_index, (update, oracle) in enumerate(
        zip(episode["update_stream"], episode["hidden_update_oracle"]),
        start=1,
    ):
        state = module._apply_delta(state, oracle["hidden_delta"])
        if stage_index < 11:
            continue
        assert update["difficulty"] == "large"
        assert oracle["hidden_delta"]["type"] == delta_type
        patch = update.get("public_data_patch") or {}
        assert len(patch.get("operations") or []) >= minimum_patch_operations


def test_late_regime_states_remain_feasible_by_construction():
    for module in (inrc, green, cloud):
        for episode in module.build_episodes()[:3]:
            state = copy.deepcopy(episode["hidden_initial_state"])
            for stage_index, oracle in enumerate(episode["hidden_update_oracle"], start=1):
                state = module._apply_delta(state, oracle["hidden_delta"])
                if stage_index < 11:
                    continue
                if module is inrc:
                    weekly_demand = sum(sum(row.values()) for row in state["coverage"].values())
                    weekly_capacity = sum(int(nurse["max_shifts"]) for nurse in state["nurses"])
                    assert weekly_capacity > weekly_demand
                    absence_days = {}
                    for row in state["unavailable"]:
                        absence_days.setdefault(row["nurse"], set()).add(row["day"])
                    usable_capacity = sum(
                        min(int(nurse["max_shifts"]), len(state["days"]) - len(absence_days.get(nurse["id"], set())))
                        for nurse in state["nurses"]
                    )
                    assert usable_capacity > weekly_demand
                    for day, demand in state["coverage"].items():
                        available = sum(
                            1
                            for nurse in state["nurses"]
                            if day not in absence_days.get(nurse["id"], set())
                        )
                        assert available >= sum(demand.values())
                elif module is green:
                    active_demand = sum(order["demand"] for order in state["orders"] if order.get("active", True))
                    fleet_capacity = sum(vehicle["capacity"] for vehicle in state["vehicles"] if vehicle.get("available", True))
                    assert fleet_capacity > active_demand
                    assert all(vehicle["shift_end"] >= 1800 for vehicle in state["vehicles"])
                else:
                    active_jobs = [job for job in state["jobs"] if job.get("active", True)]
                    assert len(active_jobs) == 64
                    assert sum(machine["cpu"] for machine in state["machines"]) > sum(job["cpu"] for job in active_jobs)
                    assert sum(machine["mem"] for machine in state["machines"]) > sum(job["mem"] for job in active_jobs)


def test_roster_late_regimes_flip_the_public_preference_basin():
    for episode in inrc.build_episodes()[:3]:
        initial_ids = {nurse["id"] for nurse in episode["hidden_initial_state"]["nurses"]}
        late_deltas = [row["hidden_delta"] for row in episode["hidden_update_oracle"][-2:]]
        cohort_a = set(late_deltas[0]["max_shifts"])
        cohort_b = set(late_deltas[1]["max_shifts"])

        assert cohort_a == cohort_b == initial_ids
        for delta in late_deltas:
            assert delta["predecessor_preference_count"] == len(delta["prefer_off"])
            assert delta["predecessor_preference_count"] > 0


def test_roster_basin_flip_keeps_predecessor_feasible_but_far_from_new_reference():
    for episode in inrc.build_episodes()[:3]:
        state = copy.deepcopy(episode["hidden_initial_state"])
        trajectory = episode["evaluation"]["reference_trajectory"]
        predecessor = trajectory[0]["solution"]
        for stage_index, oracle in enumerate(episode["hidden_update_oracle"], start=1):
            state = inrc._apply_delta(state, oracle["hidden_delta"])
            current = trajectory[stage_index]["solution"]
            if stage_index >= 11:
                old_score = inrc._score_roster(state, predecessor, None)
                new_score = inrc._score_roster(state, current, None)
                keys = set(predecessor) | set(current)
                distance = sum(predecessor.get(key) != current.get(key) for key in keys)

                assert old_score["coverage_shortage"] == 0
                assert old_score["absence_violations"] == 0
                assert old_score["fairness_penalty"] == new_score["fairness_penalty"]
                assert old_score["objective"] - new_score["objective"] > 5000
                assert distance > 150
                assert "basin_target_roster" in oracle["hidden_delta"]
                public_update = episode["update_stream"][stage_index - 1]
                assert "basin_target_roster" not in json.dumps(public_update)
            predecessor = current


def test_routing_late_regimes_activate_disjoint_prelisted_order_cohorts():
    for episode in green.build_episodes()[:3]:
        initial_orders = episode["hidden_initial_state"]["orders"]
        initial_active = {order["id"] for order in initial_orders if order.get("active", True)}
        initial_inactive = {order["id"] for order in initial_orders if not order.get("active", True)}
        late_deltas = [row["hidden_delta"] for row in episode["hidden_update_oracle"][-2:]]
        cohort_a = {order["id"] for order in late_deltas[0]["orders"] if order.get("active", True)}
        cohort_b = {order["id"] for order in late_deltas[1]["orders"] if order.get("active", True)}

        assert len(cohort_a) == len(cohort_b) == 30
        assert cohort_a.isdisjoint(cohort_b)
        assert cohort_a.isdisjoint(initial_active)
        assert cohort_b.isdisjoint(initial_active)
        assert cohort_a | cohort_b <= initial_inactive


def test_cloud_late_regimes_keep_job_ids_and_flip_capacity_roles():
    for episode in cloud.build_episodes()[:3]:
        initial_ids = {job["id"] for job in episode["hidden_initial_state"]["jobs"] if job.get("active", True)}
        late_deltas = [row["hidden_delta"] for row in episode["hidden_update_oracle"][-2:]]
        cohort_a = {job["id"] for job in late_deltas[0]["jobs"] if job.get("active", True)}
        cohort_b = {job["id"] for job in late_deltas[1]["jobs"] if job.get("active", True)}

        assert len(cohort_a) == len(cohort_b) == 64
        assert cohort_a == cohort_b
        assert initial_ids < cohort_a
        high_a = {m["id"] for m in late_deltas[0]["machines"] if m["cpu"] > 100}
        high_b = {m["id"] for m in late_deltas[1]["machines"] if m["cpu"] > 100}
        assert high_a == {"M05", "M06", "M07", "M08"}
        assert high_b == {"M01", "M02", "M03", "M04"}
        assert all(m["available"] for delta in late_deltas for m in delta["machines"])


def test_nldo_p007_p015_late_updates_carry_materialized_public_patches():
    path = Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for episode in rows[6:15]:
        for stage_index in (11, 12):
            update = episode["update_stream"][stage_index - 1]
            oracle = episode["hidden_update_oracle"][stage_index - 1]
            assert update["update_id"].endswith(f"S{stage_index:02d}")
            assert update["difficulty"] == "large"
            assert update.get("public_data_patch", {}).get("operations")
            assert oracle.get("difficulty") == "large"


def test_late_public_updates_do_not_reveal_restart_or_basin_labels():
    for module in (inrc, green, cloud):
        for episode in module.build_episodes()[:3]:
            for update in episode["update_stream"]:
                text = str(update.get("natural_language_update") or update.get("public_update") or "").lower()
                assert not any(label.lower() in text for label in RESTART_LABEL_LEAKAGE)

    path = Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for episode in rows[6:15]:
        for update in episode["update_stream"]:
            text = str(update.get("natural_language_update") or update.get("public_update") or "").lower()
            assert not any(label.lower() in text for label in RESTART_LABEL_LEAKAGE)


def test_replay_prefers_benchmark_authored_public_patch():
    benchmark_patch = {"operations": [{"op": "replace_table", "path": ["tables", "x"], "value": []}]}
    artifact_patch = {"operations": [{"op": "set_value", "path": ["tables", "x", "value"], "value": 1}]}
    update = {"public_data_patch": benchmark_patch}
    source_stage = {"restart": {"data_patch": artifact_patch}}
    assert replay_data_patch(update, source_stage, {}) == benchmark_patch
    assert replay_data_patch({}, source_stage, {}) == artifact_patch
