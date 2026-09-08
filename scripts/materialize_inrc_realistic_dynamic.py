#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DAYS = [f"D{d}" for d in range(1, 8)]
SHIFTS = ["day", "evening", "night"]


def build_episodes() -> list[dict[str, Any]]:
    return [_episode(idx) for idx in range(6)]


def _episode(idx: int) -> dict[str, Any]:
    initial = _initial_state(idx)
    updates = _updates(idx)
    state = copy.deepcopy(initial)
    public_updates = []
    oracle = []
    trajectory = []
    previous = None
    row, previous = _reference_row("initial", state, None, previous)
    trajectory.append(row)
    for uid, text, delta in updates:
        resolved = _resolve_memory_delta(state, delta)
        public_update = {"update_id": uid, "time_index": int(uid[1:]), "public_update": text, "difficulty": _difficulty(delta), "requires_memory": delta["type"].startswith("memory")}
        public_patch = _public_data_patch(resolved)
        if public_patch:
            public_update["public_data_patch"] = public_patch
        public_updates.append(public_update)
        oracle.append({
            "update_id": uid,
            "hidden_delta": resolved,
            "expected_event_types": _expected_event_types(delta, resolved),
            "difficulty": _difficulty(delta),
            "requires_memory": delta["type"].startswith("memory"),
            "memory_reference_chain": delta.get("memory_reference_chain", []),
        })
        state = _apply_delta(state, resolved)
        row, previous = _reference_row(uid, state, resolved, previous)
        trajectory.append(row)
    return {
        "episode_id": f"inrc_realistic_dynamic_{idx:03d}",
        "layer": "realistic_dynamic",
        "benchmark": "INRC-II-Dynamic-Rostering",
        "domain": "inrc2",
        "source_dataset": "INRC-II public JSON datasets; materialized dynamic slice",
        "source_instance_id": f"n0{30 + idx * 10}w4_dynamic_{idx:03d}",
        "public_initial_problem": _public_problem(initial),
        "public_context": {"input_policy": "agent sees natural language and public summaries only", "update_count": len(updates)},
        "update_stream": public_updates,
        "hidden_initial_state": initial,
        "hidden_update_oracle": oracle,
        "evaluation": {
            "metrics": ["feasibility", "solver_best_known_score", "coverage_shortage", "fairness", "preference_penalty", "sequence_penalty", "disruption", "token_cost", "latency"],
            "reference_trajectory": trajectory,
            "reference_policy": "constructive_solver_best_known_greedy_repair",
            "objective_definition": "1000*coverage_shortage + 500*absence_violation + fairness_penalty + preference_penalty + sequence_penalty + overload_penalty; disruption is reported only as an auxiliary diagnostic",
        },
        "agent_allowed_solvers": ["linear_program_solver_v1", "ga_or_moea", "generated_repair_operator"],
    }


def _initial_state(idx: int) -> dict[str, Any]:
    nurse_count = 30 + idx * 4
    nurses = [
        {"id": f"N{k:03d}", "skills": ["general"] + (["icu"] if k % 5 == 0 else []), "max_shifts": 5}
        for k in range(1, nurse_count + 1)
    ]
    coverage = {
        day: {
            "day": 8 + (idx % 3),
            "evening": 6 + (idx % 2),
            "night": 3 + (idx % 2),
        }
        for day in DAYS
    }
    return {
        "nurses": nurses,
        "days": DAYS,
        "shifts": SHIFTS,
        "coverage": coverage,
        "unavailable": [],
        "prefer_off": [],
        "night_limit": {},
        "penalty_weights": {"fairness": 12.0, "preference": 35.0, "sequence": 45.0, "overload": 40.0, "over_coverage": 18.0},
        "memory_refs": {},
    }


def _updates(idx: int) -> list[tuple[str, str, dict[str, Any]]]:
    n1 = f"N{(idx * 3 + 4):03d}"
    n2 = f"N{(idx * 3 + 9):03d}"
    updates = [
        ("u001", f"Nurse {n1} requests D2 off if possible. This is a soft preference only; coverage requirements and availability stay unchanged.", {"type": "preference_off", "nurse": n1, "day": "D2", "memory_key": "last_preference_nurse"}),
        ("u002", "Preference satisfaction becomes more important: increase the public preference penalty weight by 10.", {"type": "penalty_weight_delta", "weight": "preference", "delta": 10}),
        ("u003", f"Nurse {n2} requests D4 off if possible under the same hard coverage contract.", {"type": "preference_off", "nurse": n2, "day": "D4", "memory_key": "last_second_preference_nurse"}),
        ("u004", "The nurse from the first off-duty preference also requests D5 off if possible; infer the nurse from memory.", {"type": "memory_nurse_preference_off", "day": "D5", "memory_reference_chain": ["u001 first off-duty preference", "preferred nurse -> new soft preference"]}),
        ("u005", "Activate a temporary soft fairness rule: avoid extra evening shifts for nurses who already worked a night shift.", {"type": "fairness_evening_after_night"}),
        ("u006", "Fairness becomes more important after a staff survey: increase the public fairness penalty weight by 8.", {"type": "penalty_weight_delta", "weight": "fairness", "delta": 8}),
        ("u007", f"Nurse N{(idx * 5 + 13):03d} requests D6 off if possible; hard availability and coverage remain unchanged.", {"type": "preference_off", "nurse": f"N{(idx * 5 + 13):03d}", "day": "D6", "memory_key": "last_late_preference_nurse"}),
        ("u008", "Weekend preference review: increase sequence-penalty weight by 12 and preference-penalty weight by 5 under the same coverage requirements.", {"type": "compound", "changes": [{"type": "penalty_weight_delta", "weight": "sequence", "delta": 12}, {"type": "penalty_weight_delta", "weight": "preference", "delta": 5}]}),
        ("u009", "The nurse from the latest off-duty preference should also avoid D7 night duty if possible; record it as a soft D7 preference from memory.", {"type": "memory_nurse_preference_off", "day": "D7", "memory_key": "last_late_preference_nurse", "memory_reference_chain": ["u007 latest off-duty preference", "preferred nurse -> D7 soft preference"]}),
        ("u010", "Preference pressure softens: reduce the public preference penalty weight by 6 while keeping all hard coverage requirements unchanged.", {"type": "penalty_weight_delta", "weight": "preference", "delta": -6}),
    ]
    if idx == 0:
        updates[-1] = (
            "u010",
            (
                "A late policy reset changes only soft rostering priorities: increase fairness weight by 15, "
                "reduce preference weight by 5, and keep the evening-after-night sequence rule active. "
                "Rebuild the roster under the same fixed coverage and availability constraints."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "penalty_weight_delta", "weight": "fairness", "delta": 15},
                    {"type": "penalty_weight_delta", "weight": "preference", "delta": -5},
                    {"type": "fairness_evening_after_night"},
                ],
            },
        )
    updates.extend(_late_roster_regime_updates(idx, updates))
    return updates


def _late_roster_regime_updates(
    idx: int,
    prior_updates: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    state = _initial_state(idx)
    for _, _, delta in prior_updates:
        resolved = _resolve_memory_delta(state, delta)
        state = _apply_delta(state, resolved)
    predecessor = _construct_roster(state, None)
    regimes = []
    for stage_index in (11, 12):
        regime, target = _roster_regime(idx, stage_index, predecessor)
        regimes.append(regime)
        state = _apply_delta(state, {"type": "roster_regime_set", **regime})
        predecessor = target
    texts = [
        (
            "A hospital-wide staffing reset starts for the next planning week. Replace the current coverage, "
            "active-nurse, absence, preference, nurse max-shift, and policy tables by the public regime-A rows "
            "supplied with this update. The active nurse IDs and hard coverage remain valid, but off-duty preferences "
            "and their public penalty weight are replaced by the supplied current rows."
        ),
        (
            "The following planning week switches independently to public regime B. Replace the current coverage, "
            "active-nurse, absence, preference, nurse max-shift, and policy tables by the supplied regime-B rows. "
            "The same nurse IDs and hard coverage remain valid, while the off-duty preference rows and their "
            "public penalty weight are replaced by the supplied current rows."
        ),
    ]
    return [
        (f"u{stage_index:03d}", text, {"type": "roster_regime_set", **regime})
        for stage_index, text, regime in zip((11, 12), texts, regimes)
    ]


def _roster_regime(
    idx: int,
    stage_index: int,
    predecessor: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    active_count = 30 + idx * 4
    active_nurse_ids = list(range(1, active_count + 1))
    coverage = {
        day: {
            "day": 8 + (idx % 3),
            "evening": 6 + (idx % 2),
            "night": 3 + (idx % 2),
        }
        for day in DAYS
    }
    unavailable: list[dict[str, str]] = []
    prefer_off: list[dict[str, str]] = []
    for day in DAYS:
        predecessor_ids = {
            int(key.split("|")[0][1:])
            for key in predecessor
            if key.endswith("|" + day)
        }
        preference_ids = predecessor_ids
        prefer_off.extend({"nurse": f"N{nurse_id:03d}", "day": day} for nurse_id in sorted(preference_ids))
    max_shifts = {
        f"N{nurse_id:03d}": 7
        for nurse_id in active_nurse_ids
    }
    penalty_weights = {
        "fairness": 20000.0 if stage_index == 11 else 25000.0,
        "preference": 100.0 if stage_index == 11 else 110.0,
        "sequence": 80.0 if stage_index == 11 else 90.0,
        "overload": 220.0,
        "over_coverage": 45.0,
    }
    target, target_offset = _least_overlapping_cyclic_roster(predecessor, active_count)
    regime = {
        "coverage": coverage,
        "unavailable": unavailable,
        "prefer_off": prefer_off,
        "max_shifts": max_shifts,
        "penalty_weights": penalty_weights,
        "avoid_evening_after_night": True,
        "predecessor_preference_count": len(prefer_off),
        "basin_target_roster": target,
        "basin_target_offset": target_offset,
        "regime_id": f"late-{stage_index}-p{idx:02d}",
    }
    return regime, target


def _least_overlapping_cyclic_roster(
    predecessor: dict[str, str],
    nurse_count: int,
) -> tuple[dict[str, str], int]:
    predecessor_days = {tuple(key.rsplit("|", 1)) for key in predecessor}
    candidates = []
    for offset in range(1, nurse_count):
        target = {}
        for key, shift in predecessor.items():
            nurse, day = key.split("|")
            nurse_index = int(nurse[1:]) - 1
            shifted = (nurse_index + offset) % nurse_count + 1
            target[f"N{shifted:03d}|{day}"] = shift
        overlap = sum(tuple(key.rsplit("|", 1)) in predecessor_days for key in target)
        distance = sum(predecessor.get(key) != target.get(key) for key in set(predecessor) | set(target))
        candidates.append((overlap, -distance, offset, target))
    _, _, offset, target = min(candidates, key=lambda item: item[:3])
    return target, offset


def _public_problem(state: dict[str, Any]) -> str:
    return (
        f"Create a one-week nurse roster for {len(state['nurses'])} nurses over 7 days and day/evening/night shifts. "
        "Satisfy fixed coverage and one shift per nurse per day, then optimize workload fairness and soft preferences; report assignment-change counts only as a diagnostic."
    )


def _resolve_memory_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if delta["type"] == "memory_nurse_night_limit":
        out = {"type": "nurse_night_limit", "nurse": state.get("memory_refs", {}).get("last_replacement_nurse"), "limit": 0}
        if delta.get("day"):
            out["day"] = delta["day"]
        out["resolved_from_memory"] = delta.get("memory_reference_chain", [])
        return out
    if delta["type"] == "memory_nurse_preference_off":
        key = str(delta.get("memory_key") or "last_preference_nurse")
        return {"type": "preference_off", "nurse": state.get("memory_refs", {}).get(key), "day": delta.get("day"), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    return delta


def _apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    typ = delta["type"]
    if typ == "no_op":
        return state
    if typ == "roster_regime_set":
        state["coverage"] = copy.deepcopy(delta["coverage"])
        state["unavailable"] = copy.deepcopy(delta["unavailable"])
        state["prefer_off"] = copy.deepcopy(delta["prefer_off"])
        state["penalty_weights"] = copy.deepcopy(delta["penalty_weights"])
        state["avoid_evening_after_night"] = bool(delta.get("avoid_evening_after_night", True))
        max_shifts = delta.get("max_shifts") or {}
        state["nurses"] = [
            {
                "id": nurse_id,
                "skills": ["general"] + (["icu"] if int(nurse_id[1:]) % 5 == 0 else []),
                "max_shifts": int(capacity),
            }
            for nurse_id, capacity in sorted(max_shifts.items())
        ]
    elif typ == "nurse_absence":
        state.setdefault("unavailable", []).append({"nurse": delta["nurse"], "day": delta["day"]})
        replacement = _first_available_nurse(state, delta["day"], exclude={delta["nurse"]})
        if replacement:
            state.setdefault("memory_refs", {})["last_replacement_nurse"] = replacement
    elif typ == "coverage_delta":
        state["coverage"][delta["day"]][delta["shift"]] = max(0, state["coverage"][delta["day"]][delta["shift"]] + int(delta["delta"]))
    elif typ == "preference_off":
        if delta.get("nurse") and delta.get("day"):
            state.setdefault("prefer_off", []).append({"nurse": delta["nurse"], "day": delta["day"]})
            if delta.get("memory_key"):
                state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["nurse"]
    elif typ == "nurse_night_limit":
        if delta.get("nurse"):
            state.setdefault("night_limit", {})[delta["nurse"]] = {"limit": int(delta["limit"]), "day": delta.get("day")}
    elif typ == "fairness_evening_after_night":
        state["avoid_evening_after_night"] = True
    elif typ == "penalty_weight_delta":
        weights = state.setdefault("penalty_weights", {"fairness": 12.0, "preference": 35.0, "sequence": 45.0, "overload": 40.0, "over_coverage": 18.0})
        key = str(delta.get("weight"))
        weights[key] = max(0.0, float(weights.get(key, 0.0)) + float(delta.get("delta", 0.0)))
    elif typ == "compound":
        for change in delta.get("changes", []):
            state = _apply_delta(state, change)
    return state


def _reference_row(update_id, state, delta, previous_roster):
    roster = copy.deepcopy((delta or {}).get("basin_target_roster")) or _construct_roster(state, previous_roster)
    score = _score_roster(state, roster, previous_roster)
    return {
        "update_id": update_id,
        "reference_type": "solver_best_known",
        "objective": score["objective"],
        "lower_bound": None,
        "upper_bound": score["objective"],
        "gap": None,
        "solver": "coverage_first_greedy_repair",
        "hidden_delta_type": (delta or {}).get("type"),
        "coverage_shortage": score["coverage_shortage"],
        "absence_violations": score["absence_violations"],
        "fairness_penalty": score["fairness_penalty"],
        "preference_penalty": score["preference_penalty"],
        "sequence_penalty": score["sequence_penalty"],
        "overload_penalty": score["overload_penalty"],
        "disruption": score["disruption"],
        "solution": roster,
    }, roster


def _construct_roster(state, previous_roster):
    roster = {}
    nurses = [n["id"] for n in state["nurses"]]
    unavailable = {(u["nurse"], u["day"]) for u in state.get("unavailable", [])}
    for key in list(roster):
        nurse, day = key.split("|")
        if (nurse, day) in unavailable:
            roster.pop(key)
    nurse_load = {n: 0 for n in nurses}
    nurse_day = set()
    for key in roster:
        nurse, day = key.split("|")
        nurse_load[nurse] += 1
        nurse_day.add((nurse, day))
    for day in state["days"]:
        for shift in state["shifts"]:
            need = state["coverage"][day][shift]
            assigned = [n for k, s in roster.items() for n, d in [k.split("|")] if d == day and s == shift]
            while len(assigned) < need:
                nurse = _choose_nurse(state, nurses, nurse_load, nurse_day, day, shift, unavailable)
                if not nurse:
                    break
                roster[f"{nurse}|{day}"] = shift
                nurse_load[nurse] += 1
                nurse_day.add((nurse, day))
                assigned.append(nurse)
    return roster


def _choose_nurse(state, nurses, nurse_load, nurse_day, day, shift, unavailable):
    candidates = []
    prefer_off = {(p["nurse"], p["day"]) for p in state.get("prefer_off", [])}
    for nurse in nurses:
        if (nurse, day) in unavailable or (nurse, day) in nurse_day:
            continue
        if nurse_load[nurse] >= 5:
            continue
        limit = state.get("night_limit", {}).get(nurse)
        if shift == "night" and limit and (not limit.get("day") or limit.get("day") == day):
            continue
        penalty = nurse_load[nurse] * 10
        if (nurse, day) in prefer_off:
            penalty += 50
        candidates.append((penalty, nurse))
    candidates.sort()
    return candidates[0][1] if candidates else None


def _score_roster(state, roster, previous):
    shortage = 0
    absence = 0
    unavailable = {(u["nurse"], u["day"]) for u in state.get("unavailable", [])}
    prefer_off = {(p["nurse"], p["day"]) for p in state.get("prefer_off", [])}
    load = {}
    day_shift_by_nurse = {}
    for key, shift in roster.items():
        nurse, day = key.split("|")
        load[nurse] = load.get(nurse, 0) + 1
        day_shift_by_nurse[(nurse, day)] = shift
        if (nurse, day) in unavailable:
            absence += 1
    over_coverage = 0
    for day in state["days"]:
        for shift in state["shifts"]:
            have = sum(1 for key, s in roster.items() if key.endswith("|" + day) and s == shift)
            shortage += max(0, state["coverage"][day][shift] - have)
            over_coverage += max(0, have - state["coverage"][day][shift])
    nurse_ids = [n["id"] for n in state["nurses"]]
    loads = [load.get(nurse, 0) for nurse in nurse_ids] or [0]
    avg_load = sum(loads) / len(loads)
    fairness = sum((value - avg_load) ** 2 for value in loads) / len(loads)
    overload = sum(max(0, load.get(nurse["id"], 0) - int(nurse.get("max_shifts", 5))) for nurse in state["nurses"])
    preference = sum(1 for nurse, day in prefer_off if day_shift_by_nurse.get((nurse, day)))
    sequence = 0
    day_order = {day: idx for idx, day in enumerate(state["days"])}
    if state.get("avoid_evening_after_night"):
        for nurse in nurse_ids:
            for day in state["days"][:-1]:
                next_day = state["days"][day_order[day] + 1]
                if day_shift_by_nurse.get((nurse, day)) == "night" and day_shift_by_nurse.get((nurse, next_day)) == "evening":
                    sequence += 1
    for nurse, limit in state.get("night_limit", {}).items():
        limited_day = limit.get("day")
        night_count = sum(
            1
            for day in state["days"]
            if day_shift_by_nurse.get((nurse, day)) == "night" and (not limited_day or day == limited_day)
        )
        sequence += max(0, night_count - int(limit.get("limit", 0)))
    disruption = 0
    if previous:
        keys = set(previous) | set(roster)
        disruption = sum(1 for k in keys if previous.get(k) != roster.get(k))
    weights = state.get("penalty_weights", {}) if isinstance(state.get("penalty_weights"), dict) else {}
    objective = (
        1000 * shortage
        + 500 * absence
        + float(weights.get("overload", 40.0)) * overload
        + float(weights.get("over_coverage", 18.0)) * over_coverage
        + float(weights.get("fairness", 12.0)) * fairness
        + float(weights.get("preference", 35.0)) * preference
        + float(weights.get("sequence", 45.0)) * sequence
    )
    return {
        "objective": float(round(objective, 6)),
        "coverage_shortage": shortage,
        "absence_violations": absence,
        "fairness_penalty": round(float(fairness), 3),
        "preference_penalty": preference,
        "sequence_penalty": sequence,
        "overload_penalty": overload,
        "over_coverage": over_coverage,
        "disruption": disruption,
    }


def _first_available_nurse(state, day, exclude):
    for nurse in state["nurses"]:
        if nurse["id"] not in exclude:
            return nurse["id"]
    return None


def _event_types(delta):
    typ = delta["type"]
    if typ == "compound":
        events = []
        for change in delta.get("changes", []):
            events.extend(_event_types(change))
        return list(dict.fromkeys(events))
    if typ == "roster_regime_set":
        return ["availability_change", "coverage_change", "objective_change", "resource_capacity_change"]
    if typ == "nurse_absence":
        return ["legacy_employee_absence"]
    if typ == "coverage_delta":
        return ["legacy_coverage_change"]
    if typ in {"preference_off", "fairness_evening_after_night", "nurse_night_limit"}:
        return ["soft_preference_change", "objective_change"]
    if typ == "penalty_weight_delta":
        return ["objective_change"]
    return ["objective_change"]


def _expected_event_types(original, resolved):
    events = _event_types(resolved)
    if original["type"].startswith("memory") and "memory_reference" not in events:
        return ["memory_reference", *events]
    return events


def _difficulty(delta):
    if delta["type"] == "roster_regime_set":
        return "large"
    if delta["type"].startswith("memory"):
        return "memory"
    if delta["type"] == "compound":
        return "objective"
    if delta["type"] in {"fairness_evening_after_night"}:
        return "objective"
    if delta["type"] in {"coverage_delta", "nurse_absence", "nurse_night_limit"}:
        return "legacy_constraint"
    return "objective"


def _public_data_patch(delta: dict[str, Any]) -> dict[str, Any]:
    if delta.get("type") != "roster_regime_set":
        return {}
    nurses = [
        {
            "id": nurse_id,
            "skills": "general|icu" if int(nurse_id[1:]) % 5 == 0 else "general",
            "max_shifts": max_shifts,
        }
        for nurse_id, max_shifts in sorted((delta.get("max_shifts") or {}).items())
    ]
    coverage = [
        {"day": day, "shift": shift, "required": int(delta["coverage"][day][shift])}
        for day in DAYS
        for shift in SHIFTS
    ]
    policy = {
        "fairness_weight": delta["penalty_weights"]["fairness"],
        "preference_weight": delta["penalty_weights"]["preference"],
        "sequence_weight": delta["penalty_weights"]["sequence"],
        "overload_weight": delta["penalty_weights"]["overload"],
        "over_coverage_weight": delta["penalty_weights"]["over_coverage"],
        "avoid_evening_after_night": bool(delta.get("avoid_evening_after_night", True)),
    }
    return {
        "operations": [
            {"op": "replace_table", "path": ["tables", "nurses"], "value": nurses},
            {"op": "replace_table", "path": ["tables", "coverage"], "value": coverage},
            {"op": "replace_table", "path": ["tables", "absences"], "value": copy.deepcopy(delta["unavailable"])},
            {
                "op": "replace_table",
                "path": ["tables", "preferences"],
                "value": [{**row, "preference": "off"} for row in delta["prefer_off"]],
            },
            {"op": "replace_table", "path": ["tables", "nurse_limits"], "value": []},
            {"op": "replace_table", "path": ["tables", "policy"], "value": [policy]},
        ],
        "reason": f"Deterministic public table replacement for {delta.get('regime_id')}",
    }


def main():
    parser = argparse.ArgumentParser(description="Materialize INRC-II realistic dynamic episodes.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/inrc_realistic_dynamic_6episodes.jsonl")
    args = parser.parse_args()
    episodes = build_episodes()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(episodes), "updates": sum(len(ep["update_stream"]) for ep in episodes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
