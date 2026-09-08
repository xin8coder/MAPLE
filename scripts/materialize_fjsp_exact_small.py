#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any


def build_episodes() -> list[dict[str, Any]]:
    return [_episode(idx) for idx in range(6)]


def _episode(idx: int) -> dict[str, Any]:
    initial = _initial_state(idx)
    updates = _updates(idx)
    state = copy.deepcopy(initial)
    public_updates = []
    oracle = []
    previous_solution = None
    trajectory = []
    row, previous_solution = _reference_row("initial", state, None, previous_solution)
    trajectory.append(row)
    for uid, text, delta in updates:
        public_updates.append({"update_id": uid, "time_index": int(uid[1:]), "public_update": text, "difficulty": _difficulty(delta), "requires_memory": delta["type"].startswith("memory")})
        resolved = _resolve_memory_delta(state, delta)
        oracle.append({
            "update_id": uid,
            "hidden_delta": resolved,
            "expected_event_types": _expected_event_types(delta, resolved),
            "difficulty": _difficulty(delta),
            "requires_memory": delta["type"].startswith("memory"),
            "memory_reference_chain": delta.get("memory_reference_chain", []),
        })
        state = _apply_delta(state, resolved)
        row, previous_solution = _reference_row(uid, state, resolved, previous_solution)
        trajectory.append(row)
    return {
        "episode_id": f"fjsp_exact_small_{idx:03d}",
        "layer": "exact_small",
        "benchmark": "FJSP-Exact-Small",
        "domain": "fjsp",
        "source_dataset": "synthetic exact-small FJSP materialization based on FJSPLib-style constraints",
        "source_instance_id": f"fjsp_exact_small_{idx:03d}",
        "public_initial_problem": _public_problem(initial),
        "public_context": {"input_policy": "agent sees natural language and public summaries only", "update_count": len(updates)},
        "update_stream": public_updates,
        "hidden_initial_state": initial,
        "hidden_update_oracle": oracle,
        "evaluation": {
            "metrics": ["feasibility", "objective_gap", "exact_reference_objective", "tardiness", "disruption", "token_cost", "latency"],
            "reference_trajectory": trajectory,
            "reference_policy": "exact_optimum_by_branch_search",
            "objective_definition": "weighted_tardiness; plan-change disruption is reported only as an auxiliary diagnostic",
        },
        "agent_allowed_solvers": ["linear_program_solver_v1", "ga_or_moea", "generated_repair_operator"],
    }


def _initial_state(idx: int) -> dict[str, Any]:
    machines = [{"id": f"M{k}", "cost_rate": [1.0, 4.0, 8.0][k - 1], "available": True} for k in range(1, 4)]
    jobs = []
    for j in range(1, 4):
        ops = []
        for o in range(1, 4):
            p = 4 + ((j * 3 + o * 2 + idx) % 7)
            m1 = f"M{((j + o + idx) % 3) + 1}"
            m2 = f"M{((j + o + idx + 1) % 3) + 1}"
            ops.append({"id": f"J{j}O{o}", "eligible_machines": [m1, m2], "processing_time": p})
        jobs.append({"id": f"J{j}", "due_time": 24 + 4 * j + idx, "priority": 1, "operations": ops})
    return {"machines": machines, "jobs": jobs, "machine_downtime": [], "objective_mode": "minimize_tardiness", "memory_refs": {}}


def _updates(idx: int) -> list[tuple[str, str, dict[str, Any]]]:
    cost_machine = f"M{(idx % 3) + 1}"
    urgent = f"J{(idx % 3) + 1}"
    op_job = f"J{((idx + 1) % 3) + 1}"
    op = f"{op_job}O2"
    updates = [
        ("u001", f"Machine {cost_machine}'s public cost_rate increases by {1 + idx}. All machines remain available and operation eligibility is unchanged.", {"type": "machine_cost_rate_delta", "machine": cost_machine, "delta": 1 + idx, "memory_key": "last_cost_machine"}),
        ("u002", f"{urgent} becomes urgent: set priority to 3 and tighten its soft due time by 5 minutes. This changes tardiness cost but not schedule feasibility.", {"type": "priority_due_tighten", "job": urgent, "priority": 3, "due_delta": -5}),
        (
            "u003",
            "For the job that became urgent in the previous note, increase its priority by 1 again; infer the job from memory.",
            {"type": "memory_urgent_priority_delta", "delta": 1, "memory_reference_chain": ["u002 urgent job", "urgent job -> priority update"]},
        ),
        (
            "u004",
            "The same urgent job receives a 3-minute soft due-time relaxation after customer confirmation; infer the job from memory.",
            {"type": "memory_urgent_due_delta", "delta": 3, "memory_reference_chain": ["u003 urgent priority update", "same urgent job -> due-time patch"]},
        ),
        (
            "u005",
            "The machine from the earliest cost-rate note receives another 0.5 cost-rate increase; infer it from memory and re-optimize the same feasible schedule space.",
            {"type": "memory_machine_cost_rate_delta", "delta": 0.5, "memory_reference_chain": ["u001 cost-rate machine"]},
        ),
    ]
    if idx == 0:
        updates.extend(_maintenance_updates(6, 9))
        updates.append(
            (
                "u010",
                (
                    "A late accounting reset flips the scalar objective. All machines remain available, and due dates remain soft diagnostics. "
                    "Stop minimizing tardiness; the new objective is to minimize total machine operating cost using each machine's public cost_rate. "
                    "This may prefer cheaper machines even when the schedule finishes later."
                ),
                {
                    "type": "compound",
                    "changes": [
                        {"type": "machine_cost_rate_delta", "machine": "M1", "delta": -0.5},
                        {"type": "machine_cost_rate_delta", "machine": "M3", "delta": 1.5},
                        {"type": "objective_mode", "mode": "minimize_machine_cost"},
                    ],
                    "difficulty": "large",
                },
            )
        )
    return updates


def _maintenance_updates(start: int, end: int) -> list[tuple[str, str, dict[str, Any]]]:
    texts = {
        6: "Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.",
        7: "No new business event is reported for this stage. Refresh the solution against the current public tables.",
        8: "Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.",
        9: "The planning team wants the current public contract revalidated without adding new constraints.",
    }
    return [
        (f"u{stage:03d}", texts.get(stage, "Run a maintenance update with the same public contract."), {"type": "no_op", "reason": "maintenance review"})
        for stage in range(start, end + 1)
    ]


def _public_problem(state: dict[str, Any]) -> str:
    jobs = []
    for job in state["jobs"]:
        ops = ", ".join(f"{op['id']} can run on {'/'.join(op['eligible_machines'])} for {op['processing_time']} minutes" for op in job["operations"])
        jobs.append(f"{job['id']} due {job['due_time']}: {ops}")
    return "Schedule flexible job-shop operations with precedence and machine capacity constraints. Minimize weighted tardiness; report disruption only as an auxiliary diagnostic. " + " ".join(jobs)


def _resolve_memory_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if delta["type"] == "memory_machine_downtime":
        return {"type": "machine_downtime", "machine": state.get("memory_refs", {}).get("last_down_machine"), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_urgent_middle_processing_time":
        job = state.get("memory_refs", {}).get("last_urgent_job")
        return {"type": "processing_time_delta", "operation": f"{job}O2" if job else None, "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_follow_buffered_job_processing_time":
        job = state.get("memory_refs", {}).get("last_buffered_job")
        return {"type": "processing_time_delta", "operation": f"{job}O3" if job else None, "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_urgent_priority_delta":
        return {"type": "priority_delta", "job": state.get("memory_refs", {}).get("last_urgent_job"), "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_urgent_due_delta":
        return {"type": "due_delta", "job": state.get("memory_refs", {}).get("last_urgent_job"), "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_machine_cost_rate_delta":
        return {"type": "machine_cost_rate_delta", "machine": state.get("memory_refs", {}).get("last_cost_machine"), "delta": float(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    return delta


def _apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    typ = delta["type"]
    if typ == "no_op":
        return state
    if typ == "machine_downtime":
        machine = delta.get("machine")
        state["machine_downtime"] = [machine] if machine else []
        if machine:
            state.setdefault("memory_refs", {})["last_down_machine"] = machine
    elif typ == "machine_cost_rate_delta":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["cost_rate"] = round(float(machine.get("cost_rate", 1.0)) + float(delta["delta"]), 3)
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["machine"]
    elif typ == "priority_due_tighten":
        for job in state["jobs"]:
            if job["id"] == delta["job"]:
                job["priority"] = int(delta["priority"])
                job["due_time"] += int(delta["due_delta"])
                state.setdefault("memory_refs", {})["last_urgent_job"] = delta["job"]
    elif typ == "priority_delta":
        for job in state["jobs"]:
            if job["id"] == delta.get("job"):
                job["priority"] = int(job.get("priority", 1)) + int(delta["delta"])
    elif typ == "due_delta":
        for job in state["jobs"]:
            if job["id"] == delta.get("job"):
                job["due_time"] += int(delta["delta"])
    elif typ == "processing_time_delta":
        for op in _ops(state):
            if op["id"] == delta["operation"]:
                op["processing_time"] += int(delta["delta"])
                job_id = str(op["id"]).split("O")[0]
                state.setdefault("memory_refs", {})["last_buffered_job"] = job_id
                state.setdefault("memory_refs", {})["last_buffered_operation"] = op["id"]
    elif typ == "objective_mode":
        state["objective_mode"] = str(delta["mode"])
    elif typ == "compound":
        for change in delta.get("changes", []):
            state = _apply_delta(state, change)
    return state


def _reference_row(update_id: str, state: dict[str, Any], delta: dict[str, Any] | None, previous_solution: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    solution, metrics = solve_fjsp_exact(state, previous_solution)
    elapsed = time.perf_counter() - started
    objective = metrics["objective"]
    row = {
        "update_id": update_id,
        "reference_type": "exact_optimum",
        "objective": objective,
        "lower_bound": objective,
        "upper_bound": objective,
        "gap": 0.0,
        "solver": "depth_first_branch_search",
        "solver_time_seconds": elapsed,
        "hidden_delta_type": (delta or {}).get("type"),
        "tardiness": metrics["tardiness"],
        "disruption": metrics["disruption"],
        "solution": solution,
    }
    return row, solution


def solve_fjsp_exact(state: dict[str, Any], previous_solution: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    jobs = state["jobs"]
    down = set(state.get("machine_downtime", []))
    op_by_id = {op["id"]: op for op in _ops(state)}
    remaining = {job["id"]: 0 for job in jobs}
    machine_ready = {m["id"]: 0 for m in state["machines"]}
    job_ready = {job["id"]: 0 for job in jobs}
    assignments: dict[str, dict[str, Any]] = {}
    best = {"objective": math.inf, "solution": None, "tardiness": math.inf, "disruption": math.inf}
    suffix_min = _suffix_processing_lower_bound(jobs)

    def dfs() -> None:
        if len(assignments) == len(op_by_id):
            tard = _tardiness(state, assignments)
            dis = _disruption(previous_solution, assignments)
            cost = _machine_cost(state, assignments)
            obj = _fjsp_objective_value(state, assignments, previous_solution)
            if obj < best["objective"]:
                best.update({"objective": obj, "solution": copy.deepcopy(assignments), "tardiness": tard, "disruption": dis, "machine_cost": cost})
            return
        if str(state.get("objective_mode", "minimize_tardiness")) == "minimize_machine_cost":
            lb = _partial_machine_cost_lb(state, assignments, remaining)
        else:
            lb = _partial_tardiness_lb(state, assignments, job_ready, remaining, suffix_min)
        if lb >= best["objective"]:
            return
        candidates = []
        for job in jobs:
            idx = remaining[job["id"]]
            if idx >= len(job["operations"]):
                continue
            op = job["operations"][idx]
            for machine in op["eligible_machines"]:
                if machine in down:
                    continue
                start = max(job_ready[job["id"]], machine_ready[machine])
                end = start + int(op["processing_time"])
                candidates.append((end, start, job, op, machine))
        candidates.sort(key=lambda x: (x[0], x[1], x[3]["id"], x[4]))
        for _, start, job, op, machine in candidates:
            end = start + int(op["processing_time"])
            old_job_ready = job_ready[job["id"]]
            old_machine_ready = machine_ready[machine]
            remaining[job["id"]] += 1
            job_ready[job["id"]] = end
            machine_ready[machine] = end
            assignments[op["id"]] = {"job": job["id"], "machine": machine, "start": start, "end": end}
            dfs()
            assignments.pop(op["id"])
            machine_ready[machine] = old_machine_ready
            job_ready[job["id"]] = old_job_ready
            remaining[job["id"]] -= 1

    dfs()
    if best["solution"] is None:
        raise ValueError("FJSP exact solver found no feasible schedule")
    return {"assignments": best["solution"]}, {"objective": float(best["objective"]), "tardiness": float(best["tardiness"]), "disruption": float(best["disruption"])}


def _fjsp_objective_value(state: dict[str, Any], assignments: dict[str, Any], previous_solution: dict[str, Any] | None) -> float:
    if str(state.get("objective_mode", "minimize_tardiness")) == "minimize_machine_cost":
        return _machine_cost(state, assignments)
    return _tardiness(state, assignments)


def _machine_cost(state: dict[str, Any], assignments: dict[str, Any]) -> float:
    rates = {machine["id"]: float(machine.get("cost_rate", 1.0)) for machine in state.get("machines", [])}
    return float(sum((float(item.get("end", 0)) - float(item.get("start", 0))) * rates.get(item.get("machine"), 1.0) for item in assignments.values()))


def _partial_machine_cost_lb(state: dict[str, Any], assignments: dict[str, Any], remaining: dict[str, int]) -> float:
    total = _machine_cost(state, assignments)
    rates = {machine["id"]: float(machine.get("cost_rate", 1.0)) for machine in state.get("machines", [])}
    down = set(state.get("machine_downtime", []))
    for job in state.get("jobs", []):
        start_idx = int(remaining.get(job["id"], 0))
        for op in job.get("operations", [])[start_idx:]:
            eligible = [machine for machine in op.get("eligible_machines", []) if machine not in down]
            min_rate = min((rates.get(machine, 1.0) for machine in eligible), default=1.0)
            total += float(op.get("processing_time", 0)) * min_rate
    return total


def _ops(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [op for job in state["jobs"] for op in job["operations"]]


def _suffix_processing_lower_bound(jobs: list[dict[str, Any]]) -> dict[str, list[int]]:
    out = {}
    for job in jobs:
        suffix = [0] * (len(job["operations"]) + 1)
        for idx in range(len(job["operations"]) - 1, -1, -1):
            suffix[idx] = suffix[idx + 1] + int(job["operations"][idx]["processing_time"])
        out[job["id"]] = suffix
    return out


def _partial_tardiness_lb(state, assignments, job_ready, remaining, suffix_min) -> float:
    total = 0.0
    for job in state["jobs"]:
        idx = remaining[job["id"]]
        if idx >= len(job["operations"]):
            completion = max(assignments[op["id"]]["end"] for op in job["operations"])
        else:
            completion = job_ready[job["id"]] + suffix_min[job["id"]][idx]
        total += max(0, completion - int(job["due_time"])) * float(job.get("priority", 1))
    return total


def _partial_disruption_lb(previous_solution, assignments) -> float:
    if not previous_solution:
        return 0.0
    prev = previous_solution.get("assignments", {})
    changed = 0
    for op_id, item in assignments.items():
        old = prev.get(op_id, {})
        if old and old.get("machine") != item.get("machine"):
            changed += 1
    return float(changed)


def _tardiness(state: dict[str, Any], assignments: dict[str, Any]) -> float:
    total = 0.0
    for job in state["jobs"]:
        completion = max(assignments[op["id"]]["end"] for op in job["operations"])
        total += max(0, completion - int(job["due_time"])) * float(job.get("priority", 1))
    return total


def _disruption(previous_solution: dict[str, Any] | None, assignments: dict[str, Any]) -> float:
    if not previous_solution:
        return 0.0
    prev = previous_solution.get("assignments", {})
    changed = 0
    for op_id, item in assignments.items():
        old = prev.get(op_id, {})
        if old.get("machine") != item.get("machine") or old.get("start") != item.get("start"):
            changed += 1
    return float(changed)


def _event_types(delta: dict[str, Any]) -> list[str]:
    typ = delta["type"]
    if typ == "compound":
        events = []
        for change in delta.get("changes", []):
            events.extend(_event_types(change))
        return list(dict.fromkeys(events))
    if typ == "no_op":
        return ["re_evaluate_current_state"]
    if typ == "machine_downtime":
        return ["legacy_resource_availability_change"]
    if typ == "machine_cost_rate_delta":
        return ["cost_change", "objective_change"]
    if typ == "objective_mode":
        return ["objective_change"]
    if typ == "priority_due_tighten":
        return ["urgent_order", "objective_change"]
    if typ == "processing_time_delta":
        return ["legacy_processing_time_change"]
    if typ in {"priority_delta", "due_delta"}:
        return ["objective_change"]
    return ["objective_change"]


def _expected_event_types(original: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    events = _event_types(resolved)
    if original["type"].startswith("memory") and "memory_reference" not in events:
        return ["memory_reference", *events]
    return events


def _difficulty(delta: dict[str, Any]) -> str:
    if str(delta.get("difficulty") or "") in {"maintenance", "memory", "constraint", "large", "objective", "simple"}:
        return str(delta["difficulty"])
    if delta["type"] == "no_op":
        return "maintenance"
    if delta["type"] == "compound":
        return "objective"
    if delta["type"].startswith("memory"):
        return "memory"
    if delta["type"] == "processing_time_delta":
        return "legacy_constraint"
    if delta["type"] in {"machine_cost_rate_delta", "priority_delta", "due_delta", "priority_due_tighten"}:
        return "objective"
    return "simple"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize FJSP Exact-Small episodes with exact reference trajectories.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/fjsp_exact_small_6episodes.jsonl")
    args = parser.parse_args()
    episodes = build_episodes()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(episodes), "updates": sum(len(ep["update_stream"]) for ep in episodes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
