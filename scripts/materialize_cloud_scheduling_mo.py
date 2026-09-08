#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


WEIGHT_PROFILES = [
    {"name": "sla_first", "sla": 3.0, "latency": 1.6, "energy": 0.4, "balance": 0.5},
    {"name": "energy_first", "sla": 1.2, "latency": 0.7, "energy": 2.3, "balance": 0.8},
    {"name": "latency_first", "sla": 1.6, "latency": 2.3, "energy": 0.6, "balance": 0.6},
    {"name": "balance_first", "sla": 1.4, "latency": 0.9, "energy": 0.8, "balance": 2.4},
    {"name": "priority_first", "sla": 2.4, "latency": 1.4, "energy": 0.6, "balance": 0.7},
]


def build_episodes() -> list[dict[str, Any]]:
    return [_episode(idx) for idx in range(6)]


def _episode(idx: int) -> dict[str, Any]:
    initial = _initial_state(idx)
    updates = _updates(idx)
    state = copy.deepcopy(initial)
    public_updates = []
    oracle = []
    trajectory = []
    previous_solution = None
    row, previous_solution = _reference_row("initial", state, None, previous_solution)
    trajectory.append(row)
    for uid, text, delta in updates:
        resolved = _resolve_memory_delta(state, delta)
        public_update = {
            "update_id": uid,
            "time_index": int(uid[1:]),
            "public_update": text,
            "difficulty": _difficulty(delta),
            "requires_memory": delta["type"].startswith("memory"),
        }
        public_patch = _public_data_patch(resolved)
        if public_patch:
            public_update["public_data_patch"] = public_patch
        public_updates.append(public_update)
        oracle.append(
            {
                "update_id": uid,
                "hidden_delta": resolved,
                "expected_event_types": _expected_event_types(delta, resolved),
                "difficulty": _difficulty(delta),
                "requires_memory": delta["type"].startswith("memory"),
                "memory_reference_chain": delta.get("memory_reference_chain", []),
            }
        )
        state = _apply_delta(state, resolved)
        row, previous_solution = _reference_row(uid, state, resolved, previous_solution)
        trajectory.append(row)
    return {
        "episode_id": f"cloud_scheduling_mo_{idx:03d}",
        "layer": "realistic_dynamic",
        "benchmark": "Dynamic-Cloud-Scheduling-MO",
        "domain": "cloud_scheduling_multiobjective",
        "source_dataset": "Google/Alibaba cluster trace inspired dynamic slice; native NL exposed to agent",
        "source_instance_id": f"cloud_mo_slice_{idx:03d}",
        "public_initial_problem": _public_problem(initial),
        "public_context": {
            "input_policy": "agent sees natural language and public summaries only",
            "update_count": len(updates),
            "multi_objective": True,
            "objective_sense": "minimize_all",
        },
        "update_stream": public_updates,
        "hidden_initial_state": initial,
        "hidden_update_oracle": oracle,
        "evaluation": {
            "metrics": [
                "feasibility",
                "pareto_coverage",
                "hypervolume",
                "igd",
                "sla_violations",
                "latency",
                "energy",
                "migration_disruption",
                "load_imbalance",
                "token_cost",
                "latency_seconds",
            ],
            "objectives": ["energy", "load_imbalance"],
            "constraints": ["resource_feasibility"],
            "auxiliary_metrics": ["latency", "migration_disruption", "sla_violations"],
            "reference_trajectory": trajectory,
            "reference_policy": "constructive_pareto_solver_best_known",
            "objective_definition": (
                "Assign the fixed active job set to available machines while respecting resource feasibility. The two Pareto objectives "
                "are energy use and load imbalance, which expose a consolidation-versus-spread trade-off. SLA violation "
                "penalty, latency, and migration disruption are reported as operational diagnostics and do not change the feasible set."
            ),
        },
        "agent_allowed_solvers": ["ga_or_moea", "linear_program_solver_v1", "generated_repair_operator"],
    }


def _initial_state(idx: int) -> dict[str, Any]:
    machines = []
    for k in range(1, 9):
        cpu = 34 + 4 * ((k + idx) % 4)
        mem = 72 + 8 * ((k + 2 * idx) % 4)
        machines.append(
            {
                "id": f"M{k:02d}",
                "cpu": cpu,
                "mem": mem,
                "energy_idle": round(8.0 + 0.8 * (k % 3), 2),
                "energy_per_cpu": round(0.18 + 0.025 * ((k + idx) % 4), 3),
                "available": True,
                "gpu": k in {2, 5, 8},
            }
        )
    jobs = []
    for k in range(1, 25):
        cpu = 2 + ((k * 5 + idx) % 9)
        mem = 4 + ((k * 7 + idx * 2) % 18)
        jobs.append(
            {
                "id": f"J{k:03d}",
                "cpu": cpu,
                "mem": mem,
                "deadline": 70 + ((k * 11 + idx * 3) % 120),
                "latency_sensitivity": round(0.7 + 0.15 * ((k + idx) % 5), 2),
                "priority": 1 + ((k + idx) % 4),
                "gpu_required": k % 11 == 0,
                "active": True,
            }
        )
    for job in _future_failover_jobs(idx):
        inactive = dict(job)
        inactive["active"] = False
        jobs.append(inactive)
    jobs.extend(_late_candidate_jobs(idx))
    return {
        "machines": machines,
        "jobs": jobs,
        "energy_price": 1.0,
        "carbon_intensity": 1.0,
        "memory_refs": {},
    }


def _updates(idx: int) -> list[tuple[str, str, dict[str, Any]]]:
    burst_name = f"burst-{idx}"
    machine = f"M{2 + idx % 5:02d}"
    maintenance_machine = f"M{8 - idx % 3:02d}"
    bottleneck_machine = f"M{5 + idx % 3:02d}"
    job = f"J{5 + idx:03d}"
    burst_jobs = [f"J{k:03d}" for k in range(1 + idx, 6 + idx)]
    analytics_job = f"J{18 + idx:03d}"
    reset_jobs = [f"J{k:03d}" for k in range(10 + idx, 15 + idx)]
    failover_burst = f"failover-{idx}"
    failover_jobs = _future_failover_jobs(idx)
    updates = [
        (
            "u001",
            (
                f"A latency-sensitive workload group {burst_name} is identified among existing jobs {', '.join(burst_jobs)}. "
                "Increase their public priority by 1 and remember this group name for later updates. The job table remains fixed."
            ),
            {"type": "priority_delta_jobs", "burst": burst_name, "jobs": burst_jobs, "delta": 1, "memory_key": "last_burst_jobs"},
        ),
        (
            "u002",
            f"Machine {machine}'s energy_per_cpu increases by 0.055 because of thermal throttling, but its CPU and memory capacity stay unchanged.",
            {"type": "machine_energy_per_cpu_delta", "machine": machine, "delta": 0.055, "memory_key": "last_thermal_machine"},
        ),
        (
            "u003",
            f"Soft SLA target for job {job} is tightened by 35 minutes after an escalation. This affects latency penalty, not hard feasibility.",
            {"type": "deadline_delta", "job": job, "delta": -35},
        ),
        (
            "u004",
            "Energy price doubles for the next control window, so energy cost becomes much more important.",
            {"type": "energy_price", "factor": 2.0},
        ),
        (
            "u005",
            "The workload burst mentioned at the beginning now gets premium priority 5; infer the affected jobs from memory.",
            {"type": "memory_prioritize_burst", "priority": 5, "memory_reference_chain": ["u001 workload group name", "group name -> affected jobs", "affected jobs -> priority patch"]},
        ),
        (
            "u006",
            (
                f"A carbon-aware operating window starts: machine {maintenance_machine}'s energy_idle increases by 1.6, "
                "carbon_intensity rises to 1.80, and energy_price is set to 1.25. "
                "This changes the energy-versus-balance trade-off without changing the machine set or capacities."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "machine_energy_idle_delta", "machine": maintenance_machine, "delta": 1.6},
                    {"type": "carbon_intensity", "factor": 1.80},
                    {"type": "energy_price", "factor": 1.25},
                ],
            },
        ),
        (
            "u007",
            (
                f"Existing analytics-like job {analytics_job} becomes less urgent: reduce its priority by 1 and extend its soft SLA target by 25 minutes. "
                "The active job set remains fixed."
            ),
            {"type": "compound", "changes": [{"type": "priority_delta", "job": analytics_job, "delta": -1}, {"type": "deadline_delta", "job": analytics_job, "delta": 25}]},
        ),
        (
            "u008",
            (
                "A balancing directive follows the scarcity window: energy_price drops to 0.70 while carbon_intensity remains elevated at 1.35. "
                "The plan should now expose better load balance instead of simply minimizing energy."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "energy_price", "factor": 0.70},
                    {"type": "carbon_intensity", "factor": 1.35},
                ],
            },
        ),
        (
            "u009",
            f"Operations relaxes the analytics job {analytics_job} by extending its soft SLA target by 30 minutes, and carbon_intensity is set to 1.10.",
            {
                "type": "compound",
                "changes": [
                    {"type": "deadline_delta", "job": analytics_job, "delta": 30},
                    {"type": "carbon_intensity", "factor": 1.10},
                ],
            },
        ),
        (
            "u010",
            (
                f"A late reset changes energy and service priorities over the same jobs. Machine {machine}'s energy_per_cpu is reduced by 0.04; "
                f"machine {bottleneck_machine}'s energy_idle increases by 2.2; "
                "energy_price is set to 0.60 and carbon_intensity to 0.85. "
                f"Existing jobs {', '.join(reset_jobs)} receive priority +1 and soft SLA targets tightened by 20 minutes. "
                "The objective trade-off has moved, but current feasibility is still judged over the same fixed public resource and job set."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "machine_energy_per_cpu_delta", "machine": machine, "delta": -0.04},
                    {"type": "machine_energy_idle_delta", "machine": bottleneck_machine, "delta": 2.2},
                    {"type": "energy_price", "factor": 0.60},
                    {"type": "carbon_intensity", "factor": 0.85},
                    {"type": "priority_delta_jobs", "jobs": reset_jobs, "delta": 1},
                    {"type": "deadline_delta_jobs", "jobs": reset_jobs, "delta": -20},
                ],
            },
        ),
        _late_cloud_regime_update(idx, 11),
        _late_cloud_regime_update(idx, 12),
    ]
    return updates


def _late_cloud_regime_update(idx: int, stage_index: int) -> tuple[str, str, dict[str, Any]]:
    regime = _cloud_regime(idx, stage_index)
    label = "A" if stage_index == 11 else "B"
    text = (
        f"A cluster-wide hardware and workload reset activates public regime {label}. Activate the prelisted late-window jobs, replace every machine's CPU, "
        "memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job's CPU, memory, "
        "GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain "
        "unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy_price and "
        "carbon_intensity by the supplied current values."
    )
    return f"u{stage_index:03d}", text, {"type": "cloud_regime_set", **regime}


def _cloud_regime(idx: int, stage_index: int) -> dict[str, Any]:
    machines: list[dict[str, Any]] = []
    high_group = set(range(5, 9)) if stage_index == 11 else set(range(1, 5))
    gpu_group = {5, 7, 8} if stage_index == 11 else {1, 2, 4}
    for k in range(1, 9):
        high = k in high_group
        phase = (k * (3 if stage_index == 11 else 5) + idx) % 8
        machines.append(
            {
                "id": f"M{k:02d}",
                "cpu": (150 + 10 * (phase % 3)) if high else (24 + 4 * (phase % 3)),
                "mem": (330 + 24 * (phase % 3)) if high else (62 + 8 * (phase % 3)),
                "energy_idle": round((5.0 + 1.1 * phase) if high else (17.0 + 1.4 * phase), 3),
                "energy_per_cpu": round((0.09 + 0.035 * phase) if high else (0.48 + 0.045 * phase), 3),
                "available": True,
                "gpu": k in gpu_group,
            }
        )
    jobs: list[dict[str, Any]] = []
    multiplier = 5 if stage_index == 11 else 7
    offset = (idx * 7 + stage_index) % 24
    job_ids = [f"J{k:03d}" for k in range(1, 25)] + [f"L{k:03d}" for k in range(1, 41)]
    for k, job_id in enumerate(job_ids, start=1):
        permuted = ((k - 1) * multiplier + offset) % len(job_ids)
        jobs.append(
            {
                "id": job_id,
                "cpu": 3 + ((permuted * 5 + idx) % 12),
                "mem": 7 + ((permuted * 9 + idx * 3) % 28),
                "deadline": 32 + ((len(job_ids) - 1 - permuted) * 7 + k * 3 + idx) % 210,
                "latency_sensitivity": round(0.65 + 0.22 * ((permuted + idx) % 7), 3),
                "priority": 1 + ((3 * permuted + idx + stage_index) % 5),
                "gpu_required": (permuted + idx + stage_index) % 5 == 0,
                "active": True,
            }
        )
    return {
        "machines": machines,
        "jobs": jobs,
        "energy_price": 3.4 if stage_index == 11 else 0.48,
        "carbon_intensity": 2.6 if stage_index == 11 else 0.58,
        "regime_id": f"late-{stage_index}-p{idx:02d}",
    }


def _burst_jobs(idx: int, burst: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"B{idx:02d}_{k:02d}",
            "burst": burst,
            "cpu": 2 + ((idx + k) % 4),
            "mem": 5 + ((idx * 2 + k * 3) % 10),
            "deadline": 45 + 4 * k + idx,
            "latency_sensitivity": 1.7,
            "priority": 4,
            "gpu_required": False,
            "active": True,
        }
        for k in range(1, count + 1)
    ]


def _future_failover_jobs(idx: int) -> list[dict[str, Any]]:
    return _burst_jobs(idx + 20, f"failover-{idx}", 6)


def _late_candidate_jobs(idx: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"L{k:03d}",
            "burst": f"late-window-{idx}",
            "cpu": 3 + ((k * 5 + idx) % 12),
            "mem": 7 + ((k * 9 + idx * 3) % 28),
            "deadline": 80 + ((k * 11 + idx) % 160),
            "latency_sensitivity": round(0.7 + 0.15 * ((k + idx) % 5), 2),
            "priority": 1 + ((k + idx) % 4),
            "gpu_required": False,
            "active": False,
        }
        for k in range(1, 41)
    ]


def _single_job(idx: int, job_id: str, cpu: int, mem: int, priority: int) -> dict[str, Any]:
    return {
        "id": job_id,
        "cpu": cpu,
        "mem": mem + idx % 3,
        "deadline": 190 + idx * 4,
        "latency_sensitivity": 0.8,
        "priority": priority,
        "gpu_required": False,
        "active": True,
    }


def _format_job_rows(jobs: list[dict[str, Any]]) -> str:
    return "; ".join(_format_job_row(job) for job in jobs)


def _format_job_row(job: dict[str, Any]) -> str:
    return (
        f"{job['id']} cpu {job['cpu']} mem {job['mem']} deadline {job['deadline']} "
        f"latency_sensitivity {job['latency_sensitivity']} priority {job['priority']} "
        f"gpu_required {str(job['gpu_required']).lower()} active {str(job['active']).lower()}"
    )


def _public_problem(state: dict[str, Any]) -> str:
    machine_text = ", ".join(
        f"{m['id']} cpu {m['cpu']} mem {m['mem']} gpu {str(m['gpu']).lower()}" for m in state["machines"]
    )
    job_text = ", ".join(
        f"{j['id']} cpu {j['cpu']} mem {j['mem']} deadline {j['deadline']} priority {j['priority']}"
        for j in state["jobs"]
    )
    return (
        "Place active cloud jobs onto available machines for the next scheduling window. "
        f"Machines: {machine_text}. Jobs: {job_text}. "
        "Jobs with active=false are public candidate jobs that are not scheduled until a later natural-language update explicitly activates them. "
        "Respect CPU, memory, GPU needs, and machine availability. Optimize two main criteria: energy use "
        "and load imbalance. Also report SLA violations, latency, and migrations as diagnostics; these diagnostics do not define feasibility."
    )


def _resolve_memory_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if delta["type"] == "memory_prioritize_burst":
        return {"type": "priority_set_jobs", "jobs": state.get("memory_refs", {}).get("last_burst_jobs", []), "priority": delta["priority"], "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_restore_machine":
        return {
            "type": "restore_machine",
            "machine": state.get("memory_refs", {}).get("last_throttled_machine"),
            "cpu_factor": delta["cpu_factor"],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    return delta


def _apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    typ = delta["type"]
    if typ == "no_op":
        return state
    if typ == "cloud_regime_set":
        machine_by_id = {row["id"]: copy.deepcopy(row) for row in delta.get("machines", [])}
        for machine in state.get("machines", []):
            replacement = machine_by_id.get(machine.get("id"))
            if replacement is not None:
                machine.update(replacement)
        job_by_id = {row["id"]: copy.deepcopy(row) for row in delta.get("jobs", [])}
        for job in state.get("jobs", []):
            replacement = job_by_id.get(job.get("id"))
            if replacement is not None:
                job.update(replacement)
        state["energy_price"] = float(delta["energy_price"])
        state["carbon_intensity"] = float(delta["carbon_intensity"])
    elif typ == "new_job_burst":
        existing = {job.get("id"): job for job in state["jobs"]}
        for incoming_raw in delta.get("jobs", []):
            incoming = copy.deepcopy(incoming_raw)
            incoming["active"] = True
            current = existing.get(incoming.get("id"))
            if current is None:
                state["jobs"].append(incoming)
                existing[incoming.get("id")] = incoming
            else:
                current.update(incoming)
        state.setdefault("memory_refs", {})["last_burst"] = delta["burst"]
    elif typ == "priority_delta_jobs":
        targets = set(delta.get("jobs") or [])
        for job in state["jobs"]:
            if job["id"] in targets:
                job["priority"] = max(1, int(job.get("priority", 1)) + int(delta["delta"]))
        if delta.get("memory_key"):
            state.setdefault("memory_refs", {})[str(delta["memory_key"])] = list(delta.get("jobs") or [])
    elif typ == "priority_set_jobs":
        targets = set(delta.get("jobs") or [])
        for job in state["jobs"]:
            if job["id"] in targets:
                job["priority"] = int(delta["priority"])
    elif typ == "deadline_delta_jobs":
        targets = set(delta.get("jobs") or [])
        for job in state["jobs"]:
            if job["id"] in targets:
                job["deadline"] = max(1, int(job["deadline"] + delta["delta"]))
    elif typ == "machine_capacity_drop":
        for machine in state["machines"]:
            if machine["id"] == delta["machine"]:
                machine.setdefault("base_cpu", machine["cpu"])
                machine["cpu"] = max(1, int(round(machine["base_cpu"] * float(delta["cpu_factor"]))))
                state.setdefault("memory_refs", {})["last_throttled_machine"] = delta["machine"]
    elif typ == "machine_available_set":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["available"] = bool(delta["available"])
    elif typ == "machine_capacity_set":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                if "cpu" in delta:
                    machine["cpu"] = int(delta["cpu"])
                    machine.setdefault("base_cpu", machine["cpu"])
                if "mem" in delta:
                    machine["mem"] = int(delta["mem"])
    elif typ == "machine_energy_per_cpu_set":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["energy_per_cpu"] = round(float(delta["energy_per_cpu"]), 3)
    elif typ == "machine_energy_idle_set":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["energy_idle"] = round(float(delta["energy_idle"]), 3)
    elif typ == "machine_energy_per_cpu_delta":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["energy_per_cpu"] = round(float(machine.get("energy_per_cpu", 0.0)) + float(delta["delta"]), 3)
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["machine"]
    elif typ == "machine_energy_idle_delta":
        for machine in state["machines"]:
            if machine["id"] == delta.get("machine"):
                machine["energy_idle"] = round(float(machine.get("energy_idle", 0.0)) + float(delta["delta"]), 3)
    elif typ == "deadline_delta":
        for job in state["jobs"]:
            if job["id"] == delta["job"]:
                job["deadline"] = max(1, int(job["deadline"] + delta["delta"]))
    elif typ == "priority_delta":
        for job in state["jobs"]:
            if job["id"] == delta.get("job"):
                job["priority"] = max(1, int(job.get("priority", 1)) + int(delta["delta"]))
    elif typ == "energy_price":
        state["energy_price"] = float(delta["factor"])
    elif typ == "prioritize_burst":
        for job in state["jobs"]:
            if job.get("burst") == delta.get("burst"):
                job["priority"] = int(delta["priority"])
    elif typ == "machine_unavailable":
        for machine in state["machines"]:
            if machine["id"] == delta["machine"]:
                machine["available"] = False
    elif typ == "new_job":
        incoming = copy.deepcopy(delta["job"])
        incoming["active"] = True
        for job in state["jobs"]:
            if job.get("id") == incoming.get("id"):
                job.update(incoming)
                break
        else:
            state["jobs"].append(incoming)
    elif typ == "carbon_intensity":
        state["carbon_intensity"] = float(delta["factor"])
    elif typ == "restore_machine" and delta.get("machine"):
        for machine in state["machines"]:
            if machine["id"] == delta["machine"]:
                machine["available"] = True
                base_cpu = machine.get("base_cpu", machine["cpu"])
                machine["cpu"] = int(round(base_cpu * float(delta["cpu_factor"])))
    elif typ == "compound":
        for change in delta.get("changes", []):
            state = _apply_delta(state, change)
    return state


def _reference_row(update_id: str, state: dict[str, Any], delta: dict[str, Any] | None, previous_solution: dict[str, Any] | None):
    archive = _pareto_archive(state, previous_solution)
    representative = min(archive, key=lambda item: item["scalar_score"]) if archive else None
    next_previous = representative["solution"] if representative else previous_solution
    return {
        "update_id": update_id,
        "reference_type": "pareto_solver_best_known",
        "objectives": ["energy", "load_imbalance"],
        "constraints": ["resource_feasibility"],
        "auxiliary_metrics": ["latency", "migration_disruption", "sla_violations"],
        "pareto_archive": archive,
        "archive_size": len(archive),
        "hypervolume_proxy": _hypervolume_proxy(archive),
        "solver": "deterministic_weighted_placement_pareto_proxy",
        "hidden_delta_type": (delta or {}).get("type"),
        "representative_solution": next_previous,
    }, next_previous


def _pareto_archive(state: dict[str, Any], previous_solution: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates = []
    for profile in WEIGHT_PROFILES:
        solution = _construct_placement(state, previous_solution, profile)
        objectives = _score_solution(state, solution, previous_solution)
        scalar = _scalarize(objectives, profile)
        candidates.append({"profile": profile["name"], "solution": solution, "objectives": objectives, "scalar_score": scalar})
    feasible = [item for item in candidates if item["objectives"].get("resource_violations", 0.0) == 0.0]
    return _nondominated(feasible or candidates)


def _construct_placement(state: dict[str, Any], previous_solution: dict[str, Any] | None, profile: dict[str, float]) -> dict[str, Any]:
    machines = [m for m in state["machines"] if m.get("available", True)]
    if profile["name"] == "energy_first":
        machines.sort(key=lambda m: (m["energy_idle"] + m["energy_per_cpu"] * m["cpu"], m["id"]))
    elif profile["name"] == "balance_first":
        machines.sort(key=lambda m: (-m["cpu"], -m["mem"], m["id"]))
    else:
        machines.sort(key=lambda m: (not m.get("gpu", False), m["energy_per_cpu"], m["id"]))
    jobs = [j for j in state["jobs"] if j.get("active", True)]
    if profile["name"] in {"sla_first", "priority_first"}:
        jobs.sort(key=lambda j: (j["deadline"], -j["priority"], -j["latency_sensitivity"]))
    elif profile["name"] == "energy_first":
        jobs.sort(key=lambda j: (j["cpu"], j["mem"], j["deadline"]))
    elif profile["name"] == "latency_first":
        jobs.sort(key=lambda j: (j["deadline"], -j["latency_sensitivity"], -j["priority"]))
    else:
        jobs.sort(key=lambda j: (-j["cpu"] - j["mem"] / 4, j["deadline"]))
    remaining = {m["id"]: {"cpu": m["cpu"], "mem": m["mem"]} for m in machines}
    assignments: dict[str, str] = {}
    for job in jobs:
        best_machine = None
        best_cost = float("inf")
        for machine in machines:
            if job.get("gpu_required") and not machine.get("gpu", False):
                continue
            rem = remaining[machine["id"]]
            if rem["cpu"] < job["cpu"] or rem["mem"] < job["mem"]:
                continue
            cpu_after = rem["cpu"] - job["cpu"]
            mem_after = rem["mem"] - job["mem"]
            utilization = 1.0 - cpu_after / max(machine["cpu"], 1)
            latency = _job_latency(job, machine, utilization)
            energy = (machine["energy_idle"] + machine["energy_per_cpu"] * job["cpu"]) * state.get("energy_price", 1.0) * state.get("carbon_intensity", 1.0)
            balance = abs(cpu_after / max(machine["cpu"], 1) - mem_after / max(machine["mem"], 1)) * 10.0
            sla = max(0.0, latency - job["deadline"]) * job["priority"]
            cost = (
                sla * profile["sla"]
                + latency * profile["latency"]
                + energy * profile["energy"]
                + balance * profile["balance"]
            )
            if cost < best_cost:
                best_machine = machine["id"]
                best_cost = cost
        if best_machine is not None:
            assignments[job["id"]] = best_machine
            remaining[best_machine]["cpu"] -= job["cpu"]
            remaining[best_machine]["mem"] -= job["mem"]
    return {"assignments": assignments}


def _score_solution(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None) -> dict[str, float]:
    job_map = {j["id"]: j for j in state["jobs"] if j.get("active", True)}
    machine_map = {m["id"]: m for m in state["machines"]}
    used_cpu = {m["id"]: 0.0 for m in state["machines"]}
    used_mem = {m["id"]: 0.0 for m in state["machines"]}
    sla = 0.0
    latency = 0.0
    energy = 0.0
    infeasible = 0.0
    for job_id, machine_id in solution.get("assignments", {}).items():
        if job_id not in job_map or machine_id not in machine_map:
            infeasible += 1.0
            continue
        job = job_map[job_id]
        machine = machine_map[machine_id]
        if not machine.get("available", True) or (job.get("gpu_required") and not machine.get("gpu", False)):
            infeasible += 1.0
        used_cpu[machine_id] += job["cpu"]
        used_mem[machine_id] += job["mem"]
        util = used_cpu[machine_id] / max(machine["cpu"], 1)
        job_latency = _job_latency(job, machine, util)
        latency += job_latency * job["latency_sensitivity"]
        sla += max(0.0, job_latency - job["deadline"]) * job["priority"]
        energy += (machine["energy_idle"] + machine["energy_per_cpu"] * used_cpu[machine_id]) * state.get("energy_price", 1.0) * state.get("carbon_intensity", 1.0)
    for machine in state["machines"]:
        if used_cpu[machine["id"]] > machine["cpu"] or used_mem[machine["id"]] > machine["mem"]:
            infeasible += 5.0
    missed = set(job_map) - set(solution.get("assignments", {}))
    sla += sum(job_map[job_id]["priority"] * 80.0 for job_id in missed)
    latency += len(missed) * 120.0
    migration = _migration_disruption(state, solution, previous_solution)
    balance = _load_imbalance(state, used_cpu)
    return {
        "resource_violations": round(infeasible + float(len(missed)), 3),
        "sla_violations": round(sla + infeasible * 100.0, 3),
        "latency": round(latency, 3),
        "energy": round(energy, 3),
        "migration_disruption": round(migration, 3),
        "load_imbalance": round(balance, 3),
    }


def _job_latency(job: dict[str, Any], machine: dict[str, Any], utilization: float) -> float:
    base = job["cpu"] * 3.0 + job["mem"] * 0.45
    gpu_penalty = 0.65 if job.get("gpu_required") and machine.get("gpu", False) else 1.0
    congestion = 1.0 + max(0.0, utilization - 0.7) * 1.8
    return base * congestion * gpu_penalty


def _migration_disruption(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None) -> float:
    if not previous_solution:
        return 0.0
    current = solution.get("assignments", {})
    previous = previous_solution.get("assignments", {})
    available_machines = {machine["id"] for machine in state.get("machines", []) if machine.get("available", True)}
    keys = set(current) & set(previous)
    return float(
        sum(
            1
            for key in keys
            if previous.get(key) in available_machines
            and current.get(key) != previous.get(key)
        )
    )


def _load_imbalance(state: dict[str, Any], used_cpu: dict[str, float]) -> float:
    available = [m for m in state["machines"] if m.get("available", True)]
    if not available:
        return 0.0
    utils = [used_cpu[m["id"]] / max(m["cpu"], 1) for m in available]
    avg = sum(utils) / len(utils)
    return sum(abs(u - avg) for u in utils) * 100.0


def _scalarize(objectives: dict[str, float], profile: dict[str, float]) -> float:
    return (
        objectives["sla_violations"] * profile["sla"]
        + objectives["latency"] * profile["latency"]
        + objectives["energy"] * profile["energy"]
        + objectives["load_imbalance"] * profile["balance"]
    )


def _nondominated(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cand in candidates:
        if not any(_dominates(other["objectives"], cand["objectives"]) for other in candidates if other is not cand):
            out.append(cand)
    seen = set()
    unique = []
    names = ["energy", "load_imbalance"]
    for cand in sorted(out, key=lambda item: item["scalar_score"]):
        key = tuple(cand["objectives"][name] for name in names)
        if key not in seen:
            unique.append(cand)
            seen.add(key)
    return unique


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    names = ["energy", "load_imbalance"]
    return all(a[name] <= b[name] for name in names) and any(a[name] < b[name] for name in names)


def _hypervolume_proxy(archive: list[dict[str, Any]]) -> float:
    if not archive:
        return 0.0
    total = 0.0
    for item in archive:
        obj = item["objectives"]
        total += 1.0 / (
            1.0
            + obj["energy"]
            + obj["load_imbalance"]
            + 1000.0 * obj.get("resource_violations", 0.0)
        )
    return round(total, 6)


def _event_types(delta: dict[str, Any]) -> list[str]:
    typ = delta["type"]
    if typ == "compound":
        events: list[str] = []
        for change in delta.get("changes", []):
            for event in _event_types(change):
                if event not in events:
                    events.append(event)
        return events or ["objective_change"]
    if typ == "cloud_regime_set":
        return ["legacy_resource_change", "legacy_demand_change", "objective_change"]
    if typ in {"new_job_burst", "new_job"}:
        return ["legacy_demand_change"]
    if typ in {"machine_capacity_drop", "machine_available_set", "machine_capacity_set", "machine_unavailable", "restore_machine"}:
        return ["legacy_resource_change"]
    if typ in {"deadline_delta", "deadline_delta_jobs", "prioritize_burst", "priority_delta", "priority_delta_jobs", "priority_set_jobs"}:
        return ["sla_change", "objective_change"]
    if typ in {"energy_price", "carbon_intensity", "machine_energy_per_cpu_delta", "machine_energy_idle_delta", "machine_energy_per_cpu_set", "machine_energy_idle_set"}:
        return ["multi_objective_weight_change", "objective_change"]
    return ["objective_change"]


def _expected_event_types(original: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    events = _event_types(resolved)
    if original["type"].startswith("memory") and "memory_reference" not in events:
        return ["memory_reference", *events]
    return events


def _difficulty(delta: dict[str, Any]) -> str:
    if delta["type"].startswith("memory"):
        return "memory"
    if delta["type"] == "cloud_regime_set":
        return "large"
    if delta["type"] in {"new_job_burst", "machine_unavailable", "machine_available_set", "machine_capacity_drop", "machine_capacity_set"}:
        return "legacy_large"
    if delta["type"] in {"compound", "energy_price", "carbon_intensity", "machine_energy_per_cpu_delta", "machine_energy_idle_delta", "machine_energy_per_cpu_set", "machine_energy_idle_set", "priority_delta", "priority_delta_jobs", "priority_set_jobs", "deadline_delta", "deadline_delta_jobs"}:
        return "objective"
    return "constraint"


def _public_data_patch(delta: dict[str, Any]) -> dict[str, Any]:
    if delta.get("type") != "cloud_regime_set":
        return {}
    operations: list[dict[str, Any]] = []
    operations.extend(
        {
            "op": "update_row",
            "path": ["tables", "machines"],
            "match": {"id": machine["id"]},
            "values": {key: value for key, value in machine.items() if key != "id"},
        }
        for machine in delta.get("machines", [])
    )
    operations.extend(
        {
            "op": "update_row",
            "path": ["tables", "jobs"],
            "match": {"id": job["id"]},
            "values": {key: value for key, value in job.items() if key != "id"},
        }
        for job in delta.get("jobs", [])
    )
    operations.extend(
        [
            {
                "op": "set_value",
                "path": ["tables", "global_params", "energy_price"],
                "value": float(delta["energy_price"]),
            },
            {
                "op": "set_value",
                "path": ["tables", "global_params", "carbon_intensity"],
                "value": float(delta["carbon_intensity"]),
            },
        ]
    )
    return {
        "operations": operations,
        "reason": f"Deterministic public table replacement for {delta.get('regime_id')}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Dynamic Cloud Scheduling multi-objective episodes.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/cloud_scheduling_mo_6episodes.jsonl")
    args = parser.parse_args()
    episodes = build_episodes()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(episodes), "updates": sum(len(ep["update_stream"]) for ep in episodes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
