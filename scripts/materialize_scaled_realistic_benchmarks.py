#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import materialize_cloud_scheduling_mo as cloud
from scripts import materialize_green_vrp_mo as green
from scripts import materialize_inrc_realistic_dynamic as inrc


PROFILES = {
    "small": {
        "episodes": 6,
        "cloud_machines": 8,
        "cloud_jobs": 24,
        "cloud_burst": 6,
        "green_vehicles": 5,
        "green_orders": 18,
        "inrc_nurses": 50,
        "inrc_days": 7,
        "inrc_coverage_factor": 1.0,
    },
    "medium": {
        "episodes": 12,
        "cloud_machines": 24,
        "cloud_jobs": 120,
        "cloud_burst": 16,
        "green_vehicles": 18,
        "green_orders": 120,
        "inrc_nurses": 180,
        "inrc_days": 14,
        "inrc_coverage_factor": 2.3,
    },
    "middle": {
        "episodes": 12,
        "cloud_machines": 24,
        "cloud_jobs": 120,
        "cloud_burst": 16,
        "green_vehicles": 18,
        "green_orders": 120,
        "inrc_nurses": 180,
        "inrc_days": 14,
        "inrc_coverage_factor": 2.3,
    },
    "large": {
        "episodes": 24,
        "cloud_machines": 64,
        "cloud_jobs": 420,
        "cloud_burst": 48,
        "green_vehicles": 45,
        "green_orders": 520,
        "inrc_nurses": 700,
        "inrc_days": 28,
        "inrc_coverage_factor": 7.5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize scaled realistic dynamic benchmarks.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="medium")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-dir", default="data/evo2_dynoptbench/scaled_realistic")
    parser.add_argument("--skip-reference", action="store_true", help="Only build public/hidden streams; omit constructive references for very large instances.")
    args = parser.parse_args()

    cfg = dict(PROFILES[args.profile])
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    out_dir = Path(args.output_dir) / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for name, builder in {
        "cloud_scheduling_mo": build_cloud_episodes,
        "green_vrp_mo": build_green_episodes,
        "inrc_realistic_dynamic": build_inrc_episodes,
    }.items():
        episodes = builder(cfg, args.skip_reference)
        path = out_dir / f"{name}_{args.profile}_{cfg['episodes']}episodes.jsonl"
        write_jsonl(path, episodes)
        outputs.append(summarize_file(path, episodes))

    summary = {"profile": args.profile, "config": cfg, "outputs": outputs}
    summary_path = out_dir / "materialization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def summarize_file(path: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file": str(path),
        "episodes": len(episodes),
        "updates": sum(len(ep.get("update_stream", [])) for ep in episodes),
        "stages": sum(len(ep.get("evaluation", {}).get("reference_trajectory", [])) for ep in episodes),
        "memory_updates": sum(1 for ep in episodes for item in ep.get("hidden_update_oracle", []) if item.get("requires_memory")),
        "domain": sorted({ep.get("domain") for ep in episodes}),
    }


def build_cloud_episodes(cfg: dict[str, Any], skip_reference: bool) -> list[dict[str, Any]]:
    return [_cloud_episode(idx, cfg, skip_reference) for idx in range(int(cfg["episodes"]))]


def _cloud_episode(idx: int, cfg: dict[str, Any], skip_reference: bool) -> dict[str, Any]:
    initial = _scaled_cloud_initial(idx, int(cfg["cloud_machines"]), int(cfg["cloud_jobs"]))
    updates = cloud._updates(idx)
    updates = _scale_cloud_updates(idx, updates, int(cfg["cloud_burst"]))
    return _episode_from_parts(
        module=cloud,
        episode_id=f"cloud_scheduling_mo_scaled_{idx:03d}",
        benchmark="Dynamic-Cloud-Scheduling-MO-Scaled",
        domain="cloud_scheduling_multiobjective",
        source_instance_id=f"cloud_scaled_{idx:03d}",
        initial=initial,
        updates=updates,
        public_problem=cloud._public_problem(initial),
        evaluation_template={
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
            "constraints": ["sla_violations"],
            "auxiliary_metrics": ["latency", "migration_disruption", "sla_violations"],
            "reference_policy": "constructive_pareto_solver_best_known_scaled",
            "objective_definition": "Minimize energy and load imbalance under current-state SLA/resource feasibility; migration is diagnostic only.",
        },
        allowed_solvers=["ga_or_moea", "linear_program_solver_v1", "generated_repair_operator"],
        skip_reference=skip_reference,
    )


def _scaled_cloud_initial(idx: int, machine_count: int, job_count: int) -> dict[str, Any]:
    state = cloud._initial_state(idx)
    machines = []
    for k in range(1, machine_count + 1):
        cpu = 34 + 4 * ((k + idx) % 5)
        mem = 72 + 8 * ((k + 2 * idx) % 6)
        machines.append(
            {
                "id": f"M{k:03d}",
                "cpu": cpu,
                "mem": mem,
                "energy_idle": round(7.5 + 0.45 * (k % 7), 2),
                "energy_per_cpu": round(0.16 + 0.018 * ((k + idx) % 6), 3),
                "available": True,
                "gpu": k % 6 in {0, 2},
            }
        )
    jobs = []
    for k in range(1, job_count + 1):
        jobs.append(
            {
                "id": f"J{k:04d}",
                "cpu": 2 + ((k * 5 + idx) % 11),
                "mem": 4 + ((k * 7 + idx * 2) % 26),
                "deadline": 65 + ((k * 11 + idx * 3) % 160),
                "latency_sensitivity": round(0.65 + 0.12 * ((k + idx) % 7), 2),
                "priority": 1 + ((k + idx) % 5),
                "gpu_required": k % 17 == 0,
                "active": True,
            }
        )
    state.update({"machines": machines, "jobs": jobs, "memory_refs": {}})
    return state


def _scale_cloud_updates(idx: int, updates: list[tuple[str, str, dict[str, Any]]], burst_size: int):
    out = []
    for uid, text, delta in updates:
        delta = copy.deepcopy(delta)
        if delta.get("type") == "new_job_burst":
            delta["jobs"] = cloud._burst_jobs(idx, delta["burst"], burst_size)
        if delta.get("type") == "machine_capacity_drop":
            delta["machine"] = f"M{2 + idx % 12:03d}"
            text = f"Machine {delta['machine']} loses 40 percent CPU capacity because of thermal throttling."
        if delta.get("type") == "deadline_delta":
            delta["job"] = f"J{5 + idx:04d}"
            text = f"Deadline for job {delta['job']} is tightened by 35 minutes after an SLA escalation."
        if delta.get("type") == "machine_unavailable":
            delta["machine"] = f"M{5 + idx % 10:03d}"
            text = f"GPU capacity becomes scarce: machine {delta['machine']} goes into maintenance and cannot host new GPU-required jobs."
        if delta.get("type") == "new_job":
            delta["job"] = cloud._single_job(idx, f"JX{40 + idx:04d}", cpu=12, mem=48, priority=2)
            text = f"A batch analytics job {delta['job']['id']} arrives with high memory demand but flexible latency."
        out.append((uid, text, delta))
    return out


def build_green_episodes(cfg: dict[str, Any], skip_reference: bool) -> list[dict[str, Any]]:
    return [_green_episode(idx, cfg, skip_reference) for idx in range(int(cfg["episodes"]))]


def _green_episode(idx: int, cfg: dict[str, Any], skip_reference: bool) -> dict[str, Any]:
    initial = _scaled_green_initial(idx, int(cfg["green_vehicles"]), int(cfg["green_orders"]))
    updates = _scale_green_updates(idx, green._updates(idx))
    return _episode_from_parts(
        module=green,
        episode_id=f"green_vrp_mo_scaled_{idx:03d}",
        benchmark="Green-Dynamic-VRP-MO-Scaled",
        domain="green_vrp_multiobjective",
        source_instance_id=f"green_scaled_{idx:03d}",
        initial=initial,
        updates=updates,
        public_problem=green._public_problem(initial),
        evaluation_template={
            "metrics": ["feasibility", "pareto_coverage", "hypervolume", "igd", "distance", "lateness", "emission", "disruption", "priority_penalty", "token_cost", "latency"],
            "objectives": ["distance", "lateness", "emission"],
            "auxiliary_metrics": ["disruption", "priority_penalty"],
            "reference_policy": "constructive_pareto_solver_best_known_scaled",
            "objective_definition": "Minimize distance, lateness, and emissions while preserving route feasibility and stability.",
        },
        allowed_solvers=["ga_or_moea", "generated_repair_operator", "linear_program_solver_v1"],
        skip_reference=skip_reference,
    )


def _scaled_green_initial(idx: int, vehicle_count: int, order_count: int) -> dict[str, Any]:
    state = green._initial_state(idx)
    vehicles = [
        {
            "id": f"V{k:03d}",
            "capacity": 30 + 3 * ((k + idx) % 5),
            "emission_rate": round(0.55 + 0.045 * ((k + idx) % 8), 3),
            "available": True,
            "shift_end": 360 + 10 * (k % 4),
        }
        for k in range(1, vehicle_count + 1)
    ]
    orders = []
    for k in range(1, order_count + 1):
        ready = 10 + ((k * 9 + idx * 3) % 170)
        orders.append(
            {
                "id": f"O{k:04d}",
                "x": float(3 + ((k * 11 + idx * 7) % 95)),
                "y": float(3 + ((k * 13 + idx * 5) % 90)),
                "demand": 1 + ((k + idx) % 6),
                "ready": float(ready),
                "due": float(ready + 45 + ((k * 5 + idx) % 90)),
                "service": 3.0 + (k % 5),
                "priority": 1 + ((k + idx) % 4),
                "active": True,
            }
        )
    state.update({"vehicles": vehicles, "orders": orders, "memory_refs": {}})
    return state


def _scale_green_updates(idx: int, updates: list[tuple[str, str, dict[str, Any]]]):
    out = []
    for uid, text, delta in updates:
        delta = copy.deepcopy(delta)
        if delta.get("type") == "traffic_delay":
            delta["order"] = f"O{4 + idx:04d}"
            text = f"Traffic around order {delta['order']} slows vehicles by 25 percent; remember this customer as the congestion anchor."
        if delta.get("type") == "vehicle_unavailable":
            delta["vehicle"] = f"V{1 + idx % 8:03d}"
            text = f"Vehicle {delta['vehicle']} becomes unavailable for the rest of the horizon. Reuse other routes with minimal disruption."
        if delta.get("type") == "cancel_order":
            delta["order"] = f"O{8 + idx:04d}"
            text = f"Order {delta['order']} is cancelled by the customer. Remove it without disturbing unrelated deliveries."
        out.append((uid, text, delta))
    return out


def build_inrc_episodes(cfg: dict[str, Any], skip_reference: bool) -> list[dict[str, Any]]:
    return [_inrc_episode(idx, cfg, skip_reference) for idx in range(int(cfg["episodes"]))]


def _inrc_episode(idx: int, cfg: dict[str, Any], skip_reference: bool) -> dict[str, Any]:
    initial = _scaled_inrc_initial(idx, int(cfg["inrc_nurses"]), int(cfg["inrc_days"]), float(cfg["inrc_coverage_factor"]))
    updates = _scale_inrc_updates(idx, inrc._updates(idx), initial["days"])
    return _episode_from_parts(
        module=inrc,
        episode_id=f"inrc_realistic_dynamic_scaled_{idx:03d}",
        benchmark="INRC-II-Dynamic-Rostering-Scaled",
        domain="inrc2",
        source_instance_id=f"inrc_scaled_{idx:03d}",
        initial=initial,
        updates=updates,
        public_problem=inrc._public_problem(initial),
        evaluation_template={
            "metrics": ["feasibility", "solver_best_known_score", "coverage_shortage", "fairness", "preference_penalty", "sequence_penalty", "disruption", "token_cost", "latency"],
            "reference_policy": "constructive_solver_best_known_greedy_repair_scaled",
            "objective_definition": "1000*coverage_shortage + 500*absence_violation + fairness/preference/sequence/overload penalties; disruption is diagnostic only.",
        },
        allowed_solvers=["linear_program_solver_v1", "ga_or_moea", "generated_repair_operator"],
        skip_reference=skip_reference,
    )


def _scaled_inrc_initial(idx: int, nurse_count: int, day_count: int, coverage_factor: float) -> dict[str, Any]:
    state = inrc._initial_state(idx)
    days = [f"D{d}" for d in range(1, day_count + 1)]
    nurses = [
        {"id": f"N{k:04d}", "skills": ["general"] + (["icu"] if k % 5 == 0 else []), "max_shifts": max(5, int(day_count * 0.72))}
        for k in range(1, nurse_count + 1)
    ]
    coverage = {
        day: {
            "day": max(1, int(round((8 + (idx % 3)) * coverage_factor))),
            "evening": max(1, int(round((6 + (idx % 2)) * coverage_factor))),
            "night": max(1, int(round((3 + (idx % 2)) * coverage_factor))),
        }
        for day in days
    }
    state.update({"nurses": nurses, "days": days, "coverage": coverage, "unavailable": [], "prefer_off": [], "night_limit": {}, "memory_refs": {}})
    return state


def _scale_inrc_updates(idx: int, updates: list[tuple[str, str, dict[str, Any]]], days: list[str]):
    out = []
    for uid, text, delta in updates:
        delta = copy.deepcopy(delta)
        if delta.get("type") == "nurse_absence":
            nurse_num = (idx * 13 + int(uid[1:]) * 7) % 120 + 1
            delta["nurse"] = f"N{nurse_num:04d}"
            if delta.get("day") not in days:
                delta["day"] = days[min(1, len(days) - 1)]
            text = f"Nurse {delta['nurse']} is unavailable on {delta['day']}. Rebuild a feasible roster under the current public coverage and absence tables."
        if delta.get("type") == "preference_off":
            delta["nurse"] = f"N{(idx * 17 + 9) % 120 + 1:04d}"
            text = f"Nurse {delta['nurse']} requests {delta['day']} off if possible."
        out.append((uid, text, delta))
    return out


def _episode_from_parts(
    module: Any,
    episode_id: str,
    benchmark: str,
    domain: str,
    source_instance_id: str,
    initial: dict[str, Any],
    updates: list[tuple[str, str, dict[str, Any]]],
    public_problem: str,
    evaluation_template: dict[str, Any],
    allowed_solvers: list[str],
    skip_reference: bool,
) -> dict[str, Any]:
    state = copy.deepcopy(initial)
    public_updates = []
    oracle = []
    trajectory = []
    previous_solution = None
    if not skip_reference:
        row, previous_solution = module._reference_row("initial", state, None, previous_solution)
        trajectory.append(row)
    for uid, text, delta in updates:
        public_updates.append(
            {
                "update_id": uid,
                "time_index": int(uid[1:]),
                "public_update": text,
                "difficulty": module._difficulty(delta),
                "requires_memory": delta["type"].startswith("memory"),
            }
        )
        resolved = module._resolve_memory_delta(state, delta)
        oracle.append(
            {
                "update_id": uid,
                "hidden_delta": resolved,
                "expected_event_types": module._expected_event_types(delta, resolved),
                "difficulty": module._difficulty(delta),
                "requires_memory": delta["type"].startswith("memory"),
                "memory_reference_chain": delta.get("memory_reference_chain", []),
            }
        )
        state = module._apply_delta(state, resolved)
        if not skip_reference:
            row, previous_solution = module._reference_row(uid, state, resolved, previous_solution)
            trajectory.append(row)
    evaluation = copy.deepcopy(evaluation_template)
    evaluation["reference_trajectory"] = trajectory
    return {
        "episode_id": episode_id,
        "layer": "realistic_dynamic_scaled",
        "benchmark": benchmark,
        "domain": domain,
        "source_dataset": "scaled native natural-language dynamic benchmark generated from realistic templates",
        "source_instance_id": source_instance_id,
        "public_initial_problem": public_problem,
        "public_context": {
            "input_policy": "agent sees natural language and public summaries only",
            "update_count": len(updates),
            "scale": {
                "machines": len(initial.get("machines", [])),
                "jobs": len(initial.get("jobs", [])),
                "vehicles": len(initial.get("vehicles", [])),
                "orders": len(initial.get("orders", [])),
                "nurses": len(initial.get("nurses", [])),
                "days": len(initial.get("days", [])),
            },
        },
        "update_stream": public_updates,
        "hidden_initial_state": initial,
        "hidden_update_oracle": oracle,
        "evaluation": evaluation,
        "agent_allowed_solvers": allowed_solvers,
    }


if __name__ == "__main__":
    main()
