#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evo2.evaluation.reference_solvers.moea_reference import GREEN_OBJECTIVES, evaluate_against_reference


GREEN_MODULE = "scripts.materialize_green_vrp_mo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single AEVO/EVO2 Green VRP run against per-stage reference archives.")
    parser.add_argument("--run", required=True, help="Episode JSONL from run_episode_suite or JSON trace from run_universal_dynamic_evo2_trace.")
    parser.add_argument("--benchmark", required=True, help="Green benchmark JSONL with hidden evaluation data/reference trajectory.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode = _load_benchmark_episode(Path(args.benchmark), args.episode_index)
    run = _load_run(Path(args.run))
    stages = _extract_run_stages(run)
    metrics = evaluate_run(run, episode, stages)
    _write_outputs(out_dir, metrics)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "episode_id": metrics["episode_id"],
                "stage_count": metrics["stage_count"],
                "all_constraints_satisfied": metrics["all_constraints_satisfied"],
                "dynamic_mean_normalized_hv": metrics["dynamic_mean_normalized_hv"],
                "mean_normalized_hv": metrics["mean_normalized_hv"],
                "metrics": str(out_dir / "stage_metrics.json"),
                "curve": str(out_dir / "stage_normalized_hv.png"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def evaluate_run(run: dict[str, Any], episode: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    module = importlib.import_module(GREEN_MODULE)
    state = json.loads(json.dumps(episode["hidden_initial_state"]))
    previous_agent_solution: dict[str, Any] | None = None
    stage_rows: list[dict[str, Any]] = []
    best_solutions: list[dict[str, Any]] = []
    reference_steps = _reference_step_map(episode)
    oracle_steps = episode.get("hidden_update_oracle", [])
    for stage_index, stage in enumerate(stages):
        update_id = stage["update_id"]
        if stage_index > 0:
            delta = oracle_steps[stage_index - 1].get("hidden_delta", {})
            state = module._apply_delta(state, delta)
        raw_solution = _canonical_solution(stage.get("solution"))
        constraint_report = _green_constraint_report(state, raw_solution)
        true_objectives = _safe_green_score(module, state, raw_solution, previous_agent_solution, constraint_report)
        reference_step = reference_steps.get(update_id) or {}
        reference_archive = reference_step.get("pareto_archive", [])
        hv_ref_point = reference_step.get("hypervolume_reference_point")
        hv_ideal = reference_step.get("hypervolume_ideal_point")
        candidate_archive = [{"solution": raw_solution, "objectives": {name: true_objectives[name] for name in GREEN_OBJECTIVES}}]
        hv_metrics = evaluate_against_reference(
            candidate_archive if constraint_report["feasible"] else [],
            reference_archive,
            GREEN_OBJECTIVES,
            hv_ref_point,
            hv_ideal,
            hv_samples=0,
        )
        row = {
            "stage_index": stage_index,
            "update_id": update_id,
            "stage_type": "initial" if stage_index == 0 else "update",
            "agent_reported_feasible": bool(stage.get("agent_feasible")),
            "evaluation_feasible": constraint_report["feasible"],
            "constraint_violations": constraint_report["violations"],
            "agent_reported_objective": stage.get("agent_objective"),
            "true_objectives": true_objectives,
            "true_scalar": round(sum(float(true_objectives[name]) for name in GREEN_OBJECTIVES), 6),
            "hv": hv_metrics.get("hv", 0.0),
            "reference_hv": hv_metrics.get("reference_hv", 0.0),
            "normalized_hv": hv_metrics.get("hv_ratio", 0.0),
            "igd_to_reference": hv_metrics.get("igd"),
            "reference_archive_size": len(reference_archive),
            "solver_decision": stage.get("solver_decision") or {},
            "solver_result_metadata": stage.get("solver_result_metadata") or {},
            "latency_seconds": stage.get("latency_seconds"),
            "stage_tokens": stage.get("stage_tokens"),
        }
        stage_rows.append(row)
        best_solutions.append(
            {
                "stage_index": stage_index,
                "update_id": update_id,
                "evaluation_feasible": constraint_report["feasible"],
                "true_objectives": true_objectives,
                "normalized_hv": row["normalized_hv"],
                "solution": raw_solution,
                "reference_representative_solution": reference_step.get("representative_solution"),
            }
        )
        previous_agent_solution = raw_solution
    normalized = [float(row["normalized_hv"] or 0.0) for row in stage_rows]
    dynamic = normalized[1:]
    return {
        "run_path": run.get("_path"),
        "benchmark_path": episode.get("_path"),
        "agent_mode": run.get("agent_mode", "aevo"),
        "model": run.get("model"),
        "episode_id": episode.get("episode_id"),
        "benchmark": episode.get("benchmark"),
        "domain": episode.get("domain"),
        "objectives": GREEN_OBJECTIVES,
        "stage_count": len(stage_rows),
        "all_constraints_satisfied": all(row["evaluation_feasible"] for row in stage_rows),
        "initial_normalized_hv": normalized[0] if normalized else 0.0,
        "dynamic_mean_normalized_hv": round(mean(dynamic), 6) if dynamic else None,
        "mean_normalized_hv": round(mean(normalized), 6) if normalized else 0.0,
        "mean_igd_to_reference": round(mean([float(row["igd_to_reference"]) for row in stage_rows if _finite(row["igd_to_reference"])]), 6)
        if any(_finite(row["igd_to_reference"]) for row in stage_rows)
        else None,
        "stage_metrics": stage_rows,
        "best_solutions": best_solutions,
        "token_usage": run.get("token_usage") or {},
        "episode_metrics": run.get("episode_metrics") or {},
    }


def _load_benchmark_episode(path: Path, index: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ep = rows[index]
    ep["_path"] = str(path)
    return ep


def _load_run(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty run JSONL: {path}")
        row = rows[0]
    else:
        row = json.loads(text)
    row["_path"] = str(path)
    return row


def _extract_run_stages(run: dict[str, Any]) -> list[dict[str, Any]]:
    if "stages" in run:
        out = []
        previous_tokens = 0
        for stage in run.get("stages", []):
            usage_total = sum(int(item.get("total_tokens", 0) or 0) for item in stage.get("llm_usage", []) or [])
            out.append(
                {
                    "update_id": stage.get("stage_id"),
                    "solution": _solution_from_trace_stage(stage),
                    "agent_feasible": (stage.get("agent_verification") or {}).get("feasible"),
                    "agent_objective": stage.get("agent_objective"),
                    "solver_decision": stage.get("solver_decision"),
                    "solver_result_metadata": (stage.get("solver_result") or {}).get("metadata"),
                    "latency_seconds": None,
                    "stage_tokens": usage_total,
                }
            )
            previous_tokens += usage_total
        return out
    out = [
        {
            "update_id": "initial",
            "solution": run.get("initial_candidate"),
            "agent_feasible": run.get("initial_feasible"),
            "agent_objective": run.get("initial_objective"),
            "solver_decision": {},
            "solver_result_metadata": {},
            "latency_seconds": 0.0,
            "stage_tokens": 0,
        }
    ]
    previous_tokens = 0
    for update in run.get("update_results", []) or []:
        snapshot = update.get("token_usage_snapshot") or {}
        total_tokens = int(snapshot.get("total_tokens", 0) or 0)
        stage_tokens = max(0, total_tokens - previous_tokens)
        previous_tokens = total_tokens
        out.append(
            {
                "update_id": update.get("update_id"),
                "solution": update.get("solution"),
                "agent_feasible": update.get("feasible"),
                "agent_objective": update.get("objective_value"),
                "solver_decision": update.get("dynamic_strategy") or {},
                "solver_result_metadata": ((update.get("metrics") or {}).get("solver_result") or {}).get("metadata", {}),
                "latency_seconds": (update.get("metrics") or {}).get("update_latency_seconds"),
                "stage_tokens": stage_tokens,
            }
        )
    return out


def _solution_from_trace_stage(stage: dict[str, Any]) -> dict[str, Any]:
    result = stage.get("solver_result") or {}
    if isinstance(result.get("solution"), dict):
        return result["solution"]
    preview = stage.get("solution_preview")
    return preview if isinstance(preview, dict) else {}


def _reference_step_map(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {step.get("update_id"): step for step in episode.get("evaluation", {}).get("reference_trajectory", [])}


def _canonical_solution(solution: Any) -> dict[str, Any]:
    if not isinstance(solution, dict):
        return {}
    has_route_payload = any(isinstance(solution.get(key), (list, dict)) for key in ["routes", "vehicle_routes", "plans", "route_plan", "vehicle_route_plan"])
    if isinstance(solution.get("assignments"), dict) and not has_route_payload:
        solution = solution["assignments"]
    routes = _first_route_list(solution)
    if isinstance(routes, list):
        converted = []
        for idx, route in enumerate(routes):
            if isinstance(route, dict):
                vehicle = route.get("vehicle") or route.get("vehicle_id") or route.get("resource_id") or route.get("carrier_id") or route.get("id") or f"V{idx + 1}"
                orders = (
                    route.get("orders")
                    or route.get("order_ids")
                    or route.get("stops")
                    or route.get("decision_ids")
                    or route.get("sequence")
                    or route.get("route")
                    or []
                )
                converted.append({"vehicle": str(vehicle), "orders": [_route_order_id(item) for item in orders if _route_order_id(item)]})
            elif isinstance(route, list):
                converted.append({"vehicle": f"V{idx + 1}", "orders": [_route_order_id(item) for item in route if _route_order_id(item)]})
        return {**solution, "routes": converted}
    vehicle_routes = solution.get("vehicle_routes")
    if isinstance(vehicle_routes, dict):
        converted = [
            {"vehicle": str(vehicle), "orders": [_route_order_id(item) for item in orders or [] if _route_order_id(item)]}
            for vehicle, orders in sorted(vehicle_routes.items())
        ]
        return {**solution, "routes": converted}
    route_map = solution.get("routes")
    if isinstance(route_map, dict):
        converted = []
        for vehicle, route in sorted(route_map.items()):
            if isinstance(route, dict):
                orders = (
                    route.get("orders")
                    or route.get("order_ids")
                    or route.get("stops")
                    or route.get("decision_ids")
                    or route.get("sequence")
                    or route.get("route")
                    or []
                )
            else:
                orders = route if isinstance(route, list) else []
            converted.append({"vehicle": str(vehicle), "orders": [_route_order_id(item) for item in orders if _route_order_id(item)]})
        return {**solution, "routes": converted}
    assignments = solution.get("vehicle_assignments") or solution.get("assignments")
    if isinstance(assignments, dict):
        by_vehicle: dict[str, list[str]] = {}
        for order, vehicle in assignments.items():
            by_vehicle.setdefault(str(vehicle), []).append(str(order))
        return {**solution, "routes": [{"vehicle": vehicle, "orders": orders} for vehicle, orders in sorted(by_vehicle.items())]}
    return solution


def _first_route_list(solution: dict[str, Any]) -> list[Any] | None:
    for key in ["routes", "vehicle_routes", "plans", "route_plan", "vehicle_route_plan"]:
        value = solution.get(key)
        if isinstance(value, list):
            return value
    return None


def _route_order_id(item: Any) -> str:
    if isinstance(item, dict):
        for key in ["order_id", "order", "id", "job_id", "task_id", "item_id", "request_id"]:
            value = item.get(key)
            if value is not None:
                return str(value)
        return ""
    return str(item)


def _green_constraint_report(state: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    vehicles = {v["id"]: v for v in state.get("vehicles", [])}
    orders = {o["id"]: o for o in state.get("orders", [])}
    active = {o["id"] for o in state.get("orders", []) if o.get("active", True)}
    served: list[str] = []
    used_vehicles: list[str] = []
    violations: dict[str, Any] = {}
    for route in solution.get("routes", []) or []:
        vehicle_id = route.get("vehicle")
        used_vehicles.append(vehicle_id)
        vehicle = vehicles.get(vehicle_id)
        if vehicle is None:
            violations.setdefault("unknown_vehicle", []).append(vehicle_id)
            continue
        if not vehicle.get("available", True):
            violations.setdefault("unavailable_vehicle", []).append(vehicle_id)
        load = 0.0
        for order_id in route.get("orders", []) or []:
            order = orders.get(order_id)
            if order is None:
                violations.setdefault("unknown_order", []).append(order_id)
                continue
            if not order.get("active", True):
                violations.setdefault("inactive_order", []).append(order_id)
            load += float(order.get("demand", 0.0) or 0.0)
            served.append(order_id)
        if load > float(vehicle.get("capacity", 0.0) or 0.0) + 1e-9:
            violations.setdefault("capacity_exceeded", []).append({"vehicle": vehicle_id, "load": load, "capacity": vehicle.get("capacity")})
    served_set = set(served)
    duplicates = sorted({oid for oid in served if served.count(oid) > 1})
    duplicate_vehicles = sorted(
        {vehicle_id for vehicle_id in used_vehicles if used_vehicles.count(vehicle_id) > 1},
        key=str,
    )
    missing = sorted(active - served_set)
    if duplicates:
        violations["duplicate_orders"] = duplicates
    if duplicate_vehicles:
        violations["duplicate_vehicles"] = duplicate_vehicles
    if missing:
        violations["missing_active_orders"] = missing
    return {"feasible": not violations, "violations": violations}


def _safe_green_score(module, state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None, report: dict[str, Any]) -> dict[str, float]:
    try:
        return module._score_solution(state, solution, previous_solution)
    except Exception as exc:
        sanitized = _sanitize_green_solution(state, solution)
        report.setdefault("violations", {})["score_sanitized_after_error"] = repr(exc)
        report["feasible"] = False
        try:
            return module._score_solution(state, sanitized, previous_solution)
        except Exception:
            return {"distance": 1e9, "lateness": 1e9, "emission": 1e9, "disruption": 1e9, "priority_penalty": 1e9}


def _sanitize_green_solution(state: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    vehicles = {v["id"] for v in state.get("vehicles", [])}
    orders = {o["id"] for o in state.get("orders", [])}
    routes = []
    for route in solution.get("routes", []) or []:
        vehicle = route.get("vehicle")
        if vehicle not in vehicles:
            continue
        kept = [oid for oid in route.get("orders", []) or [] if oid in orders]
        if kept:
            routes.append({"vehicle": vehicle, "orders": kept})
    return {"routes": routes}


def _write_outputs(out_dir: Path, metrics: dict[str, Any]) -> None:
    (out_dir / "stage_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "best_solutions.json").write_text(json.dumps(metrics["best_solutions"], ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / "stage_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "stage_index",
            "update_id",
            "stage_type",
            "agent_reported_feasible",
            "evaluation_feasible",
            "agent_reported_objective",
            "true_scalar",
            "hv",
            "reference_hv",
            "normalized_hv",
            "igd_to_reference",
            "reference_archive_size",
            "latency_seconds",
            "stage_tokens",
            "distance",
            "lateness",
            "emission",
            "violations",
            "solver",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["stage_metrics"]:
            writer.writerow(
                {
                    "stage_index": row["stage_index"],
                    "update_id": row["update_id"],
                    "stage_type": row["stage_type"],
                    "agent_reported_feasible": row["agent_reported_feasible"],
                    "evaluation_feasible": row["evaluation_feasible"],
                    "agent_reported_objective": row["agent_reported_objective"],
                    "true_scalar": row["true_scalar"],
                    "hv": row["hv"],
                    "reference_hv": row["reference_hv"],
                    "normalized_hv": row["normalized_hv"],
                    "igd_to_reference": row["igd_to_reference"],
                    "reference_archive_size": row["reference_archive_size"],
                    "latency_seconds": row["latency_seconds"],
                    "stage_tokens": row["stage_tokens"],
                    "distance": row["true_objectives"].get("distance"),
                    "lateness": row["true_objectives"].get("lateness"),
                    "emission": row["true_objectives"].get("emission"),
                    "violations": json.dumps(row["constraint_violations"], ensure_ascii=False),
                    "solver": row["solver_decision"].get("skill_id"),
                }
            )
    _plot_stage_metric(metrics, "normalized_hv", "Normalized HV", out_dir / "stage_normalized_hv.png", yline=1.0)
    _plot_stage_metric(metrics, "hv", "Raw HV", out_dir / "stage_raw_hv.png")
    _plot_stage_metric(metrics, "igd_to_reference", "IGD to Reference", out_dir / "stage_igd.png")
    _plot_objectives(metrics, out_dir / "stage_objectives.png")


def _plot_stage_metric(metrics: dict[str, Any], key: str, ylabel: str, path: Path, yline: float | None = None) -> None:
    rows = metrics["stage_metrics"]
    x = [row["stage_index"] for row in rows]
    y = [float(row.get(key) or 0.0) if _finite(row.get(key)) else 0.0 for row in rows]
    labels = [row["update_id"] for row in rows]
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(x, y, marker="o", linewidth=2.2, color="#2563eb")
    if yline is not None:
        plt.axhline(yline, color="#6b7280", linestyle="--", linewidth=1.0, label="reference")
        plt.legend(loc="best")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.xlabel("Dynamic stage")
    plt.ylabel(ylabel)
    plt.title(f"{metrics['agent_mode']} on {metrics['episode_id']}: {ylabel}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_objectives(metrics: dict[str, Any], path: Path) -> None:
    rows = metrics["stage_metrics"]
    x = [row["stage_index"] for row in rows]
    labels = [row["update_id"] for row in rows]
    plt.figure(figsize=(10.5, 4.8))
    for name, color in [("distance", "#2563eb"), ("lateness", "#dc2626"), ("emission", "#059669")]:
        plt.plot(x, [row["true_objectives"].get(name, 0.0) for row in rows], marker="o", linewidth=2.0, label=name, color=color)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.xlabel("Dynamic stage")
    plt.ylabel("Objective value")
    plt.title(f"{metrics['agent_mode']} true objective trajectory")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _finite(value: Any) -> bool:
    try:
        number = float(value)
        return not (math.isnan(number) or math.isinf(number))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
