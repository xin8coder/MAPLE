#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.benchmarks.optimization_contracts import (
    append_public_objective_spec_to_text,
    append_public_update_objective_notes_to_episode,
    attach_public_optimization_contract,
)


BENCHMARKS = {
    "fjsp_exact_small": {
        "episodes": "data/evo2_dynoptbench/compressed_plan/fjsp_exact_small_6episodes.jsonl",
        "out_root": "data/evo2_dynoptbench/public_csv/fjsp_exact_small",
        "out_episodes": "data/evo2_dynoptbench/public_csv/fjsp_exact_small_6episodes_csv.jsonl",
    },
    "cobench_exact_small": {
        "episodes": "data/evo2_dynoptbench/compressed_plan/cobench_exact_small_6episodes.jsonl",
        "out_root": "data/evo2_dynoptbench/public_csv/cobench_exact_small",
        "out_episodes": "data/evo2_dynoptbench/public_csv/cobench_exact_small_6episodes_csv.jsonl",
    },
    "inrc_realistic_dynamic": {
        "episodes": "data/evo2_dynoptbench/compressed_plan/inrc_realistic_dynamic_6episodes.jsonl",
        "out_root": "data/evo2_dynoptbench/public_csv/inrc_realistic_dynamic",
        "out_episodes": "data/evo2_dynoptbench/public_csv/inrc_realistic_dynamic_6episodes_csv.jsonl",
    },
    "cloud_scheduling_mo": {
        "episodes": "data/evo2_dynoptbench/compressed_plan/cloud_scheduling_mo_6episodes.jsonl",
        "out_root": "data/evo2_dynoptbench/public_csv/cloud_scheduling_mo",
        "out_episodes": "data/evo2_dynoptbench/public_csv/cloud_scheduling_mo_6episodes_csv.jsonl",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize public CSV inputs for non-Green EVO2 benchmarks.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS) + ["all"], default="all")
    parser.add_argument("--episodes", help="Override the source episode JSONL for a single benchmark, e.g. an MOEA-reference-enhanced file.")
    parser.add_argument("--out-root", help="Override the public CSV output directory for a single benchmark.")
    parser.add_argument("--out-episodes", help="Override the output episode JSONL for a single benchmark.")
    args = parser.parse_args()
    if args.benchmark == "all" and (args.episodes or args.out_root or args.out_episodes):
        parser.error("--episodes/--out-root/--out-episodes can only be used with a single --benchmark")
    names = sorted(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    summary = {}
    for name in names:
        cfg = dict(BENCHMARKS[name])
        if args.episodes:
            cfg["episodes"] = args.episodes
        if args.out_root:
            cfg["out_root"] = args.out_root
        if args.out_episodes:
            cfg["out_episodes"] = args.out_episodes
        summary[name] = materialize(name, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def materialize(name: str, cfg: dict[str, str]) -> dict[str, Any]:
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    converted = []
    for line in Path(cfg["episodes"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        state = episode.get("hidden_initial_state") or {}
        episode_dir = out_root / episode["episode_id"]
        episode_dir.mkdir(parents=True, exist_ok=True)
        csv_tables = _write_domain_tables(name, episode_dir, state)
        public_context = dict(episode.get("public_context") or {})
        public_context.update(
            {
                "input_policy": "agent sees natural-language objective plus public CSV schema/paths; hidden references remain evaluation-only",
                "csv_tables": csv_tables,
                "csv_schema": {table: spec["columns"] for table, spec in csv_tables.items()},
            }
        )
        episode["public_context"] = public_context
        episode["structured_data_path"] = _portable_path(episode_dir)
        episode["public_initial_problem"] = _public_problem_for_csv(name, episode)
        attach_public_optimization_contract(episode, name)
        episode["public_initial_problem"] = append_public_objective_spec_to_text(
            str(episode.get("public_initial_problem") or ""),
            episode["public_context"].get("optimization_contract"),
        )
        append_public_update_objective_notes_to_episode(episode)
        converted.append(episode)
    out_episodes = Path(cfg["out_episodes"])
    out_episodes.parent.mkdir(parents=True, exist_ok=True)
    with out_episodes.open("w", encoding="utf-8") as handle:
        for episode in converted:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    return {"episodes": len(converted), "out_episodes": str(out_episodes), "out_root": str(out_root)}


def _write_domain_tables(name: str, root: Path, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if name == "fjsp_exact_small":
        _write_csv(root / "machines.csv", ["id", "cost_rate", "available"], state.get("machines", []))
        rows = []
        for job in state.get("jobs", []):
            for seq, op in enumerate(job.get("operations", [])):
                rows.append(
                    {
                        "job_id": job.get("id"),
                        "operation_id": op.get("id"),
                        "sequence_index": seq,
                        "eligible_machines": "|".join(op.get("eligible_machines", [])),
                        "processing_time": op.get("processing_time", ""),
                        "due_time": job.get("due_time", ""),
                        "priority": job.get("priority", 1),
                    }
                )
        _write_csv(root / "operations.csv", ["job_id", "operation_id", "sequence_index", "eligible_machines", "processing_time", "due_time", "priority"], rows)
        _write_csv(root / "settings.csv", ["objective_mode"], [{"objective_mode": state.get("objective_mode", "minimize_tardiness")}])
        return {
            "machines": _table_spec(root / "machines.csv", root, ["id", "cost_rate", "available"]),
            "operations": _table_spec(root / "operations.csv", root, ["job_id", "operation_id", "sequence_index", "eligible_machines", "processing_time", "due_time", "priority"]),
            "settings": _table_spec(root / "settings.csv", root, ["objective_mode"]),
        }
    if name == "cobench_exact_small":
        if state.get("family") == "facility_location":
            stale_items = root / "items.csv"
            if stale_items.exists():
                stale_items.unlink()
            stale_sets = root / "sets.csv"
            if stale_sets.exists():
                stale_sets.unlink()
            stale_elements = root / "elements.csv"
            if stale_elements.exists():
                stale_elements.unlink()
            facilities = state.get("facilities", [])
            facility_ids = [str(item.get("id")) for item in facilities]
            customer_columns = ["id", "demand"] + [f"serve_cost_{facility_id}" for facility_id in facility_ids] + [f"serve_time_{facility_id}" for facility_id in facility_ids]
            customer_rows = []
            for customer in state.get("customers", []):
                serve_cost = customer.get("serve_cost", {})
                serve_time = customer.get("serve_time", {})
                row = {"id": customer.get("id"), "demand": customer.get("demand")}
                for facility_id in facility_ids:
                    row[f"serve_cost_{facility_id}"] = serve_cost.get(facility_id, "")
                    row[f"serve_time_{facility_id}"] = serve_time.get(facility_id, serve_cost.get(facility_id, ""))
                customer_rows.append(row)
            _write_csv(root / "facilities.csv", ["id", "open_cost", "capacity"], facilities)
            _write_csv(root / "customers.csv", customer_columns, customer_rows)
            _write_csv(root / "constraints.csv", ["mandatory", "forbidden", "objective_mode"], [{"mandatory": "|".join(state.get("mandatory", [])), "forbidden": "|".join(state.get("forbidden", [])), "objective_mode": state.get("objective_mode", "minimize_cost")}])
            return {
                "facilities": _table_spec(root / "facilities.csv", root, ["id", "open_cost", "capacity"]),
                "customers": _table_spec(root / "customers.csv", root, customer_columns),
                "constraints": _table_spec(root / "constraints.csv", root, ["mandatory", "forbidden", "objective_mode"]),
            }
        if state.get("family") == "set_cover":
            stale_items = root / "items.csv"
            if stale_items.exists():
                stale_items.unlink()
            stale_facilities = root / "facilities.csv"
            if stale_facilities.exists():
                stale_facilities.unlink()
            stale_customers = root / "customers.csv"
            if stale_customers.exists():
                stale_customers.unlink()
            set_rows = []
            for row in state.get("sets", []):
                set_rows.append({"id": row.get("id"), "cost": row.get("cost"), "covers": "|".join(row.get("covers", [])), "active": row.get("active", True)})
            _write_csv(root / "sets.csv", ["id", "cost", "covers", "active"], set_rows)
            _write_csv(root / "elements.csv", ["id", "active"], state.get("elements", []))
            _write_csv(root / "constraints.csv", ["mandatory", "forbidden"], [{"mandatory": "|".join(state.get("mandatory", [])), "forbidden": "|".join(state.get("forbidden", []))}])
            return {
                "sets": _table_spec(root / "sets.csv", root, ["id", "cost", "covers", "active"]),
                "elements": _table_spec(root / "elements.csv", root, ["id", "active"]),
                "constraints": _table_spec(root / "constraints.csv", root, ["mandatory", "forbidden"]),
            }
        _write_csv(root / "items.csv", ["id", "value", "weight"], state.get("items", []))
        _write_csv(root / "constraints.csv", ["capacity", "mandatory", "forbidden", "objective_mode", "value_floor"], [{"capacity": state.get("capacity"), "mandatory": "|".join(state.get("mandatory", [])), "forbidden": "|".join(state.get("forbidden", [])), "objective_mode": state.get("objective_mode", "maximize_value"), "value_floor": state.get("value_floor", 0)}])
        return {
            "items": _table_spec(root / "items.csv", root, ["id", "value", "weight"]),
            "constraints": _table_spec(root / "constraints.csv", root, ["capacity", "mandatory", "forbidden", "objective_mode", "value_floor"]),
        }
    if name == "inrc_realistic_dynamic":
        _write_csv(root / "nurses.csv", ["id", "skills", "max_shifts"], [{"id": n.get("id"), "skills": "|".join(n.get("skills", [])), "max_shifts": n.get("max_shifts", "")} for n in state.get("nurses", [])])
        _write_csv(root / "days.csv", ["id"], [{"id": day} for day in state.get("days", [])])
        _write_csv(root / "shifts.csv", ["id"], [{"id": shift} for shift in state.get("shifts", [])])
        coverage_rows = []
        coverage = state.get("coverage", {})
        for day, shift_req in coverage.items():
            for shift, required in shift_req.items():
                coverage_rows.append({"day": day, "shift": shift, "required": required})
        _write_csv(root / "coverage.csv", ["day", "shift", "required"], coverage_rows)
        _write_csv(root / "absences.csv", ["nurse", "day"], [])
        _write_csv(root / "preferences.csv", ["nurse", "day", "preference"], [])
        _write_csv(root / "nurse_limits.csv", ["nurse", "shift", "day", "limit"], [])
        weights = state.get("penalty_weights", {}) if isinstance(state.get("penalty_weights"), dict) else {}
        _write_csv(
            root / "policy.csv",
            ["fairness_weight", "preference_weight", "sequence_weight", "overload_weight", "over_coverage_weight", "avoid_evening_after_night"],
            [
                {
                    "fairness_weight": weights.get("fairness", 12.0),
                    "preference_weight": weights.get("preference", 35.0),
                    "sequence_weight": weights.get("sequence", 45.0),
                    "overload_weight": weights.get("overload", 40.0),
                    "over_coverage_weight": weights.get("over_coverage", 18.0),
                    "avoid_evening_after_night": bool(state.get("avoid_evening_after_night", False)),
                }
            ],
        )
        return {
            "nurses": _table_spec(root / "nurses.csv", root, ["id", "skills", "max_shifts"]),
            "days": _table_spec(root / "days.csv", root, ["id"]),
            "shifts": _table_spec(root / "shifts.csv", root, ["id"]),
            "coverage": _table_spec(root / "coverage.csv", root, ["day", "shift", "required"]),
            "absences": _table_spec(root / "absences.csv", root, ["nurse", "day"], dynamic=True),
            "preferences": _table_spec(root / "preferences.csv", root, ["nurse", "day", "preference"], dynamic=True),
            "nurse_limits": _table_spec(root / "nurse_limits.csv", root, ["nurse", "shift", "day", "limit"], dynamic=True),
            "policy": _table_spec(root / "policy.csv", root, ["fairness_weight", "preference_weight", "sequence_weight", "overload_weight", "over_coverage_weight", "avoid_evening_after_night"], dynamic=True),
        }
    if name == "cloud_scheduling_mo":
        _write_csv(root / "machines.csv", ["id", "cpu", "mem", "energy_idle", "energy_per_cpu", "available", "gpu"], state.get("machines", []))
        _write_csv(root / "jobs.csv", ["id", "burst", "cpu", "mem", "deadline", "latency_sensitivity", "priority", "gpu_required", "active"], state.get("jobs", []))
        _write_csv(root / "global_params.csv", ["energy_price", "carbon_intensity"], [{"energy_price": state.get("energy_price", 1.0), "carbon_intensity": state.get("carbon_intensity", 1.0)}])
        return {
            "machines": _table_spec(root / "machines.csv", root, ["id", "cpu", "mem", "energy_idle", "energy_per_cpu", "available", "gpu"]),
            "jobs": _table_spec(root / "jobs.csv", root, ["id", "burst", "cpu", "mem", "deadline", "latency_sensitivity", "priority", "gpu_required", "active"]),
            "global_params": _table_spec(root / "global_params.csv", root, ["energy_price", "carbon_intensity"]),
        }
    raise ValueError(f"unknown benchmark: {name}")


def _public_problem_for_csv(name: str, episode: dict[str, Any]) -> str:
    base = episode.get("public_initial_problem", "")
    prefix = "The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. "
    if name == "fjsp_exact_small":
        return prefix + "Create a flexible job-shop schedule. Assign every operation to one eligible resource, respect fixed job precedence and resource capacity, and use settings.objective_mode for the current scalar objective. Later updates may revise soft due/priority values or machine cost_rate but do not change machine availability, operation eligibility, or processing requirements. Plan-change disruption is reported only as a diagnostic and never defines hard feasibility. " + base
    if name == "cobench_exact_small":
        family = (episode.get("hidden_initial_state") or {}).get("family")
        if family == "facility_location":
            return prefix + "Choose open facilities and assign each customer to an open facility using only public facilities, customers, and constraints tables. Respect fixed facility capacity; later updates revise public opening/service costs but do not change capacity, demand, or facility eligibility. The constraints table gives the current objective_mode. Expose a minimization scalar suitable for the generic solver. " + base
        if family == "set_cover":
            return prefix + "Select a minimum-cost set collection using only public sets, elements, and constraints tables. Cover the fixed active element universe; later updates revise public set costs but do not change coverage requirements or set eligibility. Expose a minimization scalar suitable for the generic solver. " + base
        return prefix + "Select a feasible subset or portfolio under fixed capacity. Later updates revise public item values but do not change capacity, item weights, or eligibility fields. The constraints table gives the current objective_mode. Expose a minimization scalar suitable for the generic solver. " + base
    if name == "inrc_realistic_dynamic":
        return prefix + "Create a staff roster over days and shifts. Respect current public coverage, absences, nurse max-shift capacities, and one assignment per staff member per day. Later updates may replace these public rows as well as soft preferences and penalty weights. Report roster disruption only as a diagnostic. " + base
    if name == "cloud_scheduling_mo":
        return prefix + "Assign active tasks to available compute resources. Rows with active=false are public candidate jobs that are not scheduled until a later update explicitly activates them. Respect current public CPU, memory, accelerator needs, and machine availability. Optimize energy and imbalance as main objectives and report SLA/latency/migration diagnostics without treating those diagnostics as hard feasibility constraints. Later updates may replace public workload requirements, machine capacity/accelerator profiles, availability, soft SLA, priority, energy, or carbon parameters. " + base
    return prefix + base


def _table_spec(path: Path, root: Path, columns: list[str], *, dynamic: bool = False) -> dict[str, Any]:
    spec = {"path": _portable_path(path), "allowed_root": _portable_path(root), "columns": columns}
    if dynamic:
        spec["dynamic"] = True
    return spec


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


if __name__ == "__main__":
    main()
