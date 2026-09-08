#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.benchmarks.optimization_contracts import append_public_objective_spec_to_text, attach_public_optimization_contract
from evo2.benchmarks.optimization_contracts import append_public_update_objective_notes_to_episode


SOURCE_SPECS = [
    {
        "path": "data/evo2_dynoptbench/public_csv/cobench_exact_small_6episodes_csv.jsonl",
        "source_id": "S01",
        "type": "scalar combinatorial selection/allocation/coverage",
        "reference": "CO-Bench/OR-Library-style combinatorial optimization",
        "evaluator_profile": "cobench_exact_small",
        "indices": [0, 3, 6],
    },
    {
        "path": "data/evo2_dynoptbench/public_csv/fjsp_exact_small_6episodes_csv.jsonl",
        "source_id": "S02",
        "type": "precedence-constrained operation scheduling",
        "reference": "Brandimarte-style flexible job-shop scheduling",
        "evaluator_profile": "fjsp_exact_small",
        "indices": [0, 1, 2],
    },
    {
        "path": "data/evo2_dynoptbench/public_csv/inrc_realistic_dynamic_6episodes_csv.jsonl",
        "source_id": "S03",
        "type": "coverage and duty scheduling",
        "reference": "INRC-II-style rostering",
        "evaluator_profile": "inrc_realistic_dynamic",
        "indices": [0, 1, 2],
    },
    {
        "path": "data/evo2_dynoptbench/public_csv/green_vrp_mo_6episodes_csv.jsonl",
        "source_id": "S04",
        "type": "multi-objective ordered service planning",
        "reference": "green and pollution-aware dynamic routing",
        "evaluator_profile": "green_vrp_mo",
        "indices": [0, 1, 2],
    },
    {
        "path": "data/evo2_dynoptbench/public_csv/cloud_scheduling_mo_6episodes_csv.jsonl",
        "source_id": "S05",
        "type": "multi-objective resource scheduling",
        "reference": "energy-aware cloud scheduling",
        "evaluator_profile": "cloud_scheduling_mo",
        "indices": [0, 1, 2],
    },
]


def build_nldo(repo_root: Path, updates_per_problem: int = 10, source_path_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    problem_index = 1
    for spec in _source_specs(source_path_overrides):
        rows = _read_jsonl(repo_root / spec["path"])
        for source_row_index in spec["indices"]:
            if source_row_index >= len(rows):
                continue
            problem_id = f"NLDO-P{problem_index:03d}"
            row = _nldo_episode(
                copy.deepcopy(rows[source_row_index]),
                spec=spec,
                problem_id=problem_id,
                updates_per_problem=updates_per_problem,
            )
            episodes.append(row)
            problem_index += 1
    return episodes


def _source_specs(source_path_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    overrides = source_path_overrides or {}
    specs: list[dict[str, Any]] = []
    for raw_spec in SOURCE_SPECS:
        spec = copy.deepcopy(raw_spec)
        source_id = str(spec.get("source_id") or "")
        if source_id in overrides:
            spec["path"] = overrides[source_id]
        specs.append(spec)
    return specs


def _nldo_episode(
    row: dict[str, Any],
    *,
    spec: dict[str, Any],
    problem_id: str,
    updates_per_problem: int,
) -> dict[str, Any]:
    source_episode_id = str(row.get("episode_id") or "")
    source_benchmark_label = str(row.get("benchmark") or row.get("base_benchmark") or "")
    stage_ids = [f"{problem_id}-S{i:02d}" for i in range(updates_per_problem + 1)]
    updates = _renumber_updates(row.get("update_stream") or [], problem_id, updates_per_problem)
    oracle = _renumber_oracle(row.get("hidden_update_oracle") or [], problem_id, updates_per_problem)
    evaluation = _renumber_evaluation(row.get("evaluation") or {}, problem_id, updates_per_problem)

    metadata = dict(row.get("metadata") or {})
    metadata["nldo"] = {
        "problem_id": problem_id,
        "source_id": spec["source_id"],
        "source_type": spec["type"],
        "stage_ids": stage_ids,
        "updates_per_problem": updates_per_problem,
        "reference_source": spec["reference"],
        "source_episode_id": source_episode_id,
        "source_label": source_benchmark_label,
    }
    public_context = row.get("public_context") if isinstance(row.get("public_context"), dict) else {}
    public_context = copy.deepcopy(public_context)
    public_context["nldo_problem_id"] = problem_id
    public_context["nldo_source_id"] = spec["source_id"]
    public_context["nldo_source_type"] = spec["type"]
    public_context["update_count"] = updates_per_problem
    public_context["input_policy"] = "NLDO public table interface: use load_table/list_tables and top-level table_updates for data changes."

    out = {
        **row,
        "episode_id": problem_id,
        "benchmark": "NLDO",
        "base_benchmark": "NLDO",
        "domain": row.get("domain") or spec["source_id"].lower(),
        "source_dataset": "NLDO",
        "source_instance_id": problem_id,
        "base_instance_id": problem_id,
        "public_context": public_context,
        "update_stream": updates,
        "hidden_update_oracle": oracle,
        "evaluation": evaluation,
        "metadata": metadata,
        "hidden_evaluator_profile": spec["evaluator_profile"],
    }
    out["hidden_initial_state"] = row.get("hidden_initial_state") or {}
    out["structured_data_path"] = row.get("structured_data_path")
    out["agent_allowed_solvers"] = row.get("agent_allowed_solvers") or ["ga_or_moea", "generated_repair_operator"]
    attach_public_optimization_contract(out, spec["evaluator_profile"])
    out["public_initial_problem"] = append_public_objective_spec_to_text(
        str(out.get("public_initial_problem") or ""),
        out["public_context"].get("optimization_contract"),
    )
    append_public_update_objective_notes_to_episode(out)
    return out


def _renumber_updates(updates: list[dict[str, Any]], problem_id: str, count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_index in range(1, count + 1):
        new_update_id = f"{problem_id}-S{stage_index:02d}"
        if stage_index <= len(updates):
            item = copy.deepcopy(updates[stage_index - 1])
            source_update_id = str(item.get("update_id") or f"u{stage_index:03d}")
            text = item.get("natural_language_update") or item.get("public_update") or ""
        else:
            item = {}
            source_update_id = f"pad{stage_index:03d}"
            text = _maintenance_update_text(stage_index)
        item["update_id"] = new_update_id
        item["time_index"] = stage_index
        item["public_update"] = str(text)
        item["natural_language_update"] = str(text)
        item["difficulty"] = item.get("difficulty") or "maintenance"
        item["requires_memory"] = bool(item.get("requires_memory", False))
        metadata = dict(item.get("metadata") or {})
        metadata.update({"stage_id": new_update_id, "source_update_id": source_update_id})
        item["metadata"] = metadata
        out.append(item)
    return out


def _maintenance_update_text(stage_index: int) -> str:
    variants = [
        "Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.",
        "No new business event is reported for this stage. Refresh the solution against the current public tables.",
        "Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.",
        "The planning team wants the current public contract revalidated without adding new constraints.",
        "Run a maintenance update with the same public contract. Revalidate feasibility and quality under the current public state.",
    ]
    return variants[(stage_index - 1) % len(variants)]


def _renumber_oracle(oracle: list[dict[str, Any]], problem_id: str, count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_index in range(1, count + 1):
        new_update_id = f"{problem_id}-S{stage_index:02d}"
        if stage_index <= len(oracle):
            item = copy.deepcopy(oracle[stage_index - 1])
            source_update_id = str(item.get("update_id") or f"u{stage_index:03d}")
        else:
            item = {
                "hidden_delta": {"type": "no_op", "reason": "NLDO maintenance re-evaluation stage"},
                "expected_event_types": ["re_evaluate_current_state"],
                "difficulty": "maintenance",
                "requires_memory": False,
                "memory_reference_chain": [],
            }
            source_update_id = f"pad{stage_index:03d}"
        item["update_id"] = new_update_id
        item["stage_id"] = new_update_id
        item["source_update_id"] = source_update_id
        out.append(item)
    return out


def _renumber_evaluation(evaluation: dict[str, Any], problem_id: str, count: int) -> dict[str, Any]:
    out = copy.deepcopy(evaluation)
    trajectory = evaluation.get("reference_trajectory") if isinstance(evaluation.get("reference_trajectory"), list) else []
    new_trajectory: list[dict[str, Any]] = []
    if trajectory:
        initial = copy.deepcopy(trajectory[0])
        initial["update_id"] = "initial"
        initial["stage_id"] = f"{problem_id}-S00"
        new_trajectory.append(initial)
    else:
        new_trajectory.append({"update_id": "initial", "stage_id": f"{problem_id}-S00"})
    last = copy.deepcopy(new_trajectory[-1])
    for stage_index in range(1, count + 1):
        if stage_index < len(trajectory):
            item = copy.deepcopy(trajectory[stage_index])
        else:
            item = copy.deepcopy(last)
            item["hidden_delta_type"] = "no_op"
            item["reference_note"] = "NLDO maintenance re-evaluation stage; hidden state unchanged."
        item["update_id"] = f"{problem_id}-S{stage_index:02d}"
        item["stage_id"] = item["update_id"]
        new_trajectory.append(item)
        last = item
    out["reference_trajectory"] = new_trajectory
    out["nldo_stage_count"] = count + 1
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_10updates_csv.jsonl")
    parser.add_argument("--updates-per-problem", type=int, default=10)
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        metavar="SOURCE_ID=JSONL",
        help="Override a source JSONL path, e.g. S04=logs/.../green_vrp_mo_6episodes_csv.jsonl. May be repeated.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    episodes = build_nldo(
        repo_root,
        updates_per_problem=args.updates_per_problem,
        source_path_overrides=_parse_source_path_overrides(args.source_path),
    )
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in episodes:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(episodes)} NLDO problems to {output}")


def _parse_source_path_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    allowed = {str(spec.get("source_id") or "") for spec in SOURCE_SPECS}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source-path must use SOURCE_ID=JSONL, got: {value}")
        source_id, path = value.split("=", 1)
        source_id = source_id.strip()
        path = path.strip()
        if source_id not in allowed:
            raise SystemExit(f"unknown source id for --source-path: {source_id}")
        if not path:
            raise SystemExit(f"empty path for --source-path {source_id}")
        overrides[source_id] = path
    return overrides


if __name__ == "__main__":
    main()
