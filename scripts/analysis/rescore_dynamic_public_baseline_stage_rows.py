#!/usr/bin/env python3
"""Re-score public-baseline stage rows with the current held-out references."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import evaluate_reference_metrics as metrics  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
DEFAULT_REFERENCE = ROOT / (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
EXPECTED_REFERENCE_POLICY = "offline_multi_seed_ga_pareto_best_known_for_hv_igd_current_state_only_union_500x500x10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def executable_solution(row: dict[str, Any]) -> bool:
    execution = row.get("code_execution") if isinstance(row.get("code_execution"), dict) else {}
    return (
        bool(execution.get("compile_success"))
        and bool(execution.get("runtime_success"))
        and isinstance(row.get("solution"), dict)
        and bool(row.get("solution"))
    )


def state_at_stage(episode: dict[str, Any], stage: int) -> tuple[str, str, dict[str, Any]]:
    benchmark = metrics.effective_benchmark("NLDO", episode)
    domain = str(episode.get("domain") or "")
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    oracle = list(episode.get("hidden_update_oracle") or [])
    for offset in range(max(0, stage)):
        state = metrics.apply_hidden_delta(
            benchmark,
            domain,
            state,
            (oracle[offset].get("hidden_delta") or {}) if offset < len(oracle) else {},
        )
    return benchmark, domain, state


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    benchmark_hash = sha256(args.benchmark)
    reference_hash = sha256(args.reference)
    episodes = {str(row.get("episode_id")): row for row in load_jsonl(args.benchmark)}
    references = {str(row.get("episode_id")): row for row in load_jsonl(args.reference)}
    for episode_id in sorted(references):
        reference = references[episode_id]
        if int(episode_id[-3:]) >= 10:
            policy = str((reference.get("evaluation") or {}).get("reference_policy") or "")
            if policy != EXPECTED_REFERENCE_POLICY:
                errors.append(f"{episode_id}: unexpected reference policy {policy!r}")
        steps = list((reference.get("evaluation") or {}).get("reference_trajectory") or [])
        if len(steps) != 13:
            errors.append(f"{episode_id}: expected 13 reference states, got {len(steps)}")

    rows = load_jsonl(args.stage_rows)
    method_order = {
        name: index
        for index, name in enumerate(
            ("react_tools", "optimus", "orlm", "optimai_2025", "or_llm_agent_2025")
        )
    }
    rows.sort(
        key=lambda row: (
            method_order.get(str(row.get("method") or ""), 99),
            str(row.get("episode_id") or ""),
            int(row.get("stage_index") or 0),
        )
    )
    rescored: list[dict[str, Any]] = []
    previous: dict[tuple[str, str], dict[str, Any] | None] = {}
    hv_values: list[float] = []
    feasibility_changes: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        method = str(row.get("method") or "")
        episode_id = str(row.get("episode_id") or "")
        stage = int(row.get("stage_index") or 0)
        episode = episodes.get(episode_id)
        reference = references.get(episode_id)
        if episode is None or reference is None:
            errors.append(f"{method}/{episode_id}/t{stage:02d}: missing benchmark/reference episode")
            rescored.append(row)
            continue
        source_metric = copy.deepcopy(row.get("hidden_evaluation") or {})
        if executable_solution(row):
            benchmark, domain, state = state_at_stage(episode, stage)
            reference_steps = metrics.reference_trajectory_for_payload(benchmark, domain, reference)
            reference_step = metrics.reference_step_for_stage(
                reference_steps,
                str(row.get("update_id") or ""),
                stage,
            )
            try:
                metric = metrics.score_solution(
                    benchmark,
                    domain,
                    str(episode.get("family") or ""),
                    state,
                    row.get("solution") or {},
                    previous.get((method, episode_id)),
                    reference_step,
                    list(row.get("candidate_archive") or []),
                )
            except Exception as exc:  # noqa: BLE001
                metric = {
                    "feasible": False,
                    "normalized_score": 0.0,
                    "rescore_error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(f"{method}/{episode_id}/t{stage:02d}: {metric['rescore_error']}")
        else:
            metric = {
                "feasible": False,
                "normalized_score": 0.0,
                "not_scored_reason": str(row.get("status") or "non_executable_output"),
            }
        metric = json.loads(json.dumps(metric, default=str))
        old_feasible = bool(source_metric.get("feasible"))
        new_feasible = bool(metric.get("feasible"))
        if old_feasible != new_feasible:
            feasibility_changes.append(
                {
                    "method": method,
                    "episode_id": episode_id,
                    "stage_index": stage,
                    "source_feasible": old_feasible,
                    "rescored_feasible": new_feasible,
                }
            )
        if new_feasible:
            canonical = metric.get("canonical_solution")
            previous[(method, episode_id)] = (
                canonical if isinstance(canonical, dict) else row.get("solution") or {}
            )
            hv = metric.get("normalized_hv")
            if hv is not None:
                value = float(hv)
                if math.isfinite(value):
                    hv_values.append(value)
        row["source_hidden_evaluation"] = source_metric
        row["hidden_evaluation"] = metric
        row["metric_provenance"] = {
            "protocol": "current_held_out_reference_rescore_v1",
            "benchmark": str(args.benchmark),
            "reference": str(args.reference),
            "reference_sha256": reference_hash,
            "reference_policy": str(
                (reference.get("evaluation") or {}).get("reference_policy") or ""
            ),
            "stage_mapping": "update_id_then_chronological_index",
        }
        rescored.append(row)

    if feasibility_changes:
        errors.append(f"held-out rescore changed feasibility in {len(feasibility_changes)} rows")
    summary = {
        "status": "passed" if not errors else "failed",
        "protocol": "dynamic_public_baseline_current_reference_rescore_v1",
        "stage_rows": str(args.stage_rows),
        "output": str(args.output),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": benchmark_hash,
        "reference": str(args.reference),
        "reference_sha256": reference_hash,
        "rows": len(rescored),
        "feasibility_changes": feasibility_changes,
        "dm_feasible_hv_cells": len(hv_values),
        "dm_hv_mean": sum(hv_values) / len(hv_values) if hv_values else None,
        "dm_hv_max": max(hv_values) if hv_values else None,
        "dm_hv_above_one": sum(value > 1.0 + 1e-12 for value in hv_values),
        "errors": errors,
    }
    atomic_jsonl(args.output, rescored)
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
