#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.moea_metrics import (
    approximate_hypervolume,
    dominates,
    expanded_reference_point,
    objective_vector,
)
from scripts.analysis import evaluate_reference_metrics as evaluator


def main() -> int:
    args = parse_args()
    box_scales = sorted({float(value) for value in args.box_scales.split(",") if value.strip()})
    if not box_scales or any(value < 1.0 for value in box_scales):
        raise ValueError("box scales must be >= 1.0 so the current evaluation box is never shrunk")
    reference_episodes = evaluator.load_benchmark(args.reference_jsonl)
    reference_cache: dict[tuple[str, int], dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    above_rows: list[dict[str, Any]] = []
    mapping_errors: list[str] = []
    metric_errors: list[str] = []

    for run_arg in args.run:
        label, raw_dir = run_arg.split("=", 1)
        run_dir = Path(raw_dir)
        metric_lookup = load_metric_lookup(run_dir)
        observed_cells: set[tuple[str, int, int]] = set()
        for jsonl_path in evaluator.iter_result_jsonls(run_dir, "NLDO"):
            for run in evaluator.load_jsonl(jsonl_path):
                episode_id = str(run.get("episode_id") or "")
                episode = reference_episodes.get(episode_id)
                if episode is None or episode.get("domain") not in {
                    "green_vrp_multiobjective",
                    "cloud_scheduling_multiobjective",
                }:
                    continue
                seed = int(run.get("run_seed") or 0)
                stages = evaluator.extract_stages(run)
                stage_lookup = {index: stage for index, stage in enumerate(stages)}
                state = copy.deepcopy(episode.get("hidden_initial_state") or {})
                oracle = list(episode.get("hidden_update_oracle") or [])
                reference_steps = list((episode.get("evaluation") or {}).get("reference_trajectory") or [])
                for stage_index in range(0, 1 + len(episode.get("update_stream") or [])):
                    if stage_index > 0:
                        state = evaluator.apply_hidden_delta(
                            evaluator.effective_benchmark("NLDO", episode),
                            str(episode.get("domain") or ""),
                            state,
                            (oracle[stage_index - 1].get("hidden_delta") or {}),
                        )
                    if stage_index < args.stage_min or (
                        args.stage_max is not None and stage_index > args.stage_max
                    ):
                        continue
                    reference_step = reference_steps[stage_index]
                    stage = stage_lookup.get(stage_index)
                    expected_update_id = "initial" if stage_index == 0 else str(
                        oracle[stage_index - 1].get("update_id") or ""
                    )
                    actual_update_id = str(stage.get("update_id") or "") if stage else ""
                    reference_update_id = str(reference_step.get("update_id") or "")
                    if stage is not None and actual_update_id != expected_update_id:
                        mapping_errors.append(
                            f"{label}:{episode_id}:seed{seed}:t{stage_index:02d} "
                            f"actual={actual_update_id} expected={expected_update_id}"
                        )
                    if reference_update_id != expected_update_id:
                        mapping_errors.append(
                            f"reference:{episode_id}:t{stage_index:02d} "
                            f"reference={reference_update_id} expected={expected_update_id}"
                        )
                    cache_key = (episode_id, stage_index)
                    reference_info = reference_cache.get(cache_key)
                    if reference_info is None:
                        reference_info = audit_reference_step(episode, state, reference_step, box_scales)
                        reference_cache[cache_key] = reference_info
                    hidden_archive = hidden_candidate_archive(episode, state, stage)
                    objectives = list(reference_step.get("objectives") or [])
                    candidate_vectors = [objective_vector(item, objectives) for item in hidden_archive]
                    reference_vectors = reference_info["reference_vectors"]
                    union_vectors = [*reference_vectors, *candidate_vectors]
                    unclipped_ideal = {
                        name: min(vector[index] for vector in union_vectors)
                        for index, name in enumerate(objectives)
                    }
                    novel_vectors = [
                        vector
                        for vector in candidate_vectors
                        if not any(dominates(other, vector) or vectors_equal(other, vector) for other in reference_vectors)
                    ]
                    better_than_reference_ideal = [
                        name
                        for index, name in enumerate(objectives)
                        if candidate_vectors
                        and min(vector[index] for vector in candidate_vectors)
                        < float(reference_info["ideal"][name]) - 1e-9
                    ]
                    cell = (episode_id, stage_index, seed)
                    observed_cells.add(cell)
                    for scale in box_scales:
                        box = reference_info["boxes"][scale]
                        candidate_hv = float(
                            approximate_hypervolume(
                                hidden_archive,
                                objectives,
                                box,
                                reference_info["ideal"],
                                samples=0,
                            )
                        )
                        reference_hv = float(reference_info["reference_hv_by_scale"][scale])
                        ratio = candidate_hv / reference_hv if reference_hv > 0 else 0.0
                        unclipped_candidate_hv = float(
                            approximate_hypervolume(
                                hidden_archive,
                                objectives,
                                box,
                                unclipped_ideal,
                                samples=0,
                            )
                        )
                        unclipped_reference_hv = float(
                            approximate_hypervolume(
                                reference_info["reference_archive"],
                                objectives,
                                box,
                                unclipped_ideal,
                                samples=0,
                            )
                        )
                        unclipped_ratio = (
                            unclipped_candidate_hv / unclipped_reference_hv
                            if unclipped_reference_hv > 0
                            else 0.0
                        )
                        detail_rows.append(
                            {
                                "method": label,
                                "episode_id": episode_id,
                                "stage_index": stage_index,
                                "run_seed": seed,
                                "update_id": expected_update_id,
                                "box_scale": scale,
                                "candidate_archive_size": len(hidden_archive),
                                "reference_archive_size": len(reference_vectors),
                                "novel_candidate_points": len(novel_vectors),
                                "candidate_hv": round(candidate_hv, 9),
                                "reference_hv": round(reference_hv, 9),
                                "hv_ratio": round(ratio, 9),
                                "unclipped_candidate_hv": round(unclipped_candidate_hv, 9),
                                "unclipped_reference_hv": round(unclipped_reference_hv, 9),
                                "unclipped_hv_ratio": round(unclipped_ratio, 9),
                                "candidate_better_than_reference_ideal": ",".join(
                                    better_than_reference_ideal
                                ),
                            }
                        )
                        if math.isclose(scale, 1.0) and ratio > 1.0 + args.above_tolerance:
                            above_rows.append(
                                {
                                    "method": label,
                                    "episode_id": episode_id,
                                    "stage_index": stage_index,
                                    "run_seed": seed,
                                    "update_id": expected_update_id,
                                    "candidate_hv": round(candidate_hv, 9),
                                    "reference_hv": round(reference_hv, 9),
                                    "hv_ratio": round(ratio, 9),
                                    "candidate_archive_size": len(hidden_archive),
                                    "reference_archive_size": len(reference_vectors),
                                    "novel_candidate_points": len(novel_vectors),
                                    "novel_objective_vectors": [list(vector) for vector in sorted(set(novel_vectors))],
                                    "candidate_better_than_reference_ideal": better_than_reference_ideal,
                                    "unclipped_hv_ratio": round(unclipped_ratio, 9),
                                    "evaluation_reference_point": box,
                                    "evaluation_ideal_point": reference_info["ideal"],
                                }
                            )
                    expected_metric = metric_lookup.get(cell)
                    if expected_metric is not None:
                        scale_one = next(
                            row
                            for row in reversed(detail_rows)
                            if row["method"] == label
                            and row["episode_id"] == episode_id
                            and row["stage_index"] == stage_index
                            and row["run_seed"] == seed
                            and math.isclose(float(row["box_scale"]), 1.0)
                        )
                        compare_metric_row(expected_metric, scale_one, metric_errors)
        expected_cells = {
            (episode_id, stage, seed)
            for episode_id, episode in reference_episodes.items()
            if episode.get("domain") in {"green_vrp_multiobjective", "cloud_scheduling_multiobjective"}
            for stage in range(args.stage_min, 1 + len(episode.get("update_stream") or []))
            if args.stage_max is None or stage <= args.stage_max
            for seed in range(args.expected_seeds)
        }
        missing = sorted(expected_cells - observed_cells)
        if missing:
            mapping_errors.append(f"{label}: missing raw cells: {missing[:12]} (total={len(missing)})")

    sensitivity_rows = aggregate_sensitivity(detail_rows)
    report = {
        "schema_version": "nldo_reference_robustness_audit_v2",
        "reference_jsonl": str(args.reference_jsonl),
        "reference_sha256": sha256_file(args.reference_jsonl),
        "runs": args.run,
        "stage_min": args.stage_min,
        "stage_max": args.stage_max,
        "box_scales": box_scales,
        "above_tolerance": args.above_tolerance,
        "detail_cells": len(detail_rows),
        "above_reference_cells": above_rows,
        "above_reference_count": len(above_rows),
        "reference_objective_max_abs_error": max(
            (float(info["objective_max_abs_error"]) for info in reference_cache.values()),
            default=0.0,
        ),
        "reference_stage_count": len(reference_cache),
        "mapping_errors": mapping_errors,
        "metric_errors": metric_errors,
        "sensitivity": sensitivity_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "box_sensitivity_cells.csv", detail_rows)
    write_csv(args.out_dir / "box_sensitivity_summary.csv", sensitivity_rows)
    (args.out_dir / "reference_robustness_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if mapping_errors or metric_errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit above-reference NLDO Pareto cells and recompute HV under fixed expanded boxes."
    )
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--box-scales", default="1.0,1.1,1.25")
    parser.add_argument("--stage-min", type=int, default=1)
    parser.add_argument("--stage-max", type=int)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--above-tolerance", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def audit_reference_step(
    episode: dict[str, Any],
    state: dict[str, Any],
    reference_step: dict[str, Any],
    box_scales: list[float],
) -> dict[str, Any]:
    objectives = list(reference_step.get("objectives") or [])
    reference_archive = list(reference_step.get("pareto_archive") or [])
    module_name = (
        "scripts.materialize_green_vrp_mo"
        if episode.get("domain") == "green_vrp_multiobjective"
        else "scripts.materialize_cloud_scheduling_mo"
    )
    module = importlib.import_module(module_name)
    max_error = 0.0
    for item in reference_archive:
        rescored = module._score_solution(state, item.get("solution") or {}, None)
        for name in objectives:
            max_error = max(
                max_error,
                abs(float(rescored.get(name, 0.0)) - float((item.get("objectives") or {}).get(name, 0.0))),
            )
    ideal = {name: float((reference_step.get("hypervolume_ideal_point") or {})[name]) for name in objectives}
    base_box = expanded_reference_point(
        reference_archive,
        objectives,
        reference_step.get("hypervolume_reference_point"),
    )
    boxes = {
        scale: {
            name: float(ideal[name] + scale * (float(base_box[name]) - float(ideal[name])))
            for name in objectives
        }
        for scale in box_scales
    }
    reference_hv_by_scale = {
        scale: approximate_hypervolume(reference_archive, objectives, boxes[scale], ideal, samples=0)
        for scale in box_scales
    }
    return {
        "reference_archive": reference_archive,
        "reference_vectors": [objective_vector(item, objectives) for item in reference_archive],
        "ideal": ideal,
        "boxes": boxes,
        "reference_hv_by_scale": reference_hv_by_scale,
        "objective_max_abs_error": max_error,
    }


def hidden_candidate_archive(
    episode: dict[str, Any], state: dict[str, Any], stage: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if stage is None:
        return []
    solution = stage.get("solution") or {}
    candidates = stage.get("candidate_archive") or []
    if episode.get("domain") == "green_vrp_multiobjective":
        module = importlib.import_module("scripts.materialize_green_vrp_mo")
        canonical = evaluator.normalize_green_solution_ids(state, evaluator._canonical_green_solution(solution))
        return evaluator.hidden_green_archive(module, state, canonical, None, candidates)
    module = importlib.import_module("scripts.materialize_cloud_scheduling_mo")
    canonical = evaluator.canonical_cloud(solution)
    return evaluator.hidden_cloud_archive(module, state, canonical, None, candidates)


def load_metric_lookup(run_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    path = run_dir / "reference_metrics" / "reference_stage_metrics.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row.get("episode_id") or ""), int(row.get("stage_index") or 0), int(row.get("run_seed") or 0)): row
        for row in rows
    }


def compare_metric_row(expected: dict[str, Any], recomputed: dict[str, Any], errors: list[str]) -> None:
    fields = (("hv", "candidate_hv"), ("reference_hv", "reference_hv"), ("normalized_hv", "hv_ratio"))
    for old_key, new_key in fields:
        old = expected.get(old_key)
        if old in (None, ""):
            continue
        if abs(float(old) - float(recomputed[new_key])) > 2e-6:
            errors.append(
                f"{recomputed['method']}:{recomputed['episode_id']}:seed{recomputed['run_seed']}:"
                f"t{recomputed['stage_index']:02d} {old_key}={old} recomputed={recomputed[new_key]}"
            )


def aggregate_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["box_scale"]))].append(float(row["hv_ratio"]))
    out = []
    for (method, scale), values in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "box_scale": scale,
                "stage_seed_cells": len(values),
                "mean_hv_ratio": mean(values) if values else 0.0,
                "above_reference_cells": sum(value > 1.0 + 1e-6 for value in values),
            }
        )
    return out


def vectors_equal(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= 1e-9 for x, y in zip(a, b))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
