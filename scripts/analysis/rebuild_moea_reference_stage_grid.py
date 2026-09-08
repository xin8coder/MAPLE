#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.reference_solvers.moea_reference import MOEAConfig
from scripts.generate_moea_references import enhance_episode, run_episode


MO_DOMAINS = {"green_vrp_multiobjective", "cloud_scheduling_multiobjective"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild multi-objective NLDO references one episode-stage at a time.")
    parser.add_argument("--input", type=Path, default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_10updates_csv.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--archive-limit", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--episode-id", action="append", default=[], help="Optional episode id to refresh. May be repeated.")
    parser.add_argument("--stage-index", action="append", type=int, default=[], help="Optional 0-based stage index to refresh. May be repeated.")
    parser.add_argument(
        "--preserve-reference",
        type=Path,
        help=(
            "Optional enhanced benchmark whose strong reference trajectories are preserved "
            "for stages not selected by --stage-index. The public problem state still comes "
            "from --input."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    preserved_reference_rows = (
        {str(row["episode_id"]): row for row in load_jsonl(args.preserve_reference)}
        if args.preserve_reference
        else {}
    )
    selected_episodes = set(args.episode_id or [])
    selected_stage_indices = set(args.stage_index or [])
    mo_rows = [
        (idx, ep)
        for idx, ep in enumerate(rows)
        if ep.get("domain") in MO_DOMAINS and (not selected_episodes or str(ep.get("episode_id")) in selected_episodes)
    ]
    stage_dir = args.out_dir / "stage_references"
    stage_dir.mkdir(parents=True, exist_ok=True)
    config = MOEAConfig(
        population_size=args.population,
        generations=args.generations,
        seeds=args.seeds,
        archive_limit=args.archive_limit,
        hv_samples=0,
    )

    tasks = []
    for source_idx, ep in mo_rows:
        stage_count = 1 + len(ep.get("update_stream") or [])
        for stage_index in range(stage_count):
            if selected_stage_indices and stage_index not in selected_stage_indices:
                continue
            path = stage_path(stage_dir, ep["episode_id"], stage_index)
            if path.exists() and not args.force:
                continue
            tasks.append((source_idx, ep, stage_index, path, config))

    print(json.dumps({"event": "start", "tasks": len(tasks), "workers": args.workers, "out_dir": str(args.out_dir)}, ensure_ascii=False), flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_stage, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps({"event": "stage_written", **result}, ensure_ascii=False), flush=True)

    enhanced_rows = merge_rows(
        rows,
        mo_rows,
        stage_dir,
        config,
        args.out_dir,
        selected_stage_indices,
        preserved_reference_rows,
        args.preserve_reference,
    )
    enhanced_path = args.out_dir / enhanced_filename(args.input, rows, config)
    write_jsonl(enhanced_path, enhanced_rows)
    summary = summarize(enhanced_rows, args.input, enhanced_path, config, args.preserve_reference)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "enhanced": str(enhanced_path), "summary": str(summary_path)}, ensure_ascii=False), flush=True)


def run_stage(source_idx: int, ep: dict[str, Any], stage_index: int, path: Path, config: MOEAConfig) -> dict[str, Any]:
    reference = run_episode(
        copy.deepcopy(ep),
        config,
        seed_offset=source_idx * 100000,
        selected_stage_indices={stage_index},
        state_independent_objectives=True,
    )
    if len(reference.get("reference_trajectory") or []) != 1:
        raise RuntimeError(f"expected one refreshed stage for {ep['episode_id']} S{stage_index:02d}")
    step = reference["reference_trajectory"][0]
    payload = {
        "episode_id": ep["episode_id"],
        "stage_index": stage_index,
        "update_id": step.get("update_id"),
        "reference_policy": reference.get("reference_policy"),
        "reference_step": step,
        "solver_config": config.__dict__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "episode_id": ep["episode_id"],
        "stage_index": stage_index,
        "update_id": step.get("update_id"),
        "archive_size": step.get("archive_size"),
        "reference_hv": step.get("reference_hv"),
        "path": str(path),
    }


def merge_rows(
    rows: list[dict[str, Any]],
    mo_rows: list[tuple[int, dict[str, Any]]],
    stage_dir: Path,
    config: MOEAConfig,
    out_dir: Path,
    selected_stage_indices: set[int],
    preserved_reference_rows: dict[str, dict[str, Any]],
    preserved_reference_path: Path | None,
) -> list[dict[str, Any]]:
    mo_ids = {ep["episode_id"] for _, ep in mo_rows}
    merged = []
    for ep in rows:
        if ep.get("episode_id") not in mo_ids:
            merged.append(copy.deepcopy(ep))
            continue
        trajectory = []
        preserved_episode = preserved_reference_rows.get(str(ep["episode_id"]))
        original_trajectory = list(
            (
                (preserved_episode.get("evaluation") or {}).get("reference_trajectory")
                if preserved_episode
                else (ep.get("evaluation") or {}).get("reference_trajectory")
            )
            or []
        )
        refreshed_indices = []
        for stage_index in range(1 + len(ep.get("update_stream") or [])):
            path = stage_path(stage_dir, ep["episode_id"], stage_index)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                trajectory.append(payload["reference_step"])
                refreshed_indices.append(stage_index)
            elif selected_stage_indices:
                if stage_index < len(original_trajectory):
                    preserved_step = original_trajectory[stage_index]
                    if not is_strong_reference_step(preserved_step, config):
                        source = str(preserved_reference_path) if preserved_reference_path else "--input"
                        raise ValueError(
                            f"cannot preserve {ep['episode_id']} S{stage_index:02d} from {source}: "
                            "the step is not a complete strong MOEA reference. Pass "
                            "--preserve-reference with the prior 500x500x10 enhanced payload."
                        )
                    trajectory.append(copy.deepcopy(preserved_step))
                else:
                    raise FileNotFoundError(f"missing original reference stage {ep['episode_id']} S{stage_index:02d}")
            else:
                raise FileNotFoundError(f"missing rebuilt reference stage {path}")
        reference = {
            "episode_id": ep["episode_id"],
            "benchmark": ep.get("benchmark"),
            "domain": ep.get("domain"),
            "reference_policy": "offline_multi_seed_ga_pareto_best_known_for_hv_igd_current_state_only_union_500x500x10",
            "reference_trajectory": trajectory,
            # ``trajectory`` is already the complete hybrid trajectory. Replace the
            # input's constructive trajectory atomically instead of asking
            # ``enhance_episode`` to merge it a second time.
            "partial_stage_refresh": False,
            "selected_stage_ids": [trajectory[index].get("update_id") for index in refreshed_indices if index < len(trajectory)],
            "selected_stage_indices": refreshed_indices,
            "solver_config": config.__dict__,
        }
        enhanced = enhance_episode(ep, reference)
        # The enhanced payload is a formal scoring artifact, not a benchmark
        # construction workspace. Keep the cheap constructive trajectory in
        # the source benchmark only; excluding it here prevents downstream
        # tools from mistaking a bootstrap archive for the held-out grid.
        enhanced.setdefault("evaluation", {}).pop("reference_trajectory_constructive_backup", None)
        if selected_stage_indices:
            enhanced.setdefault("evaluation", {})["reference_refresh"] = {
                "mode": "partial_stage_refresh",
                "selected_stage_ids": [
                    trajectory[index].get("update_id")
                    for index in refreshed_indices
                    if index < len(trajectory)
                ],
                "selected_stage_indices": refreshed_indices,
                "preserved_stage_count": len(trajectory) - len(refreshed_indices),
                "preserved_reference_source": str(preserved_reference_path)
                if preserved_reference_path
                else "input.evaluation.reference_trajectory",
            }
        enhanced.setdefault("evaluation", {})["reference_rebuild"] = {
            "stage_grid_dir": str(out_dir / "stage_references"),
            "archive_merge": "union final populations from 10 seeds, nondominated sort, no truncation below 5000",
            "constructive_bootstrap": "excluded from formal enhanced payload",
            "solver_config": config.__dict__,
        }
        merged.append(enhanced)
    return merged


def is_strong_reference_step(step: dict[str, Any], config: MOEAConfig) -> bool:
    solver_config = step.get("solver_config") if isinstance(step.get("solver_config"), dict) else {}
    try:
        budget_matches = (
            int(solver_config.get("population_size")) == int(config.population_size)
            and int(solver_config.get("generations")) == int(config.generations)
            and int(solver_config.get("seeds")) == int(config.seeds)
        )
    except (TypeError, ValueError):
        budget_matches = False
    return bool(
        step.get("solver") == "offline_multi_seed_nsga2_style_ga"
        and step.get("reference_type") == "pareto_ga_best_known"
        and budget_matches
        and step.get("pareto_archive")
        and step.get("reference_hv") not in (None, "")
        and isinstance(step.get("hypervolume_reference_point"), dict)
        and isinstance(step.get("hypervolume_ideal_point"), dict)
    )


def summarize(
    rows: list[dict[str, Any]],
    input_path: Path,
    enhanced_path: Path,
    config: MOEAConfig,
    preserved_reference_path: Path | None,
) -> dict[str, Any]:
    archives = []
    for ep in rows:
        if ep.get("domain") not in MO_DOMAINS:
            continue
        for step in (ep.get("evaluation") or {}).get("reference_trajectory") or []:
            archives.append(
                {
                    "episode_id": ep.get("episode_id"),
                    "update_id": step.get("update_id"),
                    "archive_size": step.get("archive_size"),
                    "reference_hv": step.get("reference_hv"),
                }
            )
    return {
        "input": str(input_path),
        "preserve_reference": str(preserved_reference_path) if preserved_reference_path else None,
        "enhanced_output": str(enhanced_path),
        "config": config.__dict__,
        "mo_stage_count": len(archives),
        "archives": archives,
    }


def enhanced_filename(input_path: Path, rows: list[dict[str, Any]], config: MOEAConfig) -> str:
    update_counts = sorted({len(row.get("update_stream") or []) for row in rows})
    if len(update_counts) == 1:
        update_label = f"{update_counts[0]}updates"
    else:
        update_label = "mixed_updates"
    stem = input_path.stem
    if "updates" not in stem:
        stem = f"{stem}_{update_label}"
    else:
        import re

        stem = re.sub(r"\d+updates", update_label, stem)
    return f"{stem}.strong_moea_{config.population_size}x{config.generations}x{config.seeds}.jsonl"


def stage_path(stage_dir: Path, episode_id: str, stage_index: int) -> Path:
    return stage_dir / episode_id / f"S{stage_index:02d}.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
