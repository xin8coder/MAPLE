#!/usr/bin/env python3
"""Aggregate the current P014/P015 archive-cap replay into paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    100: ROOT / "outputs/liveopt_archivecap100_p014_p015_200x200x10_20260716",
    200: ROOT / "outputs/liveopt_archivecap200_p014_p015_200x200x10_20260716",
    500: ROOT / "outputs/liveopt_archivecap500_p014_p015_200x200x10_20260716",
}
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_archive_cap_sensitivity_p014_p015_20260716"
DEFAULT_TABLE = ROOT / "release_artifacts/paper_table_exports/liveopt_archive_cap_sensitivity_rows.tex"
DEFAULT_BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
EXPECTED_BENCHMARK_SHA256 = "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303"
EXPECTED_REFERENCE_SHA256 = "50126c25eb837bdbf2dbc2e7710226472027a1a880305d331dc006438a02c5e5"
EPISODES = ("NLDO-P014", "NLDO-P015")
STAGES = tuple(range(1, 13))
SEEDS = tuple(range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", help="CAP=RUN_DIR; repeat for 100, 200, and 500")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def parse_runs(values: list[str] | None) -> dict[int, Path]:
    if not values:
        return dict(DEFAULT_RUNS)
    runs: dict[int, Path] = {}
    for value in values:
        cap_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value: {value!r}")
        runs[int(cap_text)] = Path(path_text)
    if set(runs) != {100, 200, 500}:
        raise ValueError("--run must provide caps 100, 200, and 500")
    return runs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def float_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def load_metrics(run_dir: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    path = run_dir / "reference_metrics/reference_stage_metrics.csv"
    rows: dict[tuple[str, int, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            stage = int(row.get("stage_index") or -1)
            if episode_id not in EPISODES or stage not in STAGES:
                continue
            key = (episode_id, stage, int(row["run_seed"]))
            rows[key] = {
                "hv": float_value(row.get("normalized_hv")),
                "igd": float_value(row.get("igd")),
                "feasible": float(bool(str(row.get("true_pass") or "").lower() == "true")),
            }
    return rows


def archive_cap_active(metadata: dict[str, Any], archive_size: int, cap: int) -> bool:
    runtime = metadata.get("runtime") or {}
    diagnostics = metadata.get("archive_diagnostics") or runtime.get("archive_diagnostics") or {}
    if "cap_active" in diagnostics:
        return bool(diagnostics.get("cap_active"))
    return archive_size >= cap


def load_runtime(run_dir: Path, cap: int) -> dict[tuple[str, int, int], dict[str, float]]:
    path = run_dir / "NLDO/evo2_limit0.jsonl"
    rows: dict[tuple[str, int, int], dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            run = json.loads(line)
            episode_id = str(run.get("episode_id") or "")
            if episode_id not in EPISODES:
                continue
            seed = int(run.get("run_seed", -1))
            for stage, update in enumerate(run.get("update_results") or [], start=1):
                if stage not in STAGES:
                    continue
                solver = update.get("solver_result") or {}
                metadata = solver.get("metadata") or {}
                runtime = metadata.get("runtime") or {}
                archive = metadata.get("candidate_archive") or []
                rows[(episode_id, stage, seed)] = {
                    "generations": float_value(runtime.get("executed_generations")),
                    "archive_size": float(len(archive)),
                    "cap_active": float(archive_cap_active(metadata, len(archive), cap)),
                }
    return rows


def normalized_plan(plan: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(plan))
    budget = payload.get("budget_config") or {}
    budget.pop("archive_limit", None)
    return payload


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summarize(
    cap: int,
    metrics: dict[tuple[str, int, int], dict[str, float]],
    runtime: dict[tuple[str, int, int], dict[str, float]],
) -> dict[str, Any]:
    keys = sorted(metrics)
    result: dict[str, Any] = {
        "archive_cap": cap,
        "cells": len(keys),
        "mean_hv": mean([metrics[key]["hv"] for key in keys]),
        "mean_igd": mean([metrics[key]["igd"] for key in keys]),
        "solve_rate": mean([metrics[key]["feasible"] for key in keys]),
        "mean_generations": mean([runtime[key]["generations"] for key in keys]),
        "mean_archive_size": mean([runtime[key]["archive_size"] for key in keys]),
        "max_archive_size": max((runtime[key]["archive_size"] for key in keys), default=0.0),
        "cap_active_cells": sum(runtime[key]["cap_active"] for key in keys),
    }
    for episode_id in EPISODES:
        episode_keys = [key for key in keys if key[0] == episode_id]
        result[f"{episode_id}_hv"] = mean([metrics[key]["hv"] for key in episode_keys])
        result[f"{episode_id}_igd"] = mean([metrics[key]["igd"] for key in episode_keys])
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Generated by scripts/analysis/export_liveopt_archive_cap_sensitivity.py.",
        r"\newcommand{\LiveOptArchiveCapSensitivityRows}{%",
    ]
    for row in rows:
        lines.append(
            f"{row['archive_cap']} & {row['NLDO-P014_hv']:.3f} & {row['NLDO-P015_hv']:.3f} & "
            f"{row['mean_hv']:.3f} & {row['mean_igd']:.3f} & {row['mean_generations']:.1f} & "
            f"{int(row['cap_active_cells'])}/{int(row['cells'])} " + r"\\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs = parse_runs(args.run)
    errors: list[str] = []
    metrics_by_cap: dict[int, dict[tuple[str, int, int], dict[str, float]]] = {}
    runtime_by_cap: dict[int, dict[tuple[str, int, int], dict[str, float]]] = {}
    plans: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    expected_keys = {(episode, stage, seed) for episode in EPISODES for stage in STAGES for seed in SEEDS}
    benchmark_sha256 = sha256_file(DEFAULT_BENCHMARK)
    if benchmark_sha256 != EXPECTED_BENCHMARK_SHA256:
        errors.append("benchmark checksum does not match the frozen current P014/P015 replay input")
    reference_paths: set[Path] = set()
    for cap, run_dir in sorted(runs.items()):
        status_path = run_dir / "launcher_status.json"
        plan_path = run_dir / "replay_plan.json"
        metric_path = run_dir / "reference_metrics/reference_stage_metrics.csv"
        jsonl_path = run_dir / "NLDO/evo2_limit0.jsonl"
        for path in (status_path, plan_path, metric_path, jsonl_path):
            if not path.exists():
                raise FileNotFoundError(path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        reference_exit = status.get("reference_eval_exit_code")
        if status.get("status") != "completed" or int(reference_exit if reference_exit is not None else -1) != 0:
            errors.append(f"cap {cap}: launcher/reference evaluation is not complete")
        plans[cap] = json.loads(plan_path.read_text(encoding="utf-8"))
        if int((plans[cap].get("budget_config") or {}).get("archive_limit") or -1) != cap:
            errors.append(f"cap {cap}: replay plan archive limit mismatch")
        reference_path = resolve_repo_path(str((plans[cap].get("budget_config") or {}).get("reference_jsonl") or ""))
        reference_paths.add(reference_path.resolve())
        episode_inputs: dict[str, Any] = {}
        for episode_id in EPISODES:
            summary_path = run_dir / "replay_jobs" / episode_id / "summary.json"
            command_path = run_dir / "replay_jobs" / episode_id / "command.json"
            for path in (summary_path, command_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            replay_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            replay_benchmark = resolve_repo_path(str(replay_summary.get("episodes_jsonl") or ""))
            if replay_summary.get("status") != "completed" or replay_benchmark.resolve() != DEFAULT_BENCHMARK.resolve():
                errors.append(f"cap {cap} {episode_id}: replay summary does not use the frozen current benchmark")
            episode_inputs[episode_id] = {
                "command_sha256": sha256_file(command_path),
                "summary_sha256": sha256_file(summary_path),
            }
        metrics_by_cap[cap] = load_metrics(run_dir)
        runtime_by_cap[cap] = load_runtime(run_dir, cap)
        if set(metrics_by_cap[cap]) != expected_keys:
            errors.append(f"cap {cap}: metric coverage {len(metrics_by_cap[cap])} != 240")
        if set(runtime_by_cap[cap]) != expected_keys:
            errors.append(f"cap {cap}: runtime coverage {len(runtime_by_cap[cap])} != 240")
        provenance[str(cap)] = {
            "run_dir": str(run_dir),
            "replay_plan_sha256": sha256_file(plan_path),
            "metrics_sha256": sha256_file(metric_path),
            "run_jsonl_sha256": sha256_file(jsonl_path),
            "episode_inputs": episode_inputs,
        }
    if len(reference_paths) != 1:
        errors.append("archive-cap plans do not share one held-out reference")
        reference_path = Path()
        reference_sha256 = ""
    else:
        reference_path = next(iter(reference_paths))
        reference_sha256 = sha256_file(reference_path)
        if reference_sha256 != EXPECTED_REFERENCE_SHA256:
            errors.append("held-out reference checksum does not match the frozen 500x500x10 reference")
    base_plan = normalized_plan(plans[500])
    for cap in (100, 200):
        if normalized_plan(plans[cap]) != base_plan:
            errors.append(f"cap {cap}: replay plan differs from cap 500 beyond archive_limit")

    rows = [summarize(cap, metrics_by_cap[cap], runtime_by_cap[cap]) for cap in (100, 200, 500)]
    cap500 = metrics_by_cap[500]
    for row in rows:
        cap = int(row["archive_cap"])
        row["delta_hv_vs_500"] = row["mean_hv"] - rows[-1]["mean_hv"]
        row["mean_abs_cell_hv_delta_vs_500"] = mean(
            [abs(metrics_by_cap[cap][key]["hv"] - cap500[key]["hv"]) for key in sorted(expected_keys)]
        )
        row["max_abs_cell_hv_delta_vs_500"] = max(
            [abs(metrics_by_cap[cap][key]["hv"] - cap500[key]["hv"]) for key in sorted(expected_keys)],
            default=0.0,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", rows)
    cell_rows: list[dict[str, Any]] = []
    for cap in (100, 200, 500):
        for key in sorted(expected_keys):
            cell_rows.append(
                {
                    "archive_cap": cap,
                    "episode_id": key[0],
                    "stage_index": key[1],
                    "run_seed": key[2],
                    **metrics_by_cap[cap][key],
                    **runtime_by_cap[cap][key],
                }
            )
    write_csv(args.out_dir / "cells.csv", cell_rows)
    args.paper_table.parent.mkdir(parents=True, exist_ok=True)
    args.paper_table.write_text(render_table(rows), encoding="utf-8")
    payload = {
        "status": "passed" if not errors else "failed",
        "protocol": "liveopt_archive_cap_sensitivity_p014_p015_current_benchmark_v1",
        "scope": {"episodes": list(EPISODES), "stages": list(STAGES), "seeds": list(SEEDS)},
        "fixed_budget": "population=200, generations<=200, reference-free early stop patience=25",
        "only_changed_field": "archive_limit",
        "inputs": {
            "benchmark_jsonl": str(DEFAULT_BENCHMARK),
            "benchmark_sha256": benchmark_sha256,
            "reference_jsonl": str(reference_path),
            "reference_sha256": reference_sha256,
        },
        "rows": rows,
        "errors": errors,
        "provenance": provenance,
        "paper_table": str(args.paper_table),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
