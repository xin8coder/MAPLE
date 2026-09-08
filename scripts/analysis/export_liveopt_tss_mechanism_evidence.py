#!/usr/bin/env python3
"""Export the matched typed-vs-untyped Workbench mechanism evidence.

This exporter is intentionally offline.  It reads the frozen 200x200x10
replays and their held-out evaluation rows; it never invokes a provider or
reruns optimization.  Besides the aggregate table, it replays the hidden
constraint checks on the final submitted solution of every failed update so
that the paper's failure attribution is derived from the current benchmark
rather than written by hand.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.evaluate_green_agent_run import _green_constraint_report
from scripts.analysis.evaluate_reference_metrics import canonical_cloud, normalize_green_solution_ids
from scripts.materialize_cloud_scheduling_mo import _apply_delta as apply_cloud_delta
from scripts.materialize_green_vrp_mo import _apply_delta as apply_green_delta


DEFAULT_BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
DEFAULT_LIVEOPT = ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714"
DEFAULT_NO_TSS = ROOT / "outputs/liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_tss_mechanism_rows.tex"
EPISODES = [f"NLDO-P{index:03d}" for index in range(10, 16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--liveopt-run", type=Path, default=DEFAULT_LIVEOPT)
    parser.add_argument("--no-tss-run", type=Path, default=DEFAULT_NO_TSS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in read_jsonl(path)}


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "reference_metrics/reference_stage_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("stage_type") == "update"]


def summarize_method(label: str, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = metric_rows(run_dir)
    passed = [row for row in rows if bool_value(row.get("true_pass"))]
    mismatch = [
        row
        for row in rows
        if bool_value(row.get("agent_reported_feasible")) and not bool_value(row.get("true_pass"))
    ]
    all_hv = [float_value(row.get("normalized_hv")) for row in rows]
    passed_hv = [float_value(row.get("normalized_hv")) for row in passed]
    summary = {
        "method": label,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "update_cells": len(rows),
        "hidden_pass_cells": len(passed),
        "hidden_pass_rate": len(passed) / len(rows) if rows else 0.0,
        "self_feasible_hidden_fail_cells": len(mismatch),
        "all_cell_hv": sum(all_hv) / len(all_hv) if all_hv else 0.0,
        "conditional_hv": sum(passed_hv) / len(passed_hv) if passed_hv else None,
    }
    episode_rows: list[dict[str, Any]] = []
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode_id, items in sorted(by_episode.items()):
        valid = [row for row in items if bool_value(row.get("true_pass"))]
        episode_rows.append(
            {
                "method": label,
                "episode_id": episode_id,
                "update_cells": len(items),
                "hidden_pass_cells": len(valid),
                "hidden_pass_rate": len(valid) / len(items),
                "all_cell_hv": sum(float_value(row.get("normalized_hv")) for row in items) / len(items),
                "conditional_hv": (
                    sum(float_value(row.get("normalized_hv")) for row in valid) / len(valid) if valid else None
                ),
            }
        )
    return summary, episode_rows


def cloud_violation_report(state: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    assignments = canonical_cloud(solution).get("assignments", {})
    jobs = {str(item["id"]): item for item in state.get("jobs", []) if item.get("active", True)}
    machines = {str(item["id"]): item for item in state.get("machines", [])}
    cpu = {machine_id: 0.0 for machine_id in machines}
    mem = {machine_id: 0.0 for machine_id in machines}
    violations: dict[str, Any] = defaultdict(list)
    for job_id, machine_id in assignments.items():
        if job_id not in jobs:
            violations["unknown_job"].append(job_id)
            continue
        if machine_id not in machines:
            violations["unknown_machine"].append(machine_id)
            continue
        job = jobs[job_id]
        machine = machines[machine_id]
        if not machine.get("available", True):
            violations["unavailable_machine"].append({"job": job_id, "machine": machine_id})
        if job.get("gpu_required", False) and not machine.get("gpu", False):
            violations["gpu_mismatch"].append({"job": job_id, "machine": machine_id})
        cpu[machine_id] += float(job.get("cpu", 0.0) or 0.0)
        mem[machine_id] += float(job.get("mem", 0.0) or 0.0)
    missing = sorted(set(jobs) - set(assignments))
    if missing:
        violations["missing_active_jobs"].extend(missing)
    for machine_id, machine in machines.items():
        if cpu[machine_id] > float(machine.get("cpu", 0.0) or 0.0) + 1e-9:
            violations["cpu_capacity"].append(machine_id)
        if mem[machine_id] > float(machine.get("mem", 0.0) or 0.0) + 1e-9:
            violations["mem_capacity"].append(machine_id)
    return {"feasible": not violations, "violations": dict(violations)}


def episode_replay_path(run_dir: Path, episode_id: str) -> Path:
    candidates = [
        run_dir / f"replay_jobs/{episode_id}/NLDO/evo2_limit0.jsonl.gz",
        run_dir / f"replay_jobs/{episode_id}/NLDO/evo2_limit0.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no per-episode replay for {episode_id} in {run_dir}")


def artifact_text(run: dict[str, Any], update: dict[str, Any] | None = None) -> str:
    sources = [update or {}, run]
    for source in sources:
        for key in ("generic_workbench_artifact", "fitness_code"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def analyze_no_tss_failures(
    run_dir: Path,
    episodes: dict[str, dict[str, Any]],
    metric_index: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    for episode_id in EPISODES:
        episode = episodes[episode_id]
        seen_hashes: set[str] = set()
        for run in read_jsonl(episode_replay_path(run_dir, episode_id)):
            run_seed = int(run["run_seed"])
            state = copy.deepcopy(episode["hidden_initial_state"])
            initial_code = artifact_text(run)
            if initial_code:
                seen_hashes.add(hashlib.sha256(initial_code.encode("utf-8")).hexdigest())
            for stage_index, update in enumerate(run.get("update_results", []), start=1):
                delta = episode["hidden_update_oracle"][stage_index - 1].get("hidden_delta") or {}
                if episode.get("domain") == "green_vrp_multiobjective":
                    state = apply_green_delta(state, delta)
                    solution = normalize_green_solution_ids(state, update.get("solution") or {})
                    report = _green_constraint_report(state, solution)
                elif episode.get("domain") == "cloud_scheduling_multiobjective":
                    state = apply_cloud_delta(state, delta)
                    report = cloud_violation_report(state, update.get("solution") or {})
                else:
                    raise ValueError(f"unsupported no-TSS Pareto domain: {episode.get('domain')}")
                code = artifact_text(run, update)
                if code:
                    seen_hashes.add(hashlib.sha256(code.encode("utf-8")).hexdigest())
                metric = metric_index[(episode_id, stage_index, run_seed)]
                hidden_pass = bool_value(metric.get("true_pass"))
                if not hidden_pass:
                    failures.append(
                        {
                            "episode_id": episode_id,
                            "stage_index": stage_index,
                            "run_seed": run_seed,
                            "agent_reported_feasible": bool_value(metric.get("agent_reported_feasible")),
                            "hidden_pass": hidden_pass,
                            "violation_categories": sorted(report["violations"]),
                            "violations": report["violations"],
                        }
                    )
        versions.append(
            {
                "episode_id": episode_id,
                "artifact_versions": len(seen_hashes),
                "artifact_sha256": sorted(seen_hashes),
            }
        )
    return failures, versions


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fieldnames})


def tex_number(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def tex_percent(numerator: int, denominator: int) -> str:
    return "--" if denominator <= 0 else f"{100.0 * numerator / denominator:.1f}\\%"


def write_tex(path: Path, summaries: list[dict[str, Any]], failure_counts: Counter[tuple[str, str]]) -> None:
    by_name = {row["method"]: row for row in summaries}
    liveopt = by_name["LiveOpt"]
    no_tss = by_name["LiveOpt w/o TSS"]
    p010_duplicate = failure_counts[("NLDO-P010", "duplicate_vehicles")]
    p015_gpu = failure_counts[("NLDO-P015", "gpu_mismatch")]
    text = f"""% Generated by scripts/analysis/export_liveopt_tss_mechanism_evidence.py.
\\newcommand{{\\LiveOptTSSMechanismRows}}{{%
\\rowcolor{{LiveOptRowAccent}}
\\method{{}} & {tex_percent(liveopt['hidden_pass_cells'], liveopt['update_cells'])} & {liveopt['self_feasible_hidden_fail_cells']} & {tex_number(liveopt['all_cell_hv'])} & {tex_number(liveopt['conditional_hv'])} \\\\
\\method{{}} w/o TSS & {tex_percent(no_tss['hidden_pass_cells'], no_tss['update_cells'])} & {no_tss['self_feasible_hidden_fail_cells']} & {tex_number(no_tss['all_cell_hv'])} & {tex_number(no_tss['conditional_hv'])} \\\\
}}
\\newcommand{{\\LiveOptTSSFailureRows}}{{%
P010 (routing) & {p010_duplicate}/120 & one vehicle assigned to two routes & vehicle uniqueness omitted from the evaluation function \\\\
P015 (cloud) & {p015_gpu}/120 & GPU-incompatible placements & violations recorded without clearing the feasible flag \\\\
}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (args.benchmark, args.liveopt_run, args.no_tss_run):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    liveopt_summary, liveopt_episodes = summarize_method("LiveOpt", args.liveopt_run)
    no_tss_summary, no_tss_episodes = summarize_method("LiveOpt w/o TSS", args.no_tss_run)
    summaries = [liveopt_summary, no_tss_summary]
    no_tss_metrics = metric_rows(args.no_tss_run)
    metric_index = {
        (row["episode_id"], int(row["stage_index"]), int(row["run_seed"])): row for row in no_tss_metrics
    }
    failures, versions = analyze_no_tss_failures(args.no_tss_run, load_episodes(args.benchmark), metric_index)
    failure_counts: Counter[tuple[str, str]] = Counter()
    for row in failures:
        for category in row["violation_categories"]:
            failure_counts[(row["episode_id"], category)] += 1

    errors: list[str] = []
    if liveopt_summary["update_cells"] != 720 or no_tss_summary["update_cells"] != 720:
        errors.append("expected 720 update cells per method")
    if liveopt_summary["hidden_pass_cells"] != 720:
        errors.append("LiveOpt does not pass all 720 hidden update cells")
    if no_tss_summary["hidden_pass_cells"] != 580:
        errors.append("LiveOpt w/o TSS hidden-pass count is not 580")
    if len(failures) != 140:
        errors.append(f"expected 140 no-TSS failed cells, found {len(failures)}")
    if failure_counts[("NLDO-P010", "duplicate_vehicles")] != 120:
        errors.append("P010 duplicate-vehicle attribution does not cover 120 cells")
    if failure_counts[("NLDO-P015", "gpu_mismatch")] != 20:
        errors.append("P015 GPU-mismatch attribution does not cover 20 cells")
    if sum(row["artifact_versions"] for row in versions) != 13:
        errors.append("expected 13 episode-local Workbench artifact versions")

    payload = {
        "protocol": "liveopt_tss_mechanism_evidence_p010_p015_200x200x10_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "methods": summaries,
        "failure_cell_count": len(failures),
        "failure_category_cell_counts": {
            f"{episode_id}:{category}": count
            for (episode_id, category), count in sorted(failure_counts.items())
        },
        "artifact_versions": versions,
        "provenance": {
            "benchmark": str(args.benchmark.relative_to(ROOT)),
            "benchmark_sha256": sha256_file(args.benchmark),
            "liveopt_metrics_sha256": sha256_file(args.liveopt_run / "reference_metrics/reference_stage_metrics.csv"),
            "no_tss_metrics_sha256": sha256_file(args.no_tss_run / "reference_metrics/reference_stage_metrics.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        args.out_dir / "method_summary.csv",
        summaries,
        [
            "method",
            "run_dir",
            "update_cells",
            "hidden_pass_cells",
            "hidden_pass_rate",
            "self_feasible_hidden_fail_cells",
            "all_cell_hv",
            "conditional_hv",
        ],
    )
    write_csv(
        args.out_dir / "episode_summary.csv",
        liveopt_episodes + no_tss_episodes,
        ["method", "episode_id", "update_cells", "hidden_pass_cells", "hidden_pass_rate", "all_cell_hv", "conditional_hv"],
    )
    write_csv(
        args.out_dir / "failure_cells.csv",
        failures,
        ["episode_id", "stage_index", "run_seed", "agent_reported_feasible", "hidden_pass", "violation_categories", "violations"],
    )
    write_csv(
        args.out_dir / "artifact_versions.csv",
        versions,
        ["episode_id", "artifact_versions", "artifact_sha256"],
    )
    write_tex(args.paper_table, summaries, failure_counts)
    print(json.dumps({"status": payload["status"], "out_dir": str(args.out_dir), "paper_table": str(args.paper_table), "errors": errors}, indent=2))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
