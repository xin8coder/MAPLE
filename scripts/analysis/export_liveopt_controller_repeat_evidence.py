#!/usr/bin/env python3
"""Aggregate controller trajectories on one routing and one cloud problem."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
DM_REFERENCE = ROOT / (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
SEMANTIC_DECISIONS = {
    "NLDO-P010": ROOT / "outputs/semantic_restart_gate_audit_v2_allstages_p010_20260713/decisions.json",
    "NLDO-P015": ROOT / (
        "outputs/semantic_restart_gate_audit_v2_allstages_p011_p012_p014_p015_20260713/decisions.json"
    ),
}
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_controller_repeat_p010_p015_20260717"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_controller_repeat_rows.tex"
EXPECTED_BENCHMARK_SHA256 = "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303"
EXPECTED_REFERENCE_SHA256 = "50126c25eb837bdbf2dbc2e7710226472027a1a880305d331dc006438a02c5e5"
EXPECTED_DECISION_SHA256 = {
    "NLDO-P010": "33cba37ec31f4d3436919009642eaa32aaf3f59bc1e17fd34380ae883c8c070d",
    "NLDO-P015": "cbd2428b72144e9e79b9f9267e2f64dc6d881431f9fa6829c5fe4205ceac00c9",
}
SEEDS = tuple(range(3))
STAGES = tuple(range(1, 13))


def run_dir(name: str) -> Path:
    return ROOT / "outputs" / name


SPECS = (
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 0,
        "run_dir": run_dir("liveopt_matched_full_p010_p015_200x200x10_20260714"),
        "new_run": False,
    },
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 1,
        "run_dir": run_dir("liveopt_controller_repeat_p010_liveopt_c1_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 2,
        "run_dir": run_dir("liveopt_controller_repeat_p010_liveopt_c2_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 0,
        "run_dir": run_dir("liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714"),
        "summary_path": ROOT / (
            "outputs/liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714/"
            "replay_jobs/NLDO-P010/summary.json"
        ),
        "new_run": False,
    },
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 1,
        "run_dir": run_dir("liveopt_controller_repeat_p010_no_tss_c1_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P010",
        "problem_class": "routing",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 2,
        "run_dir": run_dir("liveopt_controller_repeat_p010_no_tss_c2_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 0,
        "run_dir": run_dir("liveopt_matched_full_p010_p015_200x200x10_20260714"),
        "new_run": False,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 1,
        "run_dir": run_dir("liveopt_controller_repeat_p015_liveopt_c1_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt",
        "controller_id": 2,
        "run_dir": run_dir("liveopt_controller_repeat_p015_liveopt_c2_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 0,
        "run_dir": run_dir("liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714"),
        "summary_path": ROOT / (
            "outputs/liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714/"
            "replay_jobs/NLDO-P015/summary.json"
        ),
        "new_run": False,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 1,
        "run_dir": run_dir("liveopt_controller_repeat_p015_no_tss_c1_200x200x3_20260717"),
        "new_run": True,
    },
    {
        "episode_id": "NLDO-P015",
        "problem_class": "cloud",
        "quality_field": "normalized_hv",
        "quality_label": "HV",
        "method": "LiveOpt w/o TSS",
        "controller_id": 2,
        "run_dir": run_dir("liveopt_controller_repeat_p015_no_tss_c2_200x200x3_20260717"),
        "new_run": True,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def load_summary(spec: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    path = Path(spec.get("summary_path") or Path(spec["run_dir"]) / "summary.json")
    if not path.exists():
        return {}, path
    return json.loads(path.read_text(encoding="utf-8")), path


def failure_reason(summary: dict[str, Any], episode_id: str) -> str:
    records = summary.get("episode_records") or summary.get("episodes") or []
    for row in records:
        if str(row.get("episode_id") or "") == episode_id:
            return str(row.get("failure_reason") or "")
    return ""


def trace_record(spec: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    path = Path(spec["run_dir"]) / "trace/deepseek_calls.jsonl"
    if not spec["new_run"]:
        return {
            "trajectory_source": "existing_formal_controller_trajectory",
            "trace_path": None,
            "trace_sha256": None,
            "api_response_count": None,
            "local_cache_hit_count": None,
        }
    label = f"{spec['episode_id']} {spec['method']} c{spec['controller_id']}"
    if not path.exists():
        errors.append(f"{label}: independent-controller trace is missing")
        return {
            "trajectory_source": "new_provider_call",
            "trace_path": None,
            "trace_sha256": None,
            "api_response_count": 0,
            "local_cache_hit_count": 0,
        }
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    api_responses = sum(row.get("event") == "api_response" for row in events)
    cache_hits = sum(bool(row.get("local_cache_hit")) or bool(row.get("cache_path")) for row in events)
    if not api_responses:
        errors.append(f"{label}: trace has no provider response")
    if cache_hits:
        errors.append(f"{label}: trace contains {cache_hits} local cache hits")
    return {
        "trajectory_source": "new_provider_call",
        "trace_path": repo_path(path),
        "trace_sha256": sha256_file(path),
        "api_response_count": api_responses,
        "local_cache_hit_count": cache_hits,
    }


def metric_rows(spec: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, str]], Path]:
    path = Path(spec["run_dir"]) / "reference_metrics/reference_stage_metrics.csv"
    rows: dict[tuple[int, int], dict[str, str]] = {}
    if not path.exists():
        return rows, path
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("episode_id") or "") != spec["episode_id"]:
                continue
            seed = int(row.get("run_seed") or -1)
            stage = int(row.get("stage_index") or -1)
            if seed not in SEEDS or stage not in STAGES:
                continue
            key = (seed, stage)
            if key in rows:
                raise ValueError(f"duplicate metric cell {spec['episode_id']} seed={seed} stage={stage}")
            rows[key] = row
    return rows, path


def validate_new_budget(spec: dict[str, Any], summary: dict[str, Any], errors: list[str]) -> None:
    label = f"{spec['episode_id']} {spec['method']} c{spec['controller_id']}"
    budget = summary.get("budget") if isinstance(summary.get("budget"), dict) else {}
    expected = {
        "population_size": 200,
        "initial_population_size": 200,
        "generations": 200,
        "initial_generations": 200,
        "archive_limit": 500,
        "seed_count": 3,
        "seed_start": 0,
        "controller_seed": spec["controller_id"],
    }
    for key, value in expected.items():
        if budget.get(key) != value:
            errors.append(f"{label}: budget.{key}={budget.get(key)!r}, expected {value!r}")
    if budget.get("reference_free_early_stop") is not True:
        errors.append(f"{label}: reference-free early stopping is not recorded")
    if budget.get("restart_policy") != "frozen_verified_semantic_full_else_fixed_warm":
        errors.append(f"{label}: restart policy is {budget.get('restart_policy')!r}")
    expected_decisions = [repo_path(SEMANTIC_DECISIONS[spec["episode_id"]])]
    if summary.get("semantic_decisions_json") != expected_decisions:
        errors.append(
            f"{label}: frozen semantic decisions are {summary.get('semantic_decisions_json')!r}, "
            f"expected {expected_decisions!r}"
        )
    if spec["method"] == "LiveOpt w/o TSS" and budget.get("generic_workbench") is not True:
        errors.append(f"{label}: w/o-TSS untyped Workbench construction is not recorded")


def controller_record(spec: dict[str, Any], errors: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, summary_path = load_summary(spec)
    metrics, metrics_path = metric_rows(spec)
    label = f"{spec['episode_id']} {spec['method']} c{spec['controller_id']}"
    if spec["new_run"]:
        if not summary:
            errors.append(f"{label}: summary is missing")
        else:
            validate_new_budget(spec, summary, errors)
            if summary.get("status") == "running":
                errors.append(f"{label}: run is still running")

    complete = len(metrics) == len(SEEDS) * len(STAGES)
    if complete and summary and summary.get("status") not in {"completed", None}:
        errors.append(f"{label}: complete metrics but summary status={summary.get('status')!r}")
    if summary.get("status") == "completed" and not complete:
        errors.append(f"{label}: completed summary has only {len(metrics)}/36 metric cells")

    cell_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for stage in STAGES:
            source = metrics.get((seed, stage))
            passed = bool(source and as_bool(source.get("true_pass")))
            quality = as_float(source.get(spec["quality_field"])) if source and passed else 0.0
            cell_rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "problem_class": spec["problem_class"],
                    "method": spec["method"],
                    "controller_id": spec["controller_id"],
                    "run_seed": seed,
                    "stage_index": stage,
                    "metric": spec["quality_label"],
                    "true_pass": passed,
                    "quality": quality,
                    "metric_available": source is not None,
                    "missing_due_controller_failure": source is None and not complete,
                }
            )

    solve_rate = mean(float(row["true_pass"]) for row in cell_rows)
    quality = mean(float(row["quality"]) for row in cell_rows)
    passed_values = [float(row["quality"]) for row in cell_rows if row["true_pass"]]
    seed_means = [
        mean(float(row["quality"]) for row in cell_rows if row["run_seed"] == seed)
        for seed in SEEDS
    ]
    trace = trace_record(spec, errors)
    record = {
        "episode_id": spec["episode_id"],
        "problem_class": spec["problem_class"],
        "quality_label": spec["quality_label"],
        "method": spec["method"],
        "controller_id": spec["controller_id"],
        "controller_complete": complete,
        "summary_status": summary.get("status") or ("completed_from_metrics" if complete else "missing"),
        "failure_reason": failure_reason(summary, spec["episode_id"]),
        "available_metric_cells": len(metrics),
        "expected_metric_cells": len(cell_rows),
        "solve_rate": solve_rate,
        "quality": quality,
        "pass_only_quality": mean(passed_values) if passed_values else None,
        "within_controller_search_sd": sample_sd(seed_means) if complete else None,
        "run_dir": repo_path(Path(spec["run_dir"])),
        "summary_path": repo_path(summary_path) if summary_path and summary_path.exists() else None,
        "summary_sha256": sha256_file(summary_path) if summary_path and summary_path.exists() else None,
        "metrics_path": repo_path(metrics_path) if metrics_path.exists() else None,
        "metrics_sha256": sha256_file(metrics_path) if metrics_path.exists() else None,
        **trace,
    }
    return record, cell_rows


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    keys = sorted({(row["episode_id"], row["method"]) for row in records})
    for episode_id, method in keys:
        rows = sorted(
            (row for row in records if row["episode_id"] == episode_id and row["method"] == method),
            key=lambda row: int(row["controller_id"]),
        )
        solve_values = [float(row["solve_rate"]) for row in rows]
        quality_values = [float(row["quality"]) for row in rows]
        passed_quality_sum = 0.0
        passed_cell_count = 0.0
        for row in rows:
            controller_pass_count = float(row["solve_rate"]) * len(SEEDS) * len(STAGES)
            if row["pass_only_quality"] is not None and controller_pass_count:
                passed_quality_sum += float(row["pass_only_quality"]) * controller_pass_count
                passed_cell_count += controller_pass_count
        within_values = [
            float(row["within_controller_search_sd"])
            for row in rows
            if row["within_controller_search_sd"] is not None
        ]
        groups.append(
            {
                "episode_id": episode_id,
                "problem_class": rows[0]["problem_class"],
                "quality_label": rows[0]["quality_label"],
                "method": method,
                "controller_complete": sum(bool(row["controller_complete"]) for row in rows),
                "controller_count": len(rows),
                "solve_rate_mean": mean(solve_values),
                "solve_rate_controller_sd": sample_sd(solve_values),
                "quality_mean": mean(quality_values),
                "quality_controller_sd": sample_sd(quality_values),
                "pass_only_quality_mean_across_controllers": (
                    passed_quality_sum / passed_cell_count if passed_cell_count else None
                ),
                "mean_within_controller_search_sd": mean(within_values) if within_values else None,
            }
        )
    return groups


def tex_method(method: str) -> str:
    return "LiveOpt" if method == "LiveOpt" else r"LiveOpt w/o TSS"


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def render_tex(records: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    record_map = {(row["episode_id"], row["method"], row["controller_id"]): row for row in records}
    group_map = {(row["episode_id"], row["method"]): row for row in groups}
    lines = [
        "% Generated by scripts/analysis/export_liveopt_controller_repeat_evidence.py.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\ResultTableSetup",
        r"\setlength{\tabcolsep}{2.6pt}% fit-required override: eight columns",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"\TableHead",
        r"\textbf{Episode} & \textbf{Method} & \TightStack{\textbf{Model}\\\textbf{run}} & \TightStack{\textbf{Execution}\\\textbf{completed}} & \TightStack{\textbf{Solve}\\\textbf{rate (\%)}} & \TightStack{\textbf{Online}\\\textbf{quality}} & \TightStack{\textbf{Quality}\\\textbf{if valid}} & \TightStack{\textbf{Numerical}\\\textbf{SD}} \\",
        r"\midrule",
    ]
    for problem_index, (episode_id, label) in enumerate((("NLDO-P010", "P010 (routing)"), ("NLDO-P015", "P015 (cloud)"))):
        if problem_index:
            lines.append(r"\midrule")
        for method_index, method in enumerate(("LiveOpt", "LiveOpt w/o TSS")):
            if method_index:
                lines.append(r"\cmidrule(lr){2-8}")
            for controller_id in range(3):
                row = record_map[(episode_id, method, controller_id)]
                problem_cell = label if method_index == 0 and controller_id == 0 else ""
                method_cell = tex_method(method) if controller_id == 0 else ""
                complete = "yes" if row["controller_complete"] else "no"
                lines.append(
                    f"{problem_cell} & {method_cell} & {controller_id} & {complete} & "
                    f"{100.0 * row['solve_rate']:.1f} & {row['quality']:.3f} & {fmt(row['pass_only_quality'])} & "
                    f"{fmt(row['within_controller_search_sd'])} \\\\"
                )
            group = group_map[(episode_id, method)]
            lines.append(
                r" & & \textit{All} & "
                f"{group['controller_complete']}/{group['controller_count']} & "
                f"${100.0 * group['solve_rate_mean']:.1f} \\pm {100.0 * group['solve_rate_controller_sd']:.1f}$ & "
                f"${group['quality_mean']:.3f} \\pm {group['quality_controller_sd']:.3f}$ & "
                f"{fmt(group['pass_only_quality_mean_across_controllers'])} & "
                f"{fmt(group['mean_within_controller_search_sd'])} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Independent model runs on one routing and one cloud episode.} Each run independently generates a Workbench sequence. Numerical seeds 0--2 then search that sequence over t01--t12. Warm--Full choices remain fixed, so only Workbench construction and maintenance vary. Solve rate uses the held-out validity check. Online quality sets invalid results to zero. The \textit{All} rows report mean $\pm$ sample standard deviation across model runs. Quality if valid includes accepted states only. Numerical SD measures variation across search seeds within one model run.}",
            r"\label{tab:controller-repeat}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    benchmark_sha256 = sha256_file(BENCHMARK)
    if benchmark_sha256 != EXPECTED_BENCHMARK_SHA256:
        errors.append(f"benchmark hash changed: {benchmark_sha256}")
    for path in (DM_REFERENCE, *SEMANTIC_DECISIONS.values()):
        if not path.exists():
            errors.append(f"reference missing: {path}")
    reference_sha256 = sha256_file(DM_REFERENCE) if DM_REFERENCE.exists() else None
    if reference_sha256 != EXPECTED_REFERENCE_SHA256:
        errors.append(f"held-out reference hash changed: {reference_sha256}")
    decision_sources: dict[str, dict[str, Any]] = {}
    for episode_id, path in SEMANTIC_DECISIONS.items():
        digest = sha256_file(path) if path.exists() else None
        decision_sources[episode_id] = {"path": repo_path(path), "sha256": digest}
        if digest != EXPECTED_DECISION_SHA256[episode_id]:
            errors.append(f"{episode_id}: frozen semantic-decision hash changed: {digest}")

    records: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for spec in SPECS:
        record, controller_cells = controller_record(dict(spec), errors)
        records.append(record)
        cells.extend(controller_cells)
    groups = aggregate(records)
    if len(records) != 12:
        errors.append(f"controller record count={len(records)}, expected 12")
    if len(cells) != 432:
        errors.append(f"controller cell count={len(cells)}, expected 432")
    if len(groups) != 4:
        errors.append(f"aggregate group count={len(groups)}, expected 4")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_tex.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "controller_cells.csv", cells)
    write_csv(args.out_dir / "controller_summary.csv", records)
    write_csv(args.out_dir / "aggregate_summary.csv", groups)
    args.paper_tex.write_text(render_tex(records, groups), encoding="utf-8")
    evidence = {
        "status": "passed" if not errors else "failed",
        "protocol": "independent_controller_trajectory_p010_p015_v1",
        "benchmark": repo_path(BENCHMARK),
        "benchmark_sha256": benchmark_sha256,
        "references": {"P010/P015": repo_path(DM_REFERENCE)},
        "reference_sha256": reference_sha256,
        "semantic_decision_sources": decision_sources,
        "design": {
            "problems": {"routing": "NLDO-P010", "cloud": "NLDO-P015"},
            "methods": ["LiveOpt", "LiveOpt w/o TSS"],
            "controller_replicates": [0, 1, 2],
            "numerical_seeds_per_controller": [0, 1, 2],
            "dynamic_stages": list(STAGES),
            "population_size": 200,
            "generation_cap": 200,
            "archive_limit": 500,
            "reference_free_early_stopping": True,
            "semantic_restart_choices": "frozen verified public-only decisions",
            "controller_independence": "new trajectories use separate provider calls with zero local cache hits",
            "failed_controller_accounting": "all requested dynamic cells receive solve=0 and quality=0",
        },
        "controllers": records,
        "aggregate": groups,
        "cell_count": len(cells),
        "paper_tex": repo_path(args.paper_tex),
        "errors": errors,
    }
    (args.out_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": evidence["status"], "aggregate": groups, "errors": errors}, indent=2))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
