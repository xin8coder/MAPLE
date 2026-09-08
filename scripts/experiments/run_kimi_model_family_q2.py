#!/usr/bin/env python3
"""Archived quota-safe P010/P015 LiveOpt pilot.

The completed P010 trajectories are now appendix stability evidence.  The main
Q2 model-family experiment is ``run_table2_model_family_q2.py --provider kimi``;
do not resume the partial P015 pilot as a substitute for that experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/llm_tests/run_liveopt_dynamic_nldo_benchmark_full.py"
EVALUATOR = ROOT / "scripts/analysis/evaluate_reference_metrics.py"
EXPORTER = ROOT / "scripts/analysis/export_kimi_model_family_q2.py"
BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
REFERENCE = ROOT / (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
SEMANTIC_DECISIONS = {
    "NLDO-P010": ROOT / "outputs/semantic_restart_gate_audit_v2_allstages_p010_20260713/decisions.json",
    "NLDO-P015": ROOT
    / "outputs/semantic_restart_gate_audit_v2_allstages_p011_p012_p014_p015_20260713/decisions.json",
}
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/liveopt_model_family_kimi_k3_q2_20260717"
QUOTA_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=os.getenv("KIMI_MODEL", "k3[1m]"))
    parser.add_argument("--episode-id", action="append", choices=sorted(SEMANTIC_DECISIONS))
    parser.add_argument("--controller-seed", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    episodes = args.episode_id or ["NLDO-P010", "NLDO-P015"]
    controllers = args.controller_seed or [0, 1, 2]
    selected = []
    for episode_id in episodes:
        for controller_seed in controllers:
            selected.append(
                {
                    "job_id": f"{episode_id.lower()}_c{controller_seed}",
                    "episode_id": episode_id,
                    "controller_seed": controller_seed,
                    "run_dir": args.output_root / "jobs" / episode_id / f"c{controller_seed}",
                    "semantic_decisions": SEMANTIC_DECISIONS[episode_id],
                }
            )
    return selected[: args.max_jobs] if args.max_jobs is not None else selected


def runner_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    run_dir = Path(job["run_dir"])
    command = [
        args.python,
        str(RUNNER),
        "--episodes-jsonl",
        str(BENCHMARK),
        "--episode-id",
        str(job["episode_id"]),
        "--output-dir",
        str(run_dir),
        "--model",
        args.model,
        "--controller-seed",
        str(job["controller_seed"]),
        "--semantic-decisions-json",
        str(job["semantic_decisions"]),
        "--population-size",
        "200",
        "--initial-population-size",
        "200",
        "--generations",
        "200",
        "--initial-generations",
        "200",
        "--archive-limit",
        "500",
        "--seed-count",
        "3",
        "--seed-start",
        "0",
        "--record-metric-trace",
        "--reference-free-early-stop",
        "--early-stop-min-generations",
        "40",
        "--early-stop-patience",
        "25",
        "--early-stop-hv-epsilon",
        "0.0005",
        "--early-stop-scalar-epsilon",
        "0.0005",
        "--early-stop-epsilon-box",
        "0.01",
        "--early-stop-min-archive-size",
        "8",
        "--max-patch-repairs",
        "3",
        "--max-attempts",
        "3",
        "--total-timeout",
        "900",
        "--stop-on-episode-failure",
    ]
    if args.resume or (run_dir / "summary.json").exists():
        command.append("--resume")
    return command


def evaluator_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    run_dir = Path(job["run_dir"])
    return [
        args.python,
        str(EVALUATOR),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir / "reference_metrics"),
        "--benchmark",
        f"NLDO={REFERENCE}",
    ]


def safe_command(command: list[str]) -> list[str]:
    """Commands contain no credentials, but keep one explicit audit hook."""

    return list(command)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def initial_manifest(args: argparse.Namespace, selected_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    previous = read_json(args.output_root / "manifest.json") if args.resume else {}
    previous_jobs = {
        str(row.get("job_id") or ""): row for row in list(previous.get("jobs") or [])
    }
    return {
        "status": "running",
        "protocol": "q2_model_family_kimi_k3_p010_p015_v1",
        "model": args.model,
        "provider": "kimi",
        "design": {
            "episodes": ["NLDO-P010", "NLDO-P015"],
            "controller_seeds": [0, 1, 2],
            "numerical_seeds": [0, 1, 2],
            "dynamic_stages": list(range(1, 13)),
            "population_size": 200,
            "generation_cap": 200,
            "archive_limit": 500,
            "kimi_generation_token_ceiling": 32000,
            "reference_free_early_stopping": True,
            "restart_choices": "same frozen verified choices as the DeepSeek controller-repeat control",
            "changed_factor": "controller model family only",
        },
        "quota_policy": {
            "pause_on": ["rolling_5h", "weekly", "monthly", "unknown_period"],
            "successful_response_cache": True,
            "resume": "rerun this script with --resume after quota reset",
        },
        "jobs": [
            previous_jobs.get(
                str(job["job_id"]),
                {
                    "job_id": job["job_id"],
                    "episode_id": job["episode_id"],
                    "controller_seed": job["controller_seed"],
                    "run_dir": display_path(Path(job["run_dir"])),
                    "status": "queued",
                },
            )
            for job in selected_jobs
        ],
        "quota_limit": {},
        "paused_job": None,
        "started_at": previous.get("started_at") or time.time(),
        "updated_at": time.time(),
    }


def upsert_job(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    rows = list(manifest.get("jobs") or [])
    rows = [row for row in rows if row.get("job_id") != record.get("job_id")]
    rows.append(record)
    order = {("NLDO-P010", seed): seed for seed in range(3)}
    order.update({("NLDO-P015", seed): 3 + seed for seed in range(3)})
    rows.sort(key=lambda row: order.get((row.get("episode_id"), row.get("controller_seed")), 99))
    manifest["jobs"] = rows
    manifest["updated_at"] = time.time()


def run_subprocess(command: list[str], env: dict[str, str], *, dry_run: bool) -> int:
    print(json.dumps({"command": safe_command(command), "dry_run": dry_run}, ensure_ascii=False), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main() -> int:
    args = parse_args()
    selected_jobs = jobs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"{manifest_path} exists; use --resume to preserve completed work")
    manifest = initial_manifest(args, selected_jobs)
    atomic_json(manifest_path, manifest)
    env = dict(os.environ)
    env.update(
        {
            "LLM_PROVIDER": "kimi",
            "KIMI_MODEL": args.model,
            "KIMI_CACHE": "1",
            "KIMI_MIN_GENERATION_TOKENS": "32000",
            "KIMI_MAX_GENERATION_TOKENS": "98304",
            "KIMI_REASONING_EFFORT": "medium",
        }
    )

    for job in selected_jobs:
        run_dir = Path(job["run_dir"])
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "reference_metrics/reference_stage_metrics.csv"
        existing = read_json(summary_path)
        if existing.get("status") == "completed" and metrics_path.exists():
            upsert_job(
                manifest,
                {
                    "job_id": job["job_id"],
                    "episode_id": job["episode_id"],
                    "controller_seed": job["controller_seed"],
                    "run_dir": display_path(run_dir),
                    "status": "completed",
                    "summary": display_path(summary_path),
                    "metrics": display_path(metrics_path),
                },
            )
            atomic_json(manifest_path, manifest)
            continue

        started = time.time()
        command = runner_command(args, job)
        returncode = run_subprocess(command, env, dry_run=args.dry_run)
        summary = read_json(summary_path)
        status = "dry_run" if args.dry_run else str(summary.get("status") or "failed")
        record = {
            "job_id": job["job_id"],
            "episode_id": job["episode_id"],
            "controller_seed": job["controller_seed"],
            "run_dir": display_path(run_dir),
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": time.time() - started,
            "summary": display_path(summary_path),
            "quota_limit": summary.get("quota_limit") or {},
        }
        upsert_job(manifest, record)

        if status == "quota_limited" or returncode == QUOTA_EXIT_CODE:
            manifest["status"] = "quota_limited"
            manifest["quota_limit"] = summary.get("quota_limit") or {}
            manifest["paused_job"] = job["job_id"]
            manifest["resume_command"] = [args.python, str(Path(__file__).relative_to(ROOT)), "--resume"]
            atomic_json(manifest_path, manifest)
            print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
            return QUOTA_EXIT_CODE
        if args.dry_run:
            atomic_json(manifest_path, manifest)
            continue
        if status != "completed" or returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_job"] = job["job_id"]
            atomic_json(manifest_path, manifest)
            return returncode or 1

        evaluation_code = run_subprocess(evaluator_command(args, job), env, dry_run=False)
        if evaluation_code != 0 or not metrics_path.exists():
            record["status"] = "evaluation_failed"
            record["evaluation_returncode"] = evaluation_code
            upsert_job(manifest, record)
            manifest["status"] = "failed"
            manifest["failed_job"] = job["job_id"]
            atomic_json(manifest_path, manifest)
            return evaluation_code or 1
        record["status"] = "completed"
        record["metrics"] = display_path(metrics_path)
        upsert_job(manifest, record)
        atomic_json(manifest_path, manifest)

    if args.dry_run:
        manifest["status"] = "dry_run"
        atomic_json(manifest_path, manifest)
        return 0

    export_code = run_subprocess(
        [args.python, str(EXPORTER), "--kimi-root", str(args.output_root), "--strict"],
        env,
        dry_run=False,
    )
    manifest["status"] = "completed" if export_code == 0 else "analysis_failed"
    manifest["analysis_returncode"] = export_code
    manifest["completed_at"] = time.time()
    atomic_json(manifest_path, manifest)
    return export_code


if __name__ == "__main__":
    raise SystemExit(main())
