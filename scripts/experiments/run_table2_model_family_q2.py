#!/usr/bin/env python3
"""Run the corrected DeepSeek or full Kimi K3 external-baseline Table 2.

Each job is one method--episode trajectory.  A hidden-feasible state advances to
the next state; the first hidden-infeasible state terminates the trajectory and
the exporter scores the unexecuted suffix as zero.  The table includes t00 and
t01--t12.  Kimi quota errors stop the orchestrator with all completed jobs and
the active feasible prefix preserved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/baselines/run_nldo_dynamic_public_baselines.py"
EXPORTER = ROOT / "scripts/analysis/export_main_dynamic_hv_heatmap.py"
AUDITOR = ROOT / "scripts/analysis/audit_table2_sequential_coverage.py"
RESCORER = ROOT / "scripts/analysis/rescore_dynamic_public_baseline_stage_rows.py"
BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
REFERENCE = ROOT / (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
DEEPSEEK_PREFIX_SOURCES = (
    ROOT
    / "logs/llm_tests/tss_dynamic_public_react_optimai_orllm_equal_repair_merged_20260709"
    / "dynamic_public_baseline_stage_rows.jsonl",
    ROOT
    / "logs/llm_tests/tss_dynamic_public_optimus_orlm_equal_repair_merged_20260709"
    / "dynamic_public_baseline_stage_rows.jsonl",
)
METHODS = (
    "react_tools",
    "optimus",
    "orlm",
    "optimai_2025",
    "or_llm_agent_2025",
)
# Interleave DS and DM so a quota checkpoint is not concentrated in one view.
EPISODE_ORDER = (1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 8, 9)
STAGES = tuple(range(13))
QUOTA_EXIT_CODE = 75
INFRASTRUCTURE_EXIT_CODE = 76
TERMINAL_RUN_STATUSES = {"completed", "stopped_after_failure"}
LEGACY_INFRASTRUCTURE_FAILURE_MARKERS = (
    "Remote end closed connection without response",
    "504 Gateway Time-out",
    "502 Bad Gateway",
    "503 Service Unavailable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("deepseek", "kimi"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def default_output_root(provider: str) -> Path:
    suffix = "deepseek_continuation" if provider == "deepseek" else "kimi_k3_full"
    return ROOT / f"logs/llm_tests/table2_{suffix}_20260717"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def merged_stage_lookup(paths: tuple[Path, ...]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            key = (
                str(row.get("method") or ""),
                str(row.get("episode_id") or ""),
                int(row.get("stage_index") or 0),
            )
            rows[key] = row
    return rows


def first_uncovered_stage(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    method: str,
    episode_id: str,
) -> int | None:
    """Return a gap after a feasible prefix; None means terminal evidence exists."""

    for stage in STAGES:
        row = rows.get((method, episode_id, stage))
        if row is None:
            return stage
        if not bool((row.get("hidden_evaluation") or {}).get("feasible")):
            return None
    return None


def selected_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = set(args.method or METHODS)
    episodes = {
        value if str(value).startswith("NLDO-P") else f"NLDO-P{int(value):03d}"
        for value in (args.episode_id or [f"NLDO-P{number:03d}" for number in EPISODE_ORDER])
    }
    old_rows = merged_stage_lookup(DEEPSEEK_PREFIX_SOURCES) if args.provider == "deepseek" else {}
    jobs: list[dict[str, Any]] = []
    for number in EPISODE_ORDER:
        episode_id = f"NLDO-P{number:03d}"
        if episode_id not in episodes:
            continue
        for method in METHODS:
            if method not in methods:
                continue
            next_stage = 0
            if args.provider == "deepseek":
                next_stage = first_uncovered_stage(old_rows, method, episode_id)
                if next_stage is None:
                    continue
            jobs.append(
                {
                    "job_id": f"{method}__{episode_id}",
                    "method": method,
                    "episode_id": episode_id,
                    "resume_from_stage": next_stage,
                    "run_dir": args.output_root / "jobs" / method / episode_id,
                }
            )
    return jobs[: args.max_jobs] if args.max_jobs is not None else jobs


def runner_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    command = [
        args.python,
        str(RUNNER),
        "--episodes-jsonl",
        str(BENCHMARK),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--episode-id",
        str(job["episode_id"]),
        "--methods",
        str(job["method"]),
        "--out-dir",
        str(job["run_dir"]),
        "--use-response-cache",
        "--response-protocol",
        "json",
        "--code-repair-attempts",
        "3",
        "--max-attempts",
        "2",
        "--total-timeout",
        "900",
        "--stop-after-hidden-rejection",
    ]
    if args.provider == "kimi":
        command.extend(["--max-tokens", "32000", "--cache-dir", "outputs/kimi_cache"])
    else:
        command.extend(["--max-tokens", "6000", "--cache-dir", "outputs/deepseek_cache"])
        for source in DEEPSEEK_PREFIX_SOURCES:
            command.extend(["--prefix-stage-rows", str(source)])
    if args.cache_only:
        command.append("--cache-only")
    if args.resume or (Path(job["run_dir"]) / "dynamic_public_baseline_summary.json").exists():
        command.append("--resume")
    prior = run_record(job)
    if args.resume and is_legacy_infrastructure_failure(prior):
        command.extend(["--retry-statuses", str(prior.get("run_status") or "stopped_after_failure")])
    return command


def run_record(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["run_dir"]) / "NLDO" / f"{job['method']}_limit0.jsonl"
    rows = [
        row
        for row in load_jsonl(path)
        if str(row.get("episode_id") or "") == str(job["episode_id"])
    ]
    return rows[-1] if rows else {}


def existing_terminal_job(job: dict[str, Any]) -> dict[str, Any] | None:
    row = run_record(job)
    if is_legacy_infrastructure_failure(row):
        return None
    if str(row.get("run_status") or "") not in TERMINAL_RUN_STATUSES:
        return None
    return {
        **job,
        "run_dir": display_path(Path(job["run_dir"])),
        "status": "completed",
        "trajectory_status": row.get("run_status"),
        "saved_states": len(row.get("stage_traces") or []),
    }


def is_legacy_infrastructure_failure(row: dict[str, Any]) -> bool:
    """Requeue pre-streaming transport rows without reclassifying real solver failures."""

    for stage in row.get("stage_traces") or []:
        reason = str(stage.get("failure_reason") or "")
        if any(marker in reason for marker in LEGACY_INFRASTRUCTURE_FAILURE_MARKERS):
            return True
    return False


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def upsert_job(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    rows = [row for row in list(manifest.get("jobs") or []) if row.get("job_id") != record.get("job_id")]
    rows.append(record)
    manifest["jobs"] = rows
    manifest["updated_at"] = time.time()


def execute_job(
    args: argparse.Namespace,
    job: dict[str, Any],
    env: dict[str, str],
) -> tuple[int, dict[str, Any], list[str]]:
    command = runner_command(args, job)
    print(json.dumps({"job": job["job_id"], "command": command, "dry_run": False}), flush=True)
    return_code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    return return_code, run_record(job), command


def record_job_result(
    manifest: dict[str, Any],
    job: dict[str, Any],
    return_code: int,
    row: dict[str, Any],
) -> int | None:
    status = str(row.get("run_status") or "")
    if return_code == QUOTA_EXIT_CODE or status == "quota_limited":
        manifest.update(
            {
                "status": "quota_limited",
                "paused_job": job["job_id"],
                "quota_limit": row.get("quota_limit") or {},
                "updated_at": time.time(),
            }
        )
        upsert_job(
            manifest,
            {
                **job,
                "run_dir": display_path(Path(job["run_dir"])),
                "status": "quota_limited",
                "saved_states": len(row.get("stage_traces") or []),
                "next_stage_index": row.get("next_stage_index"),
            },
        )
        return QUOTA_EXIT_CODE
    if return_code == INFRASTRUCTURE_EXIT_CODE or status == "infrastructure_limited":
        manifest.update(
            {
                "status": "infrastructure_limited",
                "paused_job": job["job_id"],
                "infrastructure_limit": row.get("infrastructure_limit") or {},
                "updated_at": time.time(),
            }
        )
        upsert_job(
            manifest,
            {
                **job,
                "run_dir": display_path(Path(job["run_dir"])),
                "status": "infrastructure_limited",
                "saved_states": len(row.get("stage_traces") or []),
                "next_stage_index": row.get("next_stage_index"),
            },
        )
        return INFRASTRUCTURE_EXIT_CODE
    if return_code != 0 or status not in TERMINAL_RUN_STATUSES:
        manifest.update({"status": "failed", "paused_job": job["job_id"]})
        upsert_job(
            manifest,
            {
                **job,
                "run_dir": display_path(Path(job["run_dir"])),
                "status": "failed",
                "return_code": return_code,
                "trajectory_status": status,
            },
        )
        return return_code or 1
    upsert_job(
        manifest,
        {
            **job,
            "run_dir": display_path(Path(job["run_dir"])),
            "status": "completed",
            "trajectory_status": status,
            "saved_states": len(row.get("stage_traces") or []),
        },
    )
    return None


def merge_completed_stage_rows(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> Path:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.provider == "deepseek":
        merged.update(merged_stage_lookup(DEEPSEEK_PREFIX_SOURCES))
    for job in jobs:
        path = Path(job["run_dir"]) / "dynamic_public_baseline_stage_rows.jsonl"
        for row in load_jsonl(path):
            key = (
                str(row.get("method") or ""),
                str(row.get("episode_id") or ""),
                int(row.get("stage_index") or 0),
            )
            merged[key] = row
    order = {method: index for index, method in enumerate(METHODS)}
    rows = sorted(
        merged.values(),
        key=lambda row: (
            order.get(str(row.get("method") or ""), 99),
            str(row.get("episode_id") or ""),
            int(row.get("stage_index") or 0),
        ),
    )
    path = args.output_root / "merged" / "dynamic_public_baseline_stage_rows.jsonl"
    atomic_jsonl(path, rows)
    return path


def export_command(args: argparse.Namespace, merged_rows: Path) -> list[str]:
    if args.provider == "deepseek":
        return [
            args.python,
            str(EXPORTER),
            "--public-stage-rows",
            str(merged_rows),
        ]
    return [
        args.python,
        str(EXPORTER),
        "--public-stage-rows",
        str(merged_rows),
        "--output",
        str(ROOT / "paper/figures/results/kimi_k3_dynamic_hv_heatmap.pdf"),
        "--png-output",
        str(ROOT / "paper/figures/results/kimi_k3_dynamic_hv_heatmap.png"),
        "--summary-json",
        str(args.output_root / "table2_summary.json"),
        "--summary-tex",
        str(ROOT / "release_artifacts/paper_table_exports/kimi_k3_dynamic_hv_summary_rows.tex"),
        "--omit-liveopt",
    ]


def initial_manifest(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    previous = load_json(args.output_root / "manifest.json") if args.resume else {}
    return {
        "status": "running",
        "protocol": "table2_external_baselines_sequential_t00_t12_v2",
        "provider": args.provider,
        "model": args.model,
        "design": {
            "methods": list(METHODS),
            "views": {"DS": "NLDO-P001--P009", "DM": "NLDO-P010--P015"},
            "states": list(STAGES),
            "ds_denominator_per_method": 9 * len(STAGES),
            "dm_denominator_per_method": 6 * len(STAGES),
            "advance_rule": "call the next state only after hidden-feasible solution",
            "failure_rule": "first failure and every unexecuted suffix state score zero",
            "repair_attempts": 3,
            "state": "own cumulative dialogue, accepted output, and own candidate population",
            "deepseek_mode": "continue only gaps after the corrected feasible prefix",
            "kimi_mode": "fresh full method-episode trajectories",
        },
        "quota_policy": {
            "provider": "kimi",
            "pause_on": ["rolling_5h", "weekly", "monthly", "unknown_period"],
            "exit_code": QUOTA_EXIT_CODE,
            "successful_response_cache": True,
        },
        "jobs": list(previous.get("jobs") or []),
        "selected_job_count": len(jobs),
        "paused_job": None,
        "quota_limit": {},
        "resume_command": [
            args.python,
            "scripts/experiments/run_table2_model_family_q2.py",
            "--provider",
            args.provider,
            "--resume",
            *(
                ["--workers", str(args.workers)]
                if args.provider == "deepseek"
                else []
            ),
        ],
        "started_at": previous.get("started_at") or time.time(),
        "updated_at": time.time(),
    }


def main() -> int:
    args = parse_args()
    args.model = args.model or (
        os.getenv("KIMI_MODEL", "k3[1m]")
        if args.provider == "kimi"
        else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    )
    args.output_root = args.output_root or default_output_root(args.provider)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"{manifest_path} exists; use --resume")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = selected_jobs(args)
    manifest = initial_manifest(args, jobs)
    atomic_json(manifest_path, manifest)

    env = dict(os.environ)
    if args.provider == "kimi":
        env.update(
            {
                "LLM_PROVIDER": "kimi",
                "KIMI_MODEL": args.model,
                "KIMI_CACHE": "1",
                "KIMI_MIN_GENERATION_TOKENS": "32000",
                "KIMI_MAX_GENERATION_TOKENS": "98304",
                "KIMI_REASONING_EFFORT": "medium",
                "KIMI_STREAM": "1",
                "KIMI_STREAM_INCLUDE_USAGE": "1",
                "KIMI_USE_SYSTEM_PROXY": "0",
            }
        )

    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.provider == "kimi" and args.workers != 1:
        raise SystemExit("Kimi quota-safe execution requires --workers 1")

    runnable_jobs: list[dict[str, Any]] = []
    for job in jobs:
        terminal = existing_terminal_job(job) if args.resume else None
        if terminal is not None:
            upsert_job(manifest, terminal)
            atomic_json(manifest_path, manifest)
            continue
        if args.dry_run:
            command = runner_command(args, job)
            print(
                json.dumps({"job": job["job_id"], "command": command, "dry_run": True}),
                flush=True,
            )
            upsert_job(
                manifest,
                {**job, "run_dir": display_path(Path(job["run_dir"])), "status": "planned"},
            )
            atomic_json(manifest_path, manifest)
            continue
        runnable_jobs.append(job)

    if args.dry_run:
        manifest.update({"status": "dry_run", "updated_at": time.time()})
        atomic_json(manifest_path, manifest)
        return 0

    if args.workers == 1:
        for job in runnable_jobs:
            return_code, row, _ = execute_job(args, job, env)
            fatal_code = record_job_result(manifest, job, return_code, row)
            atomic_json(manifest_path, manifest)
            if fatal_code is not None:
                return fatal_code
    else:
        # DeepSeek jobs have disjoint output directories. Manifest updates stay
        # on this parent thread, so parallel provider waits cannot corrupt the
        # checkpoint. Kimi deliberately remains single-worker for quota safety.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_jobs = {
                executor.submit(execute_job, args, job, env): job for job in runnable_jobs
            }
            fatal_code: int | None = None
            for future in concurrent.futures.as_completed(future_jobs):
                job = future_jobs[future]
                try:
                    return_code, row, _ = future.result()
                except Exception as exc:  # noqa: BLE001
                    return_code, row = 1, {"run_status": "orchestrator_exception", "error": str(exc)}
                current_fatal = record_job_result(manifest, job, return_code, row)
                atomic_json(manifest_path, manifest)
                if fatal_code is None and current_fatal is not None:
                    fatal_code = current_fatal
            if fatal_code is not None:
                return fatal_code

    merged_rows = merge_completed_stage_rows(args, jobs)
    rescored_rows = args.output_root / "merged" / "dynamic_public_baseline_stage_rows.rescored.jsonl"
    rescore_summary = args.output_root / "merged" / "reference_rescore_summary.json"
    rescore_command = [
        args.python,
        str(RESCORER),
        "--stage-rows",
        str(merged_rows),
        "--output",
        str(rescored_rows),
        "--summary",
        str(rescore_summary),
        "--benchmark",
        str(BENCHMARK),
        "--reference",
        str(REFERENCE),
        "--strict",
    ]
    rescore_code = subprocess.run(rescore_command, cwd=ROOT, check=False).returncode
    if rescore_code != 0:
        manifest.update(
            {
                "status": "reference_rescore_failed",
                "merged_stage_rows_raw": display_path(merged_rows),
                "reference_rescore": display_path(rescore_summary),
                "updated_at": time.time(),
            }
        )
        atomic_json(manifest_path, manifest)
        return rescore_code
    audit_out = args.output_root / "coverage_audit"
    audit_command = [
        args.python,
        str(AUDITOR),
        "--source",
        str(rescored_rows),
        "--out-dir",
        str(audit_out),
        "--strict",
    ]
    audit_code = subprocess.run(audit_command, cwd=ROOT, check=False).returncode
    if audit_code != 0:
        manifest.update(
            {
                "status": "coverage_audit_failed",
                "merged_stage_rows_raw": display_path(merged_rows),
                "merged_stage_rows": display_path(rescored_rows),
                "reference_rescore": display_path(rescore_summary),
                "coverage_audit": display_path(audit_out / "summary.json"),
                "updated_at": time.time(),
            }
        )
        atomic_json(manifest_path, manifest)
        return audit_code
    if args.skip_export or args.cache_only:
        manifest.update(
            {
                "status": "preflight_completed" if args.cache_only else "completed",
                "merged_stage_rows_raw": display_path(merged_rows),
                "merged_stage_rows": display_path(rescored_rows),
                "reference_rescore": display_path(rescore_summary),
                "coverage_audit": display_path(audit_out / "summary.json"),
                "updated_at": time.time(),
            }
        )
        atomic_json(manifest_path, manifest)
        return 0
    command = export_command(args, rescored_rows)
    return_code = subprocess.run(command, cwd=ROOT, check=False).returncode
    manifest.update(
        {
            "status": "completed" if return_code == 0 else "export_failed",
            "merged_stage_rows_raw": display_path(merged_rows),
            "merged_stage_rows": display_path(rescored_rows),
            "reference_rescore": display_path(rescore_summary),
            "coverage_audit": display_path(audit_out / "summary.json"),
            "export_command": command,
            "updated_at": time.time(),
        }
    )
    atomic_json(manifest_path, manifest)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
