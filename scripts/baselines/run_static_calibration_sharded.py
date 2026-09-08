#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch method-by-case sharded static calibration runs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="logs/static_calibration/sharded_latest")
    parser.add_argument("--methods", default="liveopt_static,optimai_2025,or_llm_agent_2025")
    parser.add_argument("--case-shards", type=int, default=6)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-cache-misses", action="store_true")
    parser.add_argument("--retry-statuses", default="")
    parser.add_argument("--total-timeout", default="240")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--code-repair-attempts", type=int, default=3)
    parser.add_argument("--response-protocol", choices=("json", "code_block", "official"), default="json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    processes: list[tuple[str, Path, subprocess.Popen[str]]] = []
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DEEPSEEK_TOTAL_TIMEOUT"] = str(args.total_timeout)
    env["DEEPSEEK_MAX_ATTEMPTS"] = str(args.max_attempts)

    for method in methods:
        for offset in range(max(1, args.case_shards)):
            shard_name = f"{method}_shard_{offset:02d}"
            shard_dir = out_dir / shard_name
            cmd = [
                sys.executable,
                "scripts/baselines/run_static_calibration.py",
                "--manifest",
                args.manifest,
                "--out-dir",
                str(shard_dir),
                "--methods",
                method,
                "--case-stride",
                str(args.case_shards),
                "--case-offset",
                str(offset),
                "--model",
                args.model,
                "--cache-dir",
                str(cache_dir),
                "--max-tokens",
                str(args.max_tokens),
                "--official-repair-attempts",
                str(args.max_attempts),
                "--code-repair-attempts",
                str(args.code_repair_attempts),
            ]
            if args.resume:
                cmd.append("--resume")
            if args.retry_cache_misses:
                cmd.append("--retry-cache-misses")
            if args.retry_statuses:
                cmd.extend(["--retry-statuses", args.retry_statuses])
            if args.no_json_mode:
                cmd.append("--no-json-mode")
            if args.response_protocol != "json":
                cmd.extend(["--response-protocol", args.response_protocol])
            shard_env = dict(env)
            if args.trace_dir:
                shard_env["DEEPSEEK_TRACE_DIR"] = str(Path(args.trace_dir) / shard_name)
            log_path = shard_dir / "launcher_stdout.log"
            shard_dir.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=shard_env, text=True, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((shard_name, shard_dir, proc))
            print(json.dumps({"event": "started", "shard": shard_name, "pid": proc.pid, "out_dir": str(shard_dir)}, sort_keys=True), flush=True)

    try:
        while True:
            running = [(name, proc) for name, _, proc in processes if proc.poll() is None]
            print_progress(out_dir, processes, running)
            if not running:
                break
            time.sleep(max(1.0, float(args.poll_seconds)))
    except KeyboardInterrupt:
        for _, _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        raise

    failures = []
    for name, shard_dir, proc in processes:
        code = proc.poll()
        if code:
            failures.append({"shard": name, "returncode": code, "log": str(shard_dir / "launcher_stdout.log")})
    manifest = {
        "schema_version": "liveopt_static_calibration_sharded_launcher_v1",
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "methods": methods,
        "case_shards": args.case_shards,
        "response_protocol": args.response_protocol,
        "no_json_mode": bool(args.no_json_mode),
        "max_attempts": args.max_attempts,
        "code_repair_attempts": args.code_repair_attempts,
        "trace_dir": args.trace_dir,
        "shards": [
            {
                "name": name,
                "out_dir": str(shard_dir),
                "returncode": proc.poll(),
                "rows": count_rows(shard_dir / "static_calibration_rows.jsonl"),
            }
            for name, shard_dir, proc in processes
        ],
        "failures": failures,
    }
    (out_dir / "sharded_launcher_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


def print_progress(out_dir: Path, processes: list[tuple[str, Path, subprocess.Popen[str]]], running: list[tuple[str, subprocess.Popen[str]]]) -> None:
    rows_by_method: dict[str, int] = {}
    total_rows = 0
    for name, shard_dir, _ in processes:
        method = name.rsplit("_shard_", 1)[0]
        rows = count_rows(shard_dir / "static_calibration_rows.jsonl")
        total_rows += rows
        rows_by_method[method] = rows_by_method.get(method, 0) + rows
    print(
        json.dumps(
            {
                "event": "progress",
                "running": len(running),
                "total_rows": total_rows,
                "rows_by_method": rows_by_method,
                "out_dir": str(out_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
