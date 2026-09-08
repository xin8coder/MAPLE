#!/usr/bin/env python3
"""Audit paper/public NLDO text against the current benchmark and formal replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.render_nldo_full_cases import HEADER, render_episode  # noqa: E402


EXPECTED_EPISODES = tuple(f"NLDO-P{number:03d}" for number in range(1, 16))
PUBLIC_EPISODE_FIELDS = (
    "agent_allowed_solvers",
    "base_benchmark",
    "base_instance_id",
    "domain",
    "episode_id",
    "family",
    "public_initial_problem",
    "source_dataset",
    "source_instance_id",
)
PUBLIC_UPDATE_FIELDS = (
    "difficulty",
    "metadata",
    "public_update",
    "requires_memory",
    "time_index",
    "update_id",
)
FORBIDDEN_PUBLIC_KEYS = {
    "hidden_initial_state",
    "hidden_update_oracle",
    "hidden_evaluator_profile",
    "evaluation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl",
    )
    parser.add_argument(
        "--public-stream",
        type=Path,
        default=ROOT / "release_artifacts/public_preview/data/public_streams/nldo_public_streams.jsonl",
    )
    parser.add_argument(
        "--appendix",
        type=Path,
        default=ROOT / "paper/sections/nldo_full_cases.tex",
    )
    parser.add_argument(
        "--formal-run",
        type=Path,
        default=ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/NLDO/evo2_limit0.jsonl",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "logs/analysis/nldo_public_alignment_20260715/report.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def public_table_root(episode_id: str) -> str:
    return (Path("data") / "public_tables" / episode_id).as_posix()


def normalize_public_context(value: dict[str, Any], episode_id: str) -> dict[str, Any]:
    context = copy.deepcopy(value)
    root = public_table_root(episode_id)
    for table in (context.get("csv_tables") or {}).values():
        filename = Path(str(table.get("path") or "")).name
        table["allowed_root"] = root
        table["path"] = (Path(root) / filename).as_posix()
    return context


def expected_appendix(episodes: list[dict[str, Any]]) -> str:
    return HEADER + "\n".join(render_episode(episode) for episode in episodes) + "\\endgroup\n"


def audit_public_stream(
    benchmark_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    public_root: Path,
    errors: list[str],
) -> dict[str, int]:
    benchmark = {str(row.get("episode_id")): row for row in benchmark_rows}
    public = {str(row.get("episode_id")): row for row in public_rows}
    if tuple(sorted(benchmark)) != EXPECTED_EPISODES:
        errors.append(f"benchmark episode ids differ from P001--P015: {sorted(benchmark)}")
    if tuple(sorted(public)) != EXPECTED_EPISODES:
        errors.append(f"public episode ids differ from P001--P015: {sorted(public)}")

    update_strings = 0
    table_files = 0
    for episode_id in EXPECTED_EPISODES:
        source = benchmark.get(episode_id) or {}
        released = public.get(episode_id) or {}
        leaked = sorted(FORBIDDEN_PUBLIC_KEYS.intersection(released))
        if leaked:
            errors.append(f"{episode_id}: hidden keys in public stream: {leaked}")
        for field in PUBLIC_EPISODE_FIELDS:
            if released.get(field) != source.get(field):
                errors.append(f"{episode_id}: public field mismatch: {field}")
        if released.get("schema_version") != "liveopt_public_episode_stream_preview_v1":
            errors.append(f"{episode_id}: unexpected public schema version")
        expected_root = public_table_root(episode_id)
        if released.get("structured_data_path") != expected_root:
            errors.append(f"{episode_id}: non-portable structured_data_path")
        source_context = normalize_public_context(source.get("public_context") or {}, episode_id)
        released_context = normalize_public_context(released.get("public_context") or {}, episode_id)
        if released_context != source_context:
            errors.append(f"{episode_id}: public_context differs from current benchmark")

        source_updates = list(source.get("update_stream") or [])
        released_updates = list(released.get("update_stream") or [])
        if len(source_updates) != 12 or len(released_updates) != 12:
            errors.append(
                f"{episode_id}: expected 12 updates, found source={len(source_updates)}, public={len(released_updates)}"
            )
        for index, (source_update, released_update) in enumerate(
            zip(source_updates, released_updates), start=1
        ):
            if source_update.get("natural_language_update") != source_update.get("public_update"):
                errors.append(f"{episode_id}/t{index:02d}: benchmark natural/public update mismatch")
            for field in PUBLIC_UPDATE_FIELDS:
                if released_update.get(field) != source_update.get(field):
                    errors.append(f"{episode_id}/t{index:02d}: update field mismatch: {field}")
            if "natural_language_update" in released_update:
                errors.append(f"{episode_id}/t{index:02d}: duplicate natural_language_update in public stream")
            update_strings += 1

        for table in (source.get("public_context") or {}).get("csv_tables", {}).values():
            source_path = Path(str(table.get("path") or ""))
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            target_path = public_root / expected_root / source_path.name
            if not source_path.is_file() or not target_path.is_file():
                errors.append(f"{episode_id}: missing public table copy for {source_path.name}")
                continue
            if sha256_file(source_path) != sha256_file(target_path):
                errors.append(f"{episode_id}: table copy differs for {source_path.name}")
            table_files += 1
    return {"episodes": len(public), "update_strings": update_strings, "public_table_files": table_files}


def audit_formal_run(
    path: Path,
    benchmark_rows: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, int]:
    benchmark = {str(row["episode_id"]): row for row in benchmark_rows}
    rows = 0
    updates = 0
    semantic_only = 0
    warm = 0
    full = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            episode_id = str(row.get("episode_id") or "")
            seed = int(row.get("run_seed") or 0)
            expected_updates = list((benchmark.get(episode_id) or {}).get("update_stream") or [])
            actual_updates = list(row.get("update_results") or [])
            if episode_id not in {f"NLDO-P{number:03d}" for number in range(10, 16)}:
                errors.append(f"formal row {line_number}: out-of-scope episode {episode_id}")
            if seed not in range(10) or len(actual_updates) != 12:
                errors.append(f"formal row {line_number}: seed/update coverage mismatch")
            for index, update in enumerate(actual_updates, start=1):
                updates += 1
                expected = expected_updates[index - 1] if index <= len(expected_updates) else {}
                if update.get("update_id") != expected.get("update_id"):
                    errors.append(f"{episode_id}/seed{seed}/t{index:02d}: update id mismatch")
                if update.get("natural_language_update") != expected.get("public_update"):
                    errors.append(f"{episode_id}/seed{seed}/t{index:02d}: formal update text mismatch")
                restart = update.get("restart") or {}
                skill = str(restart.get("restart_skill") or "")
                warm += int(skill == "warm_restart_v1")
                full += int(skill == "full_restart_v1")
                semantic = restart.get("semantic_restart_gate") or {}
                effective_semantic_only = bool(
                    restart.get("restart_selection_rule") == "verified_semantic_full_else_fixed_warm"
                    and skill in {"warm_restart_v1", "full_restart_v1"}
                    and semantic.get("prompt_version") == "liveopt_semantic_restart_gate_v2"
                    and not restart.get("metric_restart_skill")
                    and not restart.get("restart_fusion_rule")
                    and not restart.get("objective_space_shift")
                )
                if not effective_semantic_only:
                    errors.append(f"{episode_id}/seed{seed}/t{index:02d}: effective restart is not semantic-only")
                semantic_only += int(effective_semantic_only)
    if rows != 60 or updates != 720:
        errors.append(f"formal replay coverage mismatch: rows={rows}, updates={updates}")
    return {
        "seed_rows": rows,
        "update_cells": updates,
        "effective_semantic_only_cells": semantic_only,
        "warm_cells": warm,
        "full_cells": full,
    }


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    benchmark_rows = load_jsonl(args.benchmark)
    public_rows = load_jsonl(args.public_stream)
    public_counts = audit_public_stream(
        benchmark_rows,
        public_rows,
        args.public_stream.parents[2],
        errors,
    )
    rendered = expected_appendix(benchmark_rows)
    appendix_match = args.appendix.read_text(encoding="utf-8") == rendered
    if not appendix_match:
        errors.append("paper/sections/nldo_full_cases.tex is not the exact current benchmark rendering")
    formal_counts = audit_formal_run(args.formal_run, benchmark_rows, errors)
    report = {
        "schema_version": "nldo_public_alignment_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "sources": {
            "benchmark": {"path": str(args.benchmark), "sha256": sha256_file(args.benchmark)},
            "public_stream": {"path": str(args.public_stream), "sha256": sha256_file(args.public_stream)},
            "appendix": {"path": str(args.appendix), "sha256": sha256_file(args.appendix)},
            "formal_run": {"path": str(args.formal_run), "sha256": sha256_file(args.formal_run)},
        },
        "public_alignment": public_counts,
        "appendix_exact_render": appendix_match,
        "formal_alignment": formal_counts,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
