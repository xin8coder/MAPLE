#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("liveopt", "no_tss", "fixed_warm", "fixed_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict audit for the revised four-method LiveOpt ablation.")
    for key in METHODS:
        parser.add_argument(f"--{key.replace('_', '-')}-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: dict[str, Any] = {}
    errors: list[str] = []
    for key in METHODS:
        run_dir = getattr(args, f"{key}_run_dir")
        report, method_errors = audit_method(key, run_dir)
        reports[key] = report
        errors.extend(f"{key}: {error}" for error in method_errors)
    payload = {
        "status": "passed" if not errors else "failed",
        "protocol": "liveopt_no_tss_and_restart_ablation_p010_p015_200x200x10_v2",
        "methods": reports,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


def audit_method(key: str, run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows = load_jsonl(run_dir / "NLDO" / "evo2_limit0.jsonl")
    selected = [row for row in rows if 10 <= episode_number(row.get("episode_id")) <= 15]
    if len(selected) != 60:
        errors.append(f"expected 60 seed rows, got {len(selected)}")
    cells = 0
    final_population_counts: list[int] = []
    restart_counts: dict[str, int] = {}
    source_population_kinds: dict[str, int] = {}
    generic_artifact_hashes: set[str] = set()
    for row in selected:
        budget = row.get("budget") or {}
        if int(budget.get("population_size") or 0) != 200 or int(budget.get("generations") or 0) != 200:
            errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')}: budget is not 200x200")
        if int(budget.get("archive_limit") or 0) != 500:
            errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')}: archive cap is not 500")
        stages = [(0, row.get("initial_solver_result") or {})]
        updates = list(row.get("update_results") or [])
        if len(updates) != 12:
            errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')}: expected 12 updates, got {len(updates)}")
        stages.extend((index, update.get("solver_result") or {}) for index, update in enumerate(updates, start=1))
        for _, solver in stages:
            population = list(((solver.get("metadata") or {}).get("final_population") or []))
            final_population_counts.append(len(population))
        for stage, update in enumerate(updates, start=1):
            cells += 1
            restart = update.get("restart") or {}
            skill = str(restart.get("restart_skill") or "")
            restart_counts[skill] = restart_counts.get(skill, 0) + 1
            lineage = restart.get("population_lineage") or {}
            source_kind = str(((lineage.get("source_previous_metadata") or {}).get("source_population_kind") or ""))
            if source_kind:
                source_population_kinds[source_kind] = source_population_kinds.get(source_kind, 0) + 1
            if key in {"fixed_warm", "fixed_full"} and source_kind != "final_population":
                errors.append(
                    f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: source is {source_kind or 'missing'}, not final_population"
                )
            if key == "fixed_warm" and skill != "warm_restart_v1":
                errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: expected Warm, got {skill}")
            if key == "fixed_full" and skill != "full_restart_v1":
                errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: expected Full, got {skill}")
            if key == "no_tss":
                expected = "warm_restart_v1" if stage <= 10 else "full_restart_v1"
                if skill != expected:
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: expected {expected}, got {skill}")
                if restart.get("workbench_interface") != "generic_blank_callbacks_v1":
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: Generic Workbench is not the blank callback interface")
                if restart.get("generic_workbench_origin") != "from_scratch_public_problem_and_bare_abi":
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: Generic Workbench inherits a non-blank origin")
                expected_transfer = "decoded_own_solution_reencoding" if expected == "warm_restart_v1" else "none_full_restart"
                if restart.get("source_transfer_kind") != expected_transfer:
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: Generic transfer kind is not {expected_transfer}")
                if lineage.get("uses_adaptive_population") is not False:
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: w/o TSS uses a source population")
                if lineage.get("history_origin") != "own_previous_stage_same_seed":
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: w/o TSS does not advance its own history")
                if (lineage.get("source_previous_metadata") or {}):
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: w/o TSS leaks source metadata")
                if str(lineage.get("previous_population_policy") or "").startswith("source_liveopt"):
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}: w/o TSS inherits LiveOpt history")
                if row.get("generic_workbench_origin") != "from_scratch_public_problem_and_bare_abi":
                    errors.append(f"{row.get('episode_id')} seed {row.get('run_seed')}: missing from-scratch Generic row provenance")
                artifact = str(update.get("setup_code") or "")
                generic_artifact_hashes.add(str(hash(artifact)))
                errors.extend(audit_generic_artifact(artifact, row, stage))
    if key == "liveopt" and (not final_population_counts or set(final_population_counts) != {200}):
        errors.append(f"formal LiveOpt source does not retain 200 candidates at every stage: {sorted(set(final_population_counts))}")
    metric_rows = load_csv(run_dir / "reference_metrics" / "reference_stage_metrics.csv")
    metric_rows = [
        row for row in metric_rows
        if 10 <= episode_number(row.get("episode_id")) <= 15 and 1 <= int(row.get("stage_index") or 0) <= 12
    ]
    if len(metric_rows) != 720:
        errors.append(f"expected 720 evaluated update cells, got {len(metric_rows)}")
    above_one = [float(row.get("normalized_hv") or 0.0) for row in metric_rows if float(row.get("normalized_hv") or 0.0) > 1.0 + 1e-9]
    failed = [row for row in metric_rows if str(row.get("true_pass")).lower() not in {"true", "1"}]
    failed_with_nonzero_hv = [row for row in failed if abs(float(row.get("normalized_hv") or 0.0)) > 1e-12]
    if failed_with_nonzero_hv:
        errors.append(f"{len(failed_with_nonzero_hv)} hidden-infeasible cells retain nonzero normalized HV")
    return {
        "run_dir": str(run_dir),
        "seed_rows": len(selected),
        "update_cells": cells,
        "metric_cells": len(metric_rows),
        "restart_counts": restart_counts,
        "source_population_kinds": source_population_kinds,
        "final_population_counts": sorted(set(final_population_counts)),
        "generic_artifact_versions": len(generic_artifact_hashes),
        "hidden_infeasible_cells": len(failed),
        "above_reference_cells": len(above_one),
        "max_hv_ratio": max((float(row.get("normalized_hv") or 0.0) for row in metric_rows), default=0.0),
    }, errors


def audit_generic_artifact(artifact: str, row: dict[str, Any], stage: int) -> list[str]:
    prefix = f"{row.get('episode_id')} seed {row.get('run_seed')} t{stage:02d}"
    if not artifact.strip():
        return [f"{prefix}: empty generic artifact"]
    try:
        tree = ast.parse(artifact)
    except SyntaxError as exc:
        return [f"{prefix}: generic artifact syntax error: {exc}"]
    errors: list[str] = []
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        errors.append(f"{prefix}: generic artifact imports modules")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    forbidden = sorted(names & {"SegmentSpec", "run_evolution", "run_generic_evolution", "nsga2_select"})
    if forbidden:
        errors.append(f"{prefix}: generic artifact uses typed/driver names {forbidden}")
    return errors


def episode_number(value: Any) -> int:
    text = str(value or "")
    try:
        return int(text.split("P")[-1])
    except ValueError:
        return -1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    raise SystemExit(main())
