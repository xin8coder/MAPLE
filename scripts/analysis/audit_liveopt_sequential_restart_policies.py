#!/usr/bin/env python3
"""Audit and summarize the formal sequential restart-policy controls.

This script is deliberately provider-free.  It compares the frozen LiveOpt
trajectory with independent Always-Warm and Always-Full trajectories that
start from the same typed initial artifact and then carry only their own
previous population forward.  The resulting comparison is an end-to-end
policy check; the same-incoming-population action shadows remain the causal
restart-action analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVEOPT = ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714"
DEFAULT_WARM = ROOT / "outputs/liveopt_sequential_fixed_warm_p010_p015_200x200x10_20260715"
DEFAULT_FULL = ROOT / "outputs/liveopt_sequential_fixed_full_p010_p015_200x200x10_20260715"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_sequential_restart_rows.tex"
EPISODES = tuple(f"NLDO-P{number:03d}" for number in range(10, 16))
STAGES = tuple(range(1, 13))
SEEDS = tuple(range(10))
METHODS = (
    ("LiveOpt", "liveopt", DEFAULT_LIVEOPT, None),
    ("Always Warm", "warm", DEFAULT_WARM, "warm_restart_v1"),
    ("Always Full", "full", DEFAULT_FULL, "full_restart_v1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liveopt-run", type=Path, default=DEFAULT_LIVEOPT)
    parser.add_argument("--warm-run", type=Path, default=DEFAULT_WARM)
    parser.add_argument("--full-run", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--bootstrap-repeats", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.exists():
        handle = path.open(encoding="utf-8")
    elif path.with_suffix(path.suffix + ".gz").exists():
        handle = gzip.open(path.with_suffix(path.suffix + ".gz"), "rt", encoding="utf-8")
    else:
        raise FileNotFoundError(f"missing JSONL: {path}(.gz)")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_run_rows(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "NLDO/evo2_limit0.jsonl"):
        episode = str(row.get("episode_id") or "")
        if episode not in EPISODES:
            continue
        key = (episode, int(row.get("run_seed", -1)))
        if key in rows:
            raise ValueError(f"duplicate formal run row in {run_dir}: {key}")
        rows[key] = row
    return rows


def load_metrics(run_dir: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    path = run_dir / "reference_metrics/reference_stage_metrics.csv"
    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = str(row.get("episode_id") or "")
            stage = int(row.get("stage_index") or -1)
            if episode not in EPISODES or stage not in STAGES:
                continue
            key = (episode, stage, int(row.get("run_seed") or -1))
            if key in rows:
                raise ValueError(f"duplicate formal metric in {path}: {key}")
            rows[key] = row
    return rows


def initial_fingerprint(row: dict[str, Any]) -> str:
    """Hash deterministic initial evidence while ignoring trace-retention fields."""

    result = row.get("initial_solver_result") or {}
    metadata = result.get("metadata") or {}
    payload = {
        "setup_code": row.get("setup_code"),
        "fitness_code": row.get("fitness_code"),
        "initial_candidate": row.get("initial_candidate"),
        "initial_feasible": row.get("initial_feasible"),
        "initial_objective": row.get("initial_objective"),
        "success": result.get("success"),
        "solution": result.get("solution"),
        "agent_objective_raw": metadata.get("agent_objective_raw"),
        "candidate_archive": metadata.get("candidate_archive"),
        "problem_spec": metadata.get("problem_spec"),
    }
    return stable_hash(payload)


def typed_artifact_fingerprint(row: dict[str, Any], stage: int) -> str:
    if stage == 0:
        setup = row.get("setup_code")
        fitness = row.get("fitness_code")
    else:
        update = (row.get("update_results") or [])[stage - 1]
        setup = update.get("setup_code") or row.get("setup_code")
        fitness = update.get("fitness_code") or row.get("fitness_code")
    return stable_hash({"setup_code": setup, "fitness_code": fitness})


def restart_metadata(update: dict[str, Any]) -> dict[str, Any]:
    solver = update.get("solver_result") or {}
    runtime = ((solver.get("metadata") or {}).get("runtime") or {})
    return update.get("restart") or runtime.get("restart") or {}


def runtime_metadata(update: dict[str, Any]) -> dict[str, Any]:
    solver = update.get("solver_result") or {}
    return ((solver.get("metadata") or {}).get("runtime") or {})


def expected_cells() -> set[tuple[str, int, int]]:
    return {(episode, stage, seed) for episode in EPISODES for stage in STAGES for seed in SEEDS}


def expected_rows() -> set[tuple[str, int]]:
    return {(episode, seed) for episode in EPISODES for seed in SEEDS}


def quality(row: dict[str, str]) -> float:
    if not (parse_bool(row.get("true_pass")) and parse_bool(row.get("feasible"))):
        return 0.0
    return float(row.get("normalized_hv") or 0.0)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(values: list[float], repeats: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [fmean(rng.choice(values) for _ in values) for _ in range(repeats)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def exact_sign_pvalue(values: list[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if not n:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    extreme = max(positive, n - positive)
    tail = sum(math.comb(n, count) for count in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def audit_launcher_scope(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "launcher_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    statuses = payload.get("episode_status") or {}
    in_scope = {episode: statuses.get(episode, {}).get("status") for episode in EPISODES}
    failed_outside = sorted(
        episode
        for episode, status in statuses.items()
        if episode not in EPISODES and status.get("status") == "failed"
    )
    return {
        "reported_status": payload.get("status"),
        "reference_eval_exit_code": payload.get("reference_eval_exit_code"),
        "formal_replay_rows": payload.get("formal_replay_rows"),
        "in_scope_status": in_scope,
        "out_of_scope_preflight_failures": failed_outside,
    }


def audit_method(
    label: str,
    key: str,
    run_dir: Path,
    forced_action: str | None,
    rows: dict[tuple[str, int], dict[str, Any]],
    metrics: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    if set(rows) != expected_rows():
        failures.append(f"run-row coverage {len(rows)}/60")
    if set(metrics) != expected_cells():
        failures.append(f"metric coverage {len(metrics)}/720")

    full_action_cells = 0
    actions_by_unit: dict[tuple[str, int], set[str]] = defaultdict(set)
    total_tokens = 0
    generic_calls = 0
    own_lineage_cells = 0
    runtime_rows: list[dict[str, Any]] = []
    for (episode, seed), row in sorted(rows.items()):
        budget = row.get("budget") or {}
        for field in ("population_size", "initial_population_size", "generations", "initial_generations"):
            if int(budget.get(field) or 0) != 200:
                failures.append(f"{episode}/seed{seed}: {field}={budget.get(field)}")
        usage = row.get("token_usage") or {}
        total_tokens += int(usage.get("total_tokens") or 0)
        generic_calls += int(usage.get("generic_build_calls") or 0) + int(usage.get("generic_patch_calls") or 0)
        updates = list(row.get("update_results") or [])
        if len(updates) != 12:
            failures.append(f"{episode}/seed{seed}: {len(updates)} updates")
            continue
        for stage, update in enumerate(updates, start=1):
            restart = restart_metadata(update)
            runtime = runtime_metadata(update)
            action = str(restart.get("restart_skill") or "")
            full_action_cells += int(action == "full_restart_v1")
            actions_by_unit[(episode, stage)].add(action)
            if forced_action is not None and action != forced_action:
                failures.append(f"{episode}/t{stage:02d}/seed{seed}: action={action}")
            snapshot = update.get("token_usage_snapshot") or {}
            if key in {"warm", "full"} and int(snapshot.get("total_tokens") or 0) != 0:
                failures.append(f"{episode}/t{stage:02d}/seed{seed}: nonzero tokens")
            lineage = restart.get("population_lineage") or {}
            if key in {"warm", "full"}:
                valid_lineage = (
                    lineage.get("history_origin") == "own_previous_stage_same_seed"
                    and int(lineage.get("previous_stage_index", -1)) == stage - 1
                    and not bool(restart.get("use_source_history_population", False))
                )
                own_lineage_cells += int(valid_lineage)
                if not valid_lineage:
                    failures.append(f"{episode}/t{stage:02d}/seed{seed}: invalid sequential lineage")
            executed = int(runtime.get("executed_generations") or 0)
            if not 1 <= executed <= 200:
                failures.append(f"{episode}/t{stage:02d}/seed{seed}: generations={executed}")
            runtime_rows.append(
                {
                    "method": label,
                    "episode_id": episode,
                    "stage_index": stage,
                    "run_seed": seed,
                    "restart_skill": action,
                    "executed_generations": executed,
                    "early_stopped": bool(runtime.get("early_stopped")),
                    "history_origin": lineage.get("history_origin", ""),
                    "previous_stage_index": lineage.get("previous_stage_index"),
                }
            )

    inconsistent_actions = {
        unit: sorted(actions) for unit, actions in actions_by_unit.items() if len(actions) != 1
    }
    if inconsistent_actions:
        failures.append(f"{len(inconsistent_actions)} episode-stage actions differ across seeds")
    if key in {"warm", "full"} and total_tokens != 0:
        failures.append(f"control replay used {total_tokens} tokens")
    if key in {"warm", "full"} and generic_calls != 0:
        failures.append(f"control replay made {generic_calls} generic Workbench calls")

    values = [quality(row) for row in metrics.values()]
    summary = {
        "method": label,
        "key": key,
        "run_dir": str(run_dir),
        "run_rows": len(rows),
        "metric_cells": len(metrics),
        "true_pass_cells": sum(
            parse_bool(row.get("true_pass")) and parse_bool(row.get("feasible")) for row in metrics.values()
        ),
        "full_actions": sum(
            next(iter(actions)) == "full_restart_v1"
            for actions in actions_by_unit.values()
            if len(actions) == 1
        ),
        "full_action_cells": full_action_cells,
        "decision_units": 72,
        "mean_hv": fmean(values),
        "mean_generations": fmean(row["executed_generations"] for row in runtime_rows),
        "early_stop_rate": fmean(row["early_stopped"] for row in runtime_rows),
        "max_hv_ratio": max(float(row.get("normalized_hv") or 0.0) for row in metrics.values()),
        "above_reference_cells": sum(float(row.get("normalized_hv") or 0.0) > 1.0 + 1e-9 for row in metrics.values()),
        "total_tokens": total_tokens,
        "generic_calls": generic_calls,
        "own_lineage_cells": own_lineage_cells,
        "audit_failures": failures,
    }
    return summary, runtime_rows


def render_tex(method_rows: list[dict[str, Any]]) -> str:
    live_hv = float(method_rows[0]["mean_hv"])
    lines = [
        "% Generated by scripts/analysis/audit_liveopt_sequential_restart_policies.py.",
        r"\newcommand{\LiveOptSequentialRestartRows}{%",
    ]
    for index, row in enumerate(method_rows):
        label = r"\methodours" if index == 0 else str(row["method"])
        delta = r"\textemdash{}" if index == 0 else f"${float(row['mean_hv']) - live_hv:+.3f}$"
        lines.append(
            f"{label} & {int(row['full_actions'])}/72 & "
            f"{100.0 * int(row['true_pass_cells']) / 720.0:.1f}\\% & {float(row['mean_hv']):.3f} & "
            f"{float(row['mean_generations']):.1f} & {delta} " + r"\\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    method_specs = (
        ("LiveOpt", "liveopt", args.liveopt_run, None),
        ("Always Warm", "warm", args.warm_run, "warm_restart_v1"),
        ("Always Full", "full", args.full_run, "full_restart_v1"),
    )
    run_rows = {key: load_run_rows(run_dir) for _, key, run_dir, _ in method_specs}
    metrics = {key: load_metrics(run_dir) for _, key, run_dir, _ in method_specs}

    method_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for label, key, run_dir, action in method_specs:
        summary, method_runtime = audit_method(
            label, key, run_dir, action, run_rows[key], metrics[key]
        )
        method_rows.append(summary)
        runtime_rows.extend(method_runtime)

    initial_mismatches: list[dict[str, Any]] = []
    artifact_mismatches: list[dict[str, Any]] = []
    reference_mismatches: list[dict[str, Any]] = []
    for episode in EPISODES:
        for seed in SEEDS:
            fingerprints = {
                key: initial_fingerprint(run_rows[key][(episode, seed)])
                for _, key, _, _ in method_specs
            }
            if len(set(fingerprints.values())) != 1:
                initial_mismatches.append({"episode_id": episode, "run_seed": seed, **fingerprints})
            for stage in range(13):
                artifacts = {
                    key: typed_artifact_fingerprint(run_rows[key][(episode, seed)], stage)
                    for _, key, _, _ in method_specs
                }
                if len(set(artifacts.values())) != 1:
                    artifact_mismatches.append(
                        {"episode_id": episode, "stage_index": stage, "run_seed": seed, **artifacts}
                    )
        for stage in STAGES:
            for seed in SEEDS:
                key3 = (episode, stage, seed)
                refs = {method: metrics[method][key3].get("reference_hv") for method in metrics}
                if len(set(refs.values())) != 1:
                    reference_mismatches.append(
                        {"episode_id": episode, "stage_index": stage, "run_seed": seed, **refs}
                    )

    episode_rows: list[dict[str, Any]] = []
    for episode in EPISODES:
        row: dict[str, Any] = {"episode_id": episode}
        for _, key, _, _ in method_specs:
            cells = [metrics[key][(episode, stage, seed)] for stage in STAGES for seed in SEEDS]
            row[f"{key}_hv"] = fmean(quality(cell) for cell in cells)
            gens = [
                runtime["executed_generations"]
                for runtime in runtime_rows
                if runtime["method"] == next(label for label, item, _, _ in method_specs if item == key)
                and runtime["episode_id"] == episode
            ]
            row[f"{key}_generations"] = fmean(gens)
        row["live_minus_warm"] = row["liveopt_hv"] - row["warm_hv"]
        row["live_minus_full"] = row["liveopt_hv"] - row["full_hv"]
        episode_rows.append(row)

    stage_rows: list[dict[str, Any]] = []
    for episode in EPISODES:
        for stage in STAGES:
            row = {"episode_id": episode, "stage_index": stage}
            for _, key, _, _ in method_specs:
                row[f"{key}_hv"] = fmean(quality(metrics[key][(episode, stage, seed)]) for seed in SEEDS)
            row["live_minus_warm"] = row["liveopt_hv"] - row["warm_hv"]
            row["live_minus_full"] = row["liveopt_hv"] - row["full_hv"]
            stage_rows.append(row)

    warm_deltas = [float(row["live_minus_warm"]) for row in episode_rows]
    full_deltas = [float(row["live_minus_full"]) for row in episode_rows]
    warm_ci = bootstrap_ci(warm_deltas, args.bootstrap_repeats, args.bootstrap_seed)
    full_ci = bootstrap_ci(full_deltas, args.bootstrap_repeats, args.bootstrap_seed + 1)

    launcher_scope = {
        "warm": audit_launcher_scope(args.warm_run),
        "full": audit_launcher_scope(args.full_run),
    }
    strict_failures = [
        failure
        for row in method_rows
        for failure in row["audit_failures"]
    ]
    if initial_mismatches:
        strict_failures.append(f"{len(initial_mismatches)} initial-artifact mismatches")
    if artifact_mismatches:
        strict_failures.append(f"{len(artifact_mismatches)} typed-artifact mismatches")
    if reference_mismatches:
        strict_failures.append(f"{len(reference_mismatches)} reference-HV mapping mismatches")
    for method in ("warm", "full"):
        scope = launcher_scope[method]
        if scope["reference_eval_exit_code"] != 0:
            strict_failures.append(f"{method}: reference evaluator failed")
        if any(status != "completed" for status in scope["in_scope_status"].values()):
            strict_failures.append(f"{method}: incomplete in-scope launcher job")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "method_summary.csv", method_rows)
    write_csv(args.out_dir / "episode_summary.csv", episode_rows)
    write_csv(args.out_dir / "stage_summary.csv", stage_rows)
    write_csv(args.out_dir / "runtime_cells.csv", runtime_rows)
    write_csv(args.out_dir / "initial_mismatches.csv", initial_mismatches)
    write_csv(args.out_dir / "artifact_mismatches.csv", artifact_mismatches)
    write_csv(args.out_dir / "reference_mismatches.csv", reference_mismatches)
    args.paper_table.parent.mkdir(parents=True, exist_ok=True)
    args.paper_table.write_text(render_tex(method_rows), encoding="utf-8")

    payload = {
        "protocol": "liveopt_sequential_restart_policy_audit_p010_p015_200x200x10_v1",
        "interpretation": (
            "End-to-end sequential policy check: identical typed initial artifacts, then each arm uses "
            "only its own immediately preceding population. Same-incoming-population action shadows are "
            "the separate causal action analysis."
        ),
        "scope": {"episodes": list(EPISODES), "stages": list(STAGES), "seeds": list(SEEDS)},
        "methods": method_rows,
        "paired_episode_statistics": {
            "liveopt_minus_always_warm": {
                "mean": fmean(warm_deltas),
                "episode_cluster_bootstrap_95_ci": list(warm_ci),
                "exact_two_sided_sign_p": exact_sign_pvalue(warm_deltas),
                "positive_episodes": sum(value > 0 for value in warm_deltas),
            },
            "liveopt_minus_always_full": {
                "mean": fmean(full_deltas),
                "episode_cluster_bootstrap_95_ci": list(full_ci),
                "exact_two_sided_sign_p": exact_sign_pvalue(full_deltas),
                "positive_episodes": sum(value > 0 for value in full_deltas),
            },
        },
        "identity_audit": {
            "initial_artifact_comparisons": len(EPISODES) * len(SEEDS),
            "initial_mismatches": len(initial_mismatches),
            "typed_artifact_comparisons": len(EPISODES) * len(SEEDS) * 13,
            "typed_artifact_mismatches": len(artifact_mismatches),
            "reference_mapping_comparisons": len(expected_cells()),
            "reference_mapping_mismatches": len(reference_mismatches),
        },
        "launcher_scope_note": (
            "The generic launcher also attempted P001--P009; those jobs failed at source-artifact "
            "preflight before optimization. This audit gates exclusively on the declared P010--P015 jobs."
        ),
        "launcher_scope": launcher_scope,
        "provenance": {
            key: {
                "run_dir": str(run_dir),
                "metrics_sha256": sha256_file(run_dir / "reference_metrics/reference_stage_metrics.csv"),
                "rows_sha256": sha256_file(run_dir / "NLDO/evo2_limit0.jsonl"),
            }
            for _, key, run_dir, _ in method_specs
        },
        "strict_status": "passed" if not strict_failures else "failed",
        "strict_failures": strict_failures,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and strict_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
