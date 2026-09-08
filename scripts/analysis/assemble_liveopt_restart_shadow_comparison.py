#!/usr/bin/env python3
"""Assemble the all-stage LiveOpt/Warm/Full matched restart comparison.

The experimental unit is a single ``(episode, stage, seed)`` transition.  A
sequential LiveOpt run supplies the incoming population at every transition.
Fixed Warm and Fixed Full are *shadow actions*: they read that same incoming
population, apply their forced restart action, and are scored without becoming
the predecessor of the next transition.

This script deliberately makes the source routing explicit.  It rejects the
older longitudinal-control interpretation in which a purported Fixed Full run
can inherit a Fixed Warm trajectory through earlier stages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


METHODS = ("LiveOpt", "Fixed Warm", "Fixed Full")
STAGES = tuple(range(1, 13))
EXPECTED_SEEDS = tuple(range(10))
SLOTS = (
    "liveopt_ds",
    "liveopt_dm",
    "warm_t11_ds",
    "warm_t11_dm",
    "warm_t12_ds",
    "warm_t12_dm",
    "full_t01_t10_ds",
    "full_t01_t10_dm",
    "full_t12_ds",
    "full_t12_dm",
)


def main() -> int:
    args = parse_args()
    sources = parse_sources(args.source)
    missing_slots = sorted(set(SLOTS) - set(sources))
    if missing_slots:
        raise ValueError(f"missing required source slots: {missing_slots}")

    decisions, decision_meta = load_decisions(args.decision)
    expected_decision_keys = {
        (episode_id(number), stage)
        for number in range(7, 16)
        for stage in STAGES
    }
    if set(decisions) != expected_decision_keys:
        raise ValueError(
            "semantic-decision coverage mismatch: "
            f"missing={sorted(expected_decision_keys - set(decisions))}, "
            f"extra={sorted(set(decisions) - expected_decision_keys)}"
        )

    source_metrics = {slot: load_metric_rows(path) for slot, path in sources.items()}
    source_runtime = {slot: load_runtime_rows(path) for slot, path in sources.items()}
    source_audit = audit_sources(sources, decisions, source_runtime)
    shadow_lineage = audit_shadow_runtime_lineage(source_runtime)
    assembled, manifest = assemble_rows(source_metrics, sources)
    assembled_runtime, _ = assemble_rows(source_runtime, sources)
    coverage = audit_coverage(assembled)
    runtime_coverage = audit_coverage(assembled_runtime)
    runtime_summary = summarize_runtime(assembled_runtime)
    routed_equality = audit_routed_equality(assembled, decisions)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        method_dir = out_dir / method_dir_name(method) / "reference_metrics"
        method_dir.mkdir(parents=True, exist_ok=True)
        write_csv(method_dir / "reference_stage_metrics.csv", assembled[method])
        write_csv(method_dir / "runtime_stage_metrics.csv", assembled_runtime[method])
    write_csv(out_dir / "source_manifest.csv", manifest)

    decision_counts = Counter("Full" if value else "Warm" for value in decisions.values())
    report = {
        "schema_version": "liveopt_restart_shadow_comparison_v1",
        "protocol": {
            "unit": "episode-stage-seed transition",
            "liveopt": "sequential backbone; its selected result becomes the next-stage predecessor",
            "fixed_warm": "Warm shadow action from the same LiveOpt incoming population at every stage",
            "fixed_full": "Full shadow action from the same LiveOpt incoming population at every stage",
            "shadow_continuation": "discarded after scoring; only LiveOpt continues",
            "budget": "population=200, generations<=200 with the same reference-free early-stop rule",
        },
        "scope": {
            "episodes": [episode_id(number) for number in range(7, 16)],
            "stages": list(STAGES),
            "seeds": list(EXPECTED_SEEDS),
            "expected_cells_per_method": 9 * 12 * 10,
        },
        "sources": {slot: str(path) for slot, path in sources.items()},
        "decision_files": decision_meta,
        "decision_counts": dict(decision_counts),
        "coverage": coverage,
        "runtime_coverage": runtime_coverage,
        "runtime_summary": runtime_summary,
        "source_audit": source_audit,
        "shadow_lineage": shadow_lineage,
        "routed_equality": routed_equality,
        "status": "passed",
    }
    (out_dir / "protocol_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="SLOT=RUN_DIR; required slots are: " + ", ".join(SLOTS),
    )
    parser.add_argument(
        "--decision",
        action="append",
        type=Path,
        required=True,
        help="Frozen semantic decision JSON; repeat for split episode groups.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_sources(items: Iterable[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--source must be SLOT=RUN_DIR, got {item!r}")
        slot, raw_path = item.split("=", 1)
        slot = slot.strip()
        if slot not in SLOTS:
            raise ValueError(f"unknown source slot {slot!r}; expected one of {SLOTS}")
        if slot in sources:
            raise ValueError(f"duplicate source slot: {slot}")
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"source run does not exist for {slot}: {path}")
        sources[slot] = path
    return sources


def load_decisions(paths: Iterable[Path]) -> tuple[dict[tuple[str, int], bool], list[dict[str, Any]]]:
    decisions: dict[tuple[str, int], bool] = {}
    metadata = []
    benchmark_hashes = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark_hashes.add(str(payload.get("episodes_jsonl_sha256") or ""))
        metadata.append(
            {
                "path": str(path),
                "model": payload.get("model"),
                "episodes_jsonl_sha256": payload.get("episodes_jsonl_sha256"),
                "all_stages": payload.get("all_stages"),
                "rows": len(payload.get("rows") or []),
            }
        )
        if not payload.get("all_stages"):
            raise ValueError(f"decision file is not an all-stage audit: {path}")
        for row in payload.get("rows") or []:
            key = (str(row["episode_id"]), int(row["stage"]))
            decision = row.get("decision") or {}
            full_vote = bool(decision.get("verified_full_vote"))
            if key in decisions and decisions[key] != full_vote:
                raise ValueError(f"conflicting semantic decisions for {key}")
            decisions[key] = full_vote
    benchmark_hashes.discard("")
    if len(benchmark_hashes) != 1:
        raise ValueError(f"decision files use different benchmark hashes: {sorted(benchmark_hashes)}")
    return decisions, metadata


def load_metric_rows(run_dir: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    path = run_dir / "reference_metrics" / "reference_stage_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing reference metric file: {path}")
    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["episode_id"]), int(row["stage_index"]), int(row["run_seed"]))
            if key in rows:
                raise ValueError(f"duplicate metric cell in {path}: {key}")
            rows[key] = row
    return rows


def load_runtime_rows(run_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    plain = run_dir / "NLDO" / "evo2_limit0.jsonl"
    compressed = plain.with_suffix(plain.suffix + ".gz")
    if plain.exists():
        handle = plain.open(encoding="utf-8")
        path = plain
    elif compressed.exists():
        handle = gzip.open(compressed, "rt", encoding="utf-8")
        path = compressed
    else:
        raise FileNotFoundError(f"missing merged runtime artifact: {plain}(.gz)")
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    with handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            episode = str(payload.get("episode_id") or "")
            seed = int(payload.get("run_seed"))
            for stage, update in enumerate(payload.get("update_results") or [], start=1):
                solver = update.get("solver_result") or {}
                runtime = ((solver.get("metadata") or {}).get("runtime") or {})
                early_stop = runtime.get("early_stop") or {}
                restart = update.get("restart") or ((solver.get("metadata") or {}).get("restart") or {})
                key = (episode, stage, seed)
                if key in rows:
                    raise ValueError(f"duplicate runtime cell in {path}: {key}")
                rows[key] = {
                    "episode_id": episode,
                    "stage_index": stage,
                    "run_seed": seed,
                    "restart_skill": restart.get("restart_skill")
                    or (update.get("impact") or {}).get("restart_skill"),
                    "executed_generations": int(runtime.get("executed_generations") or 0),
                    "early_stopped": bool(runtime.get("early_stopped", early_stop.get("stopped", False))),
                    "stop_generation": early_stop.get("stop_generation"),
                    "stop_reason": early_stop.get("stop_reason"),
                    "latency_seconds": update.get("latency_seconds"),
                    "population_size": int((payload.get("budget") or {}).get("population_size") or 0),
                    "initial_population_size": int((payload.get("budget") or {}).get("initial_population_size") or 0),
                    "generations": int((payload.get("budget") or {}).get("generations") or 0),
                    "initial_generations": int((payload.get("budget") or {}).get("initial_generations") or 0),
                    "use_source_history_population": bool(
                        restart.get("use_source_history_population", False)
                    ),
                    "previous_stage_index": (restart.get("population_lineage") or {}).get(
                        "previous_stage_index"
                    ),
                    "previous_population_policy": (restart.get("population_lineage") or {}).get(
                        "previous_population_policy"
                    ),
                    "source_population_kind": (
                        ((restart.get("population_lineage") or {}).get("source_previous_metadata") or {}).get(
                            "source_population_kind"
                        )
                    ),
                    "source_population_count": (
                        ((restart.get("population_lineage") or {}).get("source_previous_metadata") or {}).get(
                            "source_population_count"
                        )
                    ),
                }
    return rows


def assemble_rows(
    metrics: dict[str, dict[tuple[str, int, int], dict[str, str]]],
    sources: dict[str, Path],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    assembled: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    for method in METHODS:
        for number in range(7, 16):
            episode = episode_id(number)
            view = "DS" if number <= 9 else "DM"
            for stage in STAGES:
                slot = route_slot(method, view, stage)
                manifest.append(
                    {
                        "method": method,
                        "view": view,
                        "episode_id": episode,
                        "stage_index": stage,
                        "expected_action": expected_action(method),
                        "source_slot": slot,
                        "source_run_dir": str(sources[slot]),
                    }
                )
                for seed in EXPECTED_SEEDS:
                    key = (episode, stage, seed)
                    if key not in metrics[slot]:
                        raise ValueError(f"missing routed metric cell {key} in slot {slot}")
                    row = dict(metrics[slot][key])
                    row["assembled_method"] = method
                    row["assembled_source_slot"] = slot
                    row["assembled_source_run_dir"] = str(sources[slot])
                    assembled[method].append(row)
    return dict(assembled), manifest


def route_slot(method: str, view: str, stage: int) -> str:
    suffix = view.lower()
    if method == "LiveOpt":
        return f"liveopt_{suffix}"
    if method == "Fixed Warm":
        if stage <= 10:
            return f"liveopt_{suffix}"
        if stage == 11:
            return f"warm_t11_{suffix}"
        return f"warm_t12_{suffix}"
    if method == "Fixed Full":
        if stage <= 10:
            return f"full_t01_t10_{suffix}"
        if stage == 11:
            return f"liveopt_{suffix}"
        return f"full_t12_{suffix}"
    raise ValueError(f"unknown method: {method}")


def audit_coverage(assembled: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    expected = {
        (episode_id(number), stage, seed)
        for number in range(7, 16)
        for stage in STAGES
        for seed in EXPECTED_SEEDS
    }
    report = {}
    for method in METHODS:
        keys = {
            (str(row["episode_id"]), int(row["stage_index"]), int(row["run_seed"]))
            for row in assembled[method]
        }
        if keys != expected or len(assembled[method]) != len(expected):
            raise ValueError(
                f"assembled coverage mismatch for {method}: rows={len(assembled[method])}, "
                f"unique={len(keys)}, missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
            )
        report[method] = {
            "rows": len(assembled[method]),
            "unique_cells": len(keys),
            "ds_cells": sum(key[0] <= "NLDO-P009" for key in keys),
            "dm_cells": sum(key[0] >= "NLDO-P010" for key in keys),
        }
    return report


def summarize_runtime(assembled: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    report = []
    for method in METHODS:
        for view in ("DS", "DM"):
            rows = [
                row
                for row in assembled[method]
                if (str(row["episode_id"]) <= "NLDO-P009") == (view == "DS")
            ]
            generations = [int(row["executed_generations"]) for row in rows]
            report.append(
                {
                    "method": method,
                    "view": view,
                    "cells": len(rows),
                    "mean_executed_generations": mean(generations),
                    "early_stop_rate": mean(bool(row["early_stopped"]) for row in rows),
                }
            )
    return report


def audit_sources(
    sources: dict[str, Path],
    decisions: dict[tuple[str, int], bool],
    runtime_rows: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    report = []
    for slot, run_dir in sources.items():
        selected_stages = selected_stages_for_slot(slot)
        view = "DS" if slot.endswith("_ds") else "DM"
        episode_numbers = range(7, 10) if view == "DS" else range(10, 16)
        for number in episode_numbers:
            episode = episode_id(number)
            summary_path = run_dir / "replay_jobs" / episode / "summary.json"
            if not summary_path.exists():
                if not (run_dir / "assembly_manifest.json").exists():
                    raise FileNotFoundError(f"missing per-episode summary for {slot}: {summary_path}")
                for stage in selected_stages:
                    selected = [
                        row
                        for (row_episode, row_stage, _), row in runtime_rows[slot].items()
                        if row_episode == episode and row_stage == stage
                    ]
                    if len(selected) != 10:
                        raise ValueError(
                            f"assembled source {slot}/{episode}/t{stage} has {len(selected)} runtime rows, expected 10"
                        )
                    for key in ("population_size", "initial_population_size", "generations", "initial_generations"):
                        if {int(row.get(key) or 0) for row in selected} != {200}:
                            raise ValueError(f"assembled source {slot}/{episode}/t{stage} has non-200 {key}")
                    actual_actions = {normalize_action(row.get("restart_skill")) for row in selected}
                    expected = expected_slot_action(slot, episode, stage, decisions)
                    if actual_actions != {expected}:
                        raise ValueError(
                            f"restart action mismatch for assembled {slot}/{episode}/t{stage}: "
                            f"expected={expected}, actual={sorted(actual_actions)}"
                        )
                    source_kinds = {str(row.get("source_population_kind") or "") for row in selected}
                    source_counts = {int(row.get("source_population_count") or 0) for row in selected}
                    if is_shadow_slot(slot) and (source_kinds != {"final_population"} or source_counts != {200}):
                        raise ValueError(
                            f"assembled shadow did not use an exact 200-member final population: "
                            f"{slot}/{episode}/t{stage}: kinds={sorted(source_kinds)}, counts={sorted(source_counts)}"
                        )
                    report.append(
                        {
                            "source_slot": slot,
                            "episode_id": episode,
                            "stage_index": stage,
                            "action": expected,
                            "seed_rows": len(selected),
                            "source_population_kind": next(iter(source_kinds)) or "sequential_in_memory",
                            "source_population_count": next(iter(source_counts)) or 200,
                            "source_run_dir": str(run_dir),
                            "source_form": "audited_assembly",
                        }
                    )
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed" or int(summary.get("completed_seed_rows") or 0) != 10:
                raise ValueError(f"incomplete 10-seed source for {slot}/{episode}: {summary_path}")
            budget = summary.get("budget") or {}
            for key in ("population_size", "initial_population_size", "generations", "initial_generations"):
                if int(budget.get(key) or 0) != 200:
                    raise ValueError(f"non-200 budget {key} for {slot}/{episode}: {budget.get(key)!r}")
            stage_records = (summary.get("episode_records") or [{}])[0].get("stage_records") or []
            for stage in selected_stages:
                if stage >= len(stage_records):
                    raise ValueError(f"missing stage t{stage} summary for {slot}/{episode}")
                restart = stage_records[stage].get("restart") or {}
                actual_action = normalize_action(restart.get("restart_skill"))
                expected = expected_slot_action(slot, episode, stage, decisions)
                if actual_action != expected:
                    raise ValueError(
                        f"restart action mismatch for {slot}/{episode}/t{stage}: "
                        f"expected={expected}, actual={actual_action}"
                    )
                lineage = restart.get("population_lineage") or {}
                source_kind = ((lineage.get("source_previous_metadata") or {}).get("source_population_kind"))
                source_count = ((lineage.get("source_previous_metadata") or {}).get("source_population_count"))
                if is_shadow_slot(slot):
                    if not restart.get("use_source_history_population"):
                        raise ValueError(f"shadow source does not use LiveOpt history: {slot}/{episode}/t{stage}")
                    if int(lineage.get("previous_stage_index", -1)) != stage - 1:
                        raise ValueError(f"wrong shadow predecessor stage: {slot}/{episode}/t{stage}: {lineage}")
                    if source_kind != "final_population" or int(source_count or 0) != 200:
                        raise ValueError(
                            f"shadow did not rehydrate an exact 200-member final population: "
                            f"{slot}/{episode}/t{stage}: kind={source_kind}, count={source_count}"
                        )
                report.append(
                    {
                        "source_slot": slot,
                        "episode_id": episode,
                        "stage_index": stage,
                        "action": actual_action,
                        "seed_rows": summary.get("completed_seed_rows"),
                        "source_population_kind": source_kind or "sequential_in_memory",
                        "source_population_count": source_count or 200,
                        "source_run_dir": str(run_dir),
                    }
                )
    return report


def audit_shadow_runtime_lineage(
    runtime: dict[str, dict[tuple[str, int, int], dict[str, Any]]]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for slot, rows in runtime.items():
        if not is_shadow_slot(slot):
            continue
        selected = set(selected_stages_for_slot(slot))
        in_scope = [row for key, row in rows.items() if key[1] in selected]
        bad = []
        for row in in_scope:
            stage = int(row["stage_index"])
            if (
                not row["use_source_history_population"]
                or int(row["previous_stage_index"] if row["previous_stage_index"] is not None else -1)
                != stage - 1
                or row["source_population_kind"] != "final_population"
                or int(row["source_population_count"] or 0) != 200
            ):
                bad.append(
                    {
                        key: row.get(key)
                        for key in (
                            "episode_id",
                            "stage_index",
                            "run_seed",
                            "use_source_history_population",
                            "previous_stage_index",
                            "source_population_kind",
                            "source_population_count",
                        )
                    }
                )
        if bad:
            raise ValueError(f"non-exact per-seed shadow lineage in {slot}: {bad[:10]}")
        report[slot] = {"audited_cells": len(in_scope), "bad_cells": 0}
    return report


def audit_routed_equality(
    assembled: dict[str, list[dict[str, str]]], decisions: dict[tuple[str, int], bool]
) -> dict[str, Any]:
    indexed = {
        method: {
            (str(row["episode_id"]), int(row["stage_index"]), int(row["run_seed"])): row
            for row in rows
        }
        for method, rows in assembled.items()
    }
    comparisons = 0
    mismatches = []
    for key, liveopt_row in indexed["LiveOpt"].items():
        episode, stage, _ = key
        routed_method = "Fixed Full" if decisions[(episode, stage)] else "Fixed Warm"
        routed_row = indexed[routed_method][key]
        metric = "normalized_score" if episode <= "NLDO-P009" else "normalized_hv"
        fields = ("feasible", metric)
        if any(str(liveopt_row.get(field, "")) != str(routed_row.get(field, "")) for field in fields):
            mismatches.append(
                {
                    "cell": f"{episode}:t{stage:02d}:seed{key[2]}",
                    "routed_method": routed_method,
                    "metric": metric,
                    "liveopt": {field: liveopt_row.get(field) for field in fields},
                    "routed": {field: routed_row.get(field) for field in fields},
                }
            )
        comparisons += 1
    if mismatches:
        raise ValueError(f"LiveOpt does not equal its routed shadow action: {mismatches[:10]}")
    return {"compared_cells": comparisons, "mismatches": 0}


def selected_stages_for_slot(slot: str) -> tuple[int, ...]:
    if slot.startswith("liveopt_"):
        return STAGES
    if slot.startswith("warm_t11_"):
        return (11,)
    if slot.startswith("warm_t12_") or slot.startswith("full_t12_"):
        return (12,)
    if slot.startswith("full_t01_t10_"):
        return tuple(range(1, 11))
    raise ValueError(slot)


def expected_slot_action(
    slot: str, episode: str, stage: int, decisions: dict[tuple[str, int], bool]
) -> str:
    if slot.startswith("liveopt_"):
        return "Full" if decisions[(episode, stage)] else "Warm"
    if slot.startswith("warm_"):
        return "Warm"
    if slot.startswith("full_"):
        return "Full"
    raise ValueError(slot)


def is_shadow_slot(slot: str) -> bool:
    return slot.startswith(("warm_t12_", "full_t12_", "full_t01_t10_"))


def expected_action(method: str) -> str:
    if method == "LiveOpt":
        return "LLM-selected"
    return "Warm" if method == "Fixed Warm" else "Full"


def normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "warm_restart_v1":
        return "Warm"
    if text == "full_restart_v1":
        return "Full"
    return str(value or "")


def episode_id(number: int) -> str:
    return f"NLDO-P{number:03d}"


def method_dir_name(method: str) -> str:
    return method.lower().replace(" ", "_")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
