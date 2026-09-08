#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.agents.liveopt_dynamic_impl import apply_public_context_patch, normalize_public_context_tables
from scripts.analysis.evaluate_reference_metrics import apply_hidden_delta, effective_benchmark
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import public_context_with_loaded_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare replayed public table patches with overlapping fields in the hidden oracle state."
    )
    parser.add_argument("--episodes-jsonl", type=Path, default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"))
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    episodes = {str(row["episode_id"]): row for row in load_jsonl(args.episodes_jsonl)}
    selected = set(args.episode_id or [])
    mismatch_seeds: dict[tuple[str, str, str, str, str, str, str], set[Any]] = defaultdict(set)
    audited_rows = 0
    compared_cells = 0
    for run in load_jsonl(args.run_jsonl):
        episode_id = str(run.get("episode_id") or "")
        if selected and episode_id not in selected:
            continue
        episode = episodes[episode_id]
        public_context = normalize_public_context_tables(public_context_with_loaded_tables(episode))
        hidden_state = copy.deepcopy(episode.get("hidden_initial_state") or {})
        oracle = list(episode.get("hidden_update_oracle") or [])
        benchmark = effective_benchmark("NLDO", episode)
        domain = str(episode.get("domain") or "")
        audited_rows += 1
        stages = [("initial", {})]
        stages.extend(
            (str(stage.get("update_id") or ""), ((stage.get("restart") or {}).get("data_patch") or {}))
            for stage in run.get("update_results") or []
            if isinstance(stage, dict)
        )
        for stage_index, (update_id, patch) in enumerate(stages):
            if stage_index > 0:
                if patch:
                    public_context = normalize_public_context_tables(apply_public_context_patch(public_context, patch))
                if stage_index - 1 < len(oracle):
                    hidden_state = apply_hidden_delta(
                        benchmark,
                        domain,
                        hidden_state,
                        (oracle[stage_index - 1].get("hidden_delta") or {}),
                    )
            cells, mismatches = compare_overlapping_tables(public_context.get("tables") or {}, hidden_state)
            compared_cells += cells
            for table, entity, field, actual, expected in mismatches:
                key = (episode_id, update_id, table, entity, field, json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
                mismatch_seeds[key].add(run.get("run_seed"))

    mismatches = []
    for key, seeds in sorted(mismatch_seeds.items()):
        episode_id, update_id, table, entity, field, actual_json, expected_json = key
        mismatches.append(
            {
                "episode_id": episode_id,
                "update_id": update_id,
                "table": table,
                "entity": entity,
                "field": field,
                "actual": json.loads(actual_json),
                "expected": json.loads(expected_json),
                "seed_count": len(seeds),
                "seeds": sorted(seeds, key=lambda value: (value is None, str(value))),
            }
        )
    payload = {
        "schema_version": "nldo_public_hidden_overlap_alignment_v1",
        "run_jsonl": str(args.run_jsonl),
        "audited_rows": audited_rows,
        "compared_cells": compared_cells,
        "mismatch_count": len(mismatches),
        "passed": not mismatches,
        "mismatches": mismatches,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and mismatches else 0


def compare_overlapping_tables(
    tables: dict[str, Any], hidden_state: dict[str, Any]
) -> tuple[int, list[tuple[str, str, str, Any, Any]]]:
    compared = 0
    mismatches = []
    for table_name, actual_rows in tables.items():
        expected_rows = hidden_state.get(table_name)
        if not isinstance(actual_rows, list) or not isinstance(expected_rows, list):
            continue
        actual_by_id = {str(row.get("id")): row for row in actual_rows if isinstance(row, dict) and row.get("id") is not None}
        expected_by_id = {str(row.get("id")): row for row in expected_rows if isinstance(row, dict) and row.get("id") is not None}
        for entity in sorted(set(actual_by_id) & set(expected_by_id)):
            actual = actual_by_id[entity]
            expected = expected_by_id[entity]
            for field in sorted((set(actual) & set(expected)) - {"id"}):
                compared += 1
                if not equivalent(actual[field], expected[field]):
                    mismatches.append((str(table_name), entity, field, actual[field], expected[field]))
    return compared, mismatches


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    # Public CSV tables preserve compact collection-valued cells such as
    # ``E01|E03|E07`` while the hidden benchmark state stores the same value as
    # a JSON list.  Compare their canonical token sequences so the audit flags
    # state drift, not an equivalent serialization choice.
    if isinstance(left, str) and isinstance(right, list):
        return split_collection_cell(left) == [str(value) for value in right]
    if isinstance(left, list) and isinstance(right, str):
        return [str(value) for value in left] == split_collection_cell(right)
    return left == right


def split_collection_cell(value: str) -> list[str]:
    return [token.strip() for token in value.split("|") if token.strip()]


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
