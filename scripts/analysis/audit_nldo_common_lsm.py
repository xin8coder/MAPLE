#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "hidden_initial_state",
    "hidden_update_oracle",
    "hidden_delta",
    "reference_solution",
    "reference_trajectory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the common own-history LSM contract for public NLDO baselines.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-prompt-version", default="nldo_dynamic_common_public_lsm_v6")
    parser.add_argument("--expect-full", action="store_true", help="Require t00--t12 for all 15 episodes per observed method.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for run_dir in args.run_dir:
        path = run_dir / "dynamic_public_baseline_stage_rows.jsonl"
        if not path.exists():
            continue
        rows.extend(load_jsonl(path))
        sources.append(str(path))

    errors: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    method_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        method = str(row.get("method") or "")
        episode = str(row.get("episode_id") or "")
        stage = int(row.get("stage_index") or 0)
        key = (method, episode, stage)
        if key in seen:
            errors.append(f"duplicate stage row: {key}")
        seen.add(key)
        grouped[(method, episode)].append(row)
        method_counts[method] += 1
        prefix = f"{method}/{episode}/t{stage:02d}"
        if row.get("prompt_version") != args.expected_prompt_version:
            errors.append(f"{prefix}: prompt version is {row.get('prompt_version')!r}")
        row_lsm = row.get("shared_lsm_input") or {}
        if row_lsm.get("interface") != "common_public_lsm_dialogue_and_own_population_v1":
            errors.append(f"{prefix}: missing common LSM interface")
        if row_lsm.get("inherits_liveopt_population") is not False:
            errors.append(f"{prefix}: LiveOpt population inheritance is not explicitly false")
        if row_lsm.get("hidden_evaluation_controls_state") is not False:
            errors.append(f"{prefix}: hidden evaluation may control the public state")
        try:
            payload = json.loads(str(row.get("selected_stage_prompt") or ""))
        except json.JSONDecodeError as exc:
            errors.append(f"{prefix}: selected stage prompt is not JSON: {exc}")
            continue
        params = ((payload.get("public_context") or {}).get("parameters") or {})
        lsm = params.get("shared_lsm") or {}
        dialogue = list(lsm.get("cumulative_dialogue") or [])
        population = list(lsm.get("previous_candidate_population") or [])
        if lsm.get("interface") != "common_public_lsm_dialogue_and_own_population_v1":
            errors.append(f"{prefix}: prompt omits common LSM interface")
        if lsm.get("liveopt_population_available") is not False:
            errors.append(f"{prefix}: prompt does not forbid LiveOpt population access")
        if int(lsm.get("population_limit") or 0) != 200:
            errors.append(f"{prefix}: population limit is not 200")
        if int(lsm.get("previous_candidate_count") or 0) != len(population):
            errors.append(f"{prefix}: candidate count does not match transmitted population")
        if len(population) > 200:
            errors.append(f"{prefix}: more than 200 candidates were transmitted")
        if int(lsm.get("dialogue_turn_count") or 0) != stage + 1 or len(dialogue) != stage + 1:
            errors.append(f"{prefix}: expected {stage + 1} cumulative dialogue turns, got {len(dialogue)}")
        if not dialogue or dialogue[0].get("turn_id") != "initial":
            errors.append(f"{prefix}: initial public request is missing from dialogue")
        forbidden = sorted(find_keys(params) & FORBIDDEN_PUBLIC_KEYS)
        if forbidden:
            errors.append(f"{prefix}: forbidden evaluation keys exposed: {forbidden}")

    for (method, episode), episode_rows in grouped.items():
        episode_rows.sort(key=lambda item: int(item.get("stage_index") or 0))
        expected_previous_count = 0
        for row in episode_rows:
            stage = int(row.get("stage_index") or 0)
            prefix = f"{method}/{episode}/t{stage:02d}"
            try:
                payload = json.loads(str(row.get("selected_stage_prompt") or ""))
            except json.JSONDecodeError:
                continue
            lsm = (((payload.get("public_context") or {}).get("parameters") or {}).get("shared_lsm") or {})
            observed = int(lsm.get("previous_candidate_count") or 0)
            if observed != expected_previous_count:
                errors.append(
                    f"{prefix}: expected {expected_previous_count} candidates from own last public state, got {observed}"
                )
            if row.get("public_state_available"):
                expected_previous_count = min(len(row.get("candidate_archive") or []), 200)
        if args.expect_full:
            stages = [int(row.get("stage_index") or 0) for row in episode_rows]
            if stages != list(range(13)):
                errors.append(f"{method}/{episode}: incomplete stage sequence {stages}")

    if args.expect_full:
        for method, count in sorted(method_counts.items()):
            if count != 195:
                errors.append(f"{method}: expected 195 stage rows, got {count}")

    payload = {
        "status": "passed" if not errors else "failed",
        "protocol": "common_public_lsm_dialogue_and_own_population_v1",
        "expected_prompt_version": args.expected_prompt_version,
        "sources": sources,
        "stage_rows": len(rows),
        "method_counts": dict(sorted(method_counts.items())),
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


def find_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(find_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(find_keys(item))
    return keys


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
