#!/usr/bin/env python3
"""Audit the mixed external-baseline trajectories used by the NLDO heatmap.

The frozen source contains 127 source-aligned initial/early rows and 135
continuation rows generated with a common own-history wrapper. The latter
receive only that method's dialogue, accepted output, and candidate population;
no external method receives LiveOpt state.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_METHODS = {
    "react_tools",
    "optimus",
    "orlm",
    "optimai_2025",
    "or_llm_agent_2025",
}
SOURCE_ALIGNED_PROMPT_VERSION = "nldo_dynamic_public_code_solver_v1"
OWN_HISTORY_PROMPT_VERSION = "nldo_dynamic_common_public_lsm_v6"
EXPECTED_PROMPT_VERSIONS = {
    SOURCE_ALIGNED_PROMPT_VERSION,
    OWN_HISTORY_PROMPT_VERSION,
}
FORBIDDEN_PROMPT_KEYS = {
    "hidden_delta",
    "hidden_evaluation",
    "hidden_initial_state",
    "hidden_reference",
    "reference_archive",
    "reference_solution",
    "reference_trajectory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-rows", type=Path, action="append", required=True)
    parser.add_argument(
        "--summary-companion-stage-rows",
        type=Path,
        action="append",
        default=[],
        help="Additional heatmap source named after the audited external rows.",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def prompt_parameters(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(str(row.get("selected_stage_prompt") or ""))
    return ((payload.get("public_context") or {}).get("parameters") or {})


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in args.stage_rows:
        if not path.exists():
            errors.append(f"missing stage-row source: {path}")
            continue
        rows.extend(load_jsonl(path))

    seen: set[tuple[str, str, int]] = set()
    method_rows: Counter[str] = Counter()
    method_update_rows: Counter[str] = Counter()
    method_episodes: dict[str, set[str]] = defaultdict(set)
    method_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    nonempty_previous: Counter[str] = Counter()
    empty_or_absent_previous: Counter[str] = Counter()
    rows_with_candidate_history = 0
    rows_with_shared_lsm = 0
    repair_rows = 0
    hidden_feedback_repair_rows = 0
    reference_feedback_repair_rows = 0
    prompt_versions: Counter[str] = Counter()
    own_history_rows = 0
    own_history_with_previous_output = 0
    own_history_with_candidate_population = 0
    source_aligned_with_previous_output = 0

    for row in rows:
        method = str(row.get("method") or "")
        episode = str(row.get("episode_id") or "")
        stage = int(row.get("stage_index") or 0)
        prefix = f"{method}/{episode}/t{stage:02d}"
        key = (method, episode, stage)
        if key in seen:
            errors.append(f"duplicate merged cell: {prefix}")
        seen.add(key)
        method_rows[method] += 1
        if stage > 0:
            method_update_rows[method] += 1
        method_episodes[method].add(episode)
        method_statuses[method][str(row.get("status") or "")] += 1

        prompt_version = str(row.get("prompt_version") or "")
        prompt_versions[prompt_version] += 1
        if prompt_version not in EXPECTED_PROMPT_VERSIONS:
            errors.append(f"{prefix}: unexpected prompt version {prompt_version!r}")
        try:
            params = prompt_parameters(row)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"{prefix}: selected prompt is not valid JSON: {exc}")
            continue
        history = list(params.get("public_update_history") or [])
        if len(history) != stage:
            errors.append(f"{prefix}: expected {stage} cumulative public updates, got {len(history)}")
        if not isinstance(params.get("tables"), dict) or not params.get("tables"):
            errors.append(f"{prefix}: current materialized public tables are missing")
        forbidden = sorted(find_keys(params) & FORBIDDEN_PROMPT_KEYS)
        if forbidden:
            errors.append(f"{prefix}: hidden/reference keys exposed in prompt: {forbidden}")

        previous = params.get("previous_accepted_public_output")
        if previous:
            nonempty_previous[method] += 1
        else:
            empty_or_absent_previous[method] += 1
        shared_lsm = params.get("shared_lsm")
        if shared_lsm:
            rows_with_shared_lsm += 1
        candidate_population = params.get("previous_candidate_population") or (
            shared_lsm.get("previous_candidate_population")
            if isinstance(shared_lsm, dict)
            else None
        )
        if candidate_population:
            rows_with_candidate_history += 1
        if prompt_version == OWN_HISTORY_PROMPT_VERSION:
            own_history_rows += 1
            own_history_with_previous_output += int(bool(previous))
            own_history_with_candidate_population += int(bool(candidate_population))
            if not isinstance(shared_lsm, dict):
                errors.append(f"{prefix}: own-history continuation lacks the public state wrapper")
            else:
                if shared_lsm.get("interface") != "common_public_lsm_dialogue_and_own_population_v1":
                    errors.append(f"{prefix}: unexpected own-history interface")
                if shared_lsm.get("liveopt_population_available") is not False:
                    errors.append(f"{prefix}: LiveOpt population must be unavailable")
                if (
                    shared_lsm.get("population_origin")
                    != "this method's own most recent publicly executable stage"
                ):
                    errors.append(f"{prefix}: candidate population has an unexpected origin")
            if not previous or not candidate_population:
                errors.append(f"{prefix}: own-history continuation is missing accepted or candidate state")
        elif prompt_version == SOURCE_ALIGNED_PROMPT_VERSION:
            source_aligned_with_previous_output += int(bool(previous))
            if shared_lsm or candidate_population:
                errors.append(f"{prefix}: source-aligned row unexpectedly contains candidate state")

        repair = row.get("code_output_repair") or {}
        if repair:
            repair_rows += 1
            if int(repair.get("max_attempts") or 0) != 3:
                errors.append(f"{prefix}: recorded repair budget is not three")
            if repair.get("hidden_feedback_used"):
                hidden_feedback_repair_rows += 1
            if repair.get("reference_feedback_used"):
                reference_feedback_repair_rows += 1
        elif str(row.get("status") or "") in {"compile_failed", "runtime_failed", "schema_failed", "failed"}:
            errors.append(f"{prefix}: technical failure lacks the equal-repair record")

    observed_methods = set(method_rows)
    if observed_methods != EXPECTED_METHODS:
        errors.append(
            f"expected methods {sorted(EXPECTED_METHODS)}, got {sorted(observed_methods)}"
        )
    for method in sorted(EXPECTED_METHODS):
        episodes = method_episodes.get(method, set())
        if len(episodes) != 15:
            errors.append(f"{method}: expected 15 episodes, got {len(episodes)}")
        t0_count = sum(1 for candidate in seen if candidate[0] == method and candidate[2] == 0)
        if t0_count != 15:
            errors.append(f"{method}: expected 15 initial rows, got {t0_count}")

    if prompt_versions.get(SOURCE_ALIGNED_PROMPT_VERSION, 0) != 127:
        errors.append("expected 127 source-aligned rows")
    if prompt_versions.get(OWN_HISTORY_PROMPT_VERSION, 0) != 135:
        errors.append("expected 135 own-history continuation rows")
    if source_aligned_with_previous_output != 16:
        errors.append("expected 16 source-aligned repair rows with a preceding accepted output")
    if (
        own_history_rows != 135
        or own_history_with_previous_output != 135
        or own_history_with_candidate_population != 135
    ):
        errors.append("own-history continuation coverage is incomplete")
    if hidden_feedback_repair_rows or reference_feedback_repair_rows:
        errors.append("hidden/reference feedback was used during public code repair")

    summary_check: dict[str, Any] | None = None
    if args.summary_json:
        if not args.summary_json.exists():
            errors.append(f"missing heatmap summary: {args.summary_json}")
        else:
            summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
            expected_sources = [
                *(str(path.resolve()) for path in args.stage_rows),
                *(str(path.resolve()) for path in args.summary_companion_stage_rows),
            ]
            actual_sources = list((summary.get("sources") or {}).get("public_stage_rows") or [])
            actual_sources_resolved = [str(Path(path).resolve()) for path in actual_sources]
            summary_check = {
                "path": str(args.summary_json),
                "expected_sources": expected_sources,
                "actual_sources": actual_sources,
                "matches": actual_sources_resolved == expected_sources,
            }
            if actual_sources_resolved != expected_sources:
                errors.append("heatmap summary does not name the audited frozen sources in order")
            for display, method in {
                "ReAct": "react_tools",
                "OptiMUS": "optimus",
                "ORLM": "orlm",
                "OptimAI": "optimai_2025",
                "OR-Agent": "or_llm_agent_2025",
            }.items():
                observed = ((summary.get("methods") or {}).get(display) or {}).get("observed_public_rows")
                if observed != method_rows[method]:
                    errors.append(
                        f"{display}: heatmap records {observed} stage rows, "
                        f"audit found {method_rows[method]}"
                    )

    payload = {
        "schema_version": "nldo_frozen_external_baseline_protocol_audit_v2",
        "status": "passed" if not errors else "failed",
        "classification": "source_aligned_then_own_history_continuation",
        "not_a_uniform_common_lsm_control": True,
        "prompt_versions": dict(sorted(prompt_versions.items())),
        "sources": [str(path) for path in args.stage_rows],
        "summary_companion_sources": [
            str(path) for path in args.summary_companion_stage_rows
        ],
        "stage_rows": len(rows),
        "method_rows": dict(sorted(method_rows.items())),
        "method_update_rows": dict(sorted(method_update_rows.items())),
        "method_episode_counts": {
            method: len(episodes) for method, episodes in sorted(method_episodes.items())
        },
        "method_statuses": {
            method: dict(sorted(statuses.items())) for method, statuses in sorted(method_statuses.items())
        },
        "public_state_contract": {
            "all_rows_include_current_materialized_tables": True,
            "all_rows_include_cumulative_public_updates": True,
            "nonempty_previous_output_rows": dict(sorted(nonempty_previous.items())),
            "empty_or_absent_previous_output_rows": dict(sorted(empty_or_absent_previous.items())),
            "shared_lsm_rows": rows_with_shared_lsm,
            "candidate_history_rows": rows_with_candidate_history,
            "liveopt_private_state_rows": 0,
            "source_aligned_previous_output_rows": source_aligned_with_previous_output,
            "own_history_continuation_rows": own_history_rows,
            "own_history_previous_output_rows": own_history_with_previous_output,
            "own_history_candidate_population_rows": own_history_with_candidate_population,
        },
        "repair_contract": {
            "rows_with_explicit_equal_repair_record": repair_rows,
            "max_attempts": 3,
            "hidden_feedback_rows": hidden_feedback_repair_rows,
            "reference_feedback_rows": reference_feedback_repair_rows,
        },
        "summary_check": summary_check,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
