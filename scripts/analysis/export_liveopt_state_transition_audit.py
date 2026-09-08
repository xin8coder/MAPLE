#!/usr/bin/env python3
"""Audit LiveOpt's saved state transitions without making provider calls.

The audit deliberately separates two questions:

1. Did the LPD routing declaration agree with the action that was committed
   (data patch, setup-slot edit, fitness-slot edit)?
2. On benchmark-authored referential updates, did the committed public patch
   ground the intended entity, field, and value?

The first is a trace-consistency diagnostic rather than an independently
annotated localization-accuracy test.  The second uses evaluation-only
benchmark annotations after the run and never exposes them to the controller.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import apply_public_context_patch
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import public_context_with_loaded_tables


BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
EXPECTED_BENCHMARK_SHA256 = "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303"

SCALAR_SOURCE = ROOT / "outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713"
ROSTER_SOURCE = ROOT / (
    "outputs/liveopt_ablation_llmbackbone_sequential_p007_p009_"
    "200x200x10_trace_t11pop_20260714"
)
DM_SOURCE = ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_allpop_20260714"
DM_RETRY_SOURCE = ROOT / "outputs/liveopt_matched_full_retry_p013_p015_200x200x10_allpop_20260714"

METRIC_SOURCES = (
    SCALAR_SOURCE / "reference_metrics/reference_stage_metrics.csv",
    ROSTER_SOURCE / "reference_metrics/reference_stage_metrics.csv",
    ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
)

PROTOCOL_SOURCES = (
    ROOT / "release_artifacts/anonymous_frozen_results/evidence/baseline_protocol/report.json",
    ROOT / "release_artifacts/anonymous_frozen_results/evidence/controller_repeat/evidence.json",
    ROOT / "release_artifacts/anonymous_frozen_results/evidence/selector/summary.json",
)

DEFAULT_OUT = ROOT / "logs/analysis/liveopt_state_transition_audit_20260717"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_state_transition_audit_rows.tex"

PROFILE_NAMES = {
    "cobench": "Selection/allocation",
    "fjsp": "Precedence scheduling",
    "inrc2": "Coverage rostering",
    "green_vrp_multiobjective": "Ordered service routing",
    "cloud_scheduling_multiobjective": "Cloud placement",
}

DIFFICULTY_NAMES = {
    "maintenance": "Maint.",
    "simple": "Local",
    "memory": "Ref.",
    "objective": "Objective",
    "large": "Large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_dir(problem_number: int) -> Path:
    if problem_number <= 6:
        return SCALAR_SOURCE
    if problem_number <= 9:
        return ROSTER_SOURCE
    if problem_number in {13, 15}:
        return DM_RETRY_SOURCE
    return DM_SOURCE


def replay_path(episode_id: str) -> Path:
    problem_number = int(episode_id.rsplit("P", 1)[-1])
    directory = source_dir(problem_number) / "replay_jobs" / episode_id / "NLDO"
    candidates = (directory / "evo2_limit0.jsonl", directory / "evo2_limit0.jsonl.gz")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"no non-empty formal replay for {episode_id}: {candidates}")


def read_seed_zero(path: Path, episode_id: str) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_id") == episode_id and int(row.get("run_seed", -1)) == 0:
                return row
    raise ValueError(f"seed-0 replay missing for {episode_id}: {path}")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_hidden_passes() -> dict[tuple[str, int], bool]:
    merged: dict[tuple[str, int, int], bool] = {}
    for path in METRIC_SOURCES:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                episode_id = str(row.get("episode_id") or "")
                if not episode_id.startswith("NLDO-P"):
                    continue
                stage = int(float(row.get("stage_index") or 0))
                seed = int(float(row.get("run_seed") or 0))
                if not 1 <= stage <= 12:
                    continue
                merged[(episode_id, stage, seed)] = truthy(
                    row.get("true_pass") if row.get("true_pass") not in (None, "") else row.get("feasible")
                )
    return {
        (episode_id, stage): passed
        for (episode_id, stage, seed), passed in merged.items()
        if seed == 0
    }


def as_operations(patch: dict[str, Any]) -> list[dict[str, Any]]:
    return [operation for operation in (patch.get("operations") or []) if isinstance(operation, dict)]


def patch_has_action(patch: dict[str, Any]) -> bool:
    return bool(as_operations(patch))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return left == right
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        return left_number == right_number
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def expected_memory_specs(delta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the public-state transitions implied by a resolved memory delta."""

    kind = str(delta.get("type") or "")
    if kind == "compound":
        specs: list[dict[str, Any]] = []
        for change in delta.get("changes") or []:
            specs.extend(expected_memory_specs(change))
        return specs
    if kind == "value_delta":
        return [delta_spec("items", {"id": delta["item"]}, "value", delta["delta"])]
    if kind == "open_cost_delta":
        return [delta_spec("facilities", {"id": delta["facility"]}, "open_cost", delta["delta"])]
    if kind == "serve_cost_delta":
        field = f"serve_cost_{delta['facility']}"
        return [delta_spec("customers", {"id": delta["customer"]}, field, delta["delta"])]
    if kind == "set_cost_delta":
        return [delta_spec("sets", {"id": delta["set"]}, "cost", delta["delta"])]
    if kind == "machine_cost_rate_delta":
        return [delta_spec("machines", {"id": delta["machine"]}, "cost_rate", delta["delta"])]
    if kind == "priority_delta" and delta.get("job") is not None:
        return [delta_spec("operations", {"job_id": delta["job"]}, "priority", delta["delta"])]
    if kind == "due_delta" and delta.get("job") is not None:
        return [delta_spec("operations", {"job_id": delta["job"]}, "due_time", delta["delta"])]
    if kind == "preference_off":
        return [
            {
                "mode": "append",
                "table": "preferences",
                "row": {"day": delta["day"], "nurse": delta["nurse"], "preference": "off"},
            }
        ]
    if kind == "tighten_due":
        return [delta_spec("orders", {"id": delta["order"]}, "due", delta["delta"])]
    if kind == "vehicle_emission_delta_for_vehicle":
        return [delta_spec("vehicles", {"id": delta["vehicle"]}, "emission_rate", delta["delta"])]
    if kind == "priority_set_jobs":
        return [set_spec("jobs", {"id": job}, "priority", delta["priority"]) for job in delta["jobs"]]
    raise ValueError(f"unsupported resolved memory delta: {kind}: {delta}")


def delta_spec(table: str, match: dict[str, Any], field: str, amount: Any) -> dict[str, Any]:
    return {"mode": "delta", "table": table, "match": match, "field": field, "amount": amount}


def set_spec(table: str, match: dict[str, Any], field: str, value: Any) -> dict[str, Any]:
    return {"mode": "set", "table": table, "match": match, "field": field, "value": value}


def rows_matching(context: dict[str, Any], table: str, match: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ((context.get("tables") or {}).get(table) or [])
    return [row for row in rows if all(row.get(key) == value for key, value in match.items())]


def actual_patch_keys(patch: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for operation in as_operations(patch):
        path = list(operation.get("path") or [])
        table = str(path[1]) if len(path) >= 2 and path[0] == "tables" else stable_json(path)
        kind = str(operation.get("op") or "")
        if kind == "update_row":
            match = operation.get("match") or {}
            for field in sorted((operation.get("values") or {}).keys()):
                keys.add(stable_json({"mode": "field", "table": table, "match": match, "field": field}))
        elif kind == "append_row":
            keys.add(stable_json({"mode": "append", "table": table, "row": operation.get("row") or {}}))
        else:
            keys.add(stable_json({"mode": kind, "table": table, "operation": operation}))
    return keys


def expected_patch_keys(specs: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for spec in specs:
        if spec["mode"] == "append":
            keys.add(stable_json({"mode": "append", "table": spec["table"], "row": spec["row"]}))
        else:
            keys.add(
                stable_json(
                    {
                        "mode": "field",
                        "table": spec["table"],
                        "match": spec["match"],
                        "field": spec["field"],
                    }
                )
            )
    return keys


def transition_values_exact(
    before: dict[str, Any], after: dict[str, Any], specs: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for spec in specs:
        mode = spec["mode"]
        table = spec["table"]
        if mode == "append":
            row = spec["row"]
            before_count = sum(all(item.get(key) == value for key, value in row.items()) for item in ((before.get("tables") or {}).get(table) or []))
            after_count = sum(all(item.get(key) == value for key, value in row.items()) for item in ((after.get("tables") or {}).get(table) or []))
            if after_count != before_count + 1:
                errors.append(f"{table}: expected one appended row {row}, counts {before_count}->{after_count}")
            continue

        before_rows = rows_matching(before, table, spec["match"])
        after_rows = rows_matching(after, table, spec["match"])
        if not before_rows or len(before_rows) != len(after_rows):
            errors.append(
                f"{table}/{spec['match']}: row count mismatch {len(before_rows)}->{len(after_rows)}"
            )
            continue
        for old_row, new_row in zip(before_rows, after_rows):
            field = spec["field"]
            if mode == "delta":
                try:
                    expected = float(old_row[field]) + float(spec["amount"])
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{table}/{spec['match']}/{field}: cannot apply delta: {exc}")
                    continue
            else:
                expected = spec["value"]
            if not close(new_row.get(field), expected):
                errors.append(
                    f"{table}/{spec['match']}/{field}: expected {expected!r}, observed {new_row.get(field)!r}"
                )
    return not errors, errors


def antecedent_entities(delta: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    for key in ("item", "facility", "customer", "set", "job", "machine", "nurse", "order", "vehicle"):
        value = delta.get(key)
        if value not in (None, ""):
            entities.append(str(value))
    entities.extend(str(value) for value in (delta.get("jobs") or []))
    for change in delta.get("changes") or []:
        entities.extend(antecedent_entities(change))
    return sorted(set(entities))


def mentions_entity(text: str, entity: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(entity)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def check_schema_patch(
    patch: dict[str, Any], schema: dict[str, list[str]], initial_ids: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    schema_errors: list[str] = []
    new_ids: list[str] = []
    for operation in as_operations(patch):
        path = list(operation.get("path") or [])
        if len(path) < 2 or path[0] != "tables":
            continue
        table = str(path[1])
        if table not in schema:
            schema_errors.append(f"unknown table {table}")
            continue
        fields: set[str] = set()
        if operation.get("op") == "update_row":
            fields.update((operation.get("match") or {}).keys())
            fields.update((operation.get("values") or {}).keys())
            identifier = (operation.get("match") or {}).get("id")
            if identifier is not None and table in initial_ids and str(identifier) not in initial_ids[table]:
                new_ids.append(f"{table}/{identifier}")
        elif operation.get("op") == "append_row":
            fields.update((operation.get("row") or {}).keys())
        elif operation.get("op") == "replace_table":
            for row in operation.get("value") or []:
                fields.update(row.keys())
                identifier = row.get("id")
                if identifier is not None and table in initial_ids and str(identifier) not in initial_ids[table]:
                    new_ids.append(f"{table}/{identifier}")
        elif operation.get("op") == "set_value" and len(path) >= 3:
            fields.add(str(path[2]))
        unknown = fields - set(schema[table])
        if unknown:
            schema_errors.append(f"{table}: fields outside initial schema {sorted(unknown)}")
    return schema_errors, new_ids


def metric_counts(pairs: Iterable[tuple[bool, bool]]) -> dict[str, int | float]:
    pairs = list(pairs)
    tp = sum(predicted and actual for predicted, actual in pairs)
    fp = sum(predicted and not actual for predicted, actual in pairs)
    fn = sum(not predicted and actual for predicted, actual in pairs)
    tn = sum(not predicted and not actual for predicted, actual in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "agreement": (tp + tn) / len(pairs) if pairs else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}\\%"


def render_tex(summary: dict[str, Any], profiles: list[dict[str, Any]]) -> str:
    transition = summary["transition_audit"]
    memory = summary["referential_audit"]
    surface = transition["surface_metrics"]
    lines = [
        "% Generated by scripts/analysis/export_liveopt_state_transition_audit.py.",
        r"\newcommand{\LiveOptTransitionAuditRows}{%",
        (
            f"LPD prediction matches accepted change & {transition['updates']} & "
            f"{transition['exact_mask_count']}/{transition['updates']} "
            f"({pct(transition['exact_mask_count'], transition['updates'])}) \\\\"
        ),
        (
            f"Data/decision/evaluation agreement & {transition['updates']} & "
            f"{pct(surface['data']['tp'] + surface['data']['tn'], transition['updates'])} / "
            f"{pct(surface['setup']['tp'] + surface['setup']['tn'], transition['updates'])} / "
            f"{pct(surface['fitness']['tp'] + surface['fitness']['tn'], transition['updates'])} \\\\"
        ),
        (
            f"Both editable functions unchanged & {transition['updates']} & "
            f"{transition['slot_locality']['none']}/{transition['updates']} "
            f"({pct(transition['slot_locality']['none'], transition['updates'])}) \\\\"
        ),
        (
            f"Reference entity/field/value & {memory['updates']} & "
            f"{memory['value_exact_count']}/{memory['updates']} "
            f"({pct(memory['value_exact_count'], memory['updates'])}) \\\\"
        ),
        (
            f"Valid updated state & {transition['updates']} & "
            f"{transition['hidden_pass_count']}/{transition['updates']} "
            f"({pct(transition['hidden_pass_count'], transition['updates'])}) \\\\"
        ),
        "}",
        r"\newcommand{\LiveOptMemoryChainRows}{%",
    ]
    for chain in sorted(memory["by_chain_length"], key=int):
        row = memory["by_chain_length"][chain]
        lines.append(
            f"{chain} & {row['updates']} & {row['target_field_exact']} & {row['value_exact']} & "
            f"{row['hidden_pass']} \\\\"
        )
    lines.extend(["}", r"\newcommand{\LiveOptBenchmarkCharacterizationRows}{%"])
    for row in profiles:
        lines.append(
            f"{row['display']} & {row['episodes']} & {row['updates']} & {row['memory_updates']} & "
            f"{row['large_updates']} & {row['unique_wordings']} \\\\"
        )
    lines.extend(["}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    if sha256_file(BENCHMARK) != EXPECTED_BENCHMARK_SHA256:
        errors.append("benchmark SHA256 does not match the frozen current benchmark")

    episodes = {str(row["episode_id"]): row for row in load_jsonl(BENCHMARK)}
    hidden_passes = load_hidden_passes()
    expected_hidden_keys = {(episode_id, stage) for episode_id in episodes for stage in range(1, 13)}
    if set(hidden_passes) != expected_hidden_keys:
        errors.append(
            f"seed-0 hidden metric coverage mismatch: observed={len(hidden_passes)}, "
            f"expected={len(expected_hidden_keys)}"
        )

    stage_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    replay_sources: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    new_entity_ids: list[str] = []

    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        path = replay_path(episode_id)
        replay_sources.append({"path": repo_path(path), "sha256": sha256_file(path)})
        run = read_seed_zero(path, episode_id)
        updates = list(run.get("update_results") or [])
        if len(updates) != 12:
            errors.append(f"{episode_id}: expected 12 saved updates, got {len(updates)}")
            continue
        if str(run.get("agent_mode") or "").split("__", 1)[0] != "liveopt_workbench_v1":
            errors.append(f"{episode_id}: unexpected agent mode {run.get('agent_mode')!r}")

        context = public_context_with_loaded_tables(episode)
        schema = {key: list(value) for key, value in ((context.get("csv_schema") or {}).items())}
        initial_ids = {
            table: {str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id") is not None}
            for table, rows in ((context.get("tables") or {}).items())
        }
        previous_setup = str(run.get("setup_code") or "")
        previous_fitness = str(run.get("fitness_code") or "")

        for stage, (public_update, oracle, result) in enumerate(
            zip(episode.get("update_stream") or [], episode.get("hidden_update_oracle") or [], updates),
            start=1,
        ):
            impact = result.get("impact") or {}
            patch = (result.get("restart") or {}).get("data_patch") or {}
            current_setup = str(result.get("setup_code") or "")
            current_fitness = str(result.get("fitness_code") or "")
            declared = {
                "data": bool(impact.get("data_update")),
                "setup": bool(impact.get("patch_setup")),
                "fitness": bool(impact.get("patch_fitness")),
            }
            realized = {
                "data": patch_has_action(patch),
                "setup": current_setup != previous_setup,
                "fitness": current_fitness != previous_fitness,
            }
            exact_mask = declared == realized
            hidden_pass = bool(hidden_passes.get((episode_id, stage), False))

            before_context = context
            try:
                after_context = apply_public_context_patch(before_context, patch)
            except Exception as exc:  # pragma: no cover - strict audit path
                errors.append(f"{episode_id}/t{stage:02d}: public patch application failed: {exc}")
                after_context = before_context

            patch_schema_errors, patch_new_ids = check_schema_patch(patch, schema, initial_ids)
            schema_errors.extend(f"{episode_id}/t{stage:02d}: {item}" for item in patch_schema_errors)
            new_entity_ids.extend(f"{episode_id}/t{stage:02d}: {item}" for item in patch_new_ids)

            slot_key = (
                "both"
                if realized["setup"] and realized["fitness"]
                else "setup"
                if realized["setup"]
                else "fitness"
                if realized["fitness"]
                else "none"
            )
            stage_row = {
                "episode_id": episode_id,
                "profile": str(episode.get("domain") or episode.get("family") or "unknown"),
                "stage_index": stage,
                "update_id": str(public_update.get("update_id") or result.get("update_id") or ""),
                "difficulty": str(public_update.get("difficulty") or "unknown"),
                "requires_memory": bool(oracle.get("requires_memory")),
                "memory_chain_length": len(oracle.get("memory_reference_chain") or []),
                "delta_type": str((oracle.get("hidden_delta") or {}).get("type") or ""),
                "declared_data": declared["data"],
                "declared_setup": declared["setup"],
                "declared_fitness": declared["fitness"],
                "realized_data": realized["data"],
                "realized_setup": realized["setup"],
                "realized_fitness": realized["fitness"],
                "exact_mask": exact_mask,
                "slot_locality": slot_key,
                "hidden_pass": hidden_pass,
                "schema_preserved": not patch_schema_errors,
                "prelisted_entity_ids": not patch_new_ids,
            }
            stage_rows.append(stage_row)

            if oracle.get("requires_memory"):
                delta = oracle.get("hidden_delta") or {}
                specs = expected_memory_specs(delta)
                expected_keys = expected_patch_keys(specs)
                observed_keys = actual_patch_keys(patch)
                target_field_exact = expected_keys == observed_keys
                value_exact, value_errors = transition_values_exact(before_context, after_context, specs)
                entities = antecedent_entities(delta)
                first_paragraph = str(public_update.get("natural_language_update") or "").split("\n\n", 1)[0]
                named_entities = [entity for entity in entities if mentions_entity(first_paragraph, entity)]
                memory_rows.append(
                    {
                        "episode_id": episode_id,
                        "stage_index": stage,
                        "update_id": stage_row["update_id"],
                        "profile": stage_row["profile"],
                        "chain_length": stage_row["memory_chain_length"],
                        "entities": "|".join(entities),
                        "entities_named_in_current_update": "|".join(named_entities),
                        "target_field_exact": target_field_exact,
                        "value_exact": value_exact,
                        "hidden_pass": hidden_pass,
                        "expected_change_count": len(expected_keys),
                        "observed_change_count": len(observed_keys),
                        "value_errors": " | ".join(value_errors),
                    }
                )

            context = after_context
            previous_setup = current_setup
            previous_fitness = current_fitness

    if len(stage_rows) != 180:
        errors.append(f"expected 180 transition rows, got {len(stage_rows)}")
    if len(memory_rows) != 30:
        errors.append(f"expected 30 referential rows, got {len(memory_rows)}")
    if schema_errors:
        errors.extend(schema_errors)
    if new_entity_ids:
        errors.extend(f"new decision entity outside initial tables: {item}" for item in new_entity_ids)

    surface_metrics = {
        surface: metric_counts((bool(row[f"declared_{surface}"]), bool(row[f"realized_{surface}"])) for row in stage_rows)
        for surface in ("data", "setup", "fitness")
    }
    slot_locality = Counter(str(row["slot_locality"]) for row in stage_rows)
    exact_mask_count = sum(bool(row["exact_mask"]) for row in stage_rows)
    hidden_pass_count = sum(bool(row["hidden_pass"]) for row in stage_rows)

    by_chain: dict[str, dict[str, int]] = {}
    for chain_length in sorted({int(row["chain_length"]) for row in memory_rows}):
        selected = [row for row in memory_rows if int(row["chain_length"]) == chain_length]
        by_chain[str(chain_length)] = {
            "updates": len(selected),
            "target_field_exact": sum(bool(row["target_field_exact"]) for row in selected),
            "value_exact": sum(bool(row["value_exact"]) for row in selected),
            "hidden_pass": sum(bool(row["hidden_pass"]) for row in selected),
        }

    profile_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        profile_groups[str(row["profile"])].append(row)
    profile_rows: list[dict[str, Any]] = []
    for profile, rows in profile_groups.items():
        episode_ids = sorted({str(row["episode_id"]) for row in rows})
        wordings = {
            str(update.get("natural_language_update") or "").split("\n\n", 1)[0]
            for episode_id in episode_ids
            for update in episodes[episode_id].get("update_stream") or []
        }
        difficulty_counts = Counter(str(row["difficulty"]) for row in rows)
        profile_rows.append(
            {
                "profile": profile,
                "display": PROFILE_NAMES.get(profile, profile),
                "episodes": len(episode_ids),
                "updates": len(rows),
                "memory_updates": sum(bool(row["requires_memory"]) for row in rows),
                "large_updates": difficulty_counts.get("large", 0),
                "unique_wordings": len(wordings),
                "difficulty_counts": dict(sorted(difficulty_counts.items())),
            }
        )
    profile_order = list(PROFILE_NAMES)
    profile_rows.sort(key=lambda row: profile_order.index(row["profile"]))

    protocol_manifest: list[dict[str, Any]] = []
    for path in (*METRIC_SOURCES, *PROTOCOL_SOURCES):
        if not path.exists():
            errors.append(f"protocol input missing: {repo_path(path)}")
            continue
        protocol_manifest.append({"path": repo_path(path), "sha256": sha256_file(path)})

    baseline_protocol: dict[str, Any] = {}
    baseline_path = PROTOCOL_SOURCES[0]
    if baseline_path.exists():
        baseline_protocol = json.loads(baseline_path.read_text(encoding="utf-8"))

    all_first_paragraphs = {
        str(update.get("natural_language_update") or "").split("\n\n", 1)[0]
        for episode in episodes.values()
        for update in episode.get("update_stream") or []
    }
    public_state_contract = baseline_protocol.get("public_state_contract") or {}
    nonempty_previous = public_state_contract.get("nonempty_previous_output_rows") or {}

    summary = {
        "status": "passed" if not errors else "failed",
        "protocol": "liveopt_state_transition_trace_audit_v1",
        "provider_calls": 0,
        "benchmark": {"path": repo_path(BENCHMARK), "sha256": sha256_file(BENCHMARK)},
        "transition_audit": {
            "updates": len(stage_rows),
            "exact_mask_count": exact_mask_count,
            "exact_mask_rate": exact_mask_count / len(stage_rows) if stage_rows else 0.0,
            "surface_metrics": surface_metrics,
            "slot_locality": dict(sorted(slot_locality.items())),
            "hidden_pass_count": hidden_pass_count,
            "hidden_pass_rate": hidden_pass_count / len(stage_rows) if stage_rows else 0.0,
            "schema_extension_count": len(schema_errors),
            "new_decision_entity_count": len(new_entity_ids),
            "interpretation": (
                "Proposal-to-commit agreement is a trace diagnostic, not an independently annotated "
                "LPD localization-accuracy estimate."
            ),
        },
        "referential_audit": {
            "updates": len(memory_rows),
            "target_field_exact_count": sum(bool(row["target_field_exact"]) for row in memory_rows),
            "value_exact_count": sum(bool(row["value_exact"]) for row in memory_rows),
            "hidden_pass_count": sum(bool(row["hidden_pass"]) for row in memory_rows),
            "current_update_names_target_count": sum(
                bool(row["entities_named_in_current_update"]) for row in memory_rows
            ),
            "by_chain_length": by_chain,
            "interpretation": (
                "Grounded entities and values come from evaluation-only benchmark annotations and are "
                "joined after the saved run. This validates reference grounding with stored events, not a "
                "factorial LSM effect."
            ),
        },
        "benchmark_characterization": {
            "profiles": profile_rows,
            "episodes": len(episodes),
            "updates": len(stage_rows),
            "memory_updates": sum(bool(row["requires_memory"]) for row in stage_rows),
            "unique_first_paragraphs": len(all_first_paragraphs),
            "schema_extensions_in_formal_patches": len(schema_errors),
            "new_decision_entities_in_formal_patches": len(new_entity_ids),
        },
        "table2_protocol_snapshot": {
            "source": repo_path(baseline_path),
            "status": baseline_protocol.get("status"),
            "stage_rows": baseline_protocol.get("stage_rows"),
            "candidate_history_rows": public_state_contract.get("candidate_history_rows"),
            "preceding_accepted_output_rows": sum(int(value) for value in nonempty_previous.values()),
            "role": "source-aligned repeated-solving anchors; not a matched LSM control",
        },
        "replay_sources": replay_sources,
        "protocol_sources": protocol_manifest,
        "errors": errors,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_tex.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "transition_rows.csv", stage_rows)
    write_csv(args.out_dir / "referential_rows.csv", memory_rows)
    write_csv(args.out_dir / "benchmark_profiles.csv", profile_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": summary["benchmark"],
                "replay_sources": replay_sources,
                "protocol_sources": protocol_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args.paper_tex.write_text(render_tex(summary, profile_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
