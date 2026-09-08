#!/usr/bin/env python3
"""Standalone verifier for the anonymous LiveOpt frozen-results artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any


FORBIDDEN_PATTERNS = {
    "unix_home": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "windows_home": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+"),
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "credential": re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row.get("normalized_hv") or 0.0)
        if parse_bool(row.get("true_pass")) and parse_bool(row.get("feasible"))
        else 0.0
        for row in rows
    ]
    return {
        "cells": len(rows),
        "hidden_pass": sum(
            parse_bool(row.get("true_pass")) and parse_bool(row.get("feasible")) for row in rows
        ),
        "mean_hv": fmean(values),
        "max_hv": max(float(row.get("normalized_hv") or 0.0) for row in rows),
    }


def assert_close(actual: float, expected: float, tolerance: float, label: str, errors: list[str]) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        errors.append(f"{label}: expected {expected}, got {actual}")


def verify_checksums(root: Path, errors: list[str]) -> int:
    manifest = root / "checksums/manifest.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            errors.append(f"checksum target is missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_anonymity(root: Path, errors: list[str]) -> int:
    checked = 0
    text_suffixes = {".csv", ".json", ".jsonl", ".md", ".py", ".tex", ".txt", ".yaml", ".yml", ".sha256"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {name} leak in {relative}")
        checked += 1
    return checked


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    checksum_files = verify_checksums(root, errors)
    text_files = verify_anonymity(root, errors)

    expected = {
        "liveopt": (720, 720, 0.7789711125),
        "no_tss": (720, 580, 0.6056894430555556),
        "sequential_warm": (720, 720, 0.7689366569444445),
        "sequential_full": (720, 720, 0.7307362833333334),
        "tss_operator_control": (72, 66, 0.6177855),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, (cells, passes, mean_hv) in expected.items():
        summary = metric_summary(root / f"data/formal_metrics/{key}_dm_updates.csv")
        summaries[key] = summary
        if summary["cells"] != cells:
            errors.append(f"{key}: expected {cells} metric cells, got {summary['cells']}")
        if summary["hidden_pass"] != passes:
            errors.append(f"{key}: expected {passes} hidden passes, got {summary['hidden_pass']}")
        assert_close(summary["mean_hv"], mean_hv, 5e-12, f"{key} mean HV", errors)

    with (root / "evidence/tss/failure_cells.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    if len(failures) != 140:
        errors.append(f"expected 140 no-TSS failure cells, got {len(failures)}")
    categories: dict[str, int] = {}
    for row in failures:
        row_categories = json.loads(row.get("violation_categories") or "[]")
        for category in row_categories:
            key = f"{row['episode_id']}:{category}"
            categories[key] = categories.get(key, 0) + 1
    expected_categories = {"NLDO-P010:duplicate_vehicles": 120, "NLDO-P015:gpu_mismatch": 20}
    if categories != expected_categories:
        errors.append(f"unexpected no-TSS failure categories: {categories}")

    operator_control = json.loads(
        (root / "evidence/tss/operator_control_summary.json").read_text(encoding="utf-8")
    )
    if operator_control.get("protocol") != "tss_matched_type_agnostic_variation_v1":
        errors.append("TSS operator control protocol changed")
    operator_design = operator_control.get("design") or {}
    if operator_design.get("lineage") != "same seed-matched LiveOpt population immediately before each restart":
        errors.append("TSS operator control does not use matched LiveOpt incoming populations")
    if int(operator_design.get("population_size") or 0) != 200 or int(operator_design.get("generation_cap") or 0) != 200:
        errors.append("TSS operator control is not a 200x200 run")
    typed = operator_control.get("baseline") or {}
    agnostic = operator_control.get("type_agnostic_resampling") or {}
    assert_close(float(typed.get("solve_rate") or 0.0), 1.0, 5e-12, "typed-operator solve rate", errors)
    assert_close(float(typed.get("online_quality") or 0.0), 0.774115875, 5e-12, "typed-operator HV", errors)
    assert_close(float(agnostic.get("solve_rate") or 0.0), 11.0 / 12.0, 5e-12, "type-agnostic solve rate", errors)
    assert_close(float(agnostic.get("online_quality") or 0.0), 0.6177855, 5e-12, "type-agnostic HV", errors)

    sequential = json.loads((root / "evidence/sequential/summary.json").read_text(encoding="utf-8"))
    if sequential.get("strict_status") != "passed":
        errors.append("sequential replay audit did not pass")
    identity = sequential.get("identity_audit") or {}
    for field in ("initial_mismatches", "typed_artifact_mismatches", "reference_mapping_mismatches"):
        if int(identity.get(field) or 0) != 0:
            errors.append(f"sequential identity check failed: {field}={identity.get(field)}")

    selector = json.loads((root / "evidence/selector/summary.json").read_text(encoding="utf-8"))
    dm = (selector.get("views") or selector).get("DM") or {}
    if int(dm.get("liveopt_full_actions") or 0) != 12:
        errors.append("selector evidence does not contain 12/72 DM Full decisions")
    assert_close(
        float(dm.get("oracle_gain_recovered_over_always_warm") or 0.0),
        0.743674854825467,
        5e-12,
        "selector oracle gain recovered",
        errors,
    )
    assert_close(
        float(dm.get("episode_exact_sign_test_liveopt_minus_warm_two_sided_p") or 0.0),
        0.03125,
        5e-12,
        "selector exact sign p",
        errors,
    )

    landscape = json.loads((root / "evidence/landscape/audit.json").read_text(encoding="utf-8"))
    if landscape.get("status") != "passed" or int(landscape.get("cells") or 0) != 720:
        errors.append("objective-landscape selector audit did not pass 720-cell coverage")
    if abs(float(landscape.get("sensor_fraction") or 0.0) - 0.10) > 1e-12:
        errors.append("objective-landscape selector does not use the frozen 10% sensor fraction")
    landscape_all = next(
        (
            row
            for row in (landscape.get("method_summary") or [])
            if row.get("scope") == "all" and row.get("method") == "Sensor landscape"
        ),
        {},
    )
    if int(landscape_all.get("full_cells") or 0) != 245:
        errors.append("objective-landscape selector does not contain 245/720 Full cells")
    assert_close(
        float(landscape_all.get("mean_hv") or 0.0),
        0.7654232013888889,
        5e-12,
        "objective-landscape selector HV",
        errors,
    )

    archive_cap = json.loads((root / "evidence/archive_cap/summary.json").read_text(encoding="utf-8"))
    if archive_cap.get("status") != "passed" or archive_cap.get("only_changed_field") != "archive_limit":
        errors.append("P014/P015 archive-cap sensitivity audit did not pass")
    cap_rows = {int(row.get("archive_cap") or 0): row for row in archive_cap.get("rows") or []}
    if set(cap_rows) != {100, 200, 500} or any(int(row.get("cells") or 0) != 240 for row in cap_rows.values()):
        errors.append("archive-cap sensitivity does not contain 240 cells for each of caps 100/200/500")
    cap_inputs = archive_cap.get("inputs") or {}
    if cap_inputs.get("benchmark_sha256") != "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303":
        errors.append("archive-cap sensitivity does not use the frozen current benchmark")
    if cap_inputs.get("reference_sha256") != "50126c25eb837bdbf2dbc2e7710226472027a1a880305d331dc006438a02c5e5":
        errors.append("archive-cap sensitivity does not use the frozen 500x500x10 reference")
    expected_cap_values = {
        100: (0.7754553416666667, 240),
        200: (0.77570565, 162),
        500: (0.7741075125, 0),
    }
    for cap, (expected_hv, expected_active) in expected_cap_values.items():
        assert_close(float(cap_rows.get(cap, {}).get("mean_hv") or 0.0), expected_hv, 5e-12, f"cap {cap} HV", errors)
        if int(cap_rows.get(cap, {}).get("cap_active_cells") or 0) != expected_active:
            errors.append(f"cap {cap} activation count does not match the frozen replay")

    controller_repeat = json.loads(
        (root / "evidence/controller_repeat/evidence.json").read_text(encoding="utf-8")
    )
    if controller_repeat.get("status") != "passed" or controller_repeat.get("errors"):
        errors.append("P010/P015 independent-controller audit did not pass")
    if controller_repeat.get("protocol") != "independent_controller_trajectory_p010_p015_v1":
        errors.append("controller-repeat protocol or problem selection changed")
    if int(controller_repeat.get("cell_count") or 0) != 432:
        errors.append("controller-repeat evidence does not cover 432 requested cells")
    if controller_repeat.get("reference_sha256") != "50126c25eb837bdbf2dbc2e7710226472027a1a880305d331dc006438a02c5e5":
        errors.append("controller-repeat evidence does not use the frozen 500x500x10 reference")
    expected_decision_hashes = {
        "NLDO-P010": "33cba37ec31f4d3436919009642eaa32aaf3f59bc1e17fd34380ae883c8c070d",
        "NLDO-P015": "cbd2428b72144e9e79b9f9267e2f64dc6d881431f9fa6829c5fe4205ceac00c9",
    }
    observed_decision_hashes = {
        episode_id: str((row or {}).get("sha256") or "")
        for episode_id, row in (controller_repeat.get("semantic_decision_sources") or {}).items()
    }
    if observed_decision_hashes != expected_decision_hashes:
        errors.append("controller-repeat semantic-decision sources changed")
    controller_rows = controller_repeat.get("controllers") or []
    controller_groups = controller_repeat.get("aggregate") or []
    if len(controller_rows) != 12 or len(controller_groups) != 4:
        errors.append("controller-repeat evidence does not contain 12 trajectories and four groups")
    expected_controller_groups = {
        ("NLDO-P010", "LiveOpt"),
        ("NLDO-P010", "LiveOpt w/o TSS"),
        ("NLDO-P015", "LiveOpt"),
        ("NLDO-P015", "LiveOpt w/o TSS"),
    }
    observed_controller_groups = {
        (str(row.get("episode_id")), str(row.get("method"))) for row in controller_groups
    }
    if observed_controller_groups != expected_controller_groups:
        errors.append(f"unexpected controller-repeat groups: {observed_controller_groups}")
    expected_controller_values = {
        ("NLDO-P010", "LiveOpt"): (1.0, 0.6694616111111111, 0.10026458537044346, 0.05695068491245474),
        ("NLDO-P010", "LiveOpt w/o TSS"): (0.6666666666666666, 0.3355036203703704, 0.29850617832917903, 0.015015368923680161),
        ("NLDO-P015", "LiveOpt"): (0.888888888888889, 0.7282787870370371, 0.05649824395082113, 0.012076636718813687),
        ("NLDO-P015", "LiveOpt w/o TSS"): (0.888888888888889, 0.5129460740740741, 0.3928873604419912, 0.010599345211980821),
    }
    controller_group_map = {
        (str(row.get("episode_id")), str(row.get("method"))): row for row in controller_groups
    }
    for key, (solve_rate, quality, controller_sd, numerical_sd) in expected_controller_values.items():
        row = controller_group_map.get(key) or {}
        if int(row.get("controller_complete") or 0) != 3 or int(row.get("controller_count") or 0) != 3:
            errors.append(f"{key}: controller-repeat coverage is not 3/3")
        assert_close(float(row.get("solve_rate_mean") or 0.0), solve_rate, 5e-12, f"{key} solve rate", errors)
        assert_close(float(row.get("quality_mean") or 0.0), quality, 5e-12, f"{key} mean HV", errors)
        assert_close(
            float(row.get("quality_controller_sd") or 0.0), controller_sd, 5e-12,
            f"{key} controller HV SD", errors,
        )
        assert_close(
            float(row.get("mean_within_controller_search_sd") or 0.0), numerical_sd, 5e-12,
            f"{key} within-controller numerical SD", errors,
        )
    new_controller_rows = [
        row for row in controller_rows if row.get("trajectory_source") == "new_provider_call"
    ]
    if len(new_controller_rows) != 8:
        errors.append(f"expected eight new controller trajectories, got {len(new_controller_rows)}")
    if any(int(row.get("local_cache_hit_count") or 0) != 0 for row in new_controller_rows):
        errors.append("an independent controller trajectory reused a local response cache")
    if any(int(row.get("api_response_count") or 0) < 1 for row in new_controller_rows):
        errors.append("an independent controller trajectory has no provider response")

    patch_diff = json.loads((root / "evidence/patch_diff/evidence.json").read_text(encoding="utf-8"))
    if patch_diff.get("status") != "passed" or patch_diff.get("errors"):
        errors.append("real data/slot patch export did not pass")
    if patch_diff.get("benchmark_sha256") != "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303":
        errors.append("real patch evidence does not use the frozen current benchmark")
    data_patch = patch_diff.get("data_patch") or {}
    if (
        data_patch.get("episode_id") != "NLDO-P015"
        or int(data_patch.get("stage") or 0) != 2
        or not data_patch.get("setup_byte_identical")
        or not data_patch.get("fitness_byte_identical")
    ):
        errors.append("the frozen data-only patch is not P015-t02 with byte-identical slots")
    assert_close(float(data_patch.get("before") or 0.0), 0.23, 1e-12, "P015 data patch before", errors)
    assert_close(float(data_patch.get("after") or 0.0), 0.285, 1e-12, "P015 data patch after", errors)
    slot_patch = patch_diff.get("slot_patch") or {}
    if (
        slot_patch.get("episode_id") != "NLDO-P010"
        or int(slot_patch.get("stage") or 0) != 2
        or not slot_patch.get("setup_byte_identical")
    ):
        errors.append("the frozen evaluator-slot patch is not P010-t02 with unchanged setup.py")
    if not any("leg_dist *= 1.25" in str(line) for line in (slot_patch.get("full_unified_diff") or [])):
        errors.append("the frozen P010 slot diff no longer contains the congestion multiplier")

    transition = json.loads((root / "evidence/transition/summary.json").read_text(encoding="utf-8"))
    if transition.get("status") != "passed" or transition.get("errors"):
        errors.append("state-transition trace audit did not pass")
    if transition.get("protocol") != "liveopt_state_transition_trace_audit_v1":
        errors.append("state-transition audit protocol changed")
    if int(transition.get("provider_calls", -1)) != 0:
        errors.append("state-transition audit is not provider-free")
    if (transition.get("benchmark") or {}).get("sha256") != "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303":
        errors.append("state-transition audit does not use the frozen current benchmark")
    transition_audit = transition.get("transition_audit") or {}
    if int(transition_audit.get("updates") or 0) != 180:
        errors.append("state-transition audit does not cover 180 update states")
    if int(transition_audit.get("exact_mask_count") or 0) != 170:
        errors.append("state-transition audit exact LPD/action count changed")
    if int(transition_audit.get("hidden_pass_count") or 0) != 180:
        errors.append("state-transition audit hidden-pass count changed")
    expected_slot_locality = {"fitness": 8, "none": 166, "setup": 6}
    if transition_audit.get("slot_locality") != expected_slot_locality:
        errors.append(f"state-transition slot locality changed: {transition_audit.get('slot_locality')}")
    expected_surface_counts = {
        "data": {"tp": 132, "fp": 0, "fn": 6, "tn": 42},
        "setup": {"tp": 6, "fp": 2, "fn": 0, "tn": 172},
        "fitness": {"tp": 8, "fp": 2, "fn": 0, "tn": 170},
    }
    for surface, expected_counts in expected_surface_counts.items():
        observed = (transition_audit.get("surface_metrics") or {}).get(surface) or {}
        for field, expected_value in expected_counts.items():
            if int(observed.get(field) or 0) != expected_value:
                errors.append(
                    f"state-transition {surface} {field}: expected {expected_value}, got {observed.get(field)}"
                )
    if int(transition_audit.get("schema_extension_count") or 0) != 0:
        errors.append("formal transition patches extend the frozen public schema")
    if int(transition_audit.get("new_decision_entity_count") or 0) != 0:
        errors.append("formal transition patches introduce a non-prelisted decision entity")

    referential = transition.get("referential_audit") or {}
    for field, expected_value in {
        "updates": 30,
        "target_field_exact_count": 30,
        "value_exact_count": 30,
        "hidden_pass_count": 30,
        "current_update_names_target_count": 0,
    }.items():
        if int(referential.get(field) or 0) != expected_value:
            errors.append(f"referential audit {field}: expected {expected_value}, got {referential.get(field)}")
    expected_chain_counts = {"1": 3, "2": 21, "3": 6}
    chain_rows = referential.get("by_chain_length") or {}
    if set(chain_rows) != set(expected_chain_counts):
        errors.append(f"referential chain lengths changed: {sorted(chain_rows)}")
    for chain, expected_count in expected_chain_counts.items():
        row = chain_rows.get(chain) or {}
        for field in ("updates", "target_field_exact", "value_exact", "hidden_pass"):
            if int(row.get(field) or 0) != expected_count:
                errors.append(
                    f"referential chain {chain} {field}: expected {expected_count}, got {row.get(field)}"
                )
    with (root / "evidence/transition/referential_rows.csv").open(newline="", encoding="utf-8") as handle:
        referential_rows = list(csv.DictReader(handle))
    if len(referential_rows) != 30:
        errors.append(f"referential evidence CSV has {len(referential_rows)} rather than 30 rows")
    if any(row.get("entities_named_in_current_update") for row in referential_rows):
        errors.append("a frozen referential utterance directly names its resolved target ID")
    if any(not parse_bool(row.get("target_field_exact")) or not parse_bool(row.get("value_exact")) for row in referential_rows):
        errors.append("a frozen referential row no longer has an exact target/field/value transition")
    characterization = transition.get("benchmark_characterization") or {}
    if (
        int(characterization.get("episodes") or 0) != 15
        or int(characterization.get("updates") or 0) != 180
        or int(characterization.get("memory_updates") or 0) != 30
        or int(characterization.get("unique_first_paragraphs") or 0) != 103
    ):
        errors.append("benchmark characterization counts changed")

    binding_public = json.loads(
        (root / "evidence/state_binding/public_challenges.json").read_text(encoding="utf-8")
    )
    if (
        binding_public.get("protocol") != "state_binding_public_branches_v1"
        or binding_public.get("benchmark_sha256")
        != "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303"
    ):
        errors.append("state-binding public branch specification changed")
    public_branches = binding_public.get("branches") or []
    if len(public_branches) != 6 or any(len(row.get("paraphrases") or []) != 3 for row in public_branches):
        errors.append("state-binding public specification does not contain six three-wording branches")
    binding_preflight = json.loads(
        (root / "evidence/state_binding/preflight_audit.json").read_text(encoding="utf-8")
    )
    if (
        binding_preflight.get("status") != "passed"
        or binding_preflight.get("errors")
        or int(binding_preflight.get("branch_count") or 0) != 6
        or int(binding_preflight.get("source_seed_count") or 0) != 10
        or int((binding_preflight.get("pilot") or {}).get("compliant_count") or 0) != 18
    ):
        errors.append("state-binding provider-free preflight did not pass")

    state_binding = json.loads(
        (root / "evidence/state_binding/evidence.json").read_text(encoding="utf-8")
    )
    if state_binding.get("status") != "passed" or state_binding.get("errors"):
        errors.append("state-binding factorial audit did not pass")
    if state_binding.get("protocol") != "liveopt_state_binding_factorial_v1":
        errors.append("state-binding factorial protocol changed")
    binding_design = state_binding.get("design") or {}
    if (
        binding_design.get("profiles") != ["NLDO-P010", "NLDO-P015"]
        or int(binding_design.get("controller_paraphrases") or 0) != 3
        or binding_design.get("formal_numerical_seeds") != [0, 1, 2]
        or int(binding_design.get("formal_population_size") or 0) != 200
        or int(binding_design.get("formal_generation_cap") or 0) != 200
        or not binding_design.get("common_incoming_population")
    ):
        errors.append("state-binding factorial design or budget changed")
    binding_inputs = state_binding.get("inputs") or []
    if len(binding_inputs) != 4 or any(len(str(row.get("sha256") or "")) != 64 for row in binding_inputs):
        errors.append("state-binding factorial input manifest is incomplete")

    controller_binding = {
        str(row.get("memory_view")): row for row in state_binding.get("controller_aggregate") or []
    }
    expected_controller_binding = {
        "full": (18, (6, 6, 6)),
        "ledger_only": (6, (6, 0, 0)),
        "accepted_only": (6, (0, 6, 0)),
        "current_only": (0, (0, 0, 0)),
        "oracle": (18, (6, 6, 6)),
    }
    if set(controller_binding) != set(expected_controller_binding):
        errors.append(f"state-binding controller views changed: {sorted(controller_binding)}")
    for view, (expected_total, expected_tasks) in expected_controller_binding.items():
        row = controller_binding.get(view) or {}
        observed_tasks = tuple(
            int(row.get(f"{task}_binding_exact") or 0)
            for task in ("event_only", "accepted_only", "event_and_accepted")
        )
        task_trials = tuple(
            int(row.get(f"{task}_trials") or 0)
            for task in ("event_only", "accepted_only", "event_and_accepted")
        )
        if int(row.get("trials") or 0) != 18 or task_trials != (6, 6, 6):
            errors.append(f"state-binding controller {view} does not have 18 complete trials")
        if int(row.get("binding_exact") or 0) != expected_total or observed_tasks != expected_tasks:
            errors.append(
                f"state-binding controller {view}: expected {expected_total}/{expected_tasks}, "
                f"got {row.get('binding_exact')}/{observed_tasks}"
            )

    formal_binding = {
        str(row.get("memory_view")): row for row in state_binding.get("formal_aggregate") or []
    }
    expected_formal_binding = {
        "full": (18, 0.846),
        "ledger_only": (6, 0.261),
        "accepted_only": (6, 0.279),
        "current_only": (0, 0.000),
        "oracle": (18, 0.846),
    }
    if set(formal_binding) != set(expected_formal_binding):
        errors.append(f"state-binding formal views changed: {sorted(formal_binding)}")
    for view, (expected_valid, expected_hv) in expected_formal_binding.items():
        row = formal_binding.get(view) or {}
        if int(row.get("trials") or 0) != 18:
            errors.append(f"state-binding formal {view} does not contain 18 replay cells")
        if int(row.get("binding_exact") or 0) != expected_valid or int(row.get("valid") or 0) != expected_valid:
            errors.append(f"state-binding formal {view} exact/valid count changed")
        if round(float(row.get("mean_task_adjusted_reference_hv") or 0.0), 3) != expected_hv:
            errors.append(f"state-binding formal {view} feasibility-gated HV changed")

    reference = json.loads((root / "evidence/reference/summary.json").read_text(encoding="utf-8"))
    if reference.get("status") != "passed":
        errors.append("reference robustness export did not pass")
    above = reference.get("above_reference") or {}
    if int(above.get("cells_at_1x") or 0) != 5:
        errors.append("expected five above-reference no-TSS cells")
    assert_close(float(above.get("max_ratio_at_1x") or 0.0), 1.019938267, 5e-12, "maximum HV ratio", errors)
    archive = reference.get("archive_cap") or {}
    if int(archive.get("active_cells") or 0) != 0:
        errors.append("archive cap was active in the frozen LiveOpt evidence")

    dynamic = json.loads((root / "evidence/main_dynamic_summary.json").read_text(encoding="utf-8"))
    live_dynamic = (dynamic.get("methods") or {}).get("LiveOpt") or {}
    if (
        int((live_dynamic.get("ds") or {}).get("cells") or 0) != 117
        or int((live_dynamic.get("ds") or {}).get("solved") or 0) != 117
    ):
        errors.append("main dynamic summary does not contain 117/117 LiveOpt DS states including t00")
    if (
        int((live_dynamic.get("dm") or {}).get("cells") or 0) != 78
        or int((live_dynamic.get("dm") or {}).get("solved") or 0) != 78
    ):
        errors.append("main dynamic summary does not contain 78/78 LiveOpt DM states including t00")
    assert_close(
        float((live_dynamic.get("ds") or {}).get("mean_quality") or 0.0),
        0.9514493547008546,
        5e-12,
        "main dynamic DS quality including t00",
        errors,
    )
    assert_close(
        float((live_dynamic.get("dm") or {}).get("mean_hv") or 0.0),
        0.7786983641025641,
        5e-12,
        "main dynamic DM quality including t00",
        errors,
    )

    strata = json.loads((root / "evidence/strata/summary.json").read_text(encoding="utf-8"))
    if strata.get("status") != "passed":
        errors.append("update-stratum export did not pass")
    if int(strata.get("metric_cells") or 0) != 1800 or int(strata.get("public_states") or 0) != 180:
        errors.append("update-stratum evidence does not cover 15x12x10 metric cells")
    strata_rows = {
        (str(row.get("view")), str(row.get("stratum"))): row
        for row in strata.get("rows") or []
        if row.get("scope") == "all15"
    }
    expected_strata = {
        ("DS", "all"): (108, 0.9474034675925926),
        ("DS", "active"): (69, 0.9472028188405797),
        ("DS", "maintenance"): (39, 0.9477584615384617),
        ("DS", "large"): (6, 0.6836247833333333),
        ("DM", "large"): (12, 0.4937780916666667),
    }
    for key, (states, quality) in expected_strata.items():
        row = strata_rows.get(key) or {}
        if int(row.get("states") or 0) != states:
            errors.append(f"{key} stratum: expected {states} states, got {row.get('states')}")
        assert_close(float(row.get("quality") or 0.0), quality, 5e-12, f"{key} stratum quality", errors)

    annotation = json.loads((root / "evidence/annotation/report.json").read_text(encoding="utf-8"))
    if not annotation.get("passed") or annotation.get("annotation_leaks") or annotation.get("errors"):
        errors.append("prompt-annotation leak audit did not pass")
    if int(annotation.get("checked_strings") or 0) != 1260:
        errors.append("prompt-annotation audit does not cover 1,260 strings")

    alignment = json.loads((root / "evidence/alignment/report.json").read_text(encoding="utf-8"))
    if alignment.get("status") != "passed" or alignment.get("errors"):
        errors.append("benchmark/public alignment audit did not pass")
    public_alignment = alignment.get("public_alignment") or {}
    if (
        int(public_alignment.get("episodes") or 0) != 15
        or int(public_alignment.get("update_strings") or 0) != 180
        or int(public_alignment.get("public_table_files") or 0) != 62
    ):
        errors.append("benchmark/public alignment coverage is incomplete")
    if not alignment.get("appendix_exact_render"):
        errors.append("public-case appendix is not an exact benchmark rendering")
    formal_alignment = alignment.get("formal_alignment") or {}
    if (
        int(formal_alignment.get("update_cells") or 0) != 720
        or int(formal_alignment.get("effective_semantic_only_cells") or 0) != 720
    ):
        errors.append("formal effective-restart alignment does not cover all 720 cells")

    baseline_protocol = json.loads(
        (root / "evidence/baseline_protocol/report.json").read_text(encoding="utf-8")
    )
    if baseline_protocol.get("status") != "passed" or baseline_protocol.get("errors"):
        errors.append("frozen external-baseline protocol audit did not pass")
    if baseline_protocol.get("schema_version") != "nldo_frozen_external_baseline_protocol_audit_v2":
        errors.append("external-baseline protocol schema is stale")
    if baseline_protocol.get("classification") != "source_aligned_then_own_history_continuation":
        errors.append("external-baseline protocol has an unexpected classification")
    if int(baseline_protocol.get("stage_rows") or 0) != 262:
        errors.append("external-baseline protocol does not cover 262 frozen rows")
    expected_baseline_rows = {
        "optimai_2025": 33,
        "optimus": 68,
        "or_llm_agent_2025": 41,
        "orlm": 50,
        "react_tools": 70,
    }
    if baseline_protocol.get("method_rows") != expected_baseline_rows:
        errors.append("external-baseline method-row counts changed")
    public_state = baseline_protocol.get("public_state_contract") or {}
    if int(public_state.get("candidate_history_rows") or 0) != 135:
        errors.append("external-baseline own candidate-history count changed")
    if int(public_state.get("shared_lsm_rows") or 0) != 135:
        errors.append("external-baseline own-history continuation count changed")
    if int(public_state.get("liveopt_private_state_rows") or 0) != 0:
        errors.append("external baselines unexpectedly contain LiveOpt state")
    if int(public_state.get("source_aligned_previous_output_rows") or 0) != 16:
        errors.append("source-aligned preceding-output count changed")
    for field in (
        "own_history_continuation_rows",
        "own_history_previous_output_rows",
        "own_history_candidate_population_rows",
    ):
        if int(public_state.get(field) or 0) != 135:
            errors.append(f"external-baseline continuation contract changed: {field}")
    expected_prompt_versions = {
        "nldo_dynamic_common_public_lsm_v6": 135,
        "nldo_dynamic_public_code_solver_v1": 127,
    }
    if baseline_protocol.get("prompt_versions") != expected_prompt_versions:
        errors.append("external-baseline prompt-version counts changed")
    repair = baseline_protocol.get("repair_contract") or {}
    if int(repair.get("hidden_feedback_rows") or 0) or int(repair.get("reference_feedback_rows") or 0):
        errors.append("external-baseline public repair used hidden/reference feedback")
    if not ((baseline_protocol.get("summary_check") or {}).get("matches")):
        errors.append("external-baseline protocol does not match the heatmap sources")

    baseline_failure = json.loads(
        (root / "evidence/baseline_failure/summary.json").read_text(encoding="utf-8")
    )
    if baseline_failure.get("schema_version") != "persistent_react_first_break_attribution_v1":
        errors.append("Persistent ReAct failure-attribution schema changed")
    records = {
        str(row.get("episode_id")): row for row in (baseline_failure.get("records") or [])
    }
    expected_breaks = {
        "NLDO-P007": (3, "Incorrect constraint check", 0.0, 119.0),
        "NLDO-P008": (6, "Missing output field", None, 1.0),
        "NLDO-P009": (7, "Infeasible search result", 2.0, 2.0),
    }
    if set(records) != set(expected_breaks):
        errors.append("Persistent ReAct failure attribution does not cover P007--P009 exactly")
    for episode_id, (stage, category, local_shortage, heldout_shortage) in expected_breaks.items():
        row = records.get(episode_id) or {}
        if int(row.get("first_break_stage") or -1) != stage:
            errors.append(f"{episode_id} first-break stage changed")
        if row.get("category") != category:
            errors.append(f"{episode_id} failure category changed")
        observed_local = row.get("local_coverage_shortage")
        if local_shortage is None:
            if observed_local is not None:
                errors.append(f"{episode_id} unexpectedly has local coverage diagnostics")
        else:
            assert_close(
                float(observed_local or 0.0),
                local_shortage,
                5e-12,
                f"{episode_id} local coverage shortage",
                errors,
            )
        assert_close(
            float(row.get("heldout_coverage_shortage") or 0.0),
            heldout_shortage,
            5e-12,
            f"{episode_id} held-out coverage shortage",
            errors,
        )

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checksum_files": checksum_files,
        "anonymity_text_files": text_files,
        "formal_metric_summaries": summaries,
        "failure_categories": categories,
        "reference_sha256": reference.get("reference_sha256"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
