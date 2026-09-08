#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.benchmarks.optimization_contracts import (
    PUBLIC_OBJECTIVE_SPEC_MARKER,
    PUBLIC_UPDATE_OBJECTIVE_NOTE_MARKER,
    public_optimization_contract_for_episode,
)


EXPECTED_VERSION = "public_quantitative_objective_contract_v1"
USER_FACING_FORBIDDEN_PATTERNS = {
    "public_context": "framework API name should not appear in user-facing task text",
    "load_table": "framework tool/API name should not appear in user-facing task text",
    "benchmark_family": "benchmark metadata should not appear in user-facing task text",
    "required_solver_mode": "solver-routing metadata should not appear in user-facing task text",
    "solver_mode": "solver-routing metadata should not appear in user-facing task text",
    "public_quantitative_objective_contract_v1": "machine contract version should not appear in user-facing task text",
    "hidden_evaluator": "hidden evaluator metadata must not appear in user-facing task text",
    "hidden reference": "hidden reference metadata must not appear in user-facing task text",
    "reference_solution": "reference solution metadata must not appear in user-facing task text",
    "reference archive": "reference archive metadata must not appear in user-facing task text",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public quantitative objective contracts for NLDO episodes.")
    parser.add_argument("--episodes", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_10updates_csv.jsonl")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    episodes = _read_jsonl(REPO_ROOT / args.episodes)
    findings = []
    summary = {
        "episodes": len(episodes),
        "stages": 0,
        "missing_contract": 0,
        "invalid_contract": 0,
        "by_objective_mode": {},
    }
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        public_context = episode.get("public_context") if isinstance(episode.get("public_context"), dict) else {}
        contract = public_context.get("optimization_contract")
        generated = public_optimization_contract_for_episode(episode)
        if not isinstance(contract, dict):
            summary["missing_contract"] += 1
            findings.append(_finding(episode_id, "missing_contract", "public_context.optimization_contract is absent"))
            contract = generated
        initial_text = str(episode.get("public_initial_problem") or "")
        if PUBLIC_OBJECTIVE_SPEC_MARKER not in initial_text:
            findings.append(_finding(episode_id, "missing_initial_objective_spec_text", "public_initial_problem does not contain public objective/scoring specification text"))
        findings.extend(_user_facing_text_findings(episode_id, "initial", initial_text))
        problems = _contract_problems(episode, contract)
        for problem in problems:
            findings.append(_finding(episode_id, problem["kind"], problem["message"]))
        if problems:
            summary["invalid_contract"] += 1
        mode = str(contract.get("objective_mode") or "missing")
        summary["by_objective_mode"][mode] = summary["by_objective_mode"].get(mode, 0) + 1
        updates = episode.get("update_stream") if isinstance(episode.get("update_stream"), list) else []
        summary["stages"] += 1 + len(updates)
        for update in updates:
            text = str(update.get("public_update") or update.get("natural_language_update") or "")
            if not text.strip():
                findings.append(_finding(episode_id, "empty_update_text", f"{update.get('update_id')} has no public update text"))
            if PUBLIC_UPDATE_OBJECTIVE_NOTE_MARKER not in text:
                findings.append(_finding(episode_id, "missing_update_objective_note", f"{update.get('update_id')} lacks public update objective/scoring note"))
            findings.extend(_user_facing_text_findings(episode_id, str(update.get("update_id") or "update"), text))
            update_contract = contract.get("update_contract")
            if not isinstance(update_contract, dict) or not update_contract.get("preserve_terms_unless_explicitly_changed"):
                findings.append(_finding(episode_id, "weak_update_contract", f"{update.get('update_id')} does not preserve objective terms"))

    result = {"summary": summary, "findings": findings}
    if args.output_json:
        output = REPO_ROOT / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.strict and findings else 0


def _contract_problems(episode: dict[str, Any], contract: Any) -> list[dict[str, str]]:
    if not isinstance(contract, dict):
        return [{"kind": "invalid_contract", "message": "contract is not an object"}]
    problems: list[dict[str, str]] = []
    if contract.get("version") != EXPECTED_VERSION:
        problems.append({"kind": "contract_version", "message": f"expected version {EXPECTED_VERSION}, got {contract.get('version')!r}"})
    for key in ["objective_mode", "required_solver_mode", "objective_sense", "objective_names", "hard_constraints", "objective_terms", "scalar_formula", "solution_output_format", "update_contract"]:
        if key not in contract:
            problems.append({"kind": "missing_contract_field", "message": f"missing {key}"})
    solution_format = contract.get("solution_output_format")
    if not isinstance(solution_format, dict) or not solution_format.get("description"):
        problems.append({"kind": "missing_solution_output_format", "message": "solution_output_format must describe the user-visible submitted plan shape"})
    objective_terms = contract.get("objective_terms")
    if not isinstance(objective_terms, list) or not objective_terms:
        problems.append({"kind": "empty_objective_terms", "message": "objective_terms must be a non-empty list"})
    else:
        for idx, term in enumerate(objective_terms):
            if not isinstance(term, dict) or not term.get("name") or not term.get("formula"):
                problems.append({"kind": "weak_objective_term", "message": f"objective_terms[{idx}] must include name and formula"})
    mode = str(contract.get("objective_mode") or "")
    if mode == "multi_objective":
        names = contract.get("objective_names")
        if not isinstance(names, list) or len(names) < 2:
            problems.append({"kind": "multiobjective_names", "message": "multi_objective contract needs at least two objective_names"})
        if contract.get("required_solver_mode") != "moea":
            problems.append({"kind": "multiobjective_solver", "message": "multi_objective contract must require moea"})
    profile = str(episode.get("hidden_evaluator_profile") or episode.get("domain") or "").lower()
    if "inrc" in profile:
        required = set(contract.get("required_diagnostics") or [])
        expected = {"coverage_shortage", "absence_violations", "overload_penalty", "over_coverage", "fairness_penalty", "preference_penalty", "sequence_penalty"}
        missing = sorted(expected - required)
        if missing:
            problems.append({"kind": "inrc_required_diagnostics", "message": f"missing INRC diagnostics: {missing}"})
        if "policy." not in str(contract.get("scalar_formula") or ""):
            problems.append({"kind": "inrc_policy_weights", "message": "INRC scalar_formula must read public policy weights"})
    update_contract = contract.get("update_contract")
    if isinstance(update_contract, dict) and update_contract.get("history_dependent_hard_constraints_allowed") is not False:
        problems.append({"kind": "history_dependent_constraints", "message": "update_contract must forbid history-dependent hard constraints"})
    return problems


def _finding(episode_id: str, kind: str, message: str) -> dict[str, str]:
    return {"episode_id": episode_id, "kind": kind, "message": message}


def _user_facing_text_findings(episode_id: str, stage_id: str, text: str) -> list[dict[str, str]]:
    lowered = text.lower()
    findings = []
    for pattern, reason in USER_FACING_FORBIDDEN_PATTERNS.items():
        if pattern.lower() in lowered:
            findings.append(
                _finding(
                    episode_id,
                    "non_user_facing_internal_term",
                    f"{stage_id} contains {pattern!r}: {reason}",
                )
            )
    return findings


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
