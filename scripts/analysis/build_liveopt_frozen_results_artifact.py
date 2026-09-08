#!/usr/bin/env python3
"""Build the anonymous, provider-free LiveOpt frozen-results bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "release_artifacts/anonymous_frozen_results"
DEFAULT_ZIP = ROOT / "release_artifacts/liveopt_anonymous_frozen_results.zip"
EPISODES_DM = {f"NLDO-P{number:03d}" for number in range(10, 16)}
EPISODES_MATCHED = {f"NLDO-P{number:03d}" for number in range(7, 16)}
UPDATE_STAGES = set(range(1, 13))


FORMAL_METRICS = {
    "liveopt_dm_updates.csv": ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
    "no_tss_dm_updates.csv": ROOT / "outputs/liveopt_no_tss_generic_blank_selfhistory_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
    "sequential_warm_dm_updates.csv": ROOT / "outputs/liveopt_sequential_fixed_warm_p010_p015_200x200x10_20260715/reference_metrics/reference_stage_metrics.csv",
    "sequential_full_dm_updates.csv": ROOT / "outputs/liveopt_sequential_fixed_full_p010_p015_200x200x10_20260715/reference_metrics/reference_stage_metrics.csv",
    "tss_operator_control_dm_updates.csv": ROOT
    / "outputs/liveopt_tss_operator_control_p010_p015_200x200x3_20260727/reference_metrics/reference_stage_metrics.csv",
}

MATCHED_ROOT = ROOT / "outputs/liveopt_ablation_semantic_shadow_comparison_p007_p015_200x200x10_v2_20260714"

EVIDENCE_FILES = {
    "evidence/tss/summary.json": ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715/summary.json",
    "evidence/tss/method_summary.csv": ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715/method_summary.csv",
    "evidence/tss/episode_summary.csv": ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715/episode_summary.csv",
    "evidence/tss/failure_cells.csv": ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715/failure_cells.csv",
    "evidence/tss/artifact_versions.csv": ROOT / "logs/analysis/liveopt_tss_mechanism_evidence_20260715/artifact_versions.csv",
    "evidence/tss/operator_control_summary.json": ROOT
    / "logs/analysis/liveopt_tss_operator_control_20260727/summary.json",
    "evidence/selector/summary.json": ROOT / "logs/analysis/liveopt_restart_selector_regret_20260716/summary.json",
    "evidence/selector/method_summary.csv": ROOT / "logs/analysis/liveopt_restart_selector_regret_20260716/method_summary.csv",
    "evidence/selector/episode_deltas.csv": ROOT / "logs/analysis/liveopt_restart_selector_regret_20260716/episode_deltas.csv",
    "evidence/selector/stage_decisions.csv": ROOT / "logs/analysis/liveopt_restart_selector_regret_20260716/stage_decisions.csv",
    "evidence/landscape/audit.json": ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/audit.json",
    "evidence/landscape/landscape_decisions.csv": ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/landscape_decisions.csv",
    "evidence/landscape/stage_summary.csv": ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/stage_summary.csv",
    "evidence/landscape/method_summary.csv": ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/method_summary.csv",
    "evidence/landscape/threshold_sensitivity.csv": ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/threshold_sensitivity.csv",
    "evidence/archive_cap/summary.json": ROOT / "logs/analysis/liveopt_archive_cap_sensitivity_p014_p015_20260716/summary.json",
    "evidence/archive_cap/summary.csv": ROOT / "logs/analysis/liveopt_archive_cap_sensitivity_p014_p015_20260716/summary.csv",
    "evidence/controller_repeat/evidence.json": ROOT / "logs/analysis/liveopt_controller_repeat_p010_p015_20260717/evidence.json",
    "evidence/controller_repeat/controller_cells.csv": ROOT / "logs/analysis/liveopt_controller_repeat_p010_p015_20260717/controller_cells.csv",
    "evidence/controller_repeat/controller_summary.csv": ROOT / "logs/analysis/liveopt_controller_repeat_p010_p015_20260717/controller_summary.csv",
    "evidence/controller_repeat/aggregate_summary.csv": ROOT / "logs/analysis/liveopt_controller_repeat_p010_p015_20260717/aggregate_summary.csv",
    "evidence/patch_diff/evidence.json": ROOT / "logs/analysis/liveopt_real_patch_diff_20260717/evidence.json",
    "evidence/sequential/summary.json": ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715/summary.json",
    "evidence/sequential/method_summary.csv": ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715/method_summary.csv",
    "evidence/sequential/episode_summary.csv": ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715/episode_summary.csv",
    "evidence/sequential/stage_summary.csv": ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715/stage_summary.csv",
    "evidence/sequential/runtime_cells.csv": ROOT / "logs/analysis/liveopt_sequential_restart_audit_20260715/runtime_cells.csv",
    "evidence/reference/summary.json": ROOT / "logs/analysis/liveopt_reference_evidence_20260715/summary.json",
    "evidence/reference/reference_robustness_audit.json": ROOT / "logs/analysis/liveopt_reference_robustness_latest_20260715/reference_robustness_audit.json",
    "evidence/reference/box_sensitivity_summary.csv": ROOT / "logs/analysis/liveopt_reference_robustness_latest_20260715/box_sensitivity_summary.csv",
    "evidence/reference/thinning_sensitivity.json": ROOT / "logs/analysis/liveopt_reference_thinning_latest_20260715/sensitivity.json",
    "evidence/reference/thinning_sensitivity.csv": ROOT / "logs/analysis/liveopt_reference_thinning_latest_20260715/sensitivity.csv",
    "evidence/reference/archive_diagnostics.json": ROOT / "logs/analysis/liveopt_archive_diagnostics_latest_20260715/archive.json",
    "evidence/reference/archive_diagnostics.csv": ROOT / "logs/analysis/liveopt_archive_diagnostics_latest_20260715/archive.csv",
    "evidence/strata/summary.json": ROOT / "logs/analysis/liveopt_update_strata_current_20260715/summary.json",
    "evidence/strata/state_rows.csv": ROOT / "logs/analysis/liveopt_update_strata_current_20260715/state_rows.csv",
    "evidence/strata/strata.csv": ROOT / "logs/analysis/liveopt_update_strata_current_20260715/strata.csv",
    "evidence/strata/episodes.csv": ROOT / "logs/analysis/liveopt_update_strata_current_20260715/episodes.csv",
    "evidence/strata/profiles.csv": ROOT / "logs/analysis/liveopt_update_strata_current_20260715/profiles.csv",
    "evidence/annotation/report.json": ROOT / "logs/analysis/nldo_prompt_annotation_audit_20260715/report.json",
    "evidence/alignment/report.json": ROOT / "logs/analysis/nldo_public_alignment_20260715/report.json",
    "evidence/baseline_protocol/report.json": ROOT
    / "logs/analysis/nldo_frozen_baseline_protocol_20260715/report.json",
    "evidence/baseline_failure/summary.json": ROOT
    / "logs/analysis/persistent_react_failure_attribution_20260727/summary.json",
    "evidence/transition/summary.json": ROOT
    / "logs/analysis/liveopt_state_transition_audit_20260717/summary.json",
    "evidence/transition/input_manifest.json": ROOT
    / "logs/analysis/liveopt_state_transition_audit_20260717/input_manifest.json",
    "evidence/transition/transition_rows.csv": ROOT
    / "logs/analysis/liveopt_state_transition_audit_20260717/transition_rows.csv",
    "evidence/transition/referential_rows.csv": ROOT
    / "logs/analysis/liveopt_state_transition_audit_20260717/referential_rows.csv",
    "evidence/transition/benchmark_profiles.csv": ROOT
    / "logs/analysis/liveopt_state_transition_audit_20260717/benchmark_profiles.csv",
    "evidence/state_binding/evidence.json": ROOT
    / "logs/analysis/liveopt_state_binding_evidence_20260717/evidence.json",
    "evidence/state_binding/controller_aggregate.csv": ROOT
    / "logs/analysis/liveopt_state_binding_evidence_20260717/controller_aggregate.csv",
    "evidence/state_binding/formal_aggregate.csv": ROOT
    / "logs/analysis/liveopt_state_binding_evidence_20260717/formal_aggregate.csv",
    "evidence/state_binding/public_challenges.json": ROOT
    / "logs/analysis/liveopt_state_binding_challenge_20260717/public_challenges.json",
    "evidence/state_binding/preflight_audit.json": ROOT
    / "logs/analysis/liveopt_state_binding_challenge_20260717/audit.json",
    "evidence/main_dynamic_summary.json": ROOT / "outputs/main_dynamic_hv_heatmap_20260715/summary.json",
    "evidence/matched_restart_protocol.json": MATCHED_ROOT / "protocol_audit.json",
    "evidence/matched_restart_sources.csv": MATCHED_ROOT / "source_manifest.csv",
}

TABLE_FILES = (
    "main_dynamic_hv_summary_rows.tex",
    "main_static_grouped_rows.tex",
    "formal_benchmark_results_rows.tex",
    "liveopt_current_episode_results_rows.tex",
    "liveopt_mo_hv_variance_rows.tex",
    "liveopt_matched_stage_hv_rows.tex",
    "liveopt_tss_mechanism_rows.tex",
    "liveopt_restart_selector_regret_rows.tex",
    "liveopt_sequential_restart_rows.tex",
    "liveopt_reference_sensitivity_rows.tex",
    "liveopt_update_strata_rows.tex",
    "liveopt_landscape_restart_rows.tex",
    "liveopt_archive_cap_sensitivity_rows.tex",
    "liveopt_controller_repeat_rows.tex",
    "liveopt_real_patch_diff.tex",
    "liveopt_state_transition_audit_rows.tex",
    "liveopt_state_binding_factorial.tex",
    "liveopt_tss_operator_control_rows.tex",
    "persistent_react_failure_attribution_rows.tex",
)

METRIC_FIELDS = (
    "episode_id",
    "stage_index",
    "run_seed",
    "update_id",
    "feasible",
    "true_pass",
    "normalized_hv",
    "normalized_score",
    "hv",
    "reference_hv",
    "igd",
    "ideal_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--zip-output", type=Path, default=DEFAULT_ZIP)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_text(text: str) -> str:
    replacements = (
        (str(ROOT.resolve()) + "/", ""),
        (str(Path.home()) + "/", "<workspace>/"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def copy_sanitized(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".json", ".jsonl", ".csv", ".tex", ".md", ".txt", ".yaml", ".yml"}:
        target.write_text(sanitize_text(source.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(source, target)


def export_filtered_csv(
    source: Path,
    target: Path,
    episodes: set[str],
    fields: tuple[str, ...] | None = None,
) -> int:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if str(row.get("episode_id") or "") in episodes
            and int(float(row.get("stage_index") or 0)) in UPDATE_STAGES
        ]
        source_fields = tuple(reader.fieldnames or ())
    output_fields = fields or source_fields
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: sanitize_text(str(row.get(field, ""))) for field in output_fields})
    return len(rows)


def claim_registry() -> list[dict[str, str]]:
    return [
        {
            "claim_id": "C01",
            "paper_location": "Sec. 4.2; Table 2; App. update-only diagnostics",
            "claim": "Including t00, LiveOpt solves all 117 DS and all 78 DM states, with mean DS quality 0.951 and DM HV 0.779. Over t01--t12 alone, the corresponding means are 0.947 and 0.779.",
            "evidence": "evidence/main_dynamic_summary.json",
            "verification": "scripts/verify_frozen_results.py checks the state counts; the JSON contains the full 15x13 method matrices, including t00.",
        },
        {
            "claim_id": "C02",
            "paper_location": "Sec. 4.3; Table 4",
            "claim": "The typed Workbench passes all 720 P010--P015 update-seed cells; w/o TSS passes 580 and has 140 silent failures.",
            "evidence": "data/formal_metrics/liveopt_dm_updates.csv; data/formal_metrics/no_tss_dm_updates.csv; evidence/tss/failure_cells.csv",
            "verification": "The verifier recomputes hidden-pass counts and failure categories from rows.",
        },
        {
            "claim_id": "C03",
            "paper_location": "Sec. 4.3; Table 4",
            "claim": "w/o TSS reaches 0.752 conditional HV when hidden-valid but 0.606 all-cell HV because infeasible cells score zero.",
            "evidence": "evidence/tss/summary.json; evidence/tss/episode_summary.csv",
            "verification": "summary.json records both feasibility-gated and pass-only aggregates with source checksums.",
        },
        {
            "claim_id": "C04",
            "paper_location": "Sec. 4.3; Table 5",
            "claim": "From matched incoming populations, LiveOpt improves over Warm by 0.013 HV and recovers 74% of the post-hoc oracle gain; all six episode means are positive (exact sign p=0.03125).",
            "evidence": "evidence/selector/summary.json; data/matched_restart/*",
            "verification": "The frozen selector summary and per-stage matched metrics expose every decision unit and seed.",
        },
        {
            "claim_id": "C05",
            "paper_location": "App. A.5; sequential-policy table",
            "claim": "On independent sequential trajectories, LiveOpt, Always Warm, and Always Full obtain 0.779, 0.769, and 0.731 HV.",
            "evidence": "data/formal_metrics/*_dm_updates.csv; evidence/sequential/summary.json",
            "verification": "The verifier recomputes all four 720-cell means and checks zero identity/reference mismatches.",
        },
        {
            "claim_id": "C06",
            "paper_location": "App. A.2; reference sensitivity",
            "claim": "The five w/o-TSS cells above the finite best-known reference all map to P012-t01; the maximum HV ratio is 1.019938, and method ordering survives thinning and wider HV boxes.",
            "evidence": "evidence/reference/reference_robustness_audit.json; evidence/reference/thinning_sensitivity.json",
            "verification": "The robustness report contains re-evaluated objectives, stage mapping checks, novel points, and sensitivity summaries.",
        },
        {
            "claim_id": "C07",
            "paper_location": "App. A.2",
            "claim": "The 500-member result archive cap is inactive in all 780 LiveOpt P010--P015 initial/update seed cells.",
            "evidence": "evidence/reference/archive_diagnostics.csv; evidence/reference/summary.json",
            "verification": "The verifier requires active_cells=0.",
        },
        {
            "claim_id": "C08",
            "paper_location": "App. A.5; reproducibility statement",
            "claim": "The sequential Warm/Full controls are numerical replays with their own predecessor populations and make no new model calls.",
            "evidence": "evidence/sequential/runtime_cells.csv; evidence/sequential/summary.json",
            "verification": "The sequential summary requires all 1,440 control cells to have own-history lineage and zero token/generic-call use.",
        },
        {
            "claim_id": "C09",
            "paper_location": "Sec. 4.2; App. update-type table",
            "claim": "Removing the 39 scalar maintenance updates leaves DS quality at 0.947; the deliberately disruptive t11--t12 regime shifts are the hardest update type at 0.684 DS quality and 0.494 DM HV.",
            "evidence": "evidence/strata/summary.json; evidence/strata/state_rows.csv",
            "verification": "The verifier checks complete 15x12x10 coverage and recomputes the frozen stratum aggregates.",
        },
        {
            "claim_id": "C10",
            "paper_location": "App. update-type table; reproducibility statement",
            "claim": "Update-type labels are analysis metadata and do not appear as parenthetical annotations in the 1,260 checked public/formal prompt strings.",
            "evidence": "evidence/annotation/report.json",
            "verification": "The frozen report records source checksums, per-field counts, the exact annotation pattern, and zero detected leaks.",
        },
        {
            "claim_id": "C11",
            "paper_location": "App. public cases and runtime configuration",
            "claim": "All 15 released public streams, 180 updates, 62 public CSVs, and the rendered appendix match the current benchmark; all 720 formal Pareto update cells record the semantic-only effective restart rule.",
            "evidence": "evidence/alignment/report.json; public_preview/data/public_streams/nldo_public_streams.jsonl",
            "verification": "The alignment audit compares public fields and table bytes, regenerates the appendix in memory, and checks every formal update and effective restart record.",
        },
        {
            "claim_id": "C12",
            "paper_location": "Sec. 4.2; App. A.12",
            "claim": "The frozen external source contains 127 source-aligned initial/early rows and 135 own-history continuation rows. Every row contains cumulative public updates and current tables; the continuation rows contain only that method's accepted output and candidate population, and no row contains LiveOpt state.",
            "evidence": "evidence/baseline_protocol/report.json",
            "verification": "The verifier checks both prompt versions, all 262 row/method counts, the 135 own-history continuation records, the absence of LiveOpt state, and the heatmap-source match.",
        },
        {
            "claim_id": "C13",
            "paper_location": "Sec. 4.3; App. objective-landscape control",
            "claim": "The 10%-sensor objective-landscape gate selects Full in all 120 t11--t12 cells but also in 125 of 600 earlier cells, reaching 0.765 HV versus LiveOpt's 0.779.",
            "evidence": "evidence/landscape/audit.json; evidence/landscape/landscape_decisions.csv",
            "verification": "The audit freezes every public-only decision before joining matched Warm/Full outcomes and requires 720 exact-population cells with 20 sensors each.",
        },
        {
            "claim_id": "C14",
            "paper_location": "App. A.2; archive-cap sensitivity",
            "claim": "On the current P014/P015 benchmark, archive caps 100, 200, and 500 yield 0.775, 0.776, and 0.774 mean HV; caps 100/200 are active in 240/162 of 240 cells while cap 500 is inactive.",
            "evidence": "evidence/archive_cap/summary.json; evidence/archive_cap/summary.csv",
            "verification": "The cap audit checks the three replay plans, 240 matched update cells per cap, the common held-out reference, and cap-activation counts.",
        },
        {
            "claim_id": "C15",
            "paper_location": "Sec. 4.3; App. independent controller trajectories",
            "claim": "Across three controller trajectories and three numerical seeds on each of P010/P015, LiveOpt passes 204/216 update-seed cells with mean HV 0.699; w/o TSS passes 168/216 with mean HV 0.424.",
            "evidence": "evidence/controller_repeat/evidence.json; evidence/controller_repeat/controller_cells.csv; evidence/controller_repeat/aggregate_summary.csv",
            "verification": "The verifier checks 12 trajectories, 432 requested method cells, the frozen semantic decisions and reference, zero local response-cache hits for eight new trajectories, and the four controller-level aggregates.",
        },
        {
            "claim_id": "C16",
            "paper_location": "App. TSS Workbench and Skill Catalog",
            "claim": "The formal P015-t02 data patch changes M04 energy_per_cpu from 0.23 to 0.285 without changing either editable slot; P010-t02 changes the evaluator slot for the new congestion rule while leaving setup.py unchanged.",
            "evidence": "evidence/patch_diff/evidence.json; paper_tables/liveopt_real_patch_diff.tex",
            "verification": "The verifier checks the benchmark hash, exact before/after value, byte-identical P015 slots, unchanged P010 setup.py, and the saved congestion-multiplier diff.",
        },
        {
            "claim_id": "C17",
            "paper_location": "Sec. 4.3; App. transition-trace audit",
            "claim": "Across 180 frozen formal transitions, the LPD data/setup/fitness proposal exactly matches the committed action 170 times; 166 transitions keep both code slots byte-identical, six edit setup only, and eight edit fitness only.",
            "evidence": "evidence/transition/summary.json; evidence/transition/transition_rows.csv",
            "verification": "The verifier checks all 180 episode-stage rows, the three per-surface confusion matrices, slot byte-identity categories, current benchmark hash, and zero schema/entity expansion.",
        },
        {
            "claim_id": "C18",
            "paper_location": "Sec. 4.3; App. referential-resolution table",
            "claim": "All 30 benchmark-authored referential updates are grounded in the intended entity, field, and value; none of the current utterances directly names its target ID, including six three-link chains.",
            "evidence": "evidence/transition/summary.json; evidence/transition/referential_rows.csv",
            "verification": "The verifier checks 30 grounded updates, exact target/field/value counts, the 3/21/6 reference-chain split, zero literal target IDs, and held-out validity.",
        },
        {
            "claim_id": "C19",
            "paper_location": "Sec. 4.3; Table 3",
            "claim": "Across two profiles and three wording variants, Full LSM recovers all 18 event/accepted/linked bindings; ledger-only and accepted-only recover their six applicable bindings, while current-request-only recovers none. The 200x200x3 replay preserves the same pattern.",
            "evidence": "evidence/state_binding/public_challenges.json; evidence/state_binding/preflight_audit.json; evidence/state_binding/evidence.json; evidence/state_binding/controller_aggregate.csv; evidence/state_binding/formal_aggregate.csv",
            "verification": "The verifier checks the complete 3x6x5 controller matrix, all 90 formal cells, matched incoming populations, fixed budgets, exact binding counts, and feasibility-gated HV aggregates.",
        },
        {
            "claim_id": "C20",
            "paper_location": "Sec. 4.3; App. matched type-specific-operator control",
            "claim": "On P010/P015 with numerical seeds 0--2, replacing only type-specific variation with whole-segment resampling lowers solve rate from 100.0% to 91.7% and HV from 0.774 to 0.618.",
            "evidence": "data/formal_metrics/tss_operator_control_dm_updates.csv; evidence/tss/operator_control_summary.json",
            "verification": "The verifier checks 72 update cells, 66 hidden-valid cells, 0.6177855 gated HV, source-LiveOpt lineage, and the fixed 200x200 budget.",
        },
        {
            "claim_id": "C21",
            "paper_location": "Sec. 4.2; App. Persistent ReAct failure attribution",
            "claim": "The first scored failures of the three Persistent ReAct rostering runs occur at P007-t03, P008-t06, and P009-t07. The observed causes are an incorrect constraint check, a missing output field, and an infeasible search result, respectively.",
            "evidence": "evidence/baseline_failure/summary.json; paper_tables/persistent_react_failure_attribution_rows.tex",
            "verification": "The exporter identifies the first invalid state under the paper's prefix rule and records compile/runtime, local-coverage, held-out-coverage, schema, and repair evidence.",
        },
    ]


def write_claim_registry(target: Path, claims: list[dict[str, str]]) -> None:
    metadata = target / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "claim_registry.json").write_text(
        json.dumps({"schema_version": "liveopt_claim_registry_v1", "claims": claims}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (metadata / "claim_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(claims[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(claims)
    lines = [
        "# Claim-to-Evidence Map",
        "",
        "Each claim below points to a frozen file and an executable check. Values are retained at seed or episode-stage resolution where available.",
        "",
        "| ID | Paper | Claim | Evidence |",
        "|---|---|---|---|",
    ]
    for claim in claims:
        lines.append(
            f"| {claim['claim_id']} | {claim['paper_location']} | {claim['claim']} | `{claim['evidence']}` |"
        )
    (target / "CLAIM_TO_EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(target: Path) -> None:
    text = """# LiveOpt Anonymous Frozen Results

This provider-free bundle freezes the evidence used by the submission.  It is
designed for checking reported values without a model account or the original
workspace.

Included:

- all 15 public NLDO streams and tables from the public preview;
- seed-level P010--P015 metrics for LiveOpt, w/o TSS, sequential Warm, and
  sequential Full;
- seed-level same-incoming-population restart controls on P007--P015;
- TSS failure attribution, the matched type-specific-operator control,
  semantic and objective-landscape selector regret,
  sequential-lineage, reference robustness, thinning, HV-box, refreshed
  P014/P015 archive-cap, factorial state binding, independent controller repeats, a real data/slot
  localized update, the 180-transition edit surface, and 30-reference grounding
  audits, benchmark characterization, update-type, prompt-annotation,
  external-baseline protocol, first-break attribution for the three Persistent
  ReAct rostering trajectories, and benchmark/public alignment evidence;
- the generated LaTeX rows used by the paper; and
- a claim-to-evidence map with a standalone verifier.

Run from this directory:

```bash
python scripts/verify_frozen_results.py --root .
```

The verifier checks every file checksum, scans text files for common identity
or credential leaks, recomputes the primary 720-cell aggregates, checks failure
categories and key statistical claims, and validates the frozen reference and
sequential reports.  It does not contact a model provider.

This is a frozen-results artifact, not a release of the hidden evaluator.
Hidden deltas, hidden objective/checker implementations, raw candidate
populations, held-out reference archives, provider logs, and credentials are
excluded.  Consequently the bundle reproduces reported aggregation and
provenance checks, but it does not rerun model generation or independently
re-evaluate submitted solutions.
"""
    (target / "README.md").write_text(text, encoding="utf-8")


def write_checksums(target: Path) -> None:
    manifest = target / "checksums/manifest.sha256"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in target.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(target).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )


def write_zip(target: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(target.name) / path.relative_to(target)).as_posix())


def main() -> int:
    args = parse_args()
    target = args.target.resolve()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public_source = ROOT / "release_artifacts/public_preview"
    shutil.copytree(public_source, target / "public_preview")

    source_manifest: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}
    for filename, source in FORMAL_METRICS.items():
        destination = target / "data/formal_metrics" / filename
        row_counts[destination.relative_to(target).as_posix()] = export_filtered_csv(
            source, destination, EPISODES_DM, METRIC_FIELDS
        )
        source_manifest.append(
            {"artifact_file": destination.relative_to(target).as_posix(), "source_sha256": sha256_file(source)}
        )

    for key, directory in (("liveopt", "liveopt"), ("warm", "fixed_warm"), ("full", "fixed_full")):
        for name in ("reference_stage_metrics.csv", "runtime_stage_metrics.csv"):
            source = MATCHED_ROOT / directory / "reference_metrics" / name
            destination = target / "data/matched_restart" / f"{key}_{name}"
            row_counts[destination.relative_to(target).as_posix()] = export_filtered_csv(
                source, destination, EPISODES_MATCHED, None
            )
            source_manifest.append(
                {"artifact_file": destination.relative_to(target).as_posix(), "source_sha256": sha256_file(source)}
            )

    for relative, source in EVIDENCE_FILES.items():
        destination = target / relative
        copy_sanitized(source, destination)
        source_manifest.append(
            {"artifact_file": relative, "source_sha256": sha256_file(source)}
        )

    for filename in TABLE_FILES:
        source = ROOT / "release_artifacts/paper_table_exports" / filename
        destination = target / "paper_tables" / filename
        copy_sanitized(source, destination)
        source_manifest.append(
            {"artifact_file": destination.relative_to(target).as_posix(), "source_sha256": sha256_file(source)}
        )

    verifier_source = ROOT / "scripts/analysis/verify_liveopt_frozen_results.py"
    verifier_target = target / "scripts/verify_frozen_results.py"
    copy_sanitized(verifier_source, verifier_target)
    verifier_target.chmod(verifier_target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    claims = claim_registry()
    write_claim_registry(target, claims)
    write_readme(target)
    metadata = {
        "schema_version": "liveopt_anonymous_frozen_results_v2",
        "frozen_date": "2026-07-27",
        "provider_free_verification": True,
        "scope": {
            "public_episodes": 15,
            "mechanism_episodes": [f"NLDO-P{number:03d}" for number in range(10, 16)],
            "update_stages": 12,
            "numerical_seeds": 10,
            "controller_repeat": {
                "episodes": ["NLDO-P010", "NLDO-P015"],
                "controller_replicates": 3,
                "numerical_seeds_per_controller": 3,
            },
            "tss_operator_control": {
                "episodes": ["NLDO-P010", "NLDO-P015"],
                "numerical_seeds": 3,
                "matched_update_cells": 72,
                "provider_calls": 0,
            },
            "persistent_react_failure_attribution": {
                "episodes": ["NLDO-P007", "NLDO-P008", "NLDO-P009"],
                "controller_trajectories": 3,
                "provider_calls": 0,
            },
            "state_transition_audit": {
                "episodes": 15,
                "updates": 180,
                "referential_updates": 30,
                "provider_calls": 0,
            },
            "state_binding_factorial": {
                "episodes": ["NLDO-P010", "NLDO-P015"],
                "branches": 6,
                "wording_variants": 3,
                "memory_views": 5,
                "formal_numerical_seeds": 3,
                "formal_provider_calls": 0,
            },
        },
        "row_counts": row_counts,
        "source_manifest": source_manifest,
        "excluded": [
            "hidden deltas and hidden checker/objective implementations",
            "raw populations and candidate archives",
            "held-out reference archive contents",
            "provider logs and credentials",
        ],
    }
    (target / "metadata/artifact_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_checksums(target)
    write_zip(target, args.zip_output.resolve())
    print(
        json.dumps(
            {
                "target": str(target),
                "zip": str(args.zip_output.resolve()),
                "files": sum(path.is_file() for path in target.rglob("*")),
                "row_counts": row_counts,
                "zip_sha256": sha256_file(args.zip_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
