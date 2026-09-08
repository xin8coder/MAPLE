"""Export verified public-revision values without changing historical evidence."""

import argparse
import json
import math
from pathlib import Path

from scripts.analysis.compose_public_evaluation_revision import PUBLIC_CURRENT_DESTINATION as DESTINATION, ROOT
from scripts.analysis.rescore_routing_semantics import sha


def protocol_sensitivity_rows(records):
    methods = (("persistent_react", "Persistent ReAct"), ("react_tools", "ReAct"),
               ("optimus", "OptiMUS"), ("orlm", "ORLM"),
               ("optimai_2025", "OptimAI"), ("or_llm_agent_2025", "OR-LLM-Agent"))
    protocols = ("recorded_output_protocol", "mathematical_prefix_diagnostic")
    index = {(r["method"], r["split"], r["protocol"]): r for r in records}
    expected = {(method, split, protocol) for method, _ in methods
                for split in ("DS", "DM") for protocol in protocols}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("Incomplete or duplicated protocol-sensitivity records")
    lines = [r"\newcommand{\OutputProtocolSensitivityRows}{%"]
    for method, label in methods:
        values = []
        for split, count in (("DS", 117), ("DM", 78)):
            for protocol in protocols:
                row = index[method, split, protocol]
                if row["states"] != count or not math.isfinite(row["quality"]):
                    raise ValueError(f"Invalid sensitivity denominator or quality: {method}/{split}")
                values.append(f"{row['quality']:.3f}")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines + ["}"]) + "\n"


def export_protocol_sensitivity(metric_dir):
    source = metric_dir / "verification.json"
    verification = json.loads(source.read_text())
    manifest = metric_dir / "manifest.json"
    expected = verification["inputs"][str(manifest.relative_to(ROOT))]
    if sha(manifest) != expected:
        raise ValueError("Protocol sensitivity belongs to another result manifest")
    content = f"% Saved-plan protocol sensitivity. Source SHA256: {sha(source)}.\n"
    content += protocol_sensitivity_rows(verification["protocol_sensitivity"])
    destination = ROOT / "release_artifacts/paper_table_exports/output_protocol_sensitivity_rows.tex"
    destination.write_text(content)
    print(f"wrote {destination}")


def export(metric_dir=DESTINATION, values_only=False):
    manifest = json.loads((metric_dir / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(metric_dir / name) != expected:
            raise ValueError(f"Metric file changed: {name}")
    index = {(r["method"], r["scope"], r["window"]): r for r in manifest["summary"]}

    def row(method="LiveOpt", scope="DM", window="t00-t12"):
        return index[(method, scope, window)]

    macros = {
        "LiveOptPublicDMHV": row()["hv"],
        "PersistentPublicDMHV": row("Persistent ReAct")["hv"],
        "LiveOptPublicSMHV": row(window="t00")["hv"],
        "PersistentPublicSMHV": row("Persistent ReAct", window="t00")["hv"],
        "KimiPublicDMHV": row("LiveOpt (Kimi k2.7)")["hv"],
        "WarmPublicDMHV": row("Fixed Warm (own history)")["hv"],
        "FullPublicDMHV": row("Fixed Full (own history)")["hv"],
        "LiveOptPublicUpdateHV": row(window="t01-t12")["hv"],
        "WarmPublicUpdateHV": row("Fixed Warm (own history)", window="t01-t12")["hv"],
        "FullPublicUpdateHV": row("Fixed Full (own history)", window="t01-t12")["hv"],
        "NoTSSPublicUpdateHV": row("w/o TSS", window="t01-t12")["hv"],
        "NoTSSPublicConditionalHV": row("w/o TSS", window="t01-t12")["hv_if_valid"],
    }
    comparison = manifest["sequential_warm_comparison"]
    macros.update(PublicWarmGain=comparison["mean_gain"],
                  PublicWarmGainLow=comparison["episode_bootstrap_95ci"][0],
                  PublicWarmGainHigh=comparison["episode_bootstrap_95ci"][1],
                  PublicWarmSignP=comparison["sign_test_two_sided_p"],
                  PublicWarmDelta=-comparison["mean_gain"],
                  PublicFullGain=macros["LiveOptPublicUpdateHV"] - macros["FullPublicUpdateHV"],
                  PublicFullDelta=macros["FullPublicUpdateHV"] - macros["LiveOptPublicUpdateHV"])
    for method, prefix in (("Persistent ReAct", "Persistent"), ("ReAct", "React"), ("OptiMUS", "Optimus"),
                           ("ORLM", "ORLM"), ("OR-LLM-Agent", "ORAgent"), ("OptimAI", "OptimAI")):
        macros[f"{prefix}PublicSMHV"] = row(method, window="t00")["hv"]
        macros[f"{prefix}PublicUpdateHV"] = row(method, window="t01-t12")["hv"]
    lines = [f"% Generated from {manifest['version']}. Original saved agent plans; reference provenance in manifest."]
    lines += [rf"\newcommand{{\{name}}}{{{value:.3f}}}" for name, value in macros.items()]
    lines.append(rf"\newcommand{{\PublicWarmWins}}{{{comparison['wins']}}}")
    (ROOT / "release_artifacts/paper_table_exports/public_evaluation_values.tex").write_text("\n".join(lines) + "\n")
    if values_only:
        print(json.dumps(macros, indent=2))
        return

    lines = ["% Public-objective saved-plan revision, t00--t12.", r"\newcommand{\PublicEvaluationSummaryRows}{%"]
    for method in ("LiveOpt", "Fixed Warm (own history)", "Fixed Full (own history)", "w/o TSS"):
        overall, routing, cloud = row(method), row(method, "Routing"), row(method, "Cloud")
        name = r"\method{}" if method == "LiveOpt" else method.replace(" (own history)", "")
        if method == "LiveOpt":
            lines.append(r"\rowcolor{LiveOptRowAccent}")
        lines.append(f"{name} & {routing['hv']:.3f} & {cloud['hv']:.3f} & {overall['hv']:.3f} & {overall['igd']:.3f} " + r"\\")
    lines.append("}")
    (ROOT / "release_artifacts/paper_table_exports/public_evaluation_summary_rows.tex").write_text("\n".join(lines) + "\n")

    lines = ["% Public-objective revision. SD/variance across ten seed time-means.", r"\newcommand{\LiveOptMOHVVarianceRows}{%"]
    for ep in range(10, 16):
        r = row(scope=f"NLDO-P{ep:03d}")
        family = "Ordered service" if ep <= 12 else "Cloud resources"
        lines.append(f"P{ep:03d} & {family} & {r['hv']:.3f}$\\pm${r['hv_seed_sd']:.3f} & "
                     f"{r['igd']:.3f}$\\pm${r['igd_seed_sd']:.3f} " + r"\\")
    lines.append("}")
    (ROOT / "release_artifacts/paper_table_exports/liveopt_mo_hv_variance_rows.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-dir", type=Path, default=DESTINATION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--values-only", action="store_true", help="Preserve independently styled unchanged tables")
    mode.add_argument("--protocol-sensitivity-only", action="store_true", help="Export saved diagnostics without updating primary results")
    args = parser.parse_args()
    if args.protocol_sensitivity_only:
        export_protocol_sensitivity(args.metric_dir.resolve())
    else:
        export(args.metric_dir.resolve(), values_only=args.values_only)
