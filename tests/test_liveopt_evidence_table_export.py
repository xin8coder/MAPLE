from pathlib import Path

from scripts.experiments.export_liveopt_current_evidence_tables import (
    build_episode_rows,
    build_profile_rows,
    write_episode_tex,
    write_mo_tex,
)
from scripts.experiments.export_liveopt_representative_ablation_rows import (
    summarize_variant,
    write_tex as write_ablation_tex,
)


def _metric(episode_id: str, stage: int, **values: object) -> dict[str, str]:
    row = {
        "episode_id": episode_id,
        "stage_index": str(stage),
        "run_seed": "0",
        "feasible": "true",
    }
    row.update({key: str(value) for key, value in values.items()})
    return row


def test_episode_rows_match_current_six_column_latex_contract(tmp_path: Path) -> None:
    rows = [
        _metric("NLDO-P001", 0, normalized_score=1.0, objective_gap=0.0),
        _metric("NLDO-P001", 1, normalized_score=0.9, objective_gap=2.0),
        _metric("NLDO-P010", 0, normalized_hv=0.8, ideal_gap=0.2, igd=0.1),
        _metric("NLDO-P010", 1, normalized_hv=0.7, ideal_gap=0.3, igd=0.2),
    ]
    episode_rows = build_episode_rows(rows)
    out = tmp_path / "episode.tex"
    write_episode_tex(out, episode_rows, Path("metrics.csv"))
    data_lines = [line for line in out.read_text().splitlines() if line.startswith("P")]
    assert len(data_lines) == 2
    assert all(line.count("&") == 5 for line in data_lines)
    assert "P001 & Selection/allocation & 0.950 & 1.000 & -- & --" in data_lines[0]
    assert "P010 & NLDO-DM, ordered service & 0.750 & 0.250 & 0.750 & 0.150" in data_lines[1]


def test_mo_variance_writer_matches_current_six_column_latex_contract(tmp_path: Path) -> None:
    out = tmp_path / "mo.tex"
    write_mo_tex(
        out,
        [
            {
                "problem": "P010",
                "profile": "NLDO-DM, ordered service",
                "merged_hv": 0.8,
                "seed_hv_mean": 0.8,
                "seed_hv_std": 0.1,
                "seed_hv_var": 0.01,
                "seed_igd_mean": 0.2,
                "seed_igd_std": 0.03,
            }
        ],
        Path("metrics.csv"),
        Path("run.jsonl"),
    )
    row = next(line for line in out.read_text().splitlines() if line.startswith("P010"))
    assert row.count("&") == 5


def test_profile_summary_separates_static_and_dynamic_rows() -> None:
    rows = [
        _metric("NLDO-P001", 0, normalized_score=1.0, objective_gap=0.0),
        _metric("NLDO-P001", 1, normalized_score=0.8, objective_gap=4.0),
        _metric("NLDO-P010", 0, normalized_hv=0.7, igd=0.2),
        _metric("NLDO-P010", 1, normalized_hv=0.9, igd=0.1),
    ]
    profiles, macros = build_profile_rows(rows)
    selection = next(row for row in profiles if row["profile"] == "Selection/allocation")
    routing = next(row for row in profiles if row["profile"] == "Ordered service routing")
    assert selection["overall_mean"] == 0.9
    assert selection["dynamic_mean"] == 0.8
    assert routing["overall_mean"] == 0.8
    assert routing["dynamic_mean"] == 0.9
    assert macros["dynamic_hv"] == 0.9


def test_representative_ablation_uses_state_denominators(tmp_path: Path) -> None:
    run_dirs = []
    for episode_id, values in (("NLDO-P010", (0.8, 0.6)), ("NLDO-P014", (0.9, 0.7))):
        run_dir = tmp_path / episode_id
        metrics = run_dir / "reference_metrics" / "reference_stage_metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(
            "episode_id,stage_index,run_seed,normalized_hv,feasible\n"
            f"{episode_id},1,0,{values[0]},true\n"
            f"{episode_id},1,1,{values[1]},false\n",
            encoding="utf-8",
        )
        run_dirs.append(run_dir)
    row = summarize_variant("LiveOpt", run_dirs[0], run_dirs[1])
    assert row["solved_states"] == 2
    assert row["total_states"] == 2
    assert row["p010_hv"] == 0.7
    assert row["p014_hv"] == 0.8
    row["delta_mean_hv"] = 0.0
    out = tmp_path / "ablation.tex"
    write_ablation_tex(out, [row])
    data = next(line for line in out.read_text().splitlines() if line.startswith("LiveOpt"))
    assert data.count("&") == 5
