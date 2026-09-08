import json
from pathlib import Path

from scripts.analysis.evaluate_reference_metrics import evaluate_stage_sequence


EPISODES = Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")


def test_rejected_episode_suffix_remains_in_metric_denominator() -> None:
    episode = next(
        row
        for row in (json.loads(line) for line in EPISODES.read_text(encoding="utf-8").splitlines() if line.strip())
        if row["episode_id"] == "NLDO-P001"
    )
    stages = [
        {
            "update_id": "initial",
            "solution": {},
            "candidate_archive": [],
            "agent_feasible": False,
            "agent_objective": None,
            "stage_tokens": 0,
            "latency_seconds": 0.0,
        }
    ]

    rows = evaluate_stage_sequence("NLDO", "rejection_test", stages, episode)

    assert len(rows) == 13
    suffix = [row for row in rows if row["stage_type"] == "update"]
    assert len(suffix) == 12
    assert all(row["missing_stage"] for row in suffix)
    assert all(row["feasible"] is False and row["normalized_score"] == 0.0 for row in suffix)
