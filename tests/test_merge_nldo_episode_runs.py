import pytest

from scripts.experiments.merge_nldo_episode_runs import group_by_episode


def test_merge_rejects_duplicate_episode_seed_by_default() -> None:
    rows = [
        {"episode_id": "NLDO-P001", "run_seed": 0, "value": "first"},
        {"episode_id": "NLDO-P001", "run_seed": 0, "value": "second"},
    ]

    with pytest.raises(SystemExit, match="duplicate row"):
        group_by_episode(rows)


def test_merge_can_select_last_row_for_audited_append_only_retry() -> None:
    rows = [
        {"episode_id": "NLDO-P001", "run_seed": 0, "value": "first"},
        {"episode_id": "NLDO-P001", "run_seed": 0, "value": "second"},
    ]

    grouped = group_by_episode(rows, duplicate_seed_policy="last")

    assert grouped["NLDO-P001"] == [rows[1]]
