from __future__ import annotations

import pytest

from scripts.experiments import export_dynamic_public_baseline_view_summary as summary


def _episode(episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "update_stream": [{"update_id": f"{episode_id}-S01"}, {"update_id": f"{episode_id}-S02"}],
    }


def test_episode_filter_limits_the_scoring_denominator() -> None:
    episodes = [_episode("NLDO-P004"), _episode("NLDO-P007"), _episode("NLDO-P010")]
    selected = summary.select_episodes(episodes, ["NLDO-P004", "NLDO-P007"])

    result = summary.summarize_method("stateful", selected, {})

    assert [episode["episode_id"] for episode in selected] == ["NLDO-P004", "NLDO-P007"]
    assert result["NLDO (SS)"]["stages"] == 2
    assert result["NLDO (DS)"]["stages"] == 4
    assert result["NLDO (SM)"]["stages"] == 0
    assert result["NLDO (DM)"]["stages"] == 0


def test_episode_filter_rejects_unknown_ids() -> None:
    with pytest.raises(SystemExit, match="NLDO-P999"):
        summary.select_episodes([_episode("NLDO-P004")], ["NLDO-P999"])
