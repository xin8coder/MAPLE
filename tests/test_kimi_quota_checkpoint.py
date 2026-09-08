from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evo2.agents.kimi_client import KimiQuotaLimitError
from evo2.agents.liveopt_workbench_impl import LiveOptWorkbenchGenerator
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (
    count_jsonl_rows,
    load_resume_episode_records,
    merge_episode_records_with_queue,
    replace_episode_rows,
    run_episode,
)


class _QuotaClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        raise KimiQuotaLimitError(
            "weekly quota exhausted",
            status=403,
            detail="usage limit for this billing cycle",
            limit_scope="weekly",
        )


def test_workbench_does_not_spend_repairs_after_quota() -> None:
    client = _QuotaClient()
    generator = LiveOptWorkbenchGenerator(model="k3[1m]", client=client)
    with pytest.raises(KimiQuotaLimitError):
        generator.generate_project(
            "quota-test",
            "Choose a binary decision and minimize cost.",
            {"tables": {"items": [{"id": "A", "cost": 1}]}},
            max_repairs=3,
        )
    assert client.calls == 1


def test_episode_checkpoint_replacement_is_resume_safe(tmp_path) -> None:
    path = tmp_path / "NLDO" / "evo2_limit0.jsonl"
    replace_episode_rows(path, "NLDO-P010", [{"episode_id": "NLDO-P010", "run_seed": 0, "v": 1}])
    replace_episode_rows(path, "NLDO-P015", [{"episode_id": "NLDO-P015", "run_seed": 0, "v": 2}])
    replace_episode_rows(path, "NLDO-P010", [{"episode_id": "NLDO-P010", "run_seed": 0, "v": 3}])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count_jsonl_rows(path) == 2
    assert {(row["episode_id"], row["v"]) for row in rows} == {
        ("NLDO-P010", 3),
        ("NLDO-P015", 2),
    }


def test_resume_keeps_completed_and_requeues_limited(tmp_path) -> None:
    episodes = [{"episode_id": "NLDO-P010"}, {"episode_id": "NLDO-P015"}]
    summary = {
        "episode_records": [
            {"episode_id": "NLDO-P010", "status": "completed"},
            {"episode_id": "NLDO-P015", "status": "quota_limited"},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    prior = load_resume_episode_records(path, episodes)
    records = [prior["NLDO-P010"]]
    merged = merge_episode_records_with_queue(episodes, records)
    assert [row["status"] for row in merged] == ["completed", "queued"]


def test_formal_runner_records_quota_as_resumable(monkeypatch: pytest.MonkeyPatch) -> None:
    class LimitedGenerator:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def generate_project(self, *args, **kwargs):
            del args, kwargs
            raise KimiQuotaLimitError(
                "monthly quota exhausted",
                status=429,
                detail="monthly usage limit",
                limit_scope="monthly",
            )

    monkeypatch.setattr(
        "scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full.LiveOptWorkbenchGenerator",
        LimitedGenerator,
    )
    args = SimpleNamespace(model="k3[1m]", controller_seed=0, max_patch_repairs=3)
    episode = {
        "episode_id": "NLDO-P010",
        "domain": "green_vrp_multiobjective",
        "family": "routing",
        "public_initial_problem": "route public orders",
        "public_context": {},
        "update_stream": [],
    }
    record, rows = run_episode(args, episode, [0])
    assert rows == []
    assert record["status"] == "quota_limited"
    assert record["resumable"] is True
    assert record["quota_limit"]["limit_scope"] == "monthly"
