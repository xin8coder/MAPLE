"""Offline tests for the LiveOpt demo session and FastAPI app.

All LLM calls go through demo.fake_client.FakeDemoClient, whose canned
responses satisfy the real parsers in evo2.agents (workbench slots, localizer
JSON, data-patch JSON, semantic-gate JSON). urllib.request.urlopen is patched
to raise, so any accidental network access fails the test loudly.
"""

from __future__ import annotations

import urllib.request

import pytest

from demo.episode_upload import freeform_episode
from demo.fake_client import FakeDemoClient, fake_client_factory
from demo.session import DemoSession

KNAPSACK_PROBLEM = (
    "Select a subset of items maximizing total value subject to total weight "
    "not exceeding the capacity in the constraints table. Items live in the "
    "items table with columns id, value, weight."
)

KNAPSACK_TABLES = {
    "items": "id,value,weight\nI1,10,4\nI2,8,5\nI3,6,3\n",
    "constraints": "capacity,objective_mode\n8,maximize_value\n",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(url, *args, **kwargs):
        raise AssertionError(f"network access is forbidden in demo tests: {url!r}")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    yield


def make_session() -> DemoSession:
    episode = freeform_episode(KNAPSACK_PROBLEM, KNAPSACK_TABLES)
    return DemoSession.create(
        task_id=episode.episode_id,
        natural_language_problem=episode.problem,
        public_context=episode.public_context,
        model="fake-demo",
        population_size=12,
        generations=6,
        seed=0,
        client_factory=fake_client_factory,
    )


def test_session_create_commits_initial_state():
    session = make_session()
    state = session.state_summary()
    assert state["turn"] == 0
    best = state["accepted"]["best"]
    assert best is not None
    assert best["solution"]["selected_items"]
    assert best["feasible"] is True
    assert "setup.py" in state["slots"] and "fitness.py" in state["slots"]


def test_apply_update_advances_state_and_records_restart():
    session = make_session()
    ledger_before = len(session.runner.public_update_history)
    result = session.apply_update("u001", "Item I1's public value increases by 5.")

    assert result["update_id"] == "u001"
    assert result["turn"] == 1
    # Localization mask: the fake localizer reports a data-only update.
    assert result["localization"]["data_update"] is True
    assert result["localization"]["patch_setup"] is False
    assert result["localization"]["patch_fitness"] is False
    # The semantic gate (fake: low risk) keeps the default fixed-warm action.
    assert result["restart"]["restart_skill"] == "warm_restart_v1"
    assert result["restart"]["reason"]
    assert result["restart"]["metadata"]["restart_selection_rule"] == "verified_semantic_full_else_fixed_warm"
    # The fake data patch rewrites item I1's value in the live context.
    assert result["data_patch"]["operations"][0]["op"] == "update_row"
    items = {row["id"]: row for row in session.public_context["tables"]["items"]}
    assert float(items["I1"]["value"]) == 15.0
    # LSM ledger grows by exactly one accepted entry.
    ledger = session.runner.public_update_history
    assert len(ledger) == ledger_before + 1
    assert ledger[-1]["update_id"] == "u001"
    # A new accepted state is committed and the trace is kept per turn.
    state = session.state_summary()
    assert state["turn"] == 1
    assert state["accepted"]["best"] is not None
    trace = session.trace_record(1)
    assert trace["patch_trace"]["prompts"] and trace["patch_trace"]["raw_responses"]
    assert state["restart_history"][0]["restart_skill"] == "warm_restart_v1"


def test_fake_client_records_all_sub_agent_calls():
    session = make_session()
    session.apply_update(None, "Item I1 is now worth 15.")
    client = session.client
    assert isinstance(client, FakeDemoClient)
    systems = [
        next(m["content"] for m in payload["messages"] if m["role"] == "system")
        for payload in client.payloads
    ]
    joined = "\n".join(systems)
    assert "fill setup.py and fitness.py slots" in joined  # workbench generation
    assert "Classify dynamic optimization updates" in joined  # localizer
    assert "Patch public optimization data" in joined  # data patcher
    assert "semantic channel" in joined  # semantic restart gate
