"""Offline tests for the chat-first LiveOpt demo agent.

All LLM calls go through demo.fake_client.FakeDemoClient (router, summarizer,
plain chat, and every LiveOpt sub-agent). urllib.request.urlopen is patched
to raise, so any accidental network access fails the test loudly.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from demo.chat import ChatManager, fallback_route, route_message
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

FREEFORM_OPT = {"mode": "freeform", "problem": KNAPSACK_PROBLEM, "tables": KNAPSACK_TABLES}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(url, *args, **kwargs):
        raise AssertionError(f"network access is forbidden in demo tests: {url!r}")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    yield


@pytest.fixture
def manager(tmp_path):
    return ChatManager(tmp_path / "chats", client_factory=fake_client_factory)


def test_plain_chat_multi_turn(manager):
    chat = manager.create_chat(settings={"model": "fake-demo"})
    first = manager.handle_message(chat.chat_id, "Hello, what can you do?")
    second = manager.handle_message(chat.chat_id, "Tell me more.")

    assert first["message"]["action"] == "answer"
    assert second["message"]["action"] == "answer"
    assert "Fake assistant reply" in second["message"]["content"]
    stored = manager.get(chat.chat_id)
    assert len(stored.messages) == 4  # two user + two assistant
    assert stored.session is None


def test_chat_start_optimization_then_update(manager):
    chat = manager.create_chat(settings={"model": "fake-demo"})
    started = manager.handle_message(chat.chat_id, "", optimization=dict(FREEFORM_OPT))

    message = started["message"]
    assert message["action"] == "start_optimization"
    assert message["routed_by"] == "explicit"
    card_types = [card["type"] for card in message["cards"]]
    assert [t for t in card_types if t != "image"] == ["state", "workbench", "pareto"]
    assert "image" in card_types  # rendered result image rides along
    best = next(card["data"]["best"] for card in message["cards"] if card["type"] == "state")
    assert best["solution"]["selected_items"]
    assert manager.get(chat.chat_id).session is not None

    updated = manager.handle_message(chat.chat_id, "Item I1's public value increases by 5.")
    message = updated["message"]
    assert message["action"] == "apply_update"
    assert message["routed_by"] == "llm"  # fake router saw the active session
    card_types = [card["type"] for card in message["cards"]]
    assert [t for t in card_types if t != "image"] == ["localization", "restart", "state", "workbench", "pareto", "ledger"]
    cards = {card["type"]: card["data"] for card in message["cards"]}
    assert cards["localization"]["data_update"] is True
    assert cards["restart"]["restart_skill"] == "warm_restart_v1"
    assert len(cards["ledger"]["entries"]) == 1
    assert cards["state"]["best"]["solution"]["selected_items"]
    assert cards["workbench"]["setup.py"].strip()


def test_router_deterministic_fallback():
    class BrokenRouterClient(FakeDemoClient):
        def _respond(self, system, messages):
            if "router for a LiveOpt chat agent" in system:
                return "not json at all"
            return super()._respond(system, messages)

    client = BrokenRouterClient()
    # No active session: fall back to a plain answer.
    assert fallback_route("hello", has_session=False)["action"] == "answer"
    assert route_message(client, "hello", has_session=False)["action"] == "answer"
    # Active session: fall back to apply_update.
    route = route_message(client, "raise item I1 value by 5", has_session=True)
    assert route["action"] == "apply_update"
    assert route["routed_by"] == "fallback"


def test_fallback_path_through_manager(tmp_path):
    class BrokenRouterClient(FakeDemoClient):
        def _respond(self, system, messages):
            if "router for a LiveOpt chat agent" in system:
                return '{"action": "teleport"}'
            return super()._respond(system, messages)

    manager = ChatManager(tmp_path / "chats", client_factory=lambda **kw: BrokenRouterClient())
    chat = manager.create_chat(settings={"model": "fake-demo"})
    manager.handle_message(chat.chat_id, "", optimization=dict(FREEFORM_OPT))
    result = manager.handle_message(chat.chat_id, "Item I1's value increases by 5.")
    assert result["message"]["action"] == "apply_update"
    assert result["message"]["routed_by"] == "fallback"


def test_persistence_restore_and_reinit(tmp_path):
    data_dir = tmp_path / "chats"
    manager = ChatManager(data_dir, client_factory=fake_client_factory)
    chat = manager.create_chat(settings={"model": "fake-demo", "api_key": "secret-key"})
    manager.handle_message(chat.chat_id, "just chatting")
    manager.handle_message(chat.chat_id, "", optimization=dict(FREEFORM_OPT))

    # The api key must never hit disk.
    on_disk = (data_dir / f"{chat.chat_id}.json").read_text(encoding="utf-8")
    assert "secret-key" not in on_disk

    # A fresh manager restores the chat fully but marks LiveOpt for re-init.
    restored_manager = ChatManager(data_dir, client_factory=fake_client_factory)
    restored = restored_manager.get(chat.chat_id)
    assert restored.needs_reinit is True
    assert restored.session is None
    assert len(restored.messages) == 4
    assert restored.settings.get("api_key") is None

    restarted = restored_manager.restart_optimization(chat.chat_id)
    assert restarted["message"]["kind"] == "optimization_start"
    assert restored.needs_reinit is False
    assert restored.session is not None


def test_fastapi_full_message_pipeline(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from demo.server import create_app

    app = create_app(client_factory=fake_client_factory, data_dir=tmp_path / "chats")
    client = TestClient(app)

    assert client.get("/api/episodes").status_code == 200

    created = client.post("/api/chats", json={"settings": {"model": "fake-demo", "view_mode": "professional"}})
    assert created.status_code == 200
    chat_id = created.json()["chat"]["chat_id"]

    plain = client.post(f"/api/chats/{chat_id}/messages", json={"text": "Hi there"})
    assert plain.status_code == 200, plain.text
    assert plain.json()["message"]["action"] == "answer"

    started = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"text": "start", "optimization": FREEFORM_OPT},
    )
    assert started.status_code == 200, started.text
    assert started.json()["message"]["kind"] == "optimization_start"

    updated = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"text": "Item I1's public value increases by 5."},
    )
    assert updated.status_code == 200, updated.text
    message = updated.json()["message"]
    assert message["kind"] == "optimization_update"
    assert [card["type"] for card in message["cards"] if card["type"] != "image"] == [
        "localization", "restart", "state", "workbench", "pareto", "ledger",
    ]

    status = client.get(f"/api/chats/{chat_id}/status")
    assert status.status_code == 200
    assert status.json()["busy"] is False
    assert status.json()["has_optimization"] is True

    full = client.get(f"/api/chats/{chat_id}")
    assert len(full.json()["messages"]) == 6

    listed = client.get("/api/chats")
    assert any(row["chat_id"] == chat_id for row in listed.json()["chats"])

    deleted = client.delete(f"/api/chats/{chat_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/chats/{chat_id}").status_code == 404
