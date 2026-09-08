"""Offline tests for data ingestion, result visualization, and view modes."""

from __future__ import annotations

import base64
import urllib.request

import pytest

from demo.chat import ChatManager, filter_message_for_view
from demo.fake_client import FakeDemoClient, fake_client_factory
from demo.ingest import (
    IngestError,
    convert_data_to_episode,
    ingest_files_and_goal,
    load_imported_episode,
    parse_json_tables,
    parse_markdown_table,
    parse_table_text,
    parse_uploaded_file,
    validate_tables,
)
from demo.viz import matplotlib_available, result_image_cards

GOAL = "Maximize total value of selected items under the capacity limit."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(url, *args, **kwargs):
        raise AssertionError(f"network access is forbidden in demo tests: {url!r}")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    yield


# ------------------------------------------------------------------ parsers


def test_parse_csv_and_tsv():
    csv_tables = parse_uploaded_file("items.csv", "id,value\nI1,10\nI2,8\n")
    assert csv_tables["items"] == [{"id": "I1", "value": 10}, {"id": "I2", "value": 8}]
    tsv_tables = parse_uploaded_file("caps.tsv", "name\tcapacity\nA\t8\n")
    assert tsv_tables["caps"] == [{"name": "A", "capacity": 8}]


def test_parse_markdown_table():
    rows = parse_markdown_table("| id | value |\n| --- | --- |\n| I1 | 10 |\n| I2 | 8 |\n")
    assert rows == [{"id": "I1", "value": 10}, {"id": "I2", "value": 8}]


def test_parse_json_variants():
    as_list = parse_json_tables('[{"id": "I1", "value": 10}]')
    assert as_list["data"] == [{"id": "I1", "value": 10}]
    as_columns = parse_json_tables('{"id": ["I1", "I2"], "value": [10, 8]}')
    assert as_columns["data"] == [{"id": "I1", "value": 10}, {"id": "I2", "value": 8}]
    as_tables = parse_json_tables('{"items": [{"id": "I1"}], "constraints": [{"capacity": 8}]}')
    assert set(as_tables) == {"items", "constraints"}


def test_parse_pasted_text_autodetect():
    markdown = parse_table_text("| id | value |\n|---|---|\n| I1 | 10 |")
    assert markdown["table"][0]["id"] == "I1"
    csv_text = parse_table_text("id,value\nI1,10\n")
    assert csv_text["table"][0]["value"] == 10
    aligned = parse_table_text("id   value\nI1   10\nI2   8\n")
    assert aligned["table"][1]["id"] == "I2"


def test_xlsx_without_openpyxl():
    import importlib.util

    if importlib.util.find_spec("openpyxl") is not None:
        pytest.skip("openpyxl installed in this environment")
    with pytest.raises(IngestError, match="openpyxl"):
        parse_uploaded_file("data.xlsx", b"PK\x03\x04fake")


def test_validate_tables_rejects_non_scalar():
    with pytest.raises(IngestError, match="scalar"):
        validate_tables({"bad": [{"id": "I1", "nested": {"x": 1}}]})
    with pytest.raises(IngestError, match="non-empty"):
        validate_tables({"empty": []})


# ------------------------------------------------------- conversion (fake)


def test_llm_conversion_builds_valid_episode():
    client = FakeDemoClient()
    tables = {
        "items": [{"id": "I1", "value": 10, "weight": 4}],
        "constraints": [{"capacity": 8}],
    }
    episode = convert_data_to_episode(client, GOAL, tables)
    assert episode.episode_id == "imported_knapsack"
    assert "Scoring rules:" in episode.problem
    contract = episode.public_context["optimization_contract"]
    assert contract["version"] == "public_quantitative_objective_contract_v1"
    assert contract["objective_sense"] == "maximize"
    assert contract["objective_names"] == ["total_value"]
    assert episode.public_context["multi_objective"] is False


def test_ingest_mixed_files_and_pasted():
    client = FakeDemoClient()
    episode = ingest_files_and_goal(
        client,
        files={"items.csv": "id,value,weight\nI1,10,4\nI2,8,5\nI3,6,3\n"},
        pasted_text="| capacity | objective_mode |\n|---|---|\n| 8 | maximize_value |",
        goal=GOAL,
    )
    assert episode.public_context["tables"]["items"]
    # The fake conversion model returns its canned cleaned tables.
    assert len(episode.public_context["tables"]["items"]) == 3


def test_ingest_requires_goal_and_data():
    client = FakeDemoClient()
    with pytest.raises(IngestError, match="goal"):
        ingest_files_and_goal(client, files={"a.csv": "x,y\n1,2\n"}, goal="")
    with pytest.raises(IngestError, match="no tabular data"):
        ingest_files_and_goal(client, goal=GOAL)


def test_load_imported_episode_validates():
    with pytest.raises(IngestError, match="public_context"):
        load_imported_episode({"episode_id": "x"})
    payload = {
        "episode_id": "ok",
        "public_initial_problem": "Maximize value.",
        "public_context": {"tables": {"items": [{"id": "I1", "value": 10}]}},
    }
    episode = load_imported_episode(payload)
    assert episode.episode_id == "ok"
    assert episode.public_context["tables"]["items"] == [{"id": "I1", "value": 10}]


# ------------------------------------------------------- chat integration


def test_chat_import_preview_then_confirm(tmp_path):
    manager = ChatManager(tmp_path / "chats", client_factory=fake_client_factory)
    chat = manager.create_chat(settings={"model": "fake-demo"})

    preview = manager.handle_message(
        chat.chat_id,
        "",
        import_data={
            "goal": GOAL,
            "files": {"items.csv": "id,value,weight\nI1,10,4\nI2,8,5\nI3,6,3\n"},
            "pasted_text": "| capacity | objective_mode |\n|---|---|\n| 8 | maximize_value |",
        },
    )
    message = preview["message"]
    assert message["action"] == "import_data"
    assert message["kind"] == "import_preview"
    card = message["cards"][0]
    assert card["type"] == "episode_preview"
    assert card["data"]["tables"]["items"]["rows"] == 3
    assert chat.session is None  # preview does not start optimization

    started = manager.handle_message(
        chat.chat_id,
        "",
        optimization={"mode": "imported", "episode_payload": card["data"]["episode_payload"]},
    )
    assert started["message"]["action"] == "start_optimization"
    assert manager.get(chat.chat_id).session is not None


def test_chat_import_error_card(tmp_path):
    manager = ChatManager(tmp_path / "chats", client_factory=fake_client_factory)
    chat = manager.create_chat(settings={"model": "fake-demo"})
    result = manager.handle_message(chat.chat_id, "", import_data={"goal": "", "files": {}, "pasted_text": ""})
    assert result["message"]["kind"] == "import_error"
    assert result["message"]["cards"][0]["type"] == "error"


# ------------------------------------------------------------------- viz


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_scalar_image_card_is_png():
    accepted = {
        "best": {
            "scalar": -21.0,
            "feasible": True,
            "objectives": [-21.0],
            "solution": {"selected_items": ["I1", "I3"], "total_value": 21},
        },
        "archive": [],
    }
    cards = result_image_cards(accepted, ["negative_value"])
    assert cards and cards[0]["type"] == "image"
    payload = cards[0]["data"]["image"]
    assert payload.startswith("data:image/png;base64,")
    assert base64.b64decode(payload.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_pareto_image_card_is_png():
    points = [{"objectives": [1.0, 4.0]}, {"objectives": [2.0, 3.0]}, {"objectives": [3.5, 1.0, 0.5]}]
    accepted = {"best": {"scalar": 1.0, "feasible": True, "solution": {"plan": "x"}}, "archive": points}
    cards = result_image_cards(accepted, ["cost", "emissions"])
    assert cards[0]["data"]["caption"].startswith("Pareto front")
    assert base64.b64decode(cards[0]["data"]["image"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


# -------------------------------------------------------------- view modes


def test_view_mode_filtering():
    message = {
        "role": "assistant",
        "content": "summary",
        "cards": [
            {"type": "image", "title": "Result", "data": {"image": "data:image/png;base64,AA=="}},
            {"type": "state", "title": "Committed plan", "data": {}},
            {"type": "workbench", "title": "Workbench", "data": {}},
        ],
        "state": {"big": True},
        "result": {"big": True},
    }
    normal = filter_message_for_view(message, "normal")
    assert [card["type"] for card in normal["cards"]] == ["image"]
    assert "state" not in normal and "result" not in normal
    preview = {
        "role": "assistant",
        "content": "preview",
        "cards": [
            {"type": "episode_preview", "title": "p", "data": {}},
            {"type": "error", "title": "e", "data": {}},
            {"type": "state", "title": "s", "data": {}},
        ],
    }
    assert [card["type"] for card in filter_message_for_view(preview, "normal")["cards"]] == ["episode_preview", "error"]
    pro = filter_message_for_view(message, "professional")
    assert [card["type"] for card in pro["cards"]] == ["image", "state", "workbench"]
    assert pro["state"] == {"big": True}


def test_fastapi_view_modes_and_import(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from demo.server import create_app

    app = create_app(client_factory=fake_client_factory, data_dir=tmp_path / "chats")
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"settings": {"model": "fake-demo"}}).json()["chat"]["chat_id"]

    preview = client.post(
        f"/api/chats/{chat_id}/messages",
        json={
            "text": "import",
            "import_data": {"goal": GOAL, "files": {"items.csv": "id,value,weight\nI1,10,4\nI2,8,5\nI3,6,3\n"}},
        },
    )
    assert preview.status_code == 200, preview.text
    card = preview.json()["message"]["cards"][0]
    assert card["type"] == "episode_preview"

    started = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"text": "start", "optimization": {"mode": "imported", "episode_payload": card["data"]["episode_payload"]}},
    )
    assert started.status_code == 200, started.text

    # Default view is normal: image/preview/error cards only, no heavy payloads.
    message = started.json()["message"]
    assert message["cards"], "normal view should keep the result image"
    assert all(card["type"] == "image" for card in message["cards"])
    assert "state" not in message

    # Switch to professional: all cards plus the state payload.
    client.put(f"/api/chats/{chat_id}/settings", json={"settings": {"view_mode": "professional"}})
    updated = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"text": "Item I1's value increases by 5."},
    )
    message = updated.json()["message"]
    types = [card["type"] for card in message["cards"]]
    assert "localization" in types and "restart" in types and "image" in types
    assert "result" in message

    # GET respects the stored view as well.
    client.put(f"/api/chats/{chat_id}/settings", json={"settings": {"view_mode": "normal"}})
    messages = client.get(f"/api/chats/{chat_id}").json()["messages"]
    for msg in messages:
        assert all(card["type"] in {"image", "episode_preview", "error"} for card in msg.get("cards", []))
