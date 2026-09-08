from __future__ import annotations

import json
from io import BytesIO
import urllib.error

import pytest

from evo2.agents import kimi_client as module
from evo2.agents.kimi_client import (
    KimiGenerationLimitError,
    KimiK3Client,
    KimiK3ClientError,
    KimiQuotaLimitError,
    KimiTransportError,
)
from evo2.agents.llm_client import (
    create_llm_client,
    infer_llm_provider,
    is_llm_quota_limit_error,
    llm_quota_limit_record,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeSSE:
    def __init__(self, events: list[str]):
        self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        self.lines = [line.encode("utf-8") for event in events for line in (f"data: {event}\n", "\n")]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""


def _clean_kimi_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "KIMI_API_KEY",
        "KIMI_MODEL",
        "KIMI_BASE_URL",
        "KIMI_CONTEXT_WINDOW",
        "KIMI_AGENT_IDENTITY",
        "KIMI_AGENT_VERSION",
        "KIMI_USER_AGENT",
        "KIMI_REASONING_EFFORT",
        "KIMI_MIN_GENERATION_TOKENS",
        "KIMI_MAX_GENERATION_TOKENS",
        "KIMI_CACHE_ONLY",
        "KIMI_STREAM",
        "KIMI_STREAM_PROGRESS",
        "KIMI_STREAM_INCLUDE_USAGE",
        "KIMI_USE_SYSTEM_PROXY",
        "LLM_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KIMI_CACHE", "0")
    # Most unit tests patch urllib.request.urlopen directly. Production keeps
    # the safer direct/no-ambient-proxy default, which has dedicated coverage.
    monkeypatch.setenv("KIMI_USE_SYSTEM_PROXY", "1")


def test_kimi_client_requires_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    with pytest.raises(KimiK3ClientError, match="KIMI_API_KEY"):
        KimiK3Client()


def test_kimi_client_removes_zero_width_copy_noise(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    client = KimiK3Client(api_key="dummy\u200b-key\ufeff")
    assert client.api_key == "dummy-key"


def test_kimi_request_declares_agent_identity_and_k3_1m(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "choices": [{"message": {"content": '{"answer":15}'}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KimiK3Client(api_key="dummy", timeout=7)
    response = client.chat(
        [{"role": "user", "content": "Solve a tiny integer program."}],
        json_mode=True,
    )

    assert response["choices"][0]["message"]["content"] == '{"answer":15}'
    assert captured["url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["user-agent"] == "LiveOpt-Coding-Agent/0.1.0"
    assert captured["headers"]["x-client-name"] == "LiveOpt-Coding-Agent"
    assert captured["headers"]["x-client-type"] == "coding-agent"
    assert captured["payload"]["model"] == "k3[1m]"
    assert captured["payload"]["temperature"] == 1.0
    assert captured["payload"]["reasoning_effort"] == "medium"
    assert captured["payload"]["max_tokens"] == 32_000
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert "context_window" not in captured["payload"]
    assert client.context_window == 1_048_576
    assert client.min_generation_tokens == 32_000
    assert client.max_generation_tokens == 98_304
    assert client.last_request_headers["Authorization"] == "Bearer <redacted>"


def test_kimi_streaming_assembles_reasoning_content_usage_and_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    events = [
        json.dumps(
            {
                "id": "chat-1",
                "model": "k3[1m]",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "think "}}
                ],
            }
        ),
        json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "again", "content": "{\"ok\":"}}
                ]
            }
        ),
        json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "true}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ),
        "[DONE]",
    ]
    monkeypatch.setattr(module, "_urlopen_no_system_proxy", lambda _req, timeout: _FakeSSE(events))
    client = KimiK3Client(api_key="dummy", timeout=7, max_retries=0)
    response = client.chat([{"role": "user", "content": "return JSON"}], json_mode=True)

    message = response["choices"][0]["message"]
    assert message["content"] == '{"ok":true}'
    assert message["reasoning_content"] == "think again"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 15
    assert client.last_stream_stats["response_mode"] == "sse"
    assert client.last_stream_stats["data_event_count"] == 4
    assert client.last_stream_stats["reasoning_delta_events"] == 2
    assert client.last_stream_stats["content_delta_events"] == 2
    assert client.last_stream_stats["done_seen"] is True


def test_kimi_partial_stream_is_resumable_and_not_automatically_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    calls = 0

    def incomplete(_req, timeout):
        nonlocal calls
        del timeout
        calls += 1
        return _FakeSSE(
            [json.dumps({"choices": [{"index": 0, "delta": {"content": "partial"}}]})]
        )

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", incomplete)
    client = KimiK3Client(api_key="dummy", max_retries=5, retry_backoff=0)
    with pytest.raises(KimiTransportError, match=r"before \[DONE\]") as caught:
        client.chat([{"role": "user", "content": "hello"}])
    assert calls == 1
    assert caught.value.resumable is True
    assert caught.value.stream_stats["data_event_count"] == 1


def test_kimi_urlopen_disables_environment_proxies_by_default(monkeypatch) -> None:
    monkeypatch.delenv("KIMI_USE_SYSTEM_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(module, "_NO_PROXY_OPENER", None)
    seen = {}

    class DummyProxyHandler:
        def __init__(self, proxies):
            seen["proxies"] = proxies

    class DummyOpener:
        def open(self, request, timeout=None):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return "direct-response"

    monkeypatch.setattr(module.urllib.request, "ProxyHandler", DummyProxyHandler)
    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda handler: (seen.setdefault("handler", handler), DummyOpener())[1],
    )
    request = module.urllib.request.Request(
        "https://api.kimi.com/coding/v1/chat/completions"
    )
    assert module._urlopen_no_system_proxy(request, timeout=9) == "direct-response"
    assert seen["proxies"] == {}
    assert seen["timeout"] == 9


def test_kimi_generation_floor_is_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KimiK3Client(api_key="dummy", min_generation_tokens=9000)
    client.chat([{"role": "user", "content": "return compact JSON"}], max_tokens=1200)
    assert captured["payload"]["max_tokens"] == 9000


def test_kimi_truncation_expands_budget_instead_of_repeating_same_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    budgets = []

    def truncated_then_ok(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        budgets.append(payload["max_tokens"])
        if len(budgets) == 1:
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "thinking"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": payload["max_tokens"],
                        "total_tokens": 100 + payload["max_tokens"],
                        "completion_tokens_details": {"reasoning_tokens": 11_900},
                    },
                }
            )
        return _FakeResponse(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 8_000,
                    "total_tokens": 8_100,
                    "completion_tokens_details": {"reasoning_tokens": 7_500},
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", truncated_then_ok)
    client = KimiK3Client(
        api_key="dummy",
        min_generation_tokens=12_000,
        max_generation_tokens=32_000,
        max_retries=2,
        retry_backoff=0,
    )
    response = client.chat([{"role": "user", "content": "write code"}], max_tokens=5000)
    assert response["choices"][0]["message"]["content"] == "done"
    assert budgets == [12_000, 24_000]
    assert response["usage"] == {
        "prompt_tokens": 200,
        "completion_tokens": 20_000,
        "total_tokens": 20_200,
        "completion_tokens_details": {"reasoning_tokens": 19_400},
    }
    assert response["final_attempt_usage"]["completion_tokens"] == 8_000
    assert len(response["generation_attempt_usages"]) == 2


def test_kimi_generation_limit_preserves_provider_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)

    def truncated_at_cap(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "thinking"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": payload["max_tokens"],
                    "total_tokens": 321 + payload["max_tokens"],
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", truncated_at_cap)
    client = KimiK3Client(
        api_key="dummy",
        min_generation_tokens=32_000,
        max_generation_tokens=32_000,
        max_retries=0,
    )
    with pytest.raises(KimiGenerationLimitError) as caught:
        client.chat([{"role": "user", "content": "write code"}], max_tokens=32_000)
    assert caught.value.usage == {
        "prompt_tokens": 321,
        "completion_tokens": 32_000,
        "total_tokens": 32_321,
    }


def test_kimi_higher_cap_reuses_complete_lower_cap_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    monkeypatch.setenv("KIMI_CACHE", "1")
    monkeypatch.setenv("KIMI_CACHE_DIR", str(tmp_path / "cache"))
    network_calls = 0

    def one_lower_cap_response(_request, timeout):
        nonlocal network_calls
        del timeout
        network_calls += 1
        return _FakeResponse(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "complete lower-cap answer"}}
                ],
                "usage": {},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", one_lower_cap_response)
    messages = [{"role": "user", "content": "same prompt"}]
    lower = KimiK3Client(
        api_key="dummy",
        min_generation_tokens=12_000,
        max_generation_tokens=32_000,
    )
    lower.chat(messages, max_tokens=1200)

    higher = KimiK3Client(
        api_key="dummy",
        min_generation_tokens=32_000,
        max_generation_tokens=32_000,
    )
    response = higher.chat(messages, max_tokens=1200)
    assert response["choices"][0]["message"]["content"] == "complete lower-cap answer"
    assert network_calls == 1


def test_kimi_403_reports_actual_identity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)

    def forbidden(_request, timeout):
        del timeout
        raise urllib.error.HTTPError(
            "https://api.kimi.com/coding/v1/chat/completions",
            403,
            "Forbidden",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    client = KimiK3Client(api_key="dummy", max_retries=0)
    with pytest.raises(KimiK3ClientError, match="LiveOpt-Coding-Agent/0.1.0") as caught:
        client.chat([{"role": "user", "content": "hello"}])
    assert caught.value.status == 403


@pytest.mark.parametrize(
    ("status", "message", "scope"),
    [
        (403, "You've reached your usage limit for this billing cycle.", "weekly"),
        (429, "You've reached your usage limit for this period.", "rolling_5h"),
        (429, "You've reached kimi monthly usage limit for this billing cycle.", "monthly"),
    ],
)
def test_kimi_quota_limits_pause_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: int,
    message: str,
    scope: str,
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    calls = 0

    def limited(_request, timeout):
        nonlocal calls
        del timeout
        calls += 1
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        raise urllib.error.HTTPError(
            "https://api.kimi.com/coding/v1/chat/completions",
            status,
            "limited",
            {"Retry-After": "3600"},
            BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", limited)
    client = KimiK3Client(api_key="dummy", max_retries=5, retry_backoff=0)
    with pytest.raises(KimiQuotaLimitError) as caught:
        client.chat([{"role": "user", "content": "hello"}])

    assert calls == 1
    assert caught.value.limit_scope == scope
    assert caught.value.retry_after == "3600"
    assert is_llm_quota_limit_error(caught.value)
    assert llm_quota_limit_record(caught.value)["resumable"] is True


def test_kimi_overload_429_remains_transient(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    calls = 0

    def overloaded_then_ok(_request, timeout):
        nonlocal calls
        del timeout
        calls += 1
        if calls == 1:
            body = json.dumps(
                {"error": {"message": "The engine is currently overloaded, please try again later"}}
            ).encode("utf-8")
            raise urllib.error.HTTPError(
                "https://api.kimi.com/coding/v1/chat/completions",
                429,
                "overloaded",
                {},
                BytesIO(body),
            )
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr("urllib.request.urlopen", overloaded_then_ok)
    client = KimiK3Client(api_key="dummy", max_retries=1, retry_backoff=0)
    assert client.chat([{"role": "user", "content": "hello"}])["choices"][0]["message"]["content"] == "ok"
    assert calls == 2


def test_kimi_exhausted_504_is_resumable_transport_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)

    def gateway_timeout(_request, timeout):
        del timeout
        body = b"<html><h1>504 Gateway Time-out</h1></html>"
        raise urllib.error.HTTPError(
            "https://api.kimi.com/coding/v1/chat/completions",
            504,
            "Gateway Time-out",
            {},
            BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", gateway_timeout)
    client = KimiK3Client(api_key="dummy", max_retries=0)
    with pytest.raises(KimiTransportError, match="transient HTTP 504") as caught:
        client.chat([{"role": "user", "content": "hello"}])
    assert caught.value.resumable is True


def test_provider_factory_selects_kimi_from_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    assert infer_llm_provider("k3[1m]") == "kimi"
    assert infer_llm_provider("deepseek-v4-pro") == "deepseek"
    kimi = create_llm_client(model="k3[1m]", api_key="dummy")
    deepseek = create_llm_client(model="fake")
    assert isinstance(kimi, KimiK3Client)
    assert kimi.model == "k3[1m]"
    assert deepseek.model == "fake"


def test_react_agent_uses_kimi_factory(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clean_kimi_env(monkeypatch, tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", "dummy")
    from evo2.agents.react_optimai_planner import ReActPlanner

    planner = ReActPlanner(model="k3[1m]")
    assert isinstance(planner.client, KimiK3Client)
