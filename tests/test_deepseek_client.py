import pytest

from evo2.agents import deepseek_client as module
from evo2.agents.deepseek_client import DeepSeekClientError, DeepSeekDebugClient


def test_deepseek_urlopen_disables_environment_proxies_by_default(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_USE_SYSTEM_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(module, "_NO_PROXY_OPENER", None)
    seen = {}

    class DummyProxyHandler:
        def __init__(self, proxies):
            seen["proxies"] = proxies

    class DummyOpener:
        def open(self, req, timeout=None):
            seen["timeout"] = timeout
            return "response"

    def fake_build_opener(handler):
        seen["handler"] = handler
        return DummyOpener()

    monkeypatch.setattr(module.urllib.request, "ProxyHandler", DummyProxyHandler)
    monkeypatch.setattr(module.urllib.request, "build_opener", fake_build_opener)
    req = module.urllib.request.Request("https://api.deepseek.com/v1/chat/completions")

    assert module._urlopen_no_system_proxy(req, timeout=7) == "response"
    assert seen["proxies"] == {}
    assert seen["timeout"] == 7


def test_deepseek_urlopen_uses_system_proxy_when_enabled(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_USE_SYSTEM_PROXY", "1")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return "proxied-response"

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    req = module.urllib.request.Request("https://api.deepseek.com/v1/chat/completions")

    assert module._urlopen_no_system_proxy(req, timeout=11) == "proxied-response"
    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["timeout"] == 11


def test_deepseek_default_total_attempts_is_ten(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    monkeypatch.delenv("DEEPSEEK_MAX_RETRIES", raising=False)
    monkeypatch.setenv("DEEPSEEK_MAX_ATTEMPTS", "10")

    client = DeepSeekDebugClient(model="fake")

    assert client.max_retries == 9


def test_deepseek_default_model_is_pro(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    client = DeepSeekDebugClient(api_key="dummy-test-key")

    assert client.model == "deepseek-v4-pro"


def test_deepseek_flash_does_not_force_json_for_code_generation(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    captured = {}

    def fake_chat(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": "### setup.py\n```python\npass\n```"}}]}

    client = DeepSeekDebugClient(api_key="dummy-test-key", model="deepseek-v4-flash")
    monkeypatch.setattr(client, "_chat_payload", fake_chat)

    client.chat([{"role": "user", "content": "write code slots"}])

    assert "response_format" not in captured


def test_deepseek_flash_json_mode_remains_explicit(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    captured = {}

    def fake_chat(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client = DeepSeekDebugClient(api_key="dummy-test-key", model="deepseek-v4-flash")
    monkeypatch.setattr(client, "_chat_payload", fake_chat)

    client.chat([{"role": "user", "content": "write JSON"}], json_mode=True)

    assert captured["response_format"] == {"type": "json_object"}


def test_deepseek_explicit_max_retries_keeps_legacy_retry_semantics(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    monkeypatch.delenv("DEEPSEEK_HARD_TIMEOUT", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOTAL_TIMEOUT", raising=False)
    sleeps = []

    def fail_urlopen(req, timeout=None):
        raise module.urllib.error.URLError("boom")

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", fail_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = DeepSeekDebugClient(model="fake", max_retries=2, retry_backoff=1)

    with pytest.raises(DeepSeekClientError, match="after 3 attempts"):
        client.chat([{"role": "user", "content": "ping"}])

    assert sleeps == [1, 2]


def test_deepseek_env_max_attempts_and_backoff_cap(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    monkeypatch.setenv("DEEPSEEK_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("DEEPSEEK_RETRY_BACKOFF_MAX", "3")
    monkeypatch.delenv("DEEPSEEK_HARD_TIMEOUT", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOTAL_TIMEOUT", raising=False)
    sleeps = []

    def fail_urlopen(req, timeout=None):
        raise module.urllib.error.URLError("boom")

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", fail_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = DeepSeekDebugClient(model="fake", retry_backoff=2)

    with pytest.raises(DeepSeekClientError, match="after 4 attempts"):
        client.chat([{"role": "user", "content": "ping"}])

    assert client.max_retries == 3
    assert sleeps == [2, 3.0, 3.0]


def test_hard_timeout_success_response_is_returned(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_HARD_TIMEOUT", "5")
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")

    def fake_hard_timeout(payload, base_url, api_key, timeout, hard_timeout, cache_path):
        return {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(module, "_chat_payload_hard_timeout", fake_hard_timeout)
    client = DeepSeekDebugClient(model="fake")

    data = client.chat([{"role": "user", "content": "ping"}])

    assert data["choices"][0]["message"]["content"] == "OK"


def test_hard_timeout_child_error_is_traced(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_HARD_TIMEOUT", "5")
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    monkeypatch.setenv("DEEPSEEK_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.setenv("DEEPSEEK_MAX_ATTEMPTS", "1")

    def fake_hard_timeout(payload, base_url, api_key, timeout, hard_timeout, cache_path):
        raise DeepSeekClientError("DeepSeek returned empty message content")

    monkeypatch.setattr(module, "_chat_payload_hard_timeout", fake_hard_timeout)
    client = DeepSeekDebugClient(model="fake")

    with pytest.raises(DeepSeekClientError, match="empty message content"):
        client.chat([{"role": "user", "content": "ping"}])

    rows = [
        module.json.loads(line)
        for line in (tmp_path / "trace" / "deepseek_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "api_error"
    assert "empty message content" in rows[0]["error"]
    assert rows[0]["response"] is None


def test_local_response_cache_hit_counts_zero_actual_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "1")
    monkeypatch.setenv("DEEPSEEK_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_HARD_TIMEOUT", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOTAL_TIMEOUT", raising=False)

    calls = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"OK"}}],'
                b'"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10,'
                b'"prompt_cache_hit_tokens":4,"prompt_cache_miss_tokens":3}}'
            )

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return DummyResponse()

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", fake_urlopen)
    client = DeepSeekDebugClient(model="fake")
    messages = [{"role": "user", "content": "ping"}]

    first = client.chat(messages)
    second = client.chat(messages)

    assert len(calls) == 1
    assert first["usage"]["total_tokens"] == 10
    assert second["usage"]["total_tokens"] == 0
    assert second["usage"]["local_cache_hits"] == 1
    assert second["usage"]["local_cache_saved_tokens"] == 10
    assert second["usage"]["local_cache_provider_prompt_cache_hit_tokens"] == 4
    assert second["choices"][0]["message"]["content"] == "OK"


def test_deepseek_cache_only_blocks_uncached_request(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "1")
    monkeypatch.setenv("DEEPSEEK_CACHE_ONLY", "1")
    monkeypatch.setenv("DEEPSEEK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEPSEEK_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.delenv("DEEPSEEK_HARD_TIMEOUT", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOTAL_TIMEOUT", raising=False)

    def unexpected_urlopen(req, timeout=None):
        raise AssertionError("cache-only mode must not call the provider on miss")

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", unexpected_urlopen)
    client = DeepSeekDebugClient(model="fake")

    with pytest.raises(DeepSeekClientError, match="CACHE_ONLY"):
        client.chat([{"role": "user", "content": "uncached"}])

    trace_lines = (tmp_path / "trace" / "deepseek_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    assert '"event": "local_cache_miss_blocked"' in trace_lines[0]


def test_deepseek_trace_records_prompt_and_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "1")
    monkeypatch.setenv("DEEPSEEK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEPSEEK_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.delenv("DEEPSEEK_CACHE_ONLY", raising=False)
    monkeypatch.delenv("DEEPSEEK_HARD_TIMEOUT", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOTAL_TIMEOUT", raising=False)

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"OK"}}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", lambda req, timeout=None: DummyResponse())
    client = DeepSeekDebugClient(model="fake")
    messages = [{"role": "user", "content": "trace me"}]

    client.chat(messages)
    client.chat(messages)

    rows = [
        module.json.loads(line)
        for line in (tmp_path / "trace" / "deepseek_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["api_response", "local_cache_hit"]
    assert rows[0]["payload"]["messages"] == messages
    assert rows[0]["response"]["choices"][0]["message"]["content"] == "OK"
    assert rows[1]["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("hard_timeout", [False, True])
def test_balance_failure_stops_without_retry_and_preserves_quota_metadata(monkeypatch, hard_timeout):
    import io
    from evo2.agents.llm_client import is_llm_quota_limit_error, llm_quota_limit_record

    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(module, "_cache_only_enabled", lambda: False)
    monkeypatch.setattr(module, "_hard_timeout_seconds", lambda: 3 if hard_timeout else None)
    monkeypatch.setattr(module, "_write_deepseek_trace", lambda **k: None)
    monkeypatch.setattr(module.DeepSeekDebugClient, "_cache_path", lambda *a: None)
    monkeypatch.setattr(module.time, "sleep", lambda *a: pytest.fail("balance failures must not retry"))
    calls = []

    def quota_response(*a, **k):
        calls.append(1)
        raise module.urllib.error.HTTPError(
            "https://example.invalid", 402, "Payment Required", {},
            io.BytesIO(b'{"error":{"message":"Insufficient Balance"}}'),
        )

    monkeypatch.setattr(module, "_urlopen_no_system_proxy", quota_response)
    client = DeepSeekDebugClient(api_key="offline-test", model="fake", max_retries=4)
    with pytest.raises(module.DeepSeekQuotaLimitError) as caught:
        client.chat([{"role": "user", "content": "test"}], json_mode=True)
    if not hard_timeout:  # The guarded path invokes the stub in a child process.
        assert len(calls) == 1
    wrapper = RuntimeError("controller stopped")
    wrapper.__cause__ = caught.value
    assert is_llm_quota_limit_error(wrapper)
    record = llm_quota_limit_record(wrapper)
    assert record["http_status"] == 402
    assert record["provider"] == "deepseek"
    assert record["limit_scope"] == "balance"
    assert record["resumable"] is True
