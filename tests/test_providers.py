from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from agent_lvl.providers import (
    LOCAL_API_URL,
    LOCAL_MODEL,
    ChatResult,
    LocalProvider,
    OpenAICompatibleProvider,
    TokenUsage,
    create_provider,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ модели"))],
            usage=SimpleNamespace(
                prompt_tokens=42,
                completion_tokens=8,
                total_tokens=50,
            ),
        )


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> object:
        return self.payload


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_openai_compatible_provider_uses_model_and_history() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        client=client,
    )
    messages = [{"role": "user", "content": "Привет"}]

    assert provider.model == "test-model"
    assert provider.respond(messages) == ChatResult(
        content="Ответ модели",
        usage=TokenUsage(input_tokens=42, output_tokens=8, total_tokens=50),
    )
    assert completions.calls == [{"model": "test-model", "messages": messages}]


def test_local_provider_sends_configured_payload_and_authentication() -> None:
    response = FakeResponse(
        {
            "message": {"content": "Локальный ответ"},
            "prompt_eval_count": 37,
            "eval_count": 5,
        }
    )
    client = FakeHttpClient(response)
    provider = LocalProvider(
        url="http://localhost:11434/api/chat",
        model="custom-model",
        temperature=0.3,
        max_tokens=512,
        num_predict=128,
        think=True,
        timeout_seconds=10,
        api_key="secret",
        client=cast(httpx.Client, client),
    )
    messages = [{"role": "user", "content": "Привет"}]

    assert provider.model == "custom-model"
    assert provider.respond(messages) == ChatResult(
        content="Локальный ответ",
        usage=TokenUsage(input_tokens=37, output_tokens=5, total_tokens=42),
    )
    assert response.status_checked
    assert client.calls == [
        {
            "url": "http://localhost:11434/api/chat",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
            "json": {
                "model": "custom-model",
                "messages": messages,
                "stream": False,
                "think": True,
                "max_tokens": 512,
                "options": {"temperature": 0.3, "num_predict": 128},
            },
        }
    ]


def test_factory_creates_local_provider_with_defaults() -> None:
    provider = create_provider({"AI_PROVIDER": "local"})

    assert isinstance(provider, LocalProvider)
    assert provider._url == LOCAL_API_URL
    assert provider._model == LOCAL_MODEL


def test_factory_configures_local_context_window() -> None:
    provider = create_provider({"AI_PROVIDER": "local", "LOCAL_NUM_CTX": "512"})

    assert isinstance(provider, LocalProvider)
    assert provider._context_window == 512


def test_factory_creates_qwen_provider() -> None:
    provider = create_provider(
        {
            "AI_PROVIDER": "qwen",
            "DASHSCOPE_API_KEY": "test-key",
            "QWEN_BASE_URL": "https://example.test/v1",
        }
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._model == "qwen-plus"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"AI_PROVIDER": "unknown"}, "неизвестный провайдер"),
        ({"AI_PROVIDER": "deepseek"}, "DEEPSEEK_API_KEY"),
        (
            {"AI_PROVIDER": "qwen", "DASHSCOPE_API_KEY": "test-key"},
            "QWEN_BASE_URL",
        ),
        ({"AI_PROVIDER": "local", "LOCAL_THINK": "maybe"}, "true или false"),
        ({"AI_PROVIDER": "local", "LOCAL_MAX_TOKENS": "0"}, "больше нуля"),
        ({"AI_PROVIDER": "local", "LOCAL_NUM_CTX": "0"}, "больше нуля"),
    ],
)
def test_factory_rejects_invalid_configuration(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_provider(environment)
