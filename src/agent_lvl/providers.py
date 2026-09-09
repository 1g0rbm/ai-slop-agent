"""Адаптеры API для поддерживаемых провайдеров моделей."""

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

Message = dict[str, str]


@dataclass(frozen=True)
class TokenUsage:
    """Число токенов, фактически учтённых провайдером."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    """Ответ модели вместе с доступной статистикой токенов."""

    content: str
    usage: TokenUsage | None = None


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LOCAL_API_URL = "http://10.36.201.230:11435/api/chat"
LOCAL_MODEL = "gemma3:27b"


class ChatProvider(Protocol):
    """Интерфейс клиента, который отвечает на историю сообщений."""

    @property
    def model(self) -> str:
        """Вернуть название используемой модели."""
        ...

    def respond(self, messages: list[Message]) -> ChatResult:
        """Вернуть ответ модели и доступную статистику токенов."""
        ...


class OpenAICompatibleProvider:
    """Клиент OpenAI-совместимых API DeepSeek и Qwen."""

    def __init__(
        self, base_url: str, api_key: str, model: str, client: Any = None
    ) -> None:
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def model(self) -> str:
        """Вернуть название используемой модели."""
        return self._model

    def respond(self, messages: list[Message]) -> ChatResult:
        """Отправить историю через OpenAI-совместимый API."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("модель вернула пустой ответ")

        usage = completion.usage
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        return ChatResult(content=content, usage=token_usage)


class LocalProvider:
    """Клиент локального API в формате Ollama."""

    def __init__(
        self,
        url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        num_predict: int,
        think: bool,
        timeout_seconds: float,
        api_key: str | None = None,
        context_window: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_predict = num_predict
        self._think = think
        self._api_key = api_key
        self._context_window = context_window
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    @property
    def model(self) -> str:
        """Вернуть название используемой модели."""
        return self._model

    def respond(self, messages: list[Message]) -> ChatResult:
        """Отправить историю в локальный API и вернуть ответ."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        options = {
            "temperature": self._temperature,
            "num_predict": self._num_predict,
        }
        if self._context_window is not None:
            options["num_ctx"] = self._context_window

        response = self._client.post(
            self._url,
            headers=headers,
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": self._think,
                "max_tokens": self._max_tokens,
                "options": options,
            },
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as error:
            message = "локальный API вернул ответ в неизвестном формате"
            raise RuntimeError(message) from error

        if not isinstance(content, str) or not content:
            raise RuntimeError("локальный API вернул пустой ответ")

        input_tokens = _token_count(data.get("prompt_eval_count"))
        output_tokens = _token_count(data.get("eval_count"))
        usage = None
        if input_tokens is not None and output_tokens is not None:
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return ChatResult(content=content, usage=usage)


def create_provider(environ: Mapping[str, str] | None = None) -> ChatProvider:
    """Создать провайдера из переменных окружения."""
    environment = os.environ if environ is None else environ
    provider_name = environment.get("AI_PROVIDER", "deepseek").lower()

    if provider_name == "deepseek":
        return OpenAICompatibleProvider(
            base_url=environment.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            api_key=_required(environment, "DEEPSEEK_API_KEY"),
            model=environment.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
        )
    if provider_name == "qwen":
        return OpenAICompatibleProvider(
            base_url=_required(environment, "QWEN_BASE_URL"),
            api_key=_required(environment, "DASHSCOPE_API_KEY"),
            model=environment.get("QWEN_MODEL", "qwen-plus"),
        )
    if provider_name == "local":
        return LocalProvider(
            url=environment.get("LOCAL_API_URL", LOCAL_API_URL),
            model=environment.get("LOCAL_MODEL", LOCAL_MODEL),
            temperature=_float(environment, "LOCAL_TEMPERATURE", 0.0),
            max_tokens=_positive_int(environment, "LOCAL_MAX_TOKENS", 2048),
            num_predict=_positive_int(environment, "LOCAL_NUM_PREDICT", 300),
            think=_bool(environment, "LOCAL_THINK", False),
            timeout_seconds=_positive_float(
                environment, "LOCAL_TIMEOUT_SECONDS", 120.0
            ),
            api_key=environment.get("LOCAL_API_KEY") or None,
            context_window=_optional_positive_int(environment, "LOCAL_NUM_CTX"),
        )
    raise ValueError(f"неизвестный провайдер AI_PROVIDER: {provider_name}")


def _token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not value:
        raise ValueError(f"не задана переменная окружения {name}")
    return value


def _float(environment: Mapping[str, str], name: str, default: float) -> float:
    value = environment.get(name)
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"переменная {name} должна быть числом") from error
    if not math.isfinite(result):
        raise ValueError(f"переменная {name} должна быть конечным числом")
    return result


def _positive_float(environment: Mapping[str, str], name: str, default: float) -> float:
    result = _float(environment, name, default)
    if result <= 0:
        raise ValueError(f"переменная {name} должна быть больше нуля")
    return result


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    value = environment.get(name)
    if value is None:
        return default
    return _parse_positive_int(value, name)


def _optional_positive_int(environment: Mapping[str, str], name: str) -> int | None:
    value = environment.get(name)
    if value is None:
        return None
    return _parse_positive_int(value, name)


def _parse_positive_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"переменная {name} должна быть целым числом") from error
    if result <= 0:
        raise ValueError(f"переменная {name} должна быть больше нуля")
    return result


def _bool(environment: Mapping[str, str], name: str, default: bool) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"переменная {name} должна быть true или false")
