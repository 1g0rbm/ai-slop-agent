"""Сущность агента, хранящая историю диалога."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .history import DEFAULT_HISTORY_LIMIT, Exchange, SQLiteHistory, TokenTotals
from .providers import ChatProvider, Message, TokenUsage

DEFAULT_SYSTEM_PROMPT = "Ты полезный AI-ассистент. Отвечай ясно и кратко."


@dataclass(frozen=True)
class AgentResponse:
    """Ответ агента и статистика токенов после его сохранения."""

    content: str
    usage: TokenUsage | None
    conversation_totals: TokenTotals


class Agent:
    """Диалоговый агент с постоянной историей одного чата."""

    def __init__(
        self,
        provider: ChatProvider,
        history: SQLiteHistory | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if history_limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        self._provider = provider
        self._history = history or SQLiteHistory()
        self._history_limit = history_limit
        self._system_prompt = system_prompt
        self._clock = clock or (lambda: datetime.now(UTC))

    def respond(self, text: str) -> str:
        """Отправить сообщение, сохранить и вернуть текст ответа."""
        return self.respond_with_usage(text).content

    def respond_with_usage(self, text: str) -> AgentResponse:
        """Отправить сообщение и вернуть ответ со статистикой токенов."""
        messages: list[Message] = [{"role": "system", "content": self._system_prompt}]
        for exchange in self._history.recent(self._history_limit):
            messages.extend(
                [
                    {"role": "user", "content": exchange.user_content},
                    {"role": "assistant", "content": exchange.assistant_content},
                ]
            )

        user_created_at = self._clock()
        messages.append({"role": "user", "content": text})
        result = self._provider.respond(messages)
        usage = result.usage
        self._history.add(
            Exchange(
                user_content=text,
                assistant_content=result.content,
                user_created_at=user_created_at,
                assistant_created_at=self._clock(),
                model=self._provider.model,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            )
        )
        return AgentResponse(
            content=result.content,
            usage=usage,
            conversation_totals=self._history.token_totals(),
        )
