"""Сущность агента, хранящая историю диалога."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .context import (
    DEFAULT_SUMMARY_BATCH_MESSAGES,
    build_summary_messages,
    summary_context_message,
)
from .history import (
    DEFAULT_HISTORY_LIMIT,
    ConversationSummary,
    Exchange,
    SQLiteHistory,
    TokenTotals,
)
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
        summary_batch_messages: int = DEFAULT_SUMMARY_BATCH_MESSAGES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if history_limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if summary_batch_messages <= 0 or summary_batch_messages % 2 != 0:
            raise ValueError(
                "размер блока суммаризации должен быть положительным и чётным"
            )
        self._provider = provider
        self._history = history or SQLiteHistory()
        self._history_limit = history_limit
        self._system_prompt = system_prompt
        self._summary_batch_exchanges = summary_batch_messages // 2
        self._clock = clock or (lambda: datetime.now(UTC))

    def respond(self, text: str) -> str:
        """Отправить сообщение, сохранить и вернуть текст ответа."""
        return self.respond_with_usage(text).content

    def respond_with_usage(self, text: str) -> AgentResponse:
        """Отправить сообщение и вернуть ответ со статистикой токенов."""
        self._refresh_summary()
        summary = self._history.summary()
        checkpoint = summary.through_exchange_id if summary else 0

        messages: list[Message] = [{"role": "system", "content": self._system_prompt}]
        if summary is not None:
            messages.append(summary_context_message(summary))
        for item in self._history.recent_after(checkpoint, self._history_limit):
            exchange = item.exchange
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

    def _refresh_summary(self) -> None:
        while batch := self._history.summary_batch(
            self._history_limit, self._summary_batch_exchanges
        ):
            previous = self._history.summary()
            result = self._provider.respond(build_summary_messages(previous, batch))
            summarized_before = previous.summarized_messages if previous else 0
            self._history.save_summary(
                ConversationSummary(
                    content=result.content,
                    through_exchange_id=batch[-1].id,
                    summarized_messages=summarized_before + len(batch) * 2,
                    model=self._provider.model,
                    updated_at=self._clock(),
                )
            )
