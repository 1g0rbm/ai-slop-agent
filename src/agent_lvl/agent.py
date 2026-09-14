"""Оркестрация диалога и переключаемых стратегий контекста."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from .context import DEFAULT_SUMMARY_BATCH_MESSAGES
from .history import (
    DEFAULT_HISTORY_LIMIT,
    Exchange,
    OperationTotals,
    SQLiteHistory,
    TokenTotals,
)
from .providers import ChatProvider, TokenUsage
from .strategies import (
    BranchingStrategy,
    ContextStrategy,
    MaintenanceCall,
    RollingSummaryStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)

DEFAULT_SYSTEM_PROMPT = "Ты полезный AI-ассистент. Отвечай ясно и кратко."


@dataclass(frozen=True)
class AgentResponse:
    """Ответ агента и статистика токенов после его сохранения."""

    content: str
    usage: TokenUsage | None
    conversation_totals: TokenTotals
    maintenance_usage: dict[str, OperationTotals]


class Agent:
    """Диалоговый агент с постоянной историей и стратегиями контекста."""

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
        self._clock = clock or (lambda: datetime.now(UTC))
        common = {
            "provider": provider,
            "history": self._history,
            "history_limit": history_limit,
            "system_prompt": system_prompt,
        }
        strategies: Sequence[ContextStrategy] = (
            SlidingWindowStrategy(**common),
            StickyFactsStrategy(**common),
            BranchingStrategy(**common),
            RollingSummaryStrategy(
                **common,
                summary_batch_exchanges=summary_batch_messages // 2,
                clock=self._clock,
            ),
        )
        self._strategies = {strategy.name: strategy for strategy in strategies}

    @property
    def strategy(self) -> str:
        """Вернуть имя активной стратегии."""
        return self._history.strategy()

    def set_strategy(self, name: str) -> None:
        """Сохранить выбранную стратегию диалога."""
        self._history.set_strategy(name.lower())

    def respond(self, text: str) -> str:
        """Отправить сообщение, сохранить и вернуть текст ответа."""
        return self.respond_with_usage(text).content

    def respond_with_usage(self, text: str) -> AgentResponse:
        """Отправить сообщение и вернуть ответ со статистикой токенов."""
        if not text.strip():
            raise ValueError("сообщение пользователя не должно быть пустым")
        request_id = str(uuid4())
        user_created_at = self._clock()
        strategy = self._strategies[self._history.strategy()]
        prepared = strategy.prepare(text, request_id)
        result = self._provider.respond(prepared.messages)
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
            ),
            request_id=request_id,
            fact_operations=prepared.fact_operations,
        )
        return AgentResponse(
            content=result.content,
            usage=usage,
            conversation_totals=self._history.token_totals(),
            maintenance_usage=_maintenance_totals(prepared.maintenance_calls),
        )


def _maintenance_totals(
    calls: Sequence[MaintenanceCall],
) -> dict[str, OperationTotals]:
    """Сгруппировать служебные токены текущего ответа по операциям."""
    result: dict[str, OperationTotals] = {}
    for operation in {call.operation for call in calls}:
        operation_calls = [call for call in calls if call.operation == operation]
        usages = [call.usage for call in operation_calls if call.usage is not None]
        result[operation] = OperationTotals(
            operation=operation,
            input_tokens=sum(usage.input_tokens for usage in usages),
            output_tokens=sum(usage.output_tokens for usage in usages),
            total_tokens=sum(usage.total_tokens for usage in usages),
            untracked_operations=sum(call.usage is None for call in operation_calls),
            operations=len(operation_calls),
        )
    return result
