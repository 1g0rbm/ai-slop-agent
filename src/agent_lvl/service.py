"""Прикладной API диалога для CLI и будущего веб-интерфейса."""

from dataclasses import dataclass

from .agent import Agent, AgentResponse
from .history import (
    DEFAULT_CONVERSATION_ID,
    VALID_STRATEGIES,
    Branch,
    Checkpoint,
    SQLiteHistory,
)


@dataclass(frozen=True)
class StrategyChange:
    """Результат переключения стратегии разговора."""

    previous: str
    current: str


class ConversationService:
    """Независимый от транспорта API управления разговором."""

    def __init__(self, agent: Agent, history: SQLiteHistory) -> None:
        self._agent = agent
        self._history = history

    def respond(
        self,
        text: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> AgentResponse:
        """Ответить в указанном разговоре."""
        self._ensure_default_conversation(conversation_id)
        return self._agent.respond_with_usage(text)

    def strategy(self, conversation_id: int = DEFAULT_CONVERSATION_ID) -> str:
        """Вернуть активную стратегию разговора."""
        self._ensure_default_conversation(conversation_id)
        return self._history.strategy()

    def available_strategies(self) -> tuple[str, ...]:
        """Вернуть стратегии в порядке отображения пользователю."""
        return ("sliding", "sticky", "branching", "summary")

    def set_strategy(
        self,
        name: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> StrategyChange:
        """Переключить и сохранить стратегию разговора."""
        self._ensure_default_conversation(conversation_id)
        normalized = name.strip().lower()
        if normalized not in VALID_STRATEGIES:
            available = ", ".join(self.available_strategies())
            raise ValueError(f"неизвестная стратегия {name!r}; доступны: {available}")
        previous = self._history.strategy()
        self._agent.set_strategy(normalized)
        return StrategyChange(previous=previous, current=normalized)

    def active_branch(self, conversation_id: int = DEFAULT_CONVERSATION_ID) -> Branch:
        """Вернуть активную ветку разговора."""
        self._ensure_default_conversation(conversation_id)
        return self._history.active_branch()

    def list_branches(
        self, conversation_id: int = DEFAULT_CONVERSATION_ID
    ) -> list[Branch]:
        """Вернуть ветки разговора."""
        self._ensure_branching(conversation_id)
        return self._history.list_branches()

    def create_checkpoint(
        self,
        name: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> Checkpoint:
        """Сохранить checkpoint в вершине активной ветки."""
        self._ensure_branching(conversation_id)
        return self._history.create_checkpoint(name)

    def create_branch(
        self,
        name: str,
        checkpoint_name: str | None = None,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> Branch:
        """Создать ветку от текущей позиции или checkpoint."""
        self._ensure_branching(conversation_id)
        return self._history.create_branch(name, checkpoint_name)

    def switch_branch(
        self,
        name: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> Branch:
        """Переключить активную ветку."""
        self._ensure_branching(conversation_id)
        return self._history.switch_branch(name)

    @staticmethod
    def _ensure_default_conversation(conversation_id: int) -> None:
        if conversation_id != DEFAULT_CONVERSATION_ID:
            raise ValueError(f"разговор с id={conversation_id} не найден")

    def _ensure_branching(self, conversation_id: int) -> None:
        self._ensure_default_conversation(conversation_id)
        if self._history.strategy() != "branching":
            raise ValueError(
                "команда доступна только для стратегии 'branching'; "
                "переключитесь: /strategy branching"
            )
