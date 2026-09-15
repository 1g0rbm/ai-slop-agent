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
from .memory import MemoryEntry, MemoryType, Task


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

    def create_task(
        self,
        title: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> Task:
        """Создать активную задачу разговора."""
        self._ensure_default_conversation(conversation_id)
        return self._history.create_task(title, conversation_id)

    def active_task(
        self, conversation_id: int = DEFAULT_CONVERSATION_ID
    ) -> Task | None:
        """Вернуть активную задачу разговора."""
        self._ensure_default_conversation(conversation_id)
        return self._history.active_task(conversation_id)

    def list_tasks(self, conversation_id: int = DEFAULT_CONVERSATION_ID) -> list[Task]:
        """Вернуть задачи разговора."""
        self._ensure_default_conversation(conversation_id)
        return self._history.list_tasks(conversation_id)

    def task(
        self,
        task_id: int,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> Task:
        """Вернуть задачу по идентификатору."""
        self._ensure_default_conversation(conversation_id)
        return self._history.task(task_id, conversation_id)

    def complete_task(self, conversation_id: int = DEFAULT_CONVERSATION_ID) -> Task:
        """Завершить активную задачу."""
        self._ensure_default_conversation(conversation_id)
        return self._history.complete_task(conversation_id=conversation_id)

    def cancel_task(self, conversation_id: int = DEFAULT_CONVERSATION_ID) -> Task:
        """Отменить активную задачу."""
        self._ensure_default_conversation(conversation_id)
        return self._history.cancel_task(conversation_id=conversation_id)

    def list_memory(
        self,
        memory_type: MemoryType,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
        *,
        task_id: int | None = None,
    ) -> list[MemoryEntry]:
        """Вернуть память выбранного уровня."""
        self._ensure_default_conversation(conversation_id)
        return self._history.list_memory(
            memory_type,
            conversation_id=conversation_id,
            task_id=task_id,
        )

    def set_memory(
        self,
        memory_type: MemoryType,
        category: str,
        key: str,
        value: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> MemoryEntry:
        """Создать или обновить запись памяти."""
        self._ensure_default_conversation(conversation_id)
        return self._history.set_memory(
            memory_type,
            category,
            key,
            value,
            conversation_id=conversation_id,
        )

    def delete_memory(
        self,
        memory_type: MemoryType,
        category: str,
        key: str,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> bool:
        """Удалить запись памяти."""
        self._ensure_default_conversation(conversation_id)
        return self._history.delete_memory(
            memory_type,
            category,
            key,
            conversation_id=conversation_id,
        )

    def clear_memory(
        self,
        memory_type: MemoryType,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> int:
        """Очистить выбранный изменяемый уровень памяти."""
        self._ensure_default_conversation(conversation_id)
        return self._history.clear_memory(memory_type, conversation_id=conversation_id)

    def promote_memory(
        self,
        working_category: str,
        key: str,
        long_category: str,
        new_key: str | None = None,
        conversation_id: int = DEFAULT_CONVERSATION_ID,
    ) -> MemoryEntry:
        """Скопировать working-запись в long-term."""
        self._ensure_default_conversation(conversation_id)
        return self._history.promote_memory(
            working_category,
            key,
            long_category,
            new_key,
            conversation_id=conversation_id,
        )

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
