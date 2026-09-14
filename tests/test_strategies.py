from datetime import UTC, datetime

import pytest

from agent_lvl.agent import DEFAULT_SYSTEM_PROMPT, Agent
from agent_lvl.history import ConversationSummary, Exchange, SQLiteHistory
from agent_lvl.providers import ChatResult, TokenUsage
from agent_lvl.strategies import FACTS_CONTEXT_PREFIX


class ScriptedProvider:
    model = "test-model"

    def __init__(self, results: list[ChatResult | Exception]) -> None:
        self.results = results
        self.calls: list[list[dict[str, str]]] = []

    def respond(self, messages: list[dict[str, str]]) -> ChatResult:
        self.calls.append(messages)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_exchange(number: int) -> Exchange:
    timestamp = datetime(2026, 1, number, tzinfo=UTC)
    return Exchange(
        user_content=f"Вопрос {number}",
        assistant_content=f"Ответ {number}",
        user_created_at=timestamp,
        assistant_created_at=timestamp,
        model="old-model",
    )


def test_sliding_ignores_saved_summary_and_facts(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 4):
        history.add(make_exchange(number))
    history.save_summary(
        ConversationSummary(
            content="Старая сводка",
            through_exchange_id=1,
            summarized_messages=2,
            model="old-model",
            updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
    )
    history.set_strategy("sliding")
    provider = ScriptedProvider([ChatResult("Новый ответ")])

    Agent(provider, history=history, history_limit=2).respond("Новый вопрос")

    assert provider.calls == [
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Вопрос 2"},
            {"role": "assistant", "content": "Ответ 2"},
            {"role": "user", "content": "Вопрос 3"},
            {"role": "assistant", "content": "Ответ 3"},
            {"role": "user", "content": "Новый вопрос"},
        ]
    ]
    assert history.summary() is not None
    assert history.maintenance_totals() == {}


def test_sticky_updates_facts_before_main_request(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.set_strategy("sticky")
    facts_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    answer_usage = TokenUsage(input_tokens=20, output_tokens=3, total_tokens=23)
    provider = ScriptedProvider(
        [
            ChatResult(
                '{"operations": [{"operation": "set", '
                '"category": "constraint", "key": "python_version", '
                '"value": "3.12"}]}',
                facts_usage,
            ),
            ChatResult("Принято", answer_usage),
        ]
    )

    response = Agent(provider, history=history).respond_with_usage(
        "Используем Python 3.12"
    )

    assert len(provider.calls) == 2
    assert (
        "Новое сообщение пользователя:\nИспользуем Python 3.12"
        in (provider.calls[0][1]["content"])
    )
    facts_message = provider.calls[1][1]
    assert facts_message["role"] == "system"
    assert FACTS_CONTEXT_PREFIX in facts_message["content"]
    assert "python_version: 3.12" in facts_message["content"]
    assert history.facts()[0].value == "3.12"
    assert response.maintenance_usage["facts"].total_tokens == 15
    assert response.conversation_totals.maintenance_total_tokens == 15
    assert response.conversation_totals.overall_total_tokens == 38

    history.set_strategy("sliding")
    sliding_provider = ScriptedProvider([ChatResult("Без facts")])
    Agent(sliding_provider, history=history).respond("Следующий вопрос")
    assert FACTS_CONTEXT_PREFIX not in str(sliding_provider.calls[0])
    assert history.facts()[0].value == "3.12"


def test_sticky_does_not_commit_facts_when_main_request_fails(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.set_strategy("sticky")
    provider = ScriptedProvider(
        [
            ChatResult(
                '{"operations": [{"operation": "set", "category": "goal", '
                '"key": "task", "value": "context"}]}',
                TokenUsage(4, 2, 6),
            ),
            ConnectionError("основной запрос недоступен"),
        ]
    )

    with pytest.raises(ConnectionError, match="основной запрос недоступен"):
        Agent(provider, history=history).respond("Запомни цель")

    assert history.count() == 0
    assert history.facts() == []
    assert history.maintenance_totals()["facts"].total_tokens == 6


def test_sticky_rejects_invalid_extraction_but_accounts_for_it(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.set_strategy("sticky")
    provider = ScriptedProvider(
        [
            ChatResult(
                "не JSON",
                TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            )
        ]
    )

    with pytest.raises(ValueError, match="JSON для facts"):
        Agent(provider, history=history).respond("Сообщение")

    assert history.count() == 0
    assert history.maintenance_totals()["facts"].total_tokens == 4


def test_branching_keeps_continuations_independent(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.set_strategy("branching")
    provider = ScriptedProvider(
        [ChatResult("Общий ответ"), ChatResult("Ответ A"), ChatResult("Ответ B")]
    )
    agent = Agent(provider, history=history, history_limit=-1)
    agent.respond("Общий вопрос")
    history.create_checkpoint("choice")
    history.create_branch("branch-a", "choice")
    history.create_branch("branch-b", "choice")

    history.switch_branch("branch-a")
    agent.respond("Решение A")
    history.switch_branch("branch-b")
    agent.respond("Решение B")

    branch_b_contents = [message["content"] for message in provider.calls[2]]
    assert "Общий вопрос" in branch_b_contents
    assert "Решение B" in branch_b_contents
    assert "Решение A" not in branch_b_contents
    history.switch_branch("branch-a")
    assert [exchange.user_content for exchange in history.all()] == [
        "Общий вопрос",
        "Решение A",
    ]


def test_summary_is_preserved_while_another_strategy_is_active(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 4):
        history.add(make_exchange(number))
    original = ConversationSummary(
        content="Сводка первой пары",
        through_exchange_id=1,
        summarized_messages=2,
        model="old-model",
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    history.save_summary(original)
    history.set_strategy("sliding")
    provider = ScriptedProvider(
        [
            ChatResult("Ответ вне summary"),
            ChatResult("Обновлённая сводка", TokenUsage(5, 2, 7)),
            ChatResult("Ответ со summary"),
        ]
    )
    agent = Agent(
        provider,
        history=history,
        history_limit=1,
        summary_batch_messages=4,
    )

    agent.respond("Вопрос в sliding")
    assert history.summary() == original
    history.set_strategy("summary")
    response = agent.respond_with_usage("Вопрос в summary")

    summary = history.summary()
    assert summary is not None
    assert summary.content == "Обновлённая сводка"
    assert summary.through_exchange_id == 3
    assert (
        "Предыдущее краткое содержание:\nСводка первой пары"
        in (provider.calls[1][1]["content"])
    )
    assert response.maintenance_usage["summary"].total_tokens == 7
