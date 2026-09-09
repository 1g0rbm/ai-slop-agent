from datetime import UTC, datetime

import pytest

from agent_lvl.agent import DEFAULT_SYSTEM_PROMPT, Agent
from agent_lvl.history import Exchange, SQLiteHistory
from agent_lvl.providers import ChatResult, TokenUsage


class FakeProvider:
    def __init__(
        self,
        response: str = "Здравствуйте!",
        model: str = "test-model",
        usage: TokenUsage | None = None,
    ) -> None:
        self.response = response
        self.model = model
        self.usage = usage
        self.calls: list[list[dict[str, str]]] = []

    def respond(self, messages: list[dict[str, str]]) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(content=self.response, usage=self.usage)


def test_agent_sends_system_prompt_and_persists_history(tmp_path) -> None:
    provider = FakeProvider()
    history = SQLiteHistory(tmp_path / "history.db")
    agent = Agent(provider, history=history)

    assert agent.respond("Привет") == "Здравствуйте!"
    assert agent.respond("Как дела?") == "Здравствуйте!"

    assert provider.calls[0] == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Привет"},
    ]
    assert provider.calls[1][-3:] == [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Здравствуйте!"},
        {"role": "user", "content": "Как дела?"},
    ]
    exchanges = history.all()
    assert [exchange.user_content for exchange in exchanges] == ["Привет", "Как дела?"]
    assert all(exchange.model == "test-model" for exchange in exchanges)
    assert all(exchange.user_created_at.tzinfo == UTC for exchange in exchanges)
    assert all(exchange.assistant_created_at.tzinfo == UTC for exchange in exchanges)


def test_new_agent_continues_saved_dialog(tmp_path) -> None:
    path = tmp_path / "history.db"
    first_provider = FakeProvider(response="Первый ответ")
    Agent(first_provider, history=SQLiteHistory(path)).respond("Первый вопрос")

    second_provider = FakeProvider(response="Второй ответ", model="another-model")
    Agent(second_provider, history=SQLiteHistory(path)).respond("Второй вопрос")

    assert second_provider.calls == [
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": "Первый ответ"},
            {"role": "user", "content": "Второй вопрос"},
        ]
    ]
    assert [exchange.model for exchange in SQLiteHistory(path).all()] == [
        "test-model",
        "another-model",
    ]


@pytest.mark.parametrize(
    ("limit", "expected_previous_questions"),
    [
        (0, []),
        (2, ["Вопрос 2", "Вопрос 3"]),
        (-1, ["Вопрос 1", "Вопрос 2", "Вопрос 3"]),
    ],
)
def test_agent_limits_history_by_completed_pairs(
    tmp_path, limit: int, expected_previous_questions: list[str]
) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    for number in range(1, 4):
        history.add(
            Exchange(
                user_content=f"Вопрос {number}",
                assistant_content=f"Ответ {number}",
                user_created_at=timestamp,
                assistant_created_at=timestamp,
                model="old-model",
            )
        )
    provider = FakeProvider()

    Agent(provider, history=history, history_limit=limit).respond("Новый вопрос")

    questions = [
        message["content"] for message in provider.calls[0] if message["role"] == "user"
    ]
    assert questions == [*expected_previous_questions, "Новый вопрос"]


def test_agent_does_not_save_failed_request(tmp_path) -> None:
    class FailingProvider:
        model = "failing-model"

        def respond(self, messages: list[dict[str, str]]) -> ChatResult:
            raise ConnectionError("соединение недоступно")

    history = SQLiteHistory(tmp_path / "history.db")
    agent = Agent(FailingProvider(), history=history)

    with pytest.raises(ConnectionError, match="соединение недоступно"):
        agent.respond("Привет")

    assert history.count() == 0


def test_agent_persists_and_returns_token_usage(tmp_path) -> None:
    usage = TokenUsage(input_tokens=120, output_tokens=30, total_tokens=150)
    provider = FakeProvider(usage=usage)
    history = SQLiteHistory(tmp_path / "history.db")
    agent = Agent(provider, history=history)

    first = agent.respond_with_usage("Первый вопрос")
    second = agent.respond_with_usage("Второй вопрос")

    assert first.usage == usage
    assert first.conversation_totals.total_tokens == 150
    assert second.conversation_totals.input_tokens == 240
    assert second.conversation_totals.output_tokens == 60
    assert second.conversation_totals.total_tokens == 300
    assert second.conversation_totals.complete
    assert history.all()[0].total_tokens == 150
