from datetime import UTC, datetime

from typer.testing import CliRunner

from agent_lvl.cli import app
from agent_lvl.history import HISTORY_DB_NAME, Exchange, SQLiteHistory
from agent_lvl.providers import ChatResult, TokenUsage


class FakeProvider:
    model = "test-model"

    def respond(self, messages: list[dict[str, str]]) -> ChatResult:
        return ChatResult(
            content="Новый ответ",
            usage=TokenUsage(input_tokens=1_200, output_tokens=34, total_tokens=1_234),
        )


def saved_exchange() -> Exchange:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Exchange(
        user_content="Старый вопрос",
        assistant_content="Старый ответ",
        user_created_at=timestamp,
        assistant_created_at=timestamp,
        model="old-model",
    )


def test_cli_displays_saved_history(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())
    history = SQLiteHistory(HISTORY_DB_NAME)
    history.add(saved_exchange())

    result = CliRunner().invoke(app, input="exit\n")

    assert result.exit_code == 0
    assert "Сохранённая история: 1 пар." in result.output
    assert "Вы: Старый вопрос" in result.output
    assert "Агент: Старый ответ" in result.output


def test_cli_displays_current_and_conversation_token_usage(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())

    result = CliRunner().invoke(app, input="Новый вопрос\nexit\n")

    assert result.exit_code == 0
    assert "текущий запрос — 1 200" in result.output
    assert "ответ модели — 34" in result.output
    assert "Токены всего диалога — 1 234." in result.output


def test_cli_rejects_invalid_summary_batch(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUMMARY_BATCH_MESSAGES", "3")
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())

    result = CliRunner().invoke(app)

    assert result.exit_code == 1
    assert "Ошибка конфигурации" in result.output
    assert "SUMMARY_BATCH_MESSAGES" in result.output


def test_cli_clear_deletes_history_after_confirmation(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())
    history = SQLiteHistory(HISTORY_DB_NAME)
    history.add(saved_exchange())

    result = CliRunner().invoke(app, input="/clear\ny\nexit\n")

    assert result.exit_code == 0
    assert "История удалена." in result.output
    assert history.count() == 0


def test_cli_clear_keeps_history_when_cancelled(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())
    history = SQLiteHistory(HISTORY_DB_NAME)
    history.add(saved_exchange())

    result = CliRunner().invoke(app, input="/clear\nn\nexit\n")

    assert result.exit_code == 0
    assert "Очистка отменена." in result.output
    assert history.count() == 1
