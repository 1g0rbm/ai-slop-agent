from datetime import UTC, datetime

from typer.testing import CliRunner

from agent_lvl.cli import app
from agent_lvl.history import HISTORY_DB_NAME, Exchange, SQLiteHistory
from agent_lvl.providers import ChatResult, TokenUsage


class FakeProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def respond(self, messages: list[dict[str, str]]) -> ChatResult:
        self.calls.append(messages)
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


def test_cli_displays_current_and_cumulative_maintenance_tokens(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HISTORY_LIMIT", "0")
    monkeypatch.setenv("SUMMARY_BATCH_MESSAGES", "2")
    provider = FakeProvider()
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: provider)

    result = CliRunner().invoke(app, input="Первый\nВторой\nexit\n")

    assert result.exit_code == 0
    assert "Текущие служебные токены (summary)" in result.output
    assert "summary — 1 234" in result.output
    assert "facts — 0" in result.output
    assert "общий total — 3 702" in result.output
    assert len(provider.calls) == 3


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
    assert history.strategy() == "summary"


def test_cli_clear_keeps_history_when_cancelled(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())
    history = SQLiteHistory(HISTORY_DB_NAME)
    history.add(saved_exchange())

    result = CliRunner().invoke(app, input="/clear\nn\nexit\n")

    assert result.exit_code == 0
    assert "Очистка отменена." in result.output
    assert history.count() == 1


def test_cli_strategy_status_and_persistence(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: FakeProvider())

    changed = CliRunner().invoke(app, input="/strategy branching\nexit\n")
    status = CliRunner().invoke(app, input="/strategy\nexit\n")

    assert changed.exit_code == 0
    assert "Стратегия изменена: summary → branching." in changed.output
    assert status.exit_code == 0
    assert "Активная стратегия: branching." in status.output
    assert "sliding, sticky, branching, summary" in status.output
    assert SQLiteHistory(HISTORY_DB_NAME).strategy() == "branching"


def test_cli_invalid_strategy_does_not_call_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    provider = FakeProvider()
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: provider)

    result = CliRunner().invoke(
        app,
        input="/strategy unknown\n/strategy too many arguments\nexit\n",
    )

    assert result.exit_code == 0
    assert result.output.count("Ошибка команды:") == 2
    assert "неизвестная стратегия" in result.output
    assert "использование: /strategy [имя]" in result.output
    assert provider.calls == []
    assert SQLiteHistory(HISTORY_DB_NAME).count() == 0


def test_cli_branching_create_checkpoint_switch_and_list(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    provider = FakeProvider()
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: provider)

    result = CliRunner().invoke(
        app,
        input=(
            "/strategy branching\n"
            "Первый вопрос\n"
            "/branch checkpoint base\n"
            "/branch create feature base\n"
            "/branch switch feature\n"
            "/branch\n"
            "/branch list\n"
            "exit\n"
        ),
    )

    history = SQLiteHistory(HISTORY_DB_NAME)
    assert result.exit_code == 0
    assert "Checkpoint создан: base." in result.output
    assert "Ветка создана: feature." in result.output
    assert result.output.count("Активная ветка: feature.") == 2
    assert "- main" in result.output
    assert "- feature (активная)" in result.output
    assert [branch.name for branch in history.list_branches()] == ["main", "feature"]
    assert history.active_branch().name == "feature"
    assert history.count() == 1
    assert len(provider.calls) == 1


def test_cli_branch_error_continues_without_calling_provider(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    provider = FakeProvider()
    monkeypatch.setattr("agent_lvl.cli.create_provider", lambda: provider)

    result = CliRunner().invoke(app, input="/branch list\n/strategy\nexit\n")

    assert result.exit_code == 0
    assert "Ошибка команды:" in result.output
    assert "только для стратегии 'branching'" in result.output
    assert "Активная стратегия: summary." in result.output
    assert provider.calls == []
