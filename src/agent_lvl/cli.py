"""Интерактивный интерфейс командной строки для агента."""

import os

import typer
from dotenv import load_dotenv

from .agent import Agent, AgentResponse
from .context import summary_batch_messages
from .history import HISTORY_DB_NAME, OperationTotals, SQLiteHistory, history_limit
from .providers import create_provider
from .service import ConversationService

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def chat() -> None:
    """Запустить интерактивный сеанс чата."""
    load_dotenv()
    provider_name = os.environ.get("AI_PROVIDER", "deepseek").lower()
    try:
        history = SQLiteHistory(HISTORY_DB_NAME)
        agent = Agent(
            create_provider(),
            history=history,
            history_limit=history_limit(os.environ),
            summary_batch_messages=summary_batch_messages(os.environ),
        )
        service = ConversationService(agent, history)
    except ValueError as error:
        typer.echo(f"Ошибка конфигурации: {error}", err=True)
        raise typer.Exit(1) from error

    exchanges = history.all()
    if exchanges:
        typer.echo(f"Сохранённая история: {len(exchanges)} пар.")
        for exchange in exchanges:
            typer.echo(f"Вы: {exchange.user_content}")
            typer.echo(f"Агент: {exchange.assistant_content}")
    else:
        typer.echo("Сохранённой истории нет.")

    typer.echo(
        f"Чат запущен. Провайдер: {provider_name}. Команды: "
        "'/strategy [имя]', '/branch [list]', "
        "'/branch checkpoint <имя>', '/branch create <имя> [checkpoint]', "
        "'/branch switch <имя>', '/clear'; 'exit' или 'quit' для завершения."
    )
    while True:
        try:
            message = typer.prompt("Вы")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        stripped = message.strip()
        command = stripped.lower()
        if command in {"exit", "quit"}:
            break
        if command == "/clear":
            if typer.confirm("Удалить всю историю?", default=False):
                history.clear()
                typer.echo("История удалена.")
            else:
                typer.echo("Очистка отменена.")
            continue
        if not stripped:
            continue
        if stripped.startswith("/"):
            _handle_command(service, stripped)
            continue

        try:
            result = service.respond(message)
            typer.echo(f"Агент: {result.content}")
            _display_token_usage(result, history.maintenance_totals())
        except Exception as error:
            typer.echo(f"Ошибка запроса ({provider_name}): {error}", err=True)


def _handle_command(service: ConversationService, command: str) -> None:
    parts = command.split()
    name = parts[0].lower()
    try:
        if name == "/strategy":
            _handle_strategy_command(service, parts)
        elif name == "/branch":
            _handle_branch_command(service, parts)
        else:
            raise ValueError(f"неизвестная команда {parts[0]!r}")
    except ValueError as error:
        typer.echo(f"Ошибка команды: {error}", err=True)


def _handle_strategy_command(service: ConversationService, parts: list[str]) -> None:
    if len(parts) == 1:
        typer.echo(f"Активная стратегия: {service.strategy()}.")
        available = ", ".join(service.available_strategies())
        typer.echo(f"Доступные стратегии: {available}.")
        return
    if len(parts) != 2:
        raise ValueError("использование: /strategy [имя]")
    change = service.set_strategy(parts[1])
    typer.echo(f"Стратегия изменена: {change.previous} → {change.current}.")


def _handle_branch_command(service: ConversationService, parts: list[str]) -> None:
    if len(parts) == 1:
        service.list_branches()
        typer.echo(f"Активная ветка: {service.active_branch().name}.")
        return

    action = parts[1].lower()
    if action == "list" and len(parts) == 2:
        active = service.active_branch()
        typer.echo("Ветки:")
        for branch in service.list_branches():
            suffix = " (активная)" if branch.id == active.id else ""
            typer.echo(f"- {branch.name}{suffix}")
        return
    if action == "checkpoint" and len(parts) == 3:
        checkpoint = service.create_checkpoint(parts[2])
        typer.echo(f"Checkpoint создан: {checkpoint.name}.")
        return
    if action == "create" and len(parts) in {3, 4}:
        checkpoint_name = parts[3] if len(parts) == 4 else None
        branch = service.create_branch(parts[2], checkpoint_name)
        typer.echo(f"Ветка создана: {branch.name}.")
        return
    if action == "switch" and len(parts) == 3:
        branch = service.switch_branch(parts[2])
        typer.echo(f"Активная ветка: {branch.name}.")
        return
    raise ValueError(
        "использование: /branch [list|checkpoint <имя>|"
        "create <имя> [checkpoint]|switch <имя>]"
    )


def _display_token_usage(
    result: AgentResponse,
    maintenance_totals: dict[str, OperationTotals],
) -> None:
    if result.usage is None:
        typer.echo("Токены текущего запроса и ответа: недоступны.")
    else:
        typer.echo(
            "Токены: "
            f"текущий запрос — {_format_tokens(result.usage.input_tokens)}, "
            f"ответ модели — {_format_tokens(result.usage.output_tokens)}."
        )

    for operation in ("summary", "facts"):
        usage = result.maintenance_usage.get(operation)
        if usage is not None:
            suffix = _untracked_suffix(usage.untracked_operations, "операций")
            typer.echo(
                f"Текущие служебные токены ({operation}): "
                f"запрос — {_format_tokens(usage.input_tokens)}, "
                f"ответ — {_format_tokens(usage.output_tokens)}, "
                f"всего — {_format_tokens(usage.total_tokens)}{suffix}."
            )

    totals = result.conversation_totals
    untracked = totals.untracked_exchanges + totals.untracked_maintenance
    suffix = _untracked_suffix(untracked, "операций")
    typer.echo(
        f"Токены всего диалога — {_format_tokens(totals.overall_total_tokens)}{suffix}."
    )
    typer.echo(
        "Накопительные токены: "
        f"основные запросы — {_format_tokens(totals.input_tokens)}, "
        f"основные ответы — {_format_tokens(totals.output_tokens)}, "
        f"summary — {_operation_total(maintenance_totals, 'summary')}, "
        f"facts — {_operation_total(maintenance_totals, 'facts')}, "
        f"общий total — {_format_tokens(totals.overall_total_tokens)}."
    )


def _operation_total(totals: dict[str, OperationTotals], operation: str) -> str:
    value = totals.get(operation)
    return _format_tokens(value.total_tokens if value is not None else 0)


def _untracked_suffix(count: int, unit: str) -> str:
    if count == 0:
        return ""
    return f" (без данных для {count} {unit})"


def _format_tokens(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def main() -> None:
    """Запустить приложение Typer."""
    app()
