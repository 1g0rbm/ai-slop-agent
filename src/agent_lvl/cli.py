"""Интерактивный интерфейс командной строки для агента."""

import os

import typer
from dotenv import load_dotenv

from .agent import Agent, AgentResponse
from .context import summary_batch_messages
from .history import HISTORY_DB_NAME, OperationTotals, SQLiteHistory, history_limit
from .memory import MemoryEntry, MemoryType, TaskStatus
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
        "'/strategy [имя]', '/branch [list]', '/task', '/memory', "
        "'/clear'; 'exit' или 'quit' для завершения."
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
            if typer.confirm(
                "Удалить текущий диалог и working memory? Long-term memory сохранится.",
                default=False,
            ):
                history.clear()
                typer.echo("Текущий диалог и working memory удалены.")
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
        elif name == "/task":
            _handle_task_command(service, parts)
        elif name == "/memory":
            _handle_memory_command(service, parts)
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


def _handle_task_command(service: ConversationService, parts: list[str]) -> None:
    if len(parts) == 1:
        task = service.active_task()
        if task is None:
            typer.echo("Активной задачи нет.")
        else:
            typer.echo(f"Активная задача #{task.id}: {task.title}.")
        return

    action = parts[1].lower()
    if action == "new" and len(parts) >= 3:
        task = service.create_task(" ".join(parts[2:]))
        typer.echo(f"Задача создана: #{task.id} {task.title}.")
        return
    if action == "list" and len(parts) == 2:
        tasks = service.list_tasks()
        if not tasks:
            typer.echo("Задач нет.")
            return
        typer.echo("Задачи:")
        for task in tasks:
            typer.echo(f"- #{task.id} [{_task_status(task.status)}] {task.title}")
        return
    if action == "show" and len(parts) == 3:
        task_id = _positive_id(parts[2])
        task = service.task(task_id)
        typer.echo(f"Задача #{task.id}: {task.title} [{_task_status(task.status)}].")
        entries = service.list_memory(MemoryType.WORKING, task_id=task.id)
        _display_memory_entries(entries)
        return
    if action == "complete" and len(parts) == 2:
        task = service.complete_task()
        typer.echo(f"Задача #{task.id} завершена.")
        return
    if action == "cancel" and len(parts) == 2:
        task = service.cancel_task()
        typer.echo(f"Задача #{task.id} отменена.")
        return
    raise ValueError(
        "использование: /task [new <название...>|list|show <id>|complete|cancel]"
    )


def _handle_memory_command(service: ConversationService, parts: list[str]) -> None:
    if len(parts) == 1:
        typer.echo("Память:")
        typer.echo(
            f"- short-term: {len(service.list_memory(MemoryType.SHORT_TERM))} записей"
        )
        task = service.active_task()
        working_count = (
            len(service.list_memory(MemoryType.WORKING)) if task is not None else 0
        )
        typer.echo(f"- working: {working_count} записей")
        typer.echo(
            f"- long-term: {len(service.list_memory(MemoryType.LONG_TERM))} записей"
        )
        return

    action = parts[1].lower()
    if action == "list" and len(parts) == 3:
        memory_type = _parse_memory_type(parts[2])
        _display_memory_entries(service.list_memory(memory_type))
        return
    if action == "set" and len(parts) >= 6:
        memory_type = _parse_mutable_memory_type(parts[2])
        entry = service.set_memory(memory_type, parts[3], parts[4], " ".join(parts[5:]))
        typer.echo(f"Память обновлена: {entry.category}/{entry.key}.")
        return
    if action == "delete" and len(parts) == 5:
        memory_type = _parse_mutable_memory_type(parts[2])
        deleted = service.delete_memory(memory_type, parts[3], parts[4])
        typer.echo("Запись удалена." if deleted else "Запись не найдена.")
        return
    if action == "clear" and len(parts) == 3:
        memory_type = _parse_mutable_memory_type(parts[2])
        if memory_type is MemoryType.LONG_TERM and not typer.confirm(
            "Удалить всю long-term memory? Это действие необратимо.",
            default=False,
        ):
            typer.echo("Очистка отменена.")
            return
        count = service.clear_memory(memory_type)
        typer.echo(f"Память очищена. Удалено записей: {count}.")
        return
    if action == "promote" and len(parts) in {5, 6}:
        new_key = parts[5] if len(parts) == 6 else None
        entry = service.promote_memory(parts[2], parts[3], parts[4], new_key)
        typer.echo(f"Запись скопирована в long-term: {entry.category}/{entry.key}.")
        return
    raise ValueError(
        "использование: /memory [list <тип>|set <working|long-term> "
        "<категория> <ключ> <значение...>|delete <working|long-term> "
        "<категория> <ключ>|clear <working|long-term>|promote "
        "<working-категория> <ключ> <long-категория> [новый-ключ]]"
    )


def _display_memory_entries(entries: list[MemoryEntry]) -> None:
    if not entries:
        typer.echo("Записей памяти нет.")
        return
    typer.echo("Записи памяти:")
    for entry in entries:
        typer.echo(f"- #{entry.id} {entry.category}/{entry.key}: {entry.value}")


def _parse_memory_type(value: str) -> MemoryType:
    try:
        return MemoryType(value.lower())
    except ValueError as error:
        raise ValueError(
            "тип памяти должен быть short-term, working или long-term"
        ) from error


def _parse_mutable_memory_type(value: str) -> MemoryType:
    memory_type = _parse_memory_type(value)
    if memory_type is MemoryType.SHORT_TERM:
        raise ValueError("short-term memory доступна только для чтения")
    return memory_type


def _positive_id(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError("id задачи должен быть целым числом") from error
    if result <= 0:
        raise ValueError("id задачи должен быть положительным")
    return result


def _task_status(status: TaskStatus) -> str:
    return {
        TaskStatus.ACTIVE: "активна",
        TaskStatus.COMPLETED: "завершена",
        TaskStatus.CANCELLED: "отменена",
    }[status]


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
