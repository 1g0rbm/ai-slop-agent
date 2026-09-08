# AGENTS.md

## Project overview

`agent-lvl` is a minimal interactive AI chat agent written in Python 3.12. It supports DeepSeek, Qwen Cloud, and an Ollama-compatible local API.

User-facing setup instructions and the complete environment-variable reference live in `README.md`. Keep this file focused on repository-level guidance for coding agents.

## Development commands

Use the project-managed toolchain:

```sh
mise install
mise run install
mise run lint
mise run test
```

Run the CLI with:

```sh
mise run run
```

Do not require real API credentials for tests.

## Project structure

- `src/agent_lvl/agent.py` — conversation orchestration and prompt assembly.
- `src/agent_lvl/providers.py` — provider protocol, adapters, and environment configuration.
- `src/agent_lvl/history.py` — SQLite persistence and history-limit handling.
- `src/agent_lvl/cli.py` — Typer CLI and interactive input/output.
- `tests/` — unit and CLI tests using fake providers and clients.

## Behavioral invariants

- Persist a conversation exchange only after the provider returns successfully.
- Store complete user/assistant pairs in chronological order.
- `HISTORY_LIMIT=0` excludes previous exchanges.
- `HISTORY_LIMIT=-1` includes all previous exchanges.
- The system prompt and current user message are always sent.
- Store timestamps as timezone-aware UTC datetimes.
- Keep provider-specific behavior inside provider adapters or `create_provider`.
- Validate invalid environment configuration with clear `ValueError` messages.

## Testing guidelines

- Use `tmp_path` for SQLite databases created by tests.
- Inject fake providers or HTTP clients; never make real network requests.
- Add or update focused tests when changing agent behavior, persistence, configuration parsing, provider payloads, or CLI commands.
- Run both `mise run lint` and `mise run test` before considering a change complete.

## Code style

- Follow the Ruff configuration in `pyproject.toml`.
- Maximum line length is 88 characters.
- Keep changes small and consistent with the existing module boundaries.
- Preserve the current Russian language used for user-facing CLI messages and docstrings unless a task explicitly requests localization.

## Sensitive and generated files

- Never commit `.env`, API keys, or other secrets.
- Never commit `history.db`; it contains local conversation data.
- Do not edit generated caches or virtual-environment files.
