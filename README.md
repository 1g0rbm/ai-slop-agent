# agent-lvl

[![CI](https://github.com/1g0rbm/ai-slop-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/1g0rbm/ai-slop-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/badge/Ruff-checked-blue.svg)](https://docs.astral.sh/ruff/)

Минимальный интерактивный AI-агент на Python с поддержкой DeepSeek, Qwen Cloud и локального API в формате Ollama.

## Требования

Сначала установите [mise](https://mise.jdx.dev/). Конфигурация проекта автоматически установит Python и uv.

## Настройка

```sh
mise install
mise run install
export DEEPSEEK_API_KEY="..."
```

При необходимости выберите модель. По умолчанию используется `deepseek-chat`:

```sh
export DEEPSEEK_MODEL="deepseek-chat"
```

История единственного диалога хранится в `history.db` в корне проекта. По
умолчанию модель получает последние 20 завершённых пар запрос-ответ. Лимит
можно изменить переменной `HISTORY_LIMIT`:

```sh
export HISTORY_LIMIT=20
```

Значение `0` отключает передачу предыдущих пар, а `-1` передаёт модели всю
сохранённую историю. Текущий запрос и системный промпт передаются всегда.

## Провайдеры

Выбор провайдера выполняется переменной `AI_PROVIDER`. По умолчанию используется `deepseek`.
CLI автоматически загружает файл `.env` из корня проекта. Переменные, заданные в shell, имеют приоритет над значениями из этого файла.

### DeepSeek

```sh
export AI_PROVIDER=deepseek
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL=deepseek-chat
```

### Qwen Cloud

Укажите URL DashScope для региона своего аккаунта.

```sh
export AI_PROVIDER=qwen
export DASHSCOPE_API_KEY="..."
export QWEN_BASE_URL="https://.../compatible-mode/v1"
export QWEN_MODEL=qwen-plus
```

### Локальный API

Профиль `local` не требует ключа. Все параметры можно изменить без правок кода:

```sh
export AI_PROVIDER=local
export LOCAL_API_URL="http://10.36.201.230:11435/api/chat"
export LOCAL_MODEL=gemma3:27b
export LOCAL_TEMPERATURE=0.0
export LOCAL_MAX_TOKENS=2048
export LOCAL_NUM_PREDICT=300
export LOCAL_THINK=false
export LOCAL_TIMEOUT_SECONDS=120
```

При необходимости добавьте `LOCAL_API_KEY`: он будет отправлен в заголовке `Authorization`.

Для локального API в `.env` должно быть указано как минимум:

```sh
AI_PROVIDER=local
```

## Запуск

```sh
mise run run
```

При запуске CLI выводит всю сохранённую историю и продолжает диалог. Введите
сообщение в приглашении `Вы:`. Команда `/clear` удаляет историю после
подтверждения. Для завершения сеанса используйте `exit`, `quit`, Ctrl-D или
Ctrl-C.

## Разработка

```sh
mise run lint
mise run test
```
