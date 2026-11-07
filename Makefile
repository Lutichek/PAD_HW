SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv из PATH, иначе стандартное место установки официального скрипта
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
COMPOSE ?= docker compose

# Путь внутри бакета для make s3-ls (например: make s3-ls S3_PATH=raw/)
S3_PATH ?=

.PHONY: help setup check lint format-check typecheck pre-commit format clean \
	s3-up s3-down s3-purge s3-ls

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-14s %s\n", $$1, $$2}'

setup: ## Поднять окружение с нуля: uv, Python, зависимости, git-хуки
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --frozen
	$(UV) run --frozen pre-commit install --install-hooks

check: ## Все проверки (ничего не исправляет, при ошибках возвращает ненулевой код)
	@status=0; \
	echo ">>> ruff check";          $(UV) run --frozen ruff check . || status=1; \
	echo ">>> ruff format --check"; $(UV) run --frozen ruff format --check --diff . || status=1; \
	echo ">>> mypy";                $(UV) run --frozen mypy || status=1; \
	if [ $$status -ne 0 ]; then echo "Checks FAILED."; else echo "All checks passed."; fi; \
	exit $$status

lint: ## Линтер Ruff
	$(UV) run --frozen ruff check .

format-check: ## Проверка форматирования Ruff (без изменений файлов)
	$(UV) run --frozen ruff format --check --diff .

typecheck: ## Проверка типов mypy (strict)
	$(UV) run --frozen mypy

pre-commit: ## Прогнать pre-commit хуки по всем файлам
	$(UV) run --frozen pre-commit run --all-files --show-diff-on-failure

format: ## Автоисправление стиля и форматирование (меняет файлы!)
	$(UV) run --frozen ruff check --fix .
	$(UV) run --frozen ruff format .

clean: ## Удалить окружение и кеши
	rm -rf .venv .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------- S3 (RustFS в docker compose) ----------

.env:
	cp .env.example .env
	@echo "Created .env from .env.example"

s3-up: .env ## Поднять S3 (RustFS) и создать бакет
	$(COMPOSE) up -d --wait s3
	$(COMPOSE) run --rm s3-init
	@echo "S3 is up. Web console: http://localhost:9001/rustfs/console/ (port S3_CONSOLE_PORT from .env)"

s3-down: ## Остановить S3 (данные в volume сохраняются)
	$(COMPOSE) down

s3-purge: ## Остановить S3 и удалить volume с данными
	$(COMPOSE) down -v

s3-ls: .env ## Показать содержимое бакета (путь задаётся через S3_PATH=...)
	$(COMPOSE) run --rm --no-deps --entrypoint sh s3-init \
		-c 'aws --endpoint-url http://s3:9000 s3 ls "s3://$$S3_BUCKET/$(S3_PATH)"'
