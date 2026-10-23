SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv из PATH, иначе стандартное место установки официального скрипта
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: help setup check lint format-check typecheck pre-commit format clean

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
