FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM python:3.12.12-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
ARG PAD_GIT_COMMIT
ARG PAD_GIT_DIRTY
ENV PAD_GIT_COMMIT=${PAD_GIT_COMMIT} PAD_GIT_DIRTY=${PAD_GIT_DIRTY}
ENTRYPOINT ["/app/.venv/bin/python", "-m", "pad_hw.train"]
