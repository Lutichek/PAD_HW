"""Общие вспомогательные функции: хеши, JSON, логирование."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Пишет JSON детерминированно: одинаковые данные дают одинаковые байты."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
