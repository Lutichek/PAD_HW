"""Настройки подключения к S3 из переменных окружения (и файла .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

_REQUIRED = ("S3_ENDPOINT_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET")


class SettingsError(RuntimeError):
    """Не хватает обязательных переменных окружения."""


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str
    access_key: str
    secret_key: str = field(repr=False)
    bucket: str
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> S3Settings:
        """Читает настройки из окружения. Значения из .env не перекрывают уже заданные."""
        load_dotenv(find_dotenv(usecwd=True), override=False)
        missing = [name for name in _REQUIRED if not os.environ.get(name)]
        if missing:
            raise SettingsError(
                f"Missing environment variables: {', '.join(missing)}. "
                "Create .env from .env.example (make s3-up does it automatically)."
            )
        return cls(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_key=os.environ["S3_SECRET_KEY"],
            bucket=os.environ["S3_BUCKET"],
            region=os.environ.get("S3_REGION") or "us-east-1",
        )
