"""Работа с S3: клиент, проверки существования, загрузка без перезаписи."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from pad_hw.settings import S3Settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

_NOT_FOUND = {"404", "NoSuchKey", "NotFound"}
_ALREADY_EXISTS = {"412", "PreconditionFailed", "ConditionalRequestConflict"}

CONTENT_TYPES = {
    ".png": "image/png",
    ".json": "application/json",
    ".gz": "application/gzip",
}


class ObjectExistsError(RuntimeError):
    """Объект уже есть в S3, перезаписывать его нельзя."""


def make_client(settings: S3Settings, max_pool_connections: int = 64) -> S3Client:
    """Создаёт клиент для S3-совместимого хранилища."""
    config = Config(
        s3={"addressing_style": "path"},
        # Не все S3-совместимые хранилища поддерживают новые контрольные суммы boto3
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        max_pool_connections=max_pool_connections,
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
        region_name=settings.region,
        config=config,
    )


def _error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def object_exists(client: S3Client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if _error_code(error) in _NOT_FOUND:
            return False
        raise
    return True


def prefix_exists(client: S3Client, bucket: str, prefix: str) -> bool:
    """Есть ли в бакете хотя бы один объект с таким префиксом."""
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def put_file_if_absent(
    client: S3Client,
    bucket: str,
    key: str,
    path: Path,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Загружает файл, только если объекта ещё нет.

    Условие If-None-Match: * проверяет само хранилище, поэтому существующий объект
    не будет перезаписан даже при одновременном запуске нескольких скриптов.
    """
    content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    with path.open("rb") as body:
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=dict(metadata or {}),
                IfNoneMatch="*",
            )
        except ClientError as error:
            if _error_code(error) in _ALREADY_EXISTS:
                raise ObjectExistsError(f"s3://{bucket}/{key} already exists") from error
            raise


def upload_files_if_absent(
    client: S3Client,
    bucket: str,
    items: Sequence[tuple[Path, str]],
    workers: int = 32,
) -> None:
    """Параллельно загружает пары (локальный файл, ключ). Ни один объект не перезаписывается."""
    total = len(items)
    step = max(total // 10, 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(put_file_if_absent, client, bucket, key, path) for path, key in items
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % step == 0 or done == total:
                logger.info("Uploaded %d/%d files", done, total)


def download_file(client: S3Client, bucket: str, key: str, dest: Path) -> None:
    """Скачивает объект во временный файл и атомарно переименовывает его в dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(dest)
