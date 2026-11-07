"""Шаг ingest: скачать архив CIFAR-10 с официального сайта и положить в S3 как есть.

Идемпотентен: если архив уже лежит в S3, ничего не скачивается и не загружается.
Существующий объект в raw/ никогда не перезаписывается.

Запуск: python -m pad_hw.data.ingest [--url URL] [--key raw/cifar-10-python.tar.gz]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

from pad_hw.data.cifar import ARCHIVE_MD5, ARCHIVE_NAME, CIFAR10_URL, RAW_KEY
from pad_hw.s3 import ObjectExistsError, make_client, object_exists, put_file_if_absent
from pad_hw.settings import S3Settings
from pad_hw.utils import file_digest, setup_logging

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


def download_archive(url: str, dest: Path, expected_md5: str | None, timeout: float = 60) -> None:
    """Скачивает файл по URL и проверяет MD5 (если он задан)."""
    logger.info("Downloading %s", url)
    md5 = hashlib.md5(usedforsecurity=False)
    request = urllib.request.Request(url, headers={"User-Agent": "pad-hw-ingest"})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0.1
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            md5.update(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                logger.info("  %3.0f%% (%d MB)", 100 * done / total, done // _CHUNK)
                next_report += 0.1
    actual = md5.hexdigest()
    if expected_md5 and actual != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {url}: expected {expected_md5}, got {actual}")
    logger.info("Downloaded %d MB, md5 %s", done // _CHUNK, actual)


def ingest(settings: S3Settings, url: str, key: str, expected_md5: str | None) -> bool:
    """Возвращает True, если архив загружен, и False, если он уже был в S3."""
    client = make_client(settings)
    location = f"s3://{settings.bucket}/{key}"
    if object_exists(client, settings.bucket, key):
        logger.info("%s already exists, nothing to do", location)
        return False

    with tempfile.TemporaryDirectory(prefix="pad-ingest-") as tmp:
        archive = Path(tmp) / ARCHIVE_NAME
        download_archive(url, archive, expected_md5)
        metadata = {
            "source-url": url,
            "md5": file_digest(archive, "md5"),
            "sha256": file_digest(archive, "sha256"),
        }
        try:
            put_file_if_absent(client, settings.bucket, key, archive, metadata=metadata)
        except ObjectExistsError:
            logger.info("%s appeared while downloading, keeping the existing object", location)
            return False
    logger.info("Uploaded archive to %s", location)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=CIFAR10_URL, help="Archive URL (default: %(default)s)")
    parser.add_argument("--key", default=RAW_KEY, help="Object key in S3 (default: %(default)s)")
    parser.add_argument(
        "--expected-md5",
        default=ARCHIVE_MD5,
        help="Expected archive MD5, empty string disables the check (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    setup_logging()
    ingest(S3Settings.from_env(), args.url, args.key, args.expected_md5 or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
