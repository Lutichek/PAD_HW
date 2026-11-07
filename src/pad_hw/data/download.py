"""Шаг download: скачать сырой архив из S3 в data/raw/.

Если архив уже лежит локально и его MD5 совпадает с ожидаемым, повторно не скачивается.

Запуск: python -m pad_hw.data.download [--key raw/cifar-10-python.tar.gz] [--dest PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pad_hw.data.cifar import ARCHIVE_MD5, ARCHIVE_NAME, RAW_DIR, RAW_KEY
from pad_hw.s3 import download_file, make_client, object_exists
from pad_hw.settings import S3Settings
from pad_hw.utils import file_digest, setup_logging

logger = logging.getLogger(__name__)


def download_raw(settings: S3Settings, key: str, dest: Path, expected_md5: str | None) -> bool:
    """Возвращает True, если архив скачан, и False, если локальная копия уже актуальна."""
    if dest.exists() and (expected_md5 is None or file_digest(dest, "md5") == expected_md5):
        logger.info("%s is already present, skipping download", dest)
        return False

    client = make_client(settings)
    if not object_exists(client, settings.bucket, key):
        raise FileNotFoundError(
            f"s3://{settings.bucket}/{key} not found. Run `make ingest` first."
        )
    logger.info("Downloading s3://%s/%s -> %s", settings.bucket, key, dest)
    download_file(client, settings.bucket, key, dest)

    if expected_md5 is not None:
        actual = file_digest(dest, "md5")
        if actual != expected_md5:
            dest.unlink()
            raise RuntimeError(f"MD5 mismatch for {dest}: expected {expected_md5}, got {actual}")
    logger.info("Saved %s (%d MB)", dest, dest.stat().st_size // (1024 * 1024))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", default=RAW_KEY, help="Object key in S3 (default: %(default)s)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR / ARCHIVE_NAME,
        help="Local path for the archive (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-md5",
        default=ARCHIVE_MD5,
        help="Expected archive MD5, empty string disables the check (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    setup_logging()
    download_raw(S3Settings.from_env(), args.key, args.dest, args.expected_md5 or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
