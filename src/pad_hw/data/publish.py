"""Шаг publish: выгрузить обработанный датасет в S3 по пути с версией.

    processed/cifar10/<version>/
        images/            PNG-файлы
        annotations/       train.json, val.json, test.json
        stats.json
        manifest.json      загружается последним: его наличие = версия выгружена целиком

Версии неизменяемы: если под префиксом версии уже есть объекты, скрипт завершается
с ошибкой и просит указать новую версию. Каждый объект к тому же загружается
с условием If-None-Match: *, поэтому перезапись невозможна и при гонке.

Запуск: python -m pad_hw.data.publish --version v1 [--check-only]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pad_hw.data.cifar import (
    ARCHIVE_NAME,
    CIFAR10_URL,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_KEY,
)
from pad_hw.data.split import PARAMS_FILE
from pad_hw.s3 import make_client, prefix_exists, put_file_if_absent, upload_files_if_absent
from pad_hw.settings import S3Settings
from pad_hw.utils import file_digest, read_json, setup_logging, write_json

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
DEFAULT_PREFIX = "processed/cifar10"


class VersionExistsError(RuntimeError):
    """Версия уже есть в S3."""


def git_info() -> dict[str, Any]:
    """Коммит, на котором выполняется обработка, и есть ли незакоммиченные изменения."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": file_digest(path)}


def build_manifest(
    dataset_dir: Path, params: dict[str, Any], raw_archive: Path, version: str, location: str
) -> dict[str, Any]:
    splits: dict[str, int] = {}
    for split in SPLITS:
        data = read_json(dataset_dir / "annotations" / f"{split}.json")
        splits[split] = len(data["images"])

    files = [f"annotations/{split}.json" for split in SPLITS] + ["stats.json"]
    return {
        "dataset": "cifar10",
        "version": version,
        "location": location,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits": splits,
        "num_images": sum(splits.values()),
        "images": {"dir": "images/", "format": "PNG", "size": [32, 32], "channels": 3},
        "files": [_file_entry(dataset_dir, relative) for relative in files],
        "params": {
            "val_size": params["val_size"],
            "seed": params["seed"],
            "stratified": params["stratified"],
        },
        "source": {
            "url": CIFAR10_URL,
            "raw_key": RAW_KEY,
            "md5": file_digest(raw_archive, "md5"),
            "sha256": file_digest(raw_archive, "sha256"),
        },
        "git": git_info(),
    }


def _collect_images(dataset_dir: Path, expected: int) -> list[Path]:
    """Все картинки, на которые ссылается разметка, и проверка, что они есть на диске."""
    referenced: set[str] = set()
    for split in SPLITS:
        data = read_json(dataset_dir / "annotations" / f"{split}.json")
        referenced.update(item["file_name"] for item in data["images"])
    if len(referenced) != expected:
        raise ValueError(f"Expected {expected} unique images, got {len(referenced)}")

    # Один листинг каталога вместо проверки каждого файла: на медленных ФС это в разы быстрее
    on_disk: set[str] = set()
    for folder in {Path(name).parent for name in referenced}:
        if (dataset_dir / folder).is_dir():
            on_disk.update(
                (folder / entry.name).as_posix() for entry in (dataset_dir / folder).iterdir()
            )
    missing = sorted(referenced - on_disk)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing, e.g. {missing[0]}")
    return [dataset_dir / name for name in sorted(referenced)]


def publish(
    settings: S3Settings,
    dataset_dir: Path,
    params_file: Path,
    raw_archive: Path,
    prefix: str,
    version: str,
    workers: int,
    check_only: bool = False,
) -> None:
    client = make_client(settings, max_pool_connections=workers)
    version_prefix = f"{prefix.strip('/')}/{version}/"
    location = f"s3://{settings.bucket}/{version_prefix}"

    if prefix_exists(client, settings.bucket, version_prefix):
        raise VersionExistsError(
            f"Version {version} already exists at {location}. Versions are immutable: "
            "pick a new one, e.g. `make data VERSION=v2`."
        )
    if check_only:
        logger.info("Version %s is free: %s", version, location)
        return

    manifest = build_manifest(dataset_dir, read_json(params_file), raw_archive, version, location)
    if manifest["git"]["dirty"]:
        logger.warning(
            "Working tree has uncommitted changes, commit %s", manifest["git"]["commit"]
        )
    manifest_path = dataset_dir / "manifest.json"
    write_json(manifest_path, manifest)

    images = _collect_images(dataset_dir, manifest["num_images"])
    logger.info("Uploading %d images to %simages/", len(images), location)
    upload_files_if_absent(
        client,
        settings.bucket,
        [(path, f"{version_prefix}images/{path.name}") for path in images],
        workers=workers,
    )
    for entry in manifest["files"]:
        put_file_if_absent(
            client, settings.bucket, version_prefix + entry["path"], dataset_dir / entry["path"]
        )
    put_file_if_absent(client, settings.bucket, version_prefix + "manifest.json", manifest_path)
    logger.info("Published %s: %s", location, manifest["splits"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="Dataset version, e.g. v1")
    parser.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="Prefix in the bucket (default: %(default)s)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Local processed dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=INTERIM_DIR / PARAMS_FILE,
        help="Split parameters written by the split step (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-archive",
        type=Path,
        default=RAW_DIR / ARCHIVE_NAME,
        help="Raw archive, its hashes go to the manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, 4 * (os.cpu_count() or 1)),
        help="Parallel uploads (default: %(default)s)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check that the version does not exist yet",
    )
    args = parser.parse_args(argv)
    setup_logging()
    try:
        publish(
            S3Settings.from_env(),
            args.dataset_dir,
            args.params_file,
            args.raw_archive,
            args.prefix,
            args.version,
            args.workers,
            args.check_only,
        )
    except VersionExistsError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
