"""Файлы test из версии данных, записанной в конфиге чекпоинта."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pad_hw.data.cifar import CLASSES
from pad_hw.experiment_config import load_config
from pad_hw.s3 import download_file, make_client
from pad_hw.settings import S3Settings
from pad_hw.training_data import DatasetCache
from pad_hw.utils import read_json


def load_test_files(config_path: Path) -> tuple[list[tuple[Path, int]], dict[str, Any]]:
    checkpoint = load_config(config_path)["checkpoint"].strip("/")
    settings = S3Settings.from_env()
    client = make_client(settings)
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        for filename in ("config.yaml", "metrics.json"):
            download_file(client, settings.bucket, f"{checkpoint}/{filename}", temp / filename)
        config = load_config(temp / "config.yaml")
        reference = read_json(temp / "metrics.json")
    version = config["data"]["s3_path"].strip("/")
    cache_root = Path(os.environ.get("SERVING_CACHE", "data/cache"))
    cache = DatasetCache(settings, version, cache_root)
    with cache.lock:
        annotation = read_json(cache.fetch("annotations/test.json"))
        if tuple(annotation["classes"]) != CLASSES or not annotation["images"]:
            raise ValueError(f"Invalid CIFAR-10 test annotation in {version}")
        items = annotation["images"]
        with ThreadPoolExecutor(max_workers=24) as pool:
            paths = list(pool.map(cache.fetch, (item["file_name"] for item in items)))
    return list(zip(paths, (int(item["label"]) for item in items), strict=True)), reference
