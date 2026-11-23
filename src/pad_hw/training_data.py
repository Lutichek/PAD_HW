"""Локальный кеш опубликованной версии CIFAR-10."""

from __future__ import annotations

import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from torch.utils.data import TensorDataset

from pad_hw.s3 import download_file, make_client
from pad_hw.settings import S3Settings
from pad_hw.utils import read_json

logger = logging.getLogger(__name__)


class DatasetCache:
    def __init__(self, settings: S3Settings, version: str, root: Path) -> None:
        self.settings = settings
        self.version = version.strip("/")
        if not self.version or ".." in self.version.split("/"):
            raise ValueError(f"Invalid S3 prefix: {version!r}")
        namespace = f"{settings.endpoint_url}/{settings.bucket}/{self.version}"
        self.root = root / hashlib.sha256(namespace.encode()).hexdigest()
        self.client = make_client(settings)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.root / ".download.lock")
        with self.lock:
            self.fetch("manifest.json")
            self.stats = read_json(self.fetch("stats.json"))

    def fetch(self, name: str) -> Path:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid dataset path: {name!r}")
        path = self.root / relative
        if not path.is_file():
            download_file(self.client, self.settings.bucket, f"{self.version}/{name}", path)
        return path

    def split(
        self, name: str, seed: int, limit: int | None = None
    ) -> tuple[TensorDataset, list[str]]:
        with self.lock:
            annotation = read_json(self.fetch(f"annotations/{name}.json"))
            items = annotation["images"]
            if limit is not None and limit < len(items):
                items = random.Random(seed).sample(items, limit)
            if not items:
                raise ValueError(f"Empty split: {name}")
            logger.info("Preparing %s: %d images", name, len(items))
            names = [item["file_name"] for item in items]
            with ThreadPoolExecutor(max_workers=24) as pool:
                paths = list(pool.map(self.fetch, names))

        pixels = np.empty((len(paths), 32, 32, 3), dtype=np.uint8)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                pixels[index] = np.asarray(image.convert("RGB"))
        labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
        return (
            TensorDataset(torch.from_numpy(pixels).permute(0, 3, 1, 2), labels),
            list(annotation["classes"]),
        )
