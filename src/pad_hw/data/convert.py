"""Шаг convert: pickle-батчи CIFAR-10 -> PNG-картинки и JSON-разметка.

Pickle может выполнить произвольный код, поэтому архив открывается только после
сверки MD5 с официальным значением, а сам pickle читается ограниченным распаковщиком:
разрешены лишь классы numpy, нужные для восстановления массивов.

Результат:
    <images-dir>/train_00000.png ... test_09999.png
    <annotations-dir>/train.json, test.json  (полный train до разбиения)

Запуск: python -m pad_hw.data.convert [--archive PATH] [--images-dir DIR] [--annotations-dir DIR]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import pickle
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from pad_hw.data.cifar import (
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    CLASSES,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TEST_SIZE,
    TRAIN_SIZE,
)
from pad_hw.utils import bytes_sha256, file_digest, setup_logging, write_json

logger = logging.getLogger(__name__)

ImageArray = npt.NDArray[np.uint8]

_BATCHES_DIR = "cifar-10-batches-py"
SPLIT_MEMBERS = {
    "train": [f"{_BATCHES_DIR}/data_batch_{i}" for i in range(1, 6)],
    "test": [f"{_BATCHES_DIR}/test_batch"],
}
SPLIT_SIZES = {"train": TRAIN_SIZE, "test": TEST_SIZE}
_META_MEMBER = f"{_BATCHES_DIR}/batches.meta"
_CHUNK = 1000

_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Распаковщик, который не даёт pickle импортировать ничего, кроме numpy-массивов."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden global in pickle: {module}.{name}")


def _load_member(tar: tarfile.TarFile, name: str) -> dict[bytes, Any]:
    member = tar.extractfile(name)
    if member is None:
        raise ValueError(f"{name} is not a regular file in the archive")
    with member:
        data = RestrictedUnpickler(member, encoding="bytes").load()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {name}, got {type(data).__name__}")
    return data


def read_archive(archive: Path) -> tuple[list[str], dict[str, tuple[ImageArray, list[int]]]]:
    """Читает классы и картинки (N, 32, 32, 3) с метками для train и test прямо из архива."""
    splits: dict[str, tuple[ImageArray, list[int]]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        meta = _load_member(tar, _META_MEMBER)
        classes = [name.decode() for name in meta[b"label_names"]]
        for split, members in SPLIT_MEMBERS.items():
            arrays: list[ImageArray] = []
            labels: list[int] = []
            for name in members:
                batch = _load_member(tar, name)
                data = np.asarray(batch[b"data"], dtype=np.uint8)
                # Строка = 1024 значения R, затем G, затем B
                arrays.append(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                labels.extend(int(label) for label in batch[b"labels"])
            images = np.ascontiguousarray(np.concatenate(arrays))
            if len(images) != SPLIT_SIZES[split] or len(labels) != len(images):
                raise ValueError(f"Unexpected size of {split}: {len(images)} images")
            splits[split] = (images, labels)
    return classes, splits


def encode_png(image: ImageArray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_chunk(task: tuple[Path, list[str], ImageArray]) -> list[str]:
    """Сохраняет картинки в PNG и возвращает sha256 байтов каждого файла."""
    images_dir, image_ids, images = task
    hashes = []
    for image_id, image in zip(image_ids, images, strict=True):
        data = encode_png(image)
        (images_dir / f"{image_id}.png").write_bytes(data)
        hashes.append(bytes_sha256(data))
    return hashes


def _is_complete(images_dir: Path, annotations_dir: Path) -> bool:
    if not all((annotations_dir / f"{split}.json").exists() for split in SPLIT_SIZES):
        return False
    expected = sum(SPLIT_SIZES.values())
    return images_dir.is_dir() and sum(1 for _ in images_dir.glob("*.png")) == expected


def convert(
    archive: Path,
    images_dir: Path,
    annotations_dir: Path,
    expected_md5: str,
    workers: int,
    force: bool = False,
) -> None:
    if not force and _is_complete(images_dir, annotations_dir):
        logger.info("Images and annotations already exist, skipping (use --force to rebuild)")
        return

    actual_md5 = file_digest(archive, "md5")
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Refusing to unpickle {archive}: MD5 {actual_md5} != expected {expected_md5}"
        )

    classes, splits = read_archive(archive)
    if tuple(classes) != CLASSES:
        raise ValueError(f"Unexpected class list in the archive: {classes}")

    images_dir.mkdir(parents=True, exist_ok=True)
    for split, (images, labels) in splits.items():
        image_ids = [f"{split}_{i:05d}" for i in range(len(images))]
        tasks = [
            (images_dir, image_ids[start : start + _CHUNK], images[start : start + _CHUNK])
            for start in range(0, len(images), _CHUNK)
        ]
        logger.info("Writing %d %s images to %s", len(images), split, images_dir)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            hashes = [h for chunk in pool.map(_write_chunk, tasks) for h in chunk]

        annotation = {
            "classes": classes,
            "images": [
                {
                    "image_id": image_id,
                    "file_name": f"images/{image_id}.png",
                    "label": label,
                    "label_name": classes[label],
                    "sha256": sha256,
                }
                for image_id, label, sha256 in zip(image_ids, labels, hashes, strict=True)
            ],
        }
        write_json(annotations_dir / f"{split}.json", annotation)
        logger.info("Wrote %s", annotations_dir / f"{split}.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive",
        type=Path,
        default=RAW_DIR / ARCHIVE_NAME,
        help="Raw archive (default: %(default)s)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROCESSED_DIR / "images",
        help="Output directory for PNG files (default: %(default)s)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=INTERIM_DIR / "annotations",
        help="Output directory for full train/test annotations (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-md5",
        default=ARCHIVE_MD5,
        help="Archive is unpickled only if its MD5 matches (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of processes (default: CPU count)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args(argv)
    setup_logging()
    convert(
        args.archive,
        args.images_dir,
        args.annotations_dir,
        args.expected_md5,
        args.workers,
        args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
