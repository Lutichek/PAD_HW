"""Шаг stats: среднее и стандартное отклонение по каналам R, G, B только по train.

Пиксели нормированы в [0, 1] (значение / 255). Суммы считаются в целых числах,
поэтому результат точный и не зависит от порядка обработки файлов.

Запуск: python -m pad_hw.data.stats [--annotations PATH] [--root DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pad_hw.data.cifar import PROCESSED_DIR
from pad_hw.utils import read_json, setup_logging, write_json

logger = logging.getLogger(__name__)

_CHUNK = 1000


def _channel_sums(paths: list[Path]) -> tuple[list[int], list[int], int]:
    """Суммы и суммы квадратов значений пикселей по каналам, плюс число пикселей."""
    sums = np.zeros(3, dtype=np.int64)
    squares = np.zeros(3, dtype=np.int64)
    pixels = 0
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.int64).reshape(-1, 3)
        sums += array.sum(axis=0)
        squares += (array * array).sum(axis=0)
        pixels += array.shape[0]
    return sums.tolist(), squares.tolist(), pixels


def compute_stats(annotations: Path, root: Path, workers: int) -> dict[str, Any]:
    data: dict[str, Any] = read_json(annotations)
    paths = [root / item["file_name"] for item in data["images"]]
    chunks = [paths[i : i + _CHUNK] for i in range(0, len(paths), _CHUNK)]

    sums, squares, pixels = [0, 0, 0], [0, 0, 0], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_sums, chunk_squares, chunk_pixels in pool.map(_channel_sums, chunks):
            sums = [a + b for a, b in zip(sums, chunk_sums, strict=True)]
            squares = [a + b for a, b in zip(squares, chunk_squares, strict=True)]
            pixels += chunk_pixels

    mean = [s / pixels / 255 for s in sums]
    std = [
        math.sqrt(q / pixels - (s / pixels) ** 2) / 255 for s, q in zip(sums, squares, strict=True)
    ]
    return {
        "split": "train",
        "num_images": len(paths),
        "channels": ["R", "G", "B"],
        "pixel_scale": "value / 255",
        "mean": [round(value, 6) for value in mean],
        "std": [round(value, 6) for value in std],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROCESSED_DIR / "annotations" / "train.json",
        help="Train annotations (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROCESSED_DIR,
        help="Dataset root: file_name paths are relative to it (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "stats.json",
        help="Output file (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of processes (default: CPU count)",
    )
    args = parser.parse_args(argv)
    setup_logging()
    stats = compute_stats(args.annotations, args.root, args.workers)
    write_json(args.output, stats)
    logger.info("mean=%s std=%s -> %s", stats["mean"], stats["std"], args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
