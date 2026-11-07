"""Шаг split: стратифицированное разбиение исходного train на train и val.

Картинки не переносятся и не переименовываются: меняются только файлы разметки.
train.json и val.json вместе содержат все картинки исходного train, каждую ровно один раз.
Тот же seed и размер валидации дают побайтно те же файлы. Исходный test копируется без изменений.

Запуск: python -m pad_hw.data.split --val-size 5000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pad_hw.data.cifar import INTERIM_DIR, PROCESSED_DIR
from pad_hw.utils import read_json, setup_logging, write_json

logger = logging.getLogger(__name__)

DEFAULT_VAL_SIZE = 5000
DEFAULT_SEED = 42
PARAMS_FILE = "split_params.json"


def class_quotas(counts: dict[int, int], val_size: int) -> dict[int, int]:
    """Сколько картинок каждого класса отправить в val (метод наибольшего остатка)."""
    total = sum(counts.values())
    exact = {label: val_size * count / total for label, count in counts.items()}
    quotas = {label: int(value) for label, value in exact.items()}
    remainder = val_size - sum(quotas.values())
    by_fraction = sorted(counts, key=lambda label: (-(exact[label] - quotas[label]), label))
    for label in by_fraction[:remainder]:
        quotas[label] += 1
    return quotas


def stratified_split(
    labels: Sequence[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Возвращает отсортированные индексы (train, val)."""
    if not 0 < val_size < len(labels):
        raise ValueError(f"val_size must be between 1 and {len(labels) - 1}, got {val_size}")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    quotas = class_quotas({label: len(idx) for label, idx in by_class.items()}, val_size)

    rng = np.random.default_rng(seed)
    val: set[int] = set()
    for label in sorted(by_class):
        indices = by_class[label]
        if quotas[label] >= len(indices):
            raise ValueError(f"val_size {val_size} leaves no train images for class {label}")
        chosen = rng.permutation(len(indices))[: quotas[label]]
        val.update(indices[i] for i in chosen)

    train = [i for i in range(len(labels)) if i not in val]
    return train, sorted(val)


def split(
    source_dir: Path, output_dir: Path, params_dir: Path, val_size: int, seed: int
) -> dict[str, int]:
    full_train: dict[str, Any] = read_json(source_dir / "train.json")
    test: dict[str, Any] = read_json(source_dir / "test.json")
    images: list[dict[str, Any]] = full_train["images"]

    train_idx, val_idx = stratified_split([item["label"] for item in images], val_size, seed)
    outputs = {
        "train": {"classes": full_train["classes"], "images": [images[i] for i in train_idx]},
        "val": {"classes": full_train["classes"], "images": [images[i] for i in val_idx]},
        "test": test,
    }
    for name, data in outputs.items():
        write_json(output_dir / f"{name}.json", data)

    sizes = {name: len(data["images"]) for name, data in outputs.items()}
    params = {"val_size": val_size, "seed": seed, "stratified": True, "sizes": sizes}
    write_json(params_dir / PARAMS_FILE, params)
    logger.info("Split done: %s (val_size=%d, seed=%d)", sizes, val_size, seed)
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help="Number of validation images (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=INTERIM_DIR / "annotations",
        help="Full train/test annotations from convert (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "annotations",
        help="Output directory for train/val/test.json (default: %(default)s)",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=INTERIM_DIR,
        help="Where to save split parameters for the manifest (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    setup_logging()
    split(args.source_dir, args.output_dir, args.params_dir, args.val_size, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
