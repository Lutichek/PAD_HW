"""Сравнивает HW3 и HTTP-сервис на всей тестовой выборке."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from pad_hw.data.cifar import CLASSES
from pad_hw.serving.dataset import load_test_files


def verify(args: argparse.Namespace) -> int:
    samples, reference = load_test_files(args.config)
    class_count = len(CLASSES)
    with httpx.Client(base_url=args.url, timeout=15) as client:
        client.get("/health").raise_for_status()

        def predict(sample: tuple[Path, int]) -> tuple[float, int, int]:
            path, label = sample
            response = client.post(
                "/predict", files={"file": (path.name, path.read_bytes(), "image/png")}
            )
            response.raise_for_status()
            result = response.json()
            probabilities = np.asarray(result["probabilities"], dtype=np.float64)
            if (
                probabilities.shape != (class_count,)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1.0)
            ):
                raise ValueError(f"Invalid probabilities for {path.name}")
            predicted = int(probabilities.argmax())
            if result["class_id"] != predicted or result["class_name"] != CLASSES[predicted]:
                raise ValueError(f"Inconsistent predicted class for {path.name}")
            probability = max(float(probabilities[label]), np.finfo(np.float64).tiny)
            return float(-np.log(probability)), label, predicted

        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        loss = 0.0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for loss_value, label, predicted in pool.map(predict, samples):
                loss += loss_value
                confusion[label, predicted] += 1

    correct = confusion.diagonal()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    serving = {
        "loss": loss / len(samples),
        "accuracy": float(correct.sum() / len(samples)),
        "macro_f1": float((2 * correct / denominator.clip(min=1)).mean()),
    }
    limits = {"loss": args.loss_tol, "accuracy": args.accuracy_tol, "macro_f1": args.f1_tol}
    print(f"Test images: {len(samples)}")
    print("| Метрика | ДЗ №3 | Сервис | Абс. разница | Допуск |")
    print("|---|---:|---:|---:|---:|")
    passed = True
    for name, actual in serving.items():
        expected = reference["test"][name]
        difference = abs(actual - expected)
        print(f"| {name} | {expected:.9f} | {actual:.9f} | {difference:.9f} | {limits[name]:g} |")
        passed &= difference <= limits[name]
    print("PASS" if passed else "FAIL: test metrics exceed tolerance")
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--loss-tol", type=float, default=1e-5)
    parser.add_argument("--accuracy-tol", type=float, default=1e-4)
    parser.add_argument("--f1-tol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.workers < 1 or not all(
        np.isfinite(tol) and tol >= 0 for tol in (args.loss_tol, args.accuracy_tol, args.f1_tol)
    ):
        parser.error("workers must be positive and tolerances must be finite and nonnegative")
    raise SystemExit(verify(args))


if __name__ == "__main__":
    main()
