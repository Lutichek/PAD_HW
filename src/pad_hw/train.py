"""Один запуск обучения CIFAR-10 на CPU."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import mlflow
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pad_hw.experiment_config import check_name, config_params, config_yaml, git_info, load_config
from pad_hw.s3 import make_client, prefix_exists, put_file_if_absent
from pad_hw.settings import S3Settings
from pad_hw.training_data import DatasetCache
from pad_hw.utils import setup_logging, write_json

logger = logging.getLogger(__name__)


def build_model(spec: dict[str, Any], classes: int) -> nn.Module:
    match spec["type"]:
        case "linear":
            return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, classes))
        case "simple_cnn":
            channels = spec["params"]["channels"]
            if len(channels) not in (2, 3) or any(width <= 0 for width in channels):
                raise ValueError("simple_cnn requires two or three positive channel counts")
            layers: list[nn.Module] = []
            inputs = 3
            for width in channels:
                layers.extend(
                    [
                        nn.Conv2d(inputs, width, 3, padding=1),
                        nn.BatchNorm2d(width),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    ]
                )
                inputs = width
            layers.extend([nn.Flatten(), nn.Linear(inputs * (32 >> len(channels)) ** 2, classes)])
            return nn.Sequential(*layers)
        case other:
            raise ValueError(f"Unknown model type: {other}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    norm: tuple[Tensor, Tensor],
    class_count: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], Tensor]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    total_loss = 0.0
    mean, std = norm
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            logits = model((images.float() / 255 - mean) / std)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.numel()
            pairs = labels * class_count + logits.detach().argmax(dim=1)
            confusion += torch.bincount(pairs, minlength=class_count**2).reshape_as(confusion)

    correct = confusion.diag()
    count = int(confusion.sum().item())
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    return {
        "loss": total_loss / count,
        "accuracy": correct.sum().item() / count,
        "macro_f1": (2 * correct / denominator.clamp_min(1)).mean().item(),
    }, confusion


def train(config: dict[str, Any], directory: Path) -> dict[str, Any]:
    seed = config["seed"]
    data = config["data"]
    settings = S3Settings.from_env()
    client = make_client(settings)
    experiment, name = config["experiment_name"], config["run_name"]
    check_name(experiment)
    check_name(name)
    prefix = f"models/{experiment}/{name}"
    if prefix_exists(client, settings.bucket, f"{prefix}/"):
        raise ValueError(f"s3://{settings.bucket}/{prefix}/ exists; choose another run_name")

    params = config["train"]
    epochs, batch_size = params["epochs"], params["batch_size"]
    limit = data["max_train_samples"]
    if min(epochs, batch_size, params["lr"], params["threads"]) <= 0 or (
        limit is not None and limit <= 0
    ):
        raise ValueError("Training parameters must be positive")
    torch.set_num_threads(params["threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    revision = git_info()
    mlflow.set_experiment(experiment)
    (directory / "config.yaml").write_text(config_yaml(config), encoding="utf-8")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(config_params(config) | revision)
        mlflow.set_tag("mlflow.source.git.commit", revision["git_commit"])
        mlflow.log_artifact(str(directory / "config.yaml"))
        cache = DatasetCache(settings, data["s3_path"], Path("data/cache"))
        train_data, classes = cache.split("train", seed, limit)
        val_data, val_classes = cache.split("val", seed)
        if classes != val_classes:
            raise ValueError("Class order differs between train and validation")
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(val_data, batch_size=batch_size)
        norm = (
            torch.tensor(cache.stats["mean"]).reshape(1, 3, 1, 1),
            torch.tensor(cache.stats["std"]).reshape(1, 3, 1, 1),
        )
        model = build_model(config["model"], len(classes)).to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        checkpoint = directory / "model.pt"
        best_loss = float("inf")
        started = perf_counter()
        for epoch in range(1, epochs + 1):
            run_epoch(model, train_loader, norm, len(classes), optimizer)
            val, confusion = run_epoch(model, val_loader, norm, len(classes))
            mlflow.log_metrics({f"val_{key}": value for key, value in val.items()}, step=epoch)
            logger.info("Epoch %d/%d, val: %s", epoch, epochs, val)
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                best_epoch, best_val, best_confusion = epoch, val, confusion
                torch.save(model.state_dict(), checkpoint)
        elapsed = perf_counter() - started

        test_data, test_classes = cache.split("test", seed)
        if classes != test_classes:
            raise ValueError("Class order differs between train and test")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        test_loader = DataLoader(test_data, batch_size=batch_size)
        test, _ = run_epoch(model, test_loader, norm, len(classes))
        mlflow.log_metrics(
            {f"test_{key}": value for key, value in test.items()}
            | {f"best_val_{key}": value for key, value in best_val.items()}
            | {"training_seconds": elapsed, "best_epoch": best_epoch}
        )
        confusion_file = directory / "val_confusion.json"
        write_json(
            confusion_file,
            {
                "columns": ["actual / predicted", *classes],
                "data": [
                    [label, *counts]
                    for label, counts in zip(classes, best_confusion.tolist(), strict=True)
                ],
            },
        )
        mlflow.log_artifact(str(confusion_file))
        metrics = {
            "experiment_name": experiment,
            "run_name": name,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "git_commit": revision["git_commit"],
            "data": data["s3_path"],
            "model": config["model"]["type"],
            "lr": params["lr"],
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "training_seconds": elapsed,
            "val": best_val,
            "test": test,
            "s3_uri": f"s3://{settings.bucket}/{prefix}/",
        }
        write_json(directory / "metrics.json", metrics)
        for filename in ("model.pt", "config.yaml", "metrics.json"):
            path = directory / filename
            put_file_if_absent(client, settings.bucket, f"{prefix}/{filename}", path)
            if filename != "config.yaml":
                mlflow.log_artifact(str(path))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    setup_logging()
    with TemporaryDirectory() as directory:
        metrics = train(load_config(args.config), Path(directory))
    report = Path("outputs") / metrics["experiment_name"] / f"{metrics['run_name']}.json"
    write_json(report, metrics)
    logger.info("Test: %s; results: %s", metrics["test"], report)


if __name__ == "__main__":
    main()
