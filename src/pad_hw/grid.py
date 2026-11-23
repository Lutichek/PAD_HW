"""Запускает baseline и сетку, каждый конфиг в отдельном контейнере."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any
from uuid import uuid4

from pad_hw.experiment_config import check_name, config_yaml, load_config
from pad_hw.utils import read_json, write_json


def grid_configs(path: Path, series: str) -> list[dict[str, Any]]:
    spec = load_config(path)
    base = load_config(path.parent / spec["base"])
    baseline = load_config(path.parent / spec["baseline"])
    if baseline["experiment_name"] != base["experiment_name"]:
        raise ValueError("Baseline and grid must use the same experiment_name")
    baseline["run_name"] += f"-{series}"
    axes = spec["grid"]
    if not axes or any(not isinstance(values, list) or not values for values in axes.values()):
        raise ValueError("Every grid parameter needs a non-empty list of values")
    configs = [baseline]
    for index, values in enumerate(product(*axes.values()), start=1):
        config = deepcopy(base)
        name_parts = [base["run_name"]]
        for key, value in zip(axes, values, strict=True):
            target = config
            *parents, leaf = key.split(".")
            for parent in parents:
                target = target[parent]
            if leaf not in target:
                raise ValueError(f"Unknown grid parameter: {key}")
            target[leaf] = value
            if isinstance(value, int | float):
                name_parts.append(f"{leaf}{value}")
        config["run_name"] = "-".join([*name_parts, f"{index:02d}", series])
        configs.append(config)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--compose", default="docker compose")
    args = parser.parse_args()
    series = uuid4().hex[:8]
    configs = grid_configs(args.config, series)
    command = shlex.split(args.compose)
    command.extend(["run", "--rm", "--no-deps", "-T", "trainer", "/dev/stdin"])
    reports: list[dict[str, Any]] = []
    for config in configs:
        experiment, name = config["experiment_name"], config["run_name"]
        check_name(experiment)
        check_name(name)
        print(f"Running {name}", flush=True)
        subprocess.run(command, input=config_yaml(config).encode(), check=True)
        reports.append(read_json(Path("outputs") / experiment / f"{name}.json"))

    ranking = sorted(reports[1:], key=lambda report: report["val"]["loss"])
    best = min(reports, key=lambda report: report["val"]["loss"])
    summary = Path("outputs") / f"series-{series}.json"
    write_json(summary, {"baseline": reports[0], "grid_by_val_loss": ranking, "selected": best})
    print(f"Selected by val loss: {best['run_name']}; summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
