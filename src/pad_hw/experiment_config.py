"""Чтение YAML-конфигов экспериментов."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


def git_info() -> dict[str, str]:
    commit = os.environ.get("PAD_GIT_COMMIT")
    if commit:
        return {"git_commit": commit, "git_dirty": os.environ.get("PAD_GIT_DIRTY", "")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return {"git_commit": commit, "git_dirty": str(bool(dirty)).lower()}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def config_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def check_name(name: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,149}", name):
        raise ValueError(f"Invalid experiment or run name: {name!r}")


def config_params(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    params = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            params.update(config_params(value, f"{name}."))
        else:
            params[name] = (
                yaml.safe_dump(value, default_flow_style=True).removesuffix("...\n").strip()
            )
    return params
