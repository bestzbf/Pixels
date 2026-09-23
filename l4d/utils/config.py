"""Minimal YAML config loading with dotted access and recursive `_base_` merging."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


class Config:
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def keys(self):
        return self._data.keys()


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base_dir = os.path.dirname(os.path.abspath(path))
    bases = data.pop("_base_", None)
    if isinstance(bases, str):
        bases = [bases]
    for base in reversed(bases or []):
        parent = load_config(os.path.join(base_dir, base)).to_dict()
        data = _merge(parent, data)
    return Config(data)
