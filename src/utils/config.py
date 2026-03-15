"""YAML configuration loader with CLI override support."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Recursively convert a SimpleNamespace back to a dict."""
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = _namespace_to_dict(v)
        else:
            d[k] = v
    return d


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path, cli_args: list[str] | None = None) -> SimpleNamespace:
    """Load a YAML config file and merge with optional CLI overrides.

    CLI overrides use dotted notation, e.g.:
        --models.cnn.pretrained false --lr 0.0005 --epochs 100

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML file.
    cli_args : list[str] | None
        Raw CLI tokens; if *None*, ``sys.argv[1:]`` is **not** parsed
        (i.e. no CLI overrides are applied).

    Returns
    -------
    SimpleNamespace
        Nested namespace with all configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # --- Apply CLI overrides ---------------------------------------------------
    if cli_args is not None:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        # Dynamically add every CLI token that looks like a flag
        i = 0
        keys: list[str] = []
        while i < len(cli_args):
            token = cli_args[i]
            if token.startswith("--"):
                key = token.lstrip("-")
                parser.add_argument(token, dest=key, default=None)
                keys.append(key)
            i += 1

        parsed, _ = parser.parse_known_args(cli_args)

        for key in keys:
            value = getattr(parsed, key, None)
            if value is None:
                continue
            # Auto-cast
            value = _auto_cast(value)
            # Split dotted key and merge
            parts = key.split(".")
            nested: dict = {}
            current = nested
            for p in parts[:-1]:
                current[p] = {}
                current = current[p]
            current[parts[-1]] = value
            cfg = _deep_update(cfg, nested)

    return _dict_to_namespace(cfg)


def _auto_cast(value: str) -> Any:
    """Best-effort cast of a string to int / float / bool / list."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # List notation: "[1,5,10]"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [_auto_cast(v.strip()) for v in inner.split(",") if v.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
