from __future__ import annotations
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_text_lines(path: str | Path, max_lines: Optional[int] = None) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def maybe_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "t")
