from __future__ import annotations

from typing import Any, Tuple


def _step(obj: Any, token: str) -> Any:
    # token can be attr or integer index for list/ModuleList
    if token.isdigit():
        return obj[int(token)]
    return getattr(obj, token)


def get_by_path(root: Any, path: str) -> Any:
    cur = root
    for tok in path.split("."):
        cur = _step(cur, tok)
    return cur


def set_by_path(root: Any, path: str, new_value: Any) -> None:
    parts = path.split(".")
    if len(parts) == 1:
        setattr(root, parts[0], new_value)
        return

    parent = root
    for tok in parts[:-1]:
        parent = _step(parent, tok)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_value
    else:
        setattr(parent, last, new_value)
