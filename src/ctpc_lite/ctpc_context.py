from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import torch


class _TLS(threading.local):
    u: Optional[torch.Tensor] = None


_tls = _TLS()


def set_current_u(u: torch.Tensor) -> None:
    _tls.u = u


def get_current_u() -> Optional[torch.Tensor]:
    return _tls.u


@contextmanager
def ctpc_u_context(u: torch.Tensor):
    prev = _tls.u
    _tls.u = u
    try:
        yield
    finally:
        _tls.u = prev
