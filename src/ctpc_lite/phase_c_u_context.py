# ctpc_lite/phase_c_u_context.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class UContextState:
    u: Optional[torch.Tensor] = None  # scalar tensor on same device as activations


# Global (process-local) context. Fine for single-process experiments.
_STATE = UContextState()


def get_current_u() -> torch.Tensor:
    """
    Returns current u (scalar tensor). Raises if not set.
    """
    if _STATE.u is None:
        raise RuntimeError(
            "CTPC-Lite: current u is not set. "
            "Did you forget to wrap UNet calls with `set_u(u)`?"
        )
    return _STATE.u


@contextmanager
def set_u(u: torch.Tensor):
    """
    Context manager to set the current u for the duration of a UNet call.
    u should be a scalar tensor (or shape [1]) on the correct device.
    """
    prev = _STATE.u
    _STATE.u = u
    try:
        yield
    finally:
        _STATE.u = prev
