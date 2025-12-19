from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .logsnr import try_compute_sigma_and_logsnr


@dataclass
class UnetCallRecord:
    call_index: int
    t_raw: Any
    t_float: Optional[float]
    sigma: Optional[float]
    logsnr: Optional[float]
    logsnr_type: str
    notes: str


class SolverCallLogger:
    """
    Patches a diffusers UNet forward to record every call's timestep and derived sigma/logSNR if possible.

    Usage:
      logger = SolverCallLogger(pipe.unet, pipe.scheduler)
      with logger:
          _ = pipe(prompt, ...)
      records = logger.records
    """
    def __init__(self, unet, scheduler):
        self.unet = unet
        self.scheduler = scheduler
        self._orig_forward = None
        self.records: List[Dict[str, Any]] = []
        self._call_index = 0

    def reset(self):
        self.records = []
        self._call_index = 0

    def _wrap_forward(self, orig_forward):
        def wrapped_forward(*args, **kwargs):
            # Diffusers UNet forward signature typically:
            #   forward(sample, timestep, encoder_hidden_states, ...)
            # timestep can be passed as 2nd positional arg or as keyword.
            timestep = None
            if "timestep" in kwargs:
                timestep = kwargs["timestep"]
            elif len(args) >= 2:
                timestep = args[1]

            info = try_compute_sigma_and_logsnr(self.scheduler, timestep)

            rec = {
                "call_index": self._call_index,
                "t_raw": _serialize_timestep(timestep),
                "t_float": info.get("t_float", None),
                "sigma": info.get("sigma", None),
                "logsnr": info.get("logsnr", None),
                "logsnr_type": info.get("logsnr_type", "unknown"),
                "notes": info.get("notes", ""),
            }
            self.records.append(rec)
            self._call_index += 1
            return orig_forward(*args, **kwargs)

        return wrapped_forward

    def __enter__(self):
        if self._orig_forward is not None:
            return self
        self._orig_forward = self.unet.forward
        self.unet.forward = self._wrap_forward(self._orig_forward)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_forward is not None:
            self.unet.forward = self._orig_forward
            self._orig_forward = None
        return False  # do not suppress exceptions


def _serialize_timestep(t) -> Any:
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, torch.Tensor):
        if t.numel() == 1:
            return float(t.detach().cpu().item())
        # Don’t dump huge tensors; just record shape/dtype
        return {"tensor": True, "shape": list(t.shape), "dtype": str(t.dtype)}
    return str(type(t))
