# ctpc_lite/unet_u_patch.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import math
import torch

from .ctpc_context import ctpc_u_context
from .logsnr import try_compute_sigma_and_logsnr


def _extract_timestep(args, kwargs) -> Any:
    if "timestep" in kwargs:
        return kwargs["timestep"]
    if len(args) >= 2:
        return args[1]
    if "timesteps" in kwargs:
        return kwargs["timesteps"]
    raise RuntimeError("Could not locate timestep argument in UNet forward call.")


def _extract_sample(args, kwargs) -> Optional[torch.Tensor]:
    if len(args) >= 1 and torch.is_tensor(args[0]):
        return args[0]
    s = kwargs.get("sample", None)
    if torch.is_tensor(s):
        return s
    return None


def compute_u_from_scheduler(
    scheduler: Any,
    timestep: Any,
    lam_min: float,
    lam_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Best-effort u in [-1,1] from (scheduler, timestep).
    Uses whatever Phase B used to log logsnr (via try_compute_sigma_and_logsnr).
    """
    info = try_compute_sigma_and_logsnr(scheduler, timestep)
    lam = info.get("logsnr", None)
    if not isinstance(lam, (int, float)) or not math.isfinite(float(lam)):
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    denom = max(1e-12, float(lam_max) - float(lam_min))
    u = 2.0 * (float(lam) - float(lam_min)) / denom - 1.0
    u = max(-1.0, min(1.0, float(u)))
    return torch.tensor(u, device=device, dtype=torch.float32)


def patch_unet_forward_with_u(
    unet: Any,
    scheduler: Any,
    lam_min: float,
    lam_max: float,
) -> None:
    """
    Monkey-patch unet.forward to set ctpc_u_context(u) for the duration of each call.
    Safe for inference / validation scripts.
    """
    if hasattr(unet, "_ctpc_orig_forward"):
        return

    orig_forward = unet.forward

    def wrapped_forward(*args, **kwargs):
        timestep = _extract_timestep(args, kwargs)
        sample = _extract_sample(args, kwargs)
        dev = sample.device if (sample is not None) else torch.device("cpu")

        u = compute_u_from_scheduler(scheduler, timestep, lam_min, lam_max, device=dev)
        with ctpc_u_context(u):
            return orig_forward(*args, **kwargs)

    unet._ctpc_orig_forward = orig_forward
    unet.forward = wrapped_forward  # type: ignore[assignment]


def unpatch_unet_forward(unet: Any) -> None:
    if hasattr(unet, "_ctpc_orig_forward"):
        unet.forward = unet._ctpc_orig_forward  # type: ignore[assignment]
        delattr(unet, "_ctpc_orig_forward")
