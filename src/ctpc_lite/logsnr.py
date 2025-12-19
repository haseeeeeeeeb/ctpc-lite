# src/ctpc_lite/logsnr.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return None
    return None


def _as_1d_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().flatten().to(dtype=torch.float32, device="cpu")
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).flatten().to(dtype=torch.float32, device="cpu")
    return torch.tensor(list(x), dtype=torch.float32).flatten()


def _interp_indexed(values: torch.Tensor, idx: float) -> float:
    # values: [N] on CPU float32, idx in [0, N-1]
    n = int(values.numel())
    if n <= 0:
        raise ValueError("empty values")
    idx = float(max(0.0, min(float(n - 1), float(idx))))
    i0 = int(math.floor(idx))
    i1 = min(n - 1, i0 + 1)
    w = idx - float(i0)
    v0 = float(values[i0].item())
    v1 = float(values[i1].item())
    return (1.0 - w) * v0 + w * v1


def _nearest_index(values: torch.Tensor, target: float) -> int:
    diffs = torch.abs(values - float(target))
    return int(torch.argmin(diffs).item())


def try_compute_sigma_and_logsnr(scheduler, timestep: Any) -> Dict[str, Any]:
    """
    Best-effort extraction of sigma/logSNR from a diffusers scheduler and the UNet timestep input.

    Returns keys:
      - t_float
      - sigma
      - logsnr
      - logsnr_type
      - notes
    """
    out: Dict[str, Any] = {"sigma": None, "logsnr": None, "logsnr_type": "unknown", "notes": ""}

    t_float = _to_float(timestep)
    out["t_float"] = t_float

    # ---- Strategy 1: VP/DDPM-style via alphas_cumprod (supports fractional index via interpolation) ----
    try:
        if hasattr(scheduler, "alphas_cumprod") and t_float is not None:
            ac = _as_1d_tensor(scheduler.alphas_cumprod)  # alpha_bar(t) in [0,1]
            # interpret timestep as an index into alphas_cumprod
            alpha_bar = _interp_indexed(ac, t_float)
            alpha_bar = min(max(alpha_bar, 1e-12), 1.0 - 1e-12)

            # sigma^2 = 1 - alpha_bar ; SNR = alpha_bar/(1-alpha_bar)
            sigma2 = max(1e-12, 1.0 - alpha_bar)
            snr = alpha_bar / sigma2

            out["sigma"] = math.sqrt(sigma2)
            out["logsnr"] = math.log(max(snr, 1e-30))
            out["logsnr_type"] = "vp_interp"
            out["notes"] = "from alphas_cumprod (interp by timestep index)"
            return out
    except Exception as e:
        out["notes"] = f"alphas_cumprod failed: {e}"

    # ---- Strategy 2: Karras/EDM-style via sigmas + timesteps (nearest match) ----
    try:
        if hasattr(scheduler, "sigmas") and hasattr(scheduler, "timesteps") and t_float is not None:
            sigmas = _as_1d_tensor(scheduler.sigmas)
            timesteps = _as_1d_tensor(scheduler.timesteps)

            idx = _nearest_index(timesteps, t_float)
            sigma = float(sigmas[idx].item())

            out["sigma"] = sigma
            # proxy SNR ~ 1/sigma^2  => logSNR = -2 log sigma
            out["logsnr"] = -2.0 * math.log(max(sigma, 1e-30))
            out["logsnr_type"] = "sigma_only"
            out["notes"] = "from nearest(timestep)->sigma; logsnr=-2log(sigma)"
            return out
    except Exception as e:
        out["notes"] = f"sigmas/timesteps mapping failed: {e}"

    return out
