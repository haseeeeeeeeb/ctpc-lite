from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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


def _nearest_index(values: torch.Tensor, target: torch.Tensor) -> int:
    # values: [N], target: scalar tensor
    # Return argmin |values - target|
    diffs = torch.abs(values - target)
    return int(torch.argmin(diffs).item())


def try_compute_sigma_and_logsnr(scheduler, timestep: Any) -> Dict[str, Any]:
    """
    Best-effort extraction of sigma/logSNR from a diffusers scheduler and the UNet timestep input.

    We return a dict with:
      - sigma (float or None)
      - logsnr (float or None)
      - logsnr_type: string describing what we computed ("vp", "sigma_only", "unknown")
      - notes: string for debugging

    This is intentionally robust: even if we cannot compute sigma/logsnr reliably,
    we still log raw timestep and continue.
    """
    out: Dict[str, Any] = {"sigma": None, "logsnr": None, "logsnr_type": "unknown", "notes": ""}

    t = timestep
    t_float = _to_float(t)
    out["t_float"] = t_float

    # Strategy 1: VP-style schedules with alphas_cumprod (common in DDIM/DDPM-like schedulers).
    # If timestep looks like an integer index, we compute:
    #   alpha^2 = alphas_cumprod[idx]
    #   sigma^2 = 1 - alpha^2
    #   SNR = alpha^2 / sigma^2
    try:
        if hasattr(scheduler, "alphas_cumprod"):
            ac = scheduler.alphas_cumprod
            if isinstance(ac, np.ndarray):
                ac_t = torch.from_numpy(ac)
            else:
                ac_t = torch.tensor(ac) if not isinstance(ac, torch.Tensor) else ac

            if t_float is not None:
                idx = int(round(t_float))
                if 0 <= idx < ac_t.numel():
                    alpha2 = float(ac_t[idx].item())
                    sigma2 = max(1e-12, 1.0 - alpha2)
                    snr = alpha2 / sigma2
                    out["sigma"] = math.sqrt(sigma2)
                    out["logsnr"] = math.log(max(snr, 1e-20))
                    out["logsnr_type"] = "vp"
                    out["notes"] = "from alphas_cumprod index"
                    return out
    except Exception as e:
        out["notes"] = f"alphas_cumprod failed: {e}"

    # Strategy 2: Karras-style schedulers often have `sigmas` and `timesteps`.
    # Try to map timestep to nearest element in scheduler.timesteps, then use same index for sigmas.
    try:
        if hasattr(scheduler, "sigmas") and hasattr(scheduler, "timesteps"):
            sigmas = scheduler.sigmas
            timesteps = scheduler.timesteps

            sigmas_t = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas)
            timesteps_t = timesteps if isinstance(timesteps, torch.Tensor) else torch.tensor(timesteps)

            if isinstance(t, torch.Tensor) and t.numel() == 1:
                t_scalar = t.detach().cpu()
            else:
                t_scalar = torch.tensor([t_float]) if t_float is not None else None

            if t_scalar is not None:
                # timesteps_t could be float or int; take nearest match
                idx = _nearest_index(timesteps_t.float().cpu(), t_scalar.float().cpu().view(()))
                sigma = float(sigmas_t[idx].item())

                # In sigma-parameterized EDM/Karras view, a usable SNR proxy is 1/sigma^2.
                out["sigma"] = sigma
                out["logsnr"] = -2.0 * math.log(max(sigma, 1e-20))
                out["logsnr_type"] = "sigma_only"
                out["notes"] = "from nearest timestep->sigma; logsnr=-2log(sigma)"
                return out
    except Exception as e:
        out["notes"] = f"sigmas/timesteps mapping failed: {e}"

    return out
