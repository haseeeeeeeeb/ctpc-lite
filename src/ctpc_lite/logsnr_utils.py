from __future__ import annotations

import math
from typing import Tuple

import torch


def logsnr_from_alphas_cumprod(scheduler, t: torch.Tensor) -> torch.Tensor:
    """
    For VP-style schedulers with alphas_cumprod indexed by integer timesteps.
    logSNR = log(alpha_bar / (1 - alpha_bar))
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("Scheduler has no alphas_cumprod; implement another logsnr backend for this solver.")

    ac = scheduler.alphas_cumprod
    if not torch.is_tensor(ac):
        ac = torch.tensor(ac, dtype=torch.float32, device=t.device)

    # t can be float tensor; treat as index (as diffusers often does)
    ti = int(float(t.detach().cpu().item()))
    ti = max(0, min(ti, ac.shape[0] - 1))
    alpha_bar = ac[ti].to(dtype=torch.float32, device=t.device)

    # numerical guard
    alpha_bar = torch.clamp(alpha_bar, 1e-12, 1.0 - 1e-12)
    return torch.log(alpha_bar) - torch.log1p(-alpha_bar)


def u_from_logsnr(logsnr: torch.Tensor, logsnr_min: float, logsnr_max: float) -> torch.Tensor:
    denom = (logsnr_max - logsnr_min)
    if abs(denom) < 1e-12:
        return torch.zeros_like(logsnr)
    u = 2.0 * (logsnr - logsnr_min) / denom - 1.0
    return torch.clamp(u, -1.0, 1.0)
