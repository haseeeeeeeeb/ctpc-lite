# ctpc_lite/phase_c_unet_u_patch.py
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import math
import torch

from .phase_c_u_context import set_u


def _interp_alphas_cumprod(alphas_cumprod: torch.Tensor, t: float) -> float:
    """
    Linear interpolation of alpha_bar over integer index t in [0, T-1].
    Works for fractional timesteps too.
    """
    T = alphas_cumprod.shape[0]
    t = max(0.0, min(float(T - 1), float(t)))
    i0 = int(math.floor(t))
    i1 = min(T - 1, i0 + 1)
    w = t - i0
    a0 = float(alphas_cumprod[i0].item())
    a1 = float(alphas_cumprod[i1].item())
    return (1.0 - w) * a0 + w * a1


def compute_logsnr_vp_from_scheduler(scheduler: Any, timestep: Any) -> float:
    """
    Compute logSNR assuming VP/SD-style parameterization using scheduler.alphas_cumprod:
        alpha_bar = alphas_cumprod[t]
        logSNR = log(alpha_bar) - log(1 - alpha_bar)

    timestep may be:
      - torch.Tensor scalar
      - python float/int
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("Scheduler does not expose alphas_cumprod; cannot compute VP logSNR.")

    acp = scheduler.alphas_cumprod
    if torch.is_tensor(acp):
        acp_t = acp.detach().float().cpu()
    else:
        acp_t = torch.tensor(acp, dtype=torch.float32)

    if torch.is_tensor(timestep):
        t_val = float(timestep.detach().float().cpu().item())
    else:
        t_val = float(timestep)

    alpha_bar = _interp_alphas_cumprod(acp_t, t_val)
    # numeric guards
    alpha_bar = min(max(alpha_bar, 1e-12), 1.0 - 1e-12)
    logsnr = math.log(alpha_bar) - math.log(1.0 - alpha_bar)
    return logsnr


def compute_logsnr_range_from_scheduler(scheduler: Any) -> Tuple[float, float]:
    """
    Compute global min/max logSNR from scheduler.alphas_cumprod over all indices.
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("Scheduler does not expose alphas_cumprod; cannot compute VP logSNR range.")

    acp = scheduler.alphas_cumprod
    if torch.is_tensor(acp):
        acp_t = acp.detach().float().cpu()
    else:
        acp_t = torch.tensor(acp, dtype=torch.float32)

    vals = []
    for i in range(acp_t.shape[0]):
        a = float(acp_t[i].item())
        a = min(max(a, 1e-12), 1.0 - 1e-12)
        vals.append(math.log(a) - math.log(1.0 - a))
    return (min(vals), max(vals))


def patch_unet_forward_with_u(unet: Any, scheduler: Any, logsnr_minmax: Optional[Tuple[float, float]] = None) -> None:
    """
    Monkey-patches unet.forward so that during each forward we:
      - compute logSNR(t) using scheduler
      - normalize u in [-1, 1]
      - set_u(u) during the call
    """
    if hasattr(unet, "_ctpc_orig_forward"):
        # already patched
        return

    if logsnr_minmax is None:
        logsnr_min, logsnr_max = compute_logsnr_range_from_scheduler(scheduler)
    else:
        logsnr_min, logsnr_max = logsnr_minmax

    denom = (logsnr_max - logsnr_min) if (logsnr_max > logsnr_min) else 1.0

    orig_forward = unet.forward

    def wrapped_forward(*args, **kwargs):
        # diffusers UNet2DConditionModel forward signature typically:
        # forward(sample, timestep, encoder_hidden_states, ...)
        if len(args) >= 2:
            timestep = args[1]
        else:
            timestep = kwargs.get("timestep", None)
            if timestep is None:
                timestep = kwargs.get("timesteps", None)
        if timestep is None:
            raise RuntimeError("Could not locate 'timestep' argument in UNet forward call.")

        logsnr = compute_logsnr_vp_from_scheduler(scheduler, timestep)
        u_val = 2.0 * ((logsnr - logsnr_min) / denom) - 1.0
        u_val = max(-1.0, min(1.0, float(u_val)))

        # Put u on the same device as the sample (arg0)
        sample0 = args[0] if len(args) >= 1 else kwargs.get("sample", None)
        if sample0 is None or not torch.is_tensor(sample0):
            # fallback: cpu tensor
            u = torch.tensor(u_val, dtype=torch.float32)
        else:
            u = torch.tensor(u_val, device=sample0.device, dtype=torch.float32)

        with set_u(u):
            return orig_forward(*args, **kwargs)

    unet._ctpc_orig_forward = orig_forward
    unet.forward = wrapped_forward  # type: ignore[assignment]


def unpatch_unet_forward(unet: Any) -> None:
    if hasattr(unet, "_ctpc_orig_forward"):
        unet.forward = unet._ctpc_orig_forward  # type: ignore[assignment]
        delattr(unet, "_ctpc_orig_forward")
