from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ctpc_context import get_current_u


def fake_quant_int8_symmetric_per_tensor(x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """
    Symmetric INT8 fake-quant (per-tensor):
      q = clamp(round(x/step), -127, 127)
      x_hat = q * step
    Uses STE for round.
    """
    step = torch.clamp(step, min=torch.finfo(x.dtype).eps)

    y = x / step
    y_round = (torch.round(y) - y).detach() + y  # STE
    y_q = torch.clamp(y_round, -127.0, 127.0)
    return y_q * step


@dataclass
class WrapperStats:
    calls: int = 0
    last_scale: float = 0.0


class QuantActSiteWrapper(nn.Module):
    """
    Quantizes FIRST positional tensor input using CTPCScaleField at the current u.
    """
    def __init__(self, module: nn.Module, site_id: str, scale_field, enabled: bool = True):
        super().__init__()
        self.module = module
        self.site_id = site_id
        self.scale_field = scale_field
        self.enabled = enabled
        self.stats = WrapperStats()

    def forward(self, x, *args, **kwargs):
        self.stats.calls += 1

        if not self.enabled:
            return self.module(x, *args, **kwargs)

        u = get_current_u()
        if u is None:
            raise RuntimeError(
                "QuantActSiteWrapper: current u is None. "
                "Did you forget to set ctpc_u_context (or patch UNet forward with u)?"
            )

        step = self.scale_field.scales_dict(u)[self.site_id].to(device=x.device, dtype=x.dtype).reshape(())
        self.stats.last_scale = float(step.detach().float().cpu().item())

        xq = fake_quant_int8_symmetric_per_tensor(x, step)
        return self.module(xq, *args, **kwargs)
