from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .ctpc_context import get_current_u


def fake_quant_int8_symmetric_per_tensor(x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """
    Symmetric INT8 fake-quant: x -> step * clamp(round(x/step), -127, 127)
    Uses STE so gradients flow as identity through rounding.
    step is a scalar tensor on the same device.
    """
    # Avoid divide-by-zero
    step = torch.clamp(step, min=1e-12)

    y = x / step
    y_clamped = torch.clamp(y, -127.0, 127.0)

    # STE for round: forward uses round, backward uses identity
    y_rounded = torch.round(y_clamped)
    y_ste = (y_rounded - y_clamped).detach() + y_clamped

    return y_ste * step


@dataclass
class WrapperStats:
    calls: int = 0
    last_scale: float = 0.0


class QuantActSiteWrapper(nn.Module):
    """
    Wraps a module (Linear/Conv/etc). Quantizes the FIRST positional input (activation)
    using CTPCScaleField at the current u pulled from thread-local context.
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
            raise RuntimeError("QuantActSiteWrapper: current u is None. "
                               "Did you forget to set ctpc_u_context before UNet forward?")

        # u is scalar tensor; scale_field returns scalar tensor step
        step = self.scale_field.scales_dict(u)[self.site_id]
        self.stats.last_scale = float(step.detach().cpu().item())

        xq = fake_quant_int8_symmetric_per_tensor(x, step)
        return self.module(xq, *args, **kwargs)
