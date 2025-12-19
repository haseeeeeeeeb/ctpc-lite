# ctpc_lite/phase_c_wrappers.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .phase_c_u_context import get_current_u
from .phase_c_quant import fake_quant_int8_symmetric_per_tensor
from .phase_c_scale_field import CTPCScaleField


class QuantActSiteWrapper(nn.Module):
    """
    Wraps a module. On forward:
      - reads current u from global context
      - queries scale_field for scale at this site_id
      - applies fake-quant to the FIRST tensor input
      - forwards to wrapped module
    """
    def __init__(self, module: nn.Module, site_id: str, scale_field: CTPCScaleField, enabled: bool = True):
        super().__init__()
        self.module = module
        self.site_id = site_id
        self.scale_field = scale_field
        self.enabled = enabled

        # for Phase C validation/debug
        self.call_count: int = 0
        self.last_scale: Optional[float] = None

    def forward(self, x: torch.Tensor, *args, **kwargs):
        self.call_count += 1

        if not self.enabled:
            return self.module(x, *args, **kwargs)

        u = get_current_u()  # scalar tensor
        scale = self.scale_field.get_scale(
            self.site_id,
            u=u,
            dtype=x.dtype,
            device=x.device,
        )
        # store for debugging
        try:
            self.last_scale = float(scale.detach().float().cpu().item())
        except Exception:
            self.last_scale = None

        xq = fake_quant_int8_symmetric_per_tensor(x, scale)
        return self.module(xq, *args, **kwargs)
