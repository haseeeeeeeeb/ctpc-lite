# ctpc_lite/phase_c_wrappers.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .phase_c_u_context import get_current_u
from .phase_c_quant import fake_quant_int8_symmetric_per_tensor


class QuantActSiteWrapper(nn.Module):
    """
    Wraps a module. On forward:
      - reads current u from global context
      - queries scale_field for scale at this site_id
      - applies fake-quant to the FIRST tensor input
      - forwards to wrapped module

    Supports two scale_field APIs:
      (A) legacy phase_c_scale_field.CTPCScaleField: get_scale(site_id, u, dtype, device)
      (B) ctpc_scale_field.CTPCScaleField: scales_dict(u) or scales_tensor(u)
    """

    def __init__(self, module: nn.Module, site_id: str, scale_field, enabled: bool = True):
        super().__init__()
        self.module = module
        self.site_id = site_id
        self.scale_field = scale_field
        self.enabled = enabled

        # for Phase C validation/debug
        self.call_count: int = 0
        self.last_scale: Optional[float] = None

    def _get_scale(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        sf = self.scale_field

        # API (A): legacy
        if hasattr(sf, "get_scale"):
            scale = sf.get_scale(self.site_id, u=u, dtype=x.dtype, device=x.device)
            return scale

        # API (B): ctpc_scale_field
        if hasattr(sf, "scales_dict"):
            d = sf.scales_dict(u)
            scale = d[self.site_id]
            return scale.to(device=x.device, dtype=x.dtype)

        raise RuntimeError("scale_field does not expose get_scale() or scales_dict().")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        self.call_count += 1

        if not self.enabled:
            return self.module(x, *args, **kwargs)

        u = get_current_u()  # scalar tensor
        scale = self._get_scale(u, x)

        # store for debugging
        try:
            self.last_scale = float(scale.detach().float().cpu().item())
        except Exception:
            self.last_scale = None

        xq = fake_quant_int8_symmetric_per_tensor(x, scale)
        return self.module(xq, *args, **kwargs)
