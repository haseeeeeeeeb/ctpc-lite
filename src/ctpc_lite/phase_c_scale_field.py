# ctpc_lite/phase_c_scale_field.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def chebyshev_features(u: torch.Tensor, r: int = 4) -> torch.Tensor:
    """
    Returns Chebyshev T0..T_{r-1} at u in [-1,1].
    u: scalar tensor
    Output: shape [r]
    """
    # Ensure scalar
    u = u.reshape(())
    feats = []
    if r >= 1:
        feats.append(torch.ones_like(u))         # T0
    if r >= 2:
        feats.append(u)                          # T1
    if r >= 3:
        feats.append(2 * u * u - 1)              # T2
    if r >= 4:
        feats.append(4 * u * u * u - 3 * u)      # T3
    # For r>4, recurrence
    for k in range(4, r):
        # T_k = 2*u*T_{k-1} - T_{k-2}
        feats.append(2 * u * feats[-1] - feats[-2])
    return torch.stack(feats, dim=0)  # [r]


@dataclass
class ScaleFieldConfig:
    rank: int = 4
    clamp_delta: float = 2.0  # clamp log_s around log_s0 ± delta


class CTPCScaleField(nn.Module):
    """
    log s_l(u) = log s0_l + a_l^T z(u)
    where z(u) is Chebyshev basis.
    """
    def __init__(self, site_ids: List[str], log_s0: Dict[str, float], cfg: ScaleFieldConfig):
        super().__init__()
        self.site_ids = list(site_ids)
        self.rank = int(cfg.rank)
        self.clamp_delta = float(cfg.clamp_delta)

        # Buffers of log_s0
        # store as tensor aligned to site index for speed
        log_s0_vec = []
        for sid in self.site_ids:
            if sid not in log_s0:
                raise KeyError(f"Missing log_s0 for site_id='{sid}'")
            log_s0_vec.append(float(log_s0[sid]))
        self.register_buffer("log_s0_vec", torch.tensor(log_s0_vec, dtype=torch.float32))

        # Learnable A: [num_sites, rank], init zeros => start at s0
        self.A = nn.Parameter(torch.zeros(len(self.site_ids), self.rank, dtype=torch.float32))

        # Map site_id -> index
        self._idx = {sid: i for i, sid in enumerate(self.site_ids)}

    def get_scale(self, site_id: str, u: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Returns scale s_l(u) as scalar tensor on (device, dtype).
        """
        i = self._idx[site_id]
        # z(u): [r] float32 on u device
        z = chebyshev_features(u.to(torch.float32), r=self.rank)  # float32
        log_s0 = self.log_s0_vec[i]  # float32 scalar tensor
        a = self.A[i]                # [r] float32
        log_s = log_s0 + torch.dot(a, z)

        # clamp around log_s0 ± delta
        log_s = torch.clamp(log_s, log_s0 - self.clamp_delta, log_s0 + self.clamp_delta)

        s = torch.exp(log_s)  # float32
        return s.to(device=device, dtype=dtype)

    def get_scales(self, u: torch.Tensor, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Returns dict site_id -> scale scalar tensor
        """
        out = {}
        for sid in self.site_ids:
            out[sid] = self.get_scale(sid, u=u, dtype=dtype, device=device)
        return out
