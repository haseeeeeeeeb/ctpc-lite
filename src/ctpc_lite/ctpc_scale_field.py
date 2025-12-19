from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def chebyshev_features(u: torch.Tensor, r: int = 4) -> torch.Tensor:
    """
    Chebyshev T_k(u) for k=0..r-1.
    u: tensor shape (...) with values in [-1, 1] ideally.
    returns: tensor shape (..., r)
    """
    if r <= 0:
        raise ValueError("r must be >= 1")
    # Ensure float
    u = u.to(dtype=torch.float32)
    feats = []
    # T0 = 1
    feats.append(torch.ones_like(u))
    if r == 1:
        return torch.stack(feats, dim=-1)
    # T1 = u
    feats.append(u)
    if r == 2:
        return torch.stack(feats, dim=-1)
    # Recur: T_{k+1} = 2u T_k - T_{k-1}
    Tkm1 = feats[0]
    Tk = feats[1]
    for _k in range(2, r):
        Tkp1 = 2.0 * u * Tk - Tkm1
        feats.append(Tkp1)
        Tkm1, Tk = Tk, Tkp1
    return torch.stack(feats, dim=-1)


@dataclass
class ScaleFieldConfig:
    r: int = 4
    delta: float = 2.0  # clamp log s within [log_s0-delta, log_s0+delta]
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32


class CTPCScaleField(nn.Module):
    """
    CTPC-Lite scale field:
        log s_l(u) = log s0_l + A_l^T z(u)
    with z(u) = Chebyshev features.

    Stores:
      - site_ids: ordered list
      - log_s0: (S,)
      - A: (S, r) learnable
    """

    def __init__(
        self,
        site_ids: Sequence[str],
        s0_scale_step: Dict[str, float],
        cfg: ScaleFieldConfig = ScaleFieldConfig(),
    ):
        super().__init__()
        self.site_ids = list(site_ids)
        self.r = int(cfg.r)
        self.delta = float(cfg.delta)

        if self.r < 1:
            raise ValueError("r must be >= 1")
        if not self.site_ids:
            raise ValueError("site_ids must be non-empty")

        # Build log_s0 in the fixed site order
        s0 = []
        missing = []
        for sid in self.site_ids:
            if sid not in s0_scale_step:
                missing.append(sid)
            else:
                s0.append(float(s0_scale_step[sid]))
        if missing:
            raise KeyError(f"Missing s0 for sites: {missing}")

        log_s0 = torch.log(torch.tensor(s0, dtype=torch.float32))
        self.register_buffer("log_s0", log_s0)

        # Learnable coefficients A initialized to zero => s(u)=s0
        A = torch.zeros((len(self.site_ids), self.r), dtype=torch.float32)
        self.A = nn.Parameter(A)

        # Optional dtype/device move
        if cfg.device is not None:
            self.to(device=cfg.device, dtype=cfg.dtype)
        else:
            self.to(dtype=cfg.dtype)

    @torch.no_grad()
    def set_delta(self, delta: float):
        self.delta = float(delta)

    def _compute_log_s(self, u: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Returns log_s with shape:
          - (S,) if u is scalar
          - (B,S) if u is tensor shape (B,)
        """
        if not torch.is_tensor(u):
            u = torch.tensor(u, device=self.log_s0.device, dtype=torch.float32)
        u = u.to(device=self.log_s0.device, dtype=torch.float32)

        # scalar u
        if u.dim() == 0:
            z = chebyshev_features(u, self.r)  # (r,)
            # (S,r) @ (r,) -> (S,)
            log_s = self.log_s0 + (self.A * z).sum(dim=-1)
            # clamp around log_s0
            lo = self.log_s0 - self.delta
            hi = self.log_s0 + self.delta
            return torch.clamp(log_s, lo, hi)

        # vector u: shape (B,)
        if u.dim() != 1:
            raise ValueError(f"u must be scalar or 1D tensor, got shape={tuple(u.shape)}")
        z = chebyshev_features(u, self.r)  # (B,r)
        # log_s[b,s] = log_s0[s] + sum_r A[s,r]*z[b,r]
        log_s = self.log_s0[None, :] + torch.matmul(z, self.A.t())  # (B,S)
        lo = (self.log_s0 - self.delta)[None, :]
        hi = (self.log_s0 + self.delta)[None, :]
        return torch.clamp(log_s, lo, hi)

    def scales_tensor(self, u: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Returns positive scale step sizes (same semantics as s0_scale_step) as tensor.
        """
        log_s = self._compute_log_s(u)
        return torch.exp(log_s)

    def scales_dict(self, u: Union[float, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns {site_id: scale_tensor} where each scale_tensor is scalar (if u scalar)
        or shape (B,) (if u shape (B,)).
        """
        s = self.scales_tensor(u)
        out: Dict[str, torch.Tensor] = {}
        if s.dim() == 1:
            for i, sid in enumerate(self.site_ids):
                out[sid] = s[i]
        elif s.dim() == 2:
            for i, sid in enumerate(self.site_ids):
                out[sid] = s[:, i]
        else:
            raise RuntimeError(f"Unexpected scales_tensor shape: {tuple(s.shape)}")
        return out

    def lip_penalty(
        self,
        u: torch.Tensor,
        delta_u: float = 0.02,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        First-derivative (finite difference) penalty on log s(u):
          || (log s(u+du) - log s(u)) / du ||^2
        u: (B,) values in [-1,1] recommended
        returns scalar
        """
        if u.dim() != 1:
            raise ValueError("u must be 1D (B,)")
        du = float(delta_u)
        u = u.to(device=self.log_s0.device, dtype=torch.float32)

        up = torch.clamp(u + du, -1.0, 1.0)
        lo = torch.clamp(u, -1.0, 1.0)

        log_s_up = self._compute_log_s(up)  # (B,S)
        log_s_lo = self._compute_log_s(lo)  # (B,S)

        diff = (log_s_up - log_s_lo) / du  # (B,S)
        val = (diff * diff).sum(dim=-1)    # (B,)

        if reduction == "mean":
            return val.mean()
        if reduction == "sum":
            return val.sum()
        raise ValueError("reduction must be 'mean' or 'sum'")
