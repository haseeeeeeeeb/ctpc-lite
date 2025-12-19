# ctpc_lite/phase_c_quant.py
from __future__ import annotations

import torch


def fake_quant_int8_symmetric_per_tensor(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric INT8 fake-quant per tensor:
      q = clamp(round(x/scale), -127, 127)
      x_hat = q * scale

    Uses STE for round via (round - x).detach() trick.
    """
    # scale: scalar tensor
    scale = torch.clamp(scale, min=torch.finfo(x.dtype).eps)

    y = x / scale
    # STE round:
    y_round = (torch.round(y) - y).detach() + y
    y_q = torch.clamp(y_round, -127.0, 127.0)
    return y_q * scale
