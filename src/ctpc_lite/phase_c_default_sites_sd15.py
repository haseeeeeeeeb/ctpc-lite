# ctpc_lite/phase_c_default_sites_sd15.py
from __future__ import annotations

from typing import Dict, List, Tuple


def default_wrap_map_sd15() -> Dict[str, str]:
    """
    Returns module_path -> site_id for SD1.5 UNet2DConditionModel.

    2 blocks:
      - high-res: down_blocks.0.attentions.0.transformer_blocks.0
      - mid     : mid_block.attentions.0.transformer_blocks.0

    4 sites per block:
      - attn1, attn2
      - ff.net.0 (GEGLU up-proj), ff.net.2 (down-proj)
    """
    hi = "down_blocks.0.attentions.0.transformer_blocks.0"
    mid = "mid_block.attentions.0.transformer_blocks.0"

    wrap_map = {
        # High-res block
        f"{hi}.attn1": "hi_attn1",
        f"{hi}.attn2": "hi_attn2",
        f"{hi}.ff.net.0": "hi_ff_up",
        f"{hi}.ff.net.2": "hi_ff_down",
        # Mid block
        f"{mid}.attn1": "mid_attn1",
        f"{mid}.attn2": "mid_attn2",
        f"{mid}.ff.net.0": "mid_ff_up",
        f"{mid}.ff.net.2": "mid_ff_down",
    }
    return wrap_map


def default_site_ids_sd15() -> List[str]:
    # stable ordering (useful for scale field)
    return [
        "hi_attn1", "hi_attn2", "hi_ff_up", "hi_ff_down",
        "mid_attn1", "mid_attn2", "mid_ff_up", "mid_ff_down",
    ]


def make_dummy_log_s0(site_ids: List[str], log_s0_value: float = 0.0) -> Dict[str, float]:
    """
    For Phase C validation: initialize all priors to log(scale)=0 => scale=1.
    Phase D will replace this with real priors from percentile calibration.
    """
    return {sid: float(log_s0_value) for sid in site_ids}
