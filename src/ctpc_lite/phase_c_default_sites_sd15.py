# ctpc_lite/phase_c_default_sites_sd15.py
from __future__ import annotations

from typing import Dict, List


def default_wrap_map_sd15() -> Dict[str, str]:
    """
    module_path -> site_id for SD1.5 UNet2DConditionModel.

    Two transformer blocks:
      - high-res: down_blocks.0.attentions.0.transformer_blocks.0
      - mid     : mid_block.attentions.0.transformer_blocks.0

    We quantize the *inputs* to these submodules (post-LN hidden_states):
      - attn1, attn2 (attention)
      - ff.net.0 (GEGLU up-proj path)
      - ff.net.2 (FFN down-proj)
    """
    hi = "down_blocks.0.attentions.0.transformer_blocks.0"
    mid = "mid_block.attentions.0.transformer_blocks.0"

    return {
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


def default_site_ids_sd15() -> List[str]:
    # Stable ordering for scale-field tensors
    return [
        "hi_attn1", "hi_attn2", "hi_ff_up", "hi_ff_down",
        "mid_attn1", "mid_attn2", "mid_ff_up", "mid_ff_down",
    ]


def default_site_paths_sd15() -> Dict[str, str]:
    """
    site_id -> module_path (inverse of wrap_map).
    This is what Phase D (priors) and Phase F (wrapping) should use.
    """
    wm = default_wrap_map_sd15()
    return {site_id: module_path for (module_path, site_id) in wm.items()}


def make_dummy_s0_scale_step(site_ids: List[str], step: float = 1.0) -> Dict[str, float]:
    """
    Phase C validation only. Step is the symmetric INT8 step size.
    (Real priors come from Phase D.)
    """
    return {sid: float(step) for sid in site_ids}
