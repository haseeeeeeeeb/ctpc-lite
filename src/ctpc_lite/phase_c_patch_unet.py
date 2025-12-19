# ctpc_lite/phase_c_patch_unet.py
from __future__ import annotations

from typing import Dict, List, Tuple

import torch.nn as nn

from .phase_c_wrappers import QuantActSiteWrapper
from .phase_c_scale_field import CTPCScaleField


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur = root
    if name == "":
        return cur
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    return cur


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    leaf = parts[-1]
    parent = get_module_by_name(root, parent_name) if parent_name else root
    if leaf.isdigit():
        parent[int(leaf)] = new_module  # type: ignore[index]
    else:
        setattr(parent, leaf, new_module)


def wrap_modules(unet: nn.Module, wrap_map: Dict[str, str], scale_field: CTPCScaleField, enabled: bool = True) -> List[str]:
    """
    wrap_map: module_path -> site_id
    Replaces module at module_path with QuantActSiteWrapper(module, site_id, scale_field)
    Returns list of wrapped module paths.
    """
    wrapped = []
    for module_path, site_id in wrap_map.items():
        mod = get_module_by_name(unet, module_path)
        set_module_by_name(unet, module_path, QuantActSiteWrapper(mod, site_id, scale_field, enabled=enabled))
        wrapped.append(module_path)
    return wrapped
