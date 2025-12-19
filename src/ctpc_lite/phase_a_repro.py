from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ReproConfig:
    seed: int = 12345
    deterministic: bool = True
    benchmark: bool = False


def set_reproducibility(cfg: ReproConfig) -> None:
    """
    Best-effort reproducibility:
    - Seeds: python/random, numpy, torch (cpu/cuda)
    - Determinism flags: cudnn deterministic, torch deterministic algorithms
    Note: some ops remain nondeterministic depending on hardware/kernels.
    """
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(cfg.benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)

    if cfg.deterministic:
        # Can raise errors if you hit nondeterministic ops. That’s desired for debugging.
        torch.use_deterministic_algorithms(True, warn_only=True)


def _run_cmd(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        return out.strip()
    except Exception:
        return None


def collect_env_metadata() -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    md["timestamp"] = datetime.utcnow().isoformat() + "Z"
    md["python"] = sys.version.replace("\n", " ")
    md["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    md["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        md["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        }

    # Optional: git info if inside a git repo
    md["git"] = {
        "commit": _run_cmd(["git", "rev-parse", "HEAD"]),
        "status": _run_cmd(["git", "status", "--porcelain"]),
    }

    # pip freeze can be large; keep optional
    md["pip_freeze"] = None
    return md


def save_run_metadata(path: str, config: Dict[str, Any]) -> None:
    md = collect_env_metadata()
    md["config"] = config
    with open(path, "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2, sort_keys=True)
