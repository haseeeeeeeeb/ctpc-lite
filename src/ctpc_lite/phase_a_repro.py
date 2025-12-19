# src/ctpc_lite/phase_a_repro.py
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ReproConfig:
    seed: int = 12345

    # Determinism controls
    deterministic: bool = True
    deterministic_warn_only: bool = True  # avoid hard-crashing on non-deterministic ops

    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    # TF32 can change numerics => default OFF for “exact reruns”
    allow_tf32: bool = False

    # cuBLAS workspace config helps determinism for matmul on CUDA
    set_cublas_workspace: bool = True


def set_reproducibility(cfg: ReproConfig) -> None:
    """
    Phase A: make runs as reproducible as practical.
    Note: diffusion pipelines can still be non-bitwise-identical on some kernels
    unless you disable certain fused/flash attention backends.
    We log everything so you can diagnose drift.
    """
    seed = int(cfg.seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if cfg.set_cublas_workspace:
        # common deterministic setting; safe to set if not already set
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)

    if bool(cfg.deterministic):
        # warn_only avoids breaking on ops without deterministic implementations
        torch.use_deterministic_algorithms(True, warn_only=bool(cfg.deterministic_warn_only))

    # If available, lock float32 matmul precision (PyTorch 2.x)
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def _try_git(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
        return out or None
    except Exception:
        return None


def collect_run_metadata(resolved_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    meta["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    meta["platform"] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }

    meta["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }

    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            meta["cuda_device"] = {
                "index": int(dev),
                "name": torch.cuda.get_device_name(dev),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(dev))),
                "total_mem_gb": round(torch.cuda.get_device_properties(dev).total_memory / (1024**3), 3),
            }
        except Exception:
            meta["cuda_device"] = {"note": "failed to query device properties"}

    # Optional libs (don’t crash if absent)
    for pkg in ("diffusers", "transformers", "accelerate", "safetensors", "xformers"):
        try:
            mod = __import__(pkg)
            meta[pkg] = {"version": getattr(mod, "__version__", "unknown")}
        except Exception:
            meta[pkg] = {"version": None}

    # Git snapshot if available
    meta["git"] = {
        "commit": _try_git(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": bool(_try_git(["git", "status", "--porcelain"])),
    }

    if resolved_cfg is not None:
        meta["config_resolved"] = resolved_cfg

    return meta


def save_run_metadata(path: str | Path, resolved_cfg: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = collect_run_metadata(resolved_cfg=resolved_cfg)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
