from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .io_utils import ensure_dir, load_yaml, read_text_lines, save_json
from .phase_a_repro import ReproConfig, save_run_metadata, set_reproducibility
from .schedulers import create_scheduler


# -----------------------------
# Utilities
# -----------------------------

def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


def resolve_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resolve a module from a dotted path with integer indexes, e.g.
      "down_blocks.0.attentions.0.transformer_blocks.0.attn1"
    """
    cur: Any = root
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    if not isinstance(cur, torch.nn.Module):
        raise TypeError(f"Resolved object at '{path}' is not a torch.nn.Module (got {type(cur)})")
    return cur


def default_sd15_site_paths() -> Dict[str, str]:
    """
    8-site default consistent with your Phase C IDs.

    hi_*  : highest-resolution transformer block (down_blocks[0], channels=320)
    mid_* : mid_block transformer block (channels=1280)

    attn sites tap Attention module input (post-norm hidden_states).
    ff sites tap GEGLU.proj input (up-proj) and FF net[2] input (down-proj).
    """
    hi_block = "down_blocks.0.attentions.0.transformer_blocks.0"
    mid_block = "mid_block.attentions.0.transformer_blocks.0"

    return {
        "hi_attn1": f"{hi_block}.attn1",
        "hi_attn2": f"{hi_block}.attn2",
        "hi_ff_up": f"{hi_block}.ff.net.0.proj",
        "hi_ff_down": f"{hi_block}.ff.net.2",
        "mid_attn1": f"{mid_block}.attn1",
        "mid_attn2": f"{mid_block}.attn2",
        "mid_ff_up": f"{mid_block}.ff.net.0.proj",
        "mid_ff_down": f"{mid_block}.ff.net.2",
    }


def sample_abs_values(
    x: torch.Tensor,
    n: int,
    gen: torch.Generator,
) -> np.ndarray:
    """
    Sample n absolute values from tensor x without materializing abs(x) for all elements.
    Returns float32 numpy array on CPU.
    """
    if not torch.is_tensor(x):
        return np.empty((0,), dtype=np.float32)
    if x.numel() == 0:
        return np.empty((0,), dtype=np.float32)

    flat = x.detach().reshape(-1)
    m = flat.numel()
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    if n >= m:
        vals = flat.abs().float().cpu().numpy()
        return vals.astype(np.float32, copy=False)

    # sample indices without touching global RNG
    idx = torch.randint(
        low=0,
        high=m,
        size=(n,),
        generator=gen,
        device=flat.device,
        dtype=torch.int64,
    )
    vals = flat.index_select(0, idx).abs().float().cpu().numpy()
    return vals.astype(np.float32, copy=False)


class Reservoir:
    """
    Cheap approximate reservoir: append until 2*max_size then randomly downsample back to max_size.
    Good enough for robust percentile estimation.
    """

    def __init__(self, max_size: int, np_rng: np.random.Generator):
        self.max_size = int(max_size)
        self.np_rng = np_rng
        self._buf: List[np.ndarray] = []
        self.total_seen: int = 0  # total sampled values attempted (after per-call subsampling)

    def add(self, vals: np.ndarray):
        if vals.size == 0:
            return
        self.total_seen += int(vals.size)
        self._buf.append(vals)

        # compact occasionally
        cur = self.size
        if cur >= 2 * self.max_size and self.max_size > 0:
            arr = self.as_array()
            if arr.size > self.max_size:
                idx = self.np_rng.choice(arr.size, size=self.max_size, replace=False)
                arr = arr[idx]
            self._buf = [arr]

    @property
    def size(self) -> int:
        return int(sum(int(a.size) for a in self._buf))

    def as_array(self) -> np.ndarray:
        if not self._buf:
            return np.empty((0,), dtype=np.float32)
        if len(self._buf) == 1:
            return self._buf[0]
        return np.concatenate(self._buf, axis=0)

    def finalize(self) -> np.ndarray:
        """
        Return a final array <= max_size (randomly downsample if needed).
        """
        arr = self.as_array()
        if self.max_size > 0 and arr.size > self.max_size:
            idx = self.np_rng.choice(arr.size, size=self.max_size, replace=False)
            arr = arr[idx]
        return arr


@dataclass
class PriorResult:
    clip_p: float
    scale_step: float  # clip/127 for symmetric int8 with clamp [-127,127]
    n_samples: int
    n_seen: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_name = cfg["run"]["name"]
    out_dir = Path(cfg["run"]["out_dir"]) / run_name
    ensure_dir(out_dir)

    # Phase A: reproducibility + metadata
    repro = ReproConfig(**cfg["repro"])
    set_reproducibility(repro)
    save_json(out_dir / "config.resolved.json", cfg)
    save_run_metadata(str(out_dir / "run_meta.json"), cfg)

    # Load prompts
    prompts_file = cfg["logging"]["prompts_file"]
    max_prompts = int(cfg["logging"].get("max_prompts", 512))
    prompts = read_text_lines(prompts_file, max_lines=max_prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {prompts_file}")

    # Load pipeline
    model_id = cfg["model"]["model_id"]
    torch_dtype = _dtype_from_str(cfg["model"]["torch_dtype"])
    device = cfg["model"]["device"]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Configure scheduler/solver
    solver_name = cfg["solver"]["name"]
    solver_kwargs = cfg["solver"].get("kwargs", {}) or {}
    pipe.scheduler = create_scheduler(solver_name, pipe.scheduler, solver_kwargs)

    # Save scheduler config snapshot
    try:
        save_json(out_dir / "scheduler_config.json", dict(pipe.scheduler.config))
    except Exception:
        save_json(out_dir / "scheduler_config.json", {"note": "scheduler.config not serializable"})

    # Sampling config
    height = int(cfg["sample"]["height"])
    width = int(cfg["sample"]["width"])
    steps = int(cfg["sample"]["num_inference_steps"])
    guidance = float(cfg["sample"]["guidance_scale"])
    n_imgs = int(cfg["sample"]["num_images_per_prompt"])

    pipe.scheduler.set_timesteps(steps, device=device)

    # Prior collection config
    percentile_p = float(cfg["priors"].get("percentile", 99.9))
    per_call_elems = int(cfg["priors"].get("per_call_elems", 4096))
    reservoir_size = int(cfg["priors"].get("reservoir_size", 200_000))

    # Dedicated RNGs so we do not perturb pipeline randomness
    base_seed = int(cfg["repro"]["seed"])
    np_rng = np.random.default_rng(base_seed + 777)
    sample_gen = torch.Generator(device=device).manual_seed(base_seed + 888)

    # Site mapping
    site_paths = cfg.get("sites", {}).get("mapping", None)
    if not site_paths:
        site_paths = default_sd15_site_paths()

    # Resolve modules
    sites: Dict[str, torch.nn.Module] = {}
    for site_id, path in site_paths.items():
        try:
            sites[site_id] = resolve_module_by_path(pipe.unet, path)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve site '{site_id}' at path '{path}': {e}") from e

    save_json(out_dir / "sites.resolved.json", site_paths)

    # Reservoir per site
    reservoirs: Dict[str, Reservoir] = {
        sid: Reservoir(max_size=reservoir_size, np_rng=np_rng) for sid in sites.keys()
    }

    # Hook functions
    handles = []

    def make_pre_hook(site_id: str):
        def _hook(module, args, kwargs):
            # We always target the first tensor argument as the activation.
            x = None
            if args and torch.is_tensor(args[0]):
                x = args[0]
            elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                x = kwargs["hidden_states"]
            if x is None:
                return

            vals = sample_abs_values(x, n=per_call_elems, gen=sample_gen)
            reservoirs[site_id].add(vals)

        return _hook

    # Register hooks
    for sid, mod in sites.items():
        h = mod.register_forward_pre_hook(make_pre_hook(sid), with_kwargs=True)
        handles.append(h)

    # Run teacher sampling (no quant) and collect activations
    rows_meta = []
    try:
        for i, prompt in enumerate(tqdm(prompts, desc="phase_d: collecting priors")):
            seed_i = base_seed + i
            g = torch.Generator(device=device).manual_seed(seed_i)

            with torch.no_grad():
                _ = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=n_imgs,
                    generator=g,
                    output_type="latent",
                )

            if (i + 1) % int(cfg["priors"].get("log_every", 50)) == 0:
                # lightweight progress snapshot
                snap = {
                    "i": i,
                    "site_sizes": {sid: reservoirs[sid].size for sid in reservoirs},
                }
                rows_meta.append(snap)

    finally:
        for h in handles:
            h.remove()

    save_json(out_dir / "reservoir_progress.json", {"snapshots": rows_meta})

    # Compute priors: clip=percentile(|X|), scale_step = clip/127
    priors: Dict[str, PriorResult] = {}
    priors_json: Dict[str, Dict[str, Any]] = {}

    for sid, res in reservoirs.items():
        arr = res.finalize()
        if arr.size == 0:
            clip = float("nan")
            step = float("nan")
        else:
            clip = float(np.quantile(arr, percentile_p / 100.0))
            step = float(clip / 127.0)

        priors[sid] = PriorResult(
            clip_p=clip,
            scale_step=step,
            n_samples=int(arr.size),
            n_seen=int(res.total_seen),
        )

        priors_json[sid] = {
            "clip_p": clip,
            "scale_step": step,
            "percentile": percentile_p,
            "n_samples_final": int(arr.size),
            "n_seen_total": int(res.total_seen),
            "per_call_elems": per_call_elems,
            "reservoir_size": reservoir_size,
        }

    # Save: torch + json
    torch_out = {
        "meta": {
            "model_id": model_id,
            "solver": solver_name,
            "steps": steps,
            "guidance_scale": guidance,
            "height": height,
            "width": width,
            "percentile": percentile_p,
            "per_call_elems": per_call_elems,
            "reservoir_size": reservoir_size,
        },
        "site_paths": site_paths,
        # These are what later modules should use as s0 (INT8 step size)
        "s0_scale_step": {sid: priors[sid].scale_step for sid in priors},
        # Also save clip thresholds for debugging/plots
        "s0_clip": {sid: priors[sid].clip_p for sid in priors},
    }

    torch.save(torch_out, out_dir / "s0.pt")
    save_json(out_dir / "s0.json", {"meta": torch_out["meta"], "priors": priors_json})

    # Human-readable summary
    print(f"[phase_d] wrote {out_dir/'s0.pt'} and {out_dir/'s0.json'}")
    print("[phase_d] priors summary (clip_p and scale_step=clip/127):")
    for sid in sorted(priors.keys()):
        pr = priors[sid]
        print(
            f"  {sid:12s} clip_p={pr.clip_p:.6g}  step={pr.scale_step:.6g}  "
            f"n_final={pr.n_samples}  n_seen={pr.n_seen}"
        )


if __name__ == "__main__":
    main()
