from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .ctpc_context import ctpc_u_context
from .ctpc_scale_field import CTPCScaleField, ScaleFieldConfig
from .io_utils import ensure_dir, load_yaml, read_text_lines, save_json
from .logsnr_utils import logsnr_from_alphas_cumprod, u_from_logsnr
from .module_path import get_by_path, set_by_path
from .phase_a_repro import ReproConfig, save_run_metadata, set_reproducibility
from .quant_wrappers import QuantActSiteWrapper
from .schedulers import create_scheduler


class EarlyStopSampling(Exception):
    pass


class DetachingScheduler:
    """
    Proxy scheduler that detaches prev_sample after each scheduler.step(),
    preventing backprop through sampling dynamics.
    """
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, *args, **kwargs):
        out = self.inner.step(*args, **kwargs)

        # diffusers typically returns a SchedulerOutput with .prev_sample
        if hasattr(out, "prev_sample") and torch.is_tensor(out.prev_sample):
            out.prev_sample = out.prev_sample.detach()
            return out

        # sometimes dict-like
        if isinstance(out, dict) and "prev_sample" in out and torch.is_tensor(out["prev_sample"]):
            out["prev_sample"] = out["prev_sample"].detach()
            return out

        return out


class DistillUNetWrapper(nn.Module):
    """
    Wraps a UNet so that each call computes:
      teacher output (wrappers disabled) in no_grad
      student output (wrappers enabled) with grad
    and accumulates MSE loss on .sample.
    Also early-stops after max_calls UNet invocations.
    """

    def __init__(
        self,
        base_unet: nn.Module,
        scheduler,
        wrappers: Dict[str, QuantActSiteWrapper],
        logsnr_min: float,
        logsnr_max: float,
        max_calls: int,
    ):
        super().__init__()
        self.base_unet = base_unet
        self.scheduler = scheduler
        self.wrappers = wrappers
        self.logsnr_min = float(logsnr_min)
        self.logsnr_max = float(logsnr_max)
        self.max_calls = int(max_calls)

        self.reset()

    def reset(self):
        self.call_count = 0
        self.loss_accum = None

    def _set_wrappers_enabled(self, enabled: bool):
        for w in self.wrappers.values():
            w.enabled = enabled

    def forward(self, sample, timestep, *args, **kwargs):
        # timestep can be int/float tensor
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=sample.device, dtype=torch.float32)
        else:
            t = t.to(device=sample.device)

        logsnr = logsnr_from_alphas_cumprod(self.scheduler, t)
        u = u_from_logsnr(logsnr, self.logsnr_min, self.logsnr_max)

        # u is scalar tensor; ensure scalar shape
        if u.dim() != 0:
            u = u.reshape([])

        # --- teacher (no quant, no grad)
        self._set_wrappers_enabled(False)
        with torch.no_grad():
            out_t = self.base_unet(sample, timestep, *args, **kwargs)
            eps_t = out_t.sample

        # --- student (quant enabled, grad)
        self._set_wrappers_enabled(True)
        with ctpc_u_context(u):
            out_s = self.base_unet(sample, timestep, *args, **kwargs)
            eps_s = out_s.sample

        mse = F.mse_loss(eps_s, eps_t, reduction="mean")
        self.loss_accum = mse if self.loss_accum is None else (self.loss_accum + mse)

        self.call_count += 1
        if self.call_count >= self.max_calls:
            # stop pipeline early after K UNet calls
            raise EarlyStopSampling()

        return out_s


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


def _compute_logsnr_range_from_calls_raw(calls_raw_jsonl: str) -> Tuple[float, float]:
    mn = float("inf")
    mx = float("-inf")
    with open(calls_raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for c in obj["calls"]:
                v = c.get("logsnr", None)
                if v is None:
                    continue
                mn = min(mn, float(v))
                mx = max(mx, float(v))
    if not (mn < float("inf") and mx > float("-inf")):
        raise RuntimeError(f"Could not find logsnr values in {calls_raw_jsonl}")
    if mx - mn < 1e-9:
        raise RuntimeError(f"logsnr range is degenerate: [{mn},{mx}]")
    return mn, mx


def _wrap_sites_from_s0(unet: nn.Module, site_paths: Dict[str, str], scale_field: CTPCScaleField) -> Dict[str, QuantActSiteWrapper]:
    wrappers: Dict[str, QuantActSiteWrapper] = {}
    for site_id, path in site_paths.items():
        mod = get_by_path(unet, path)
        w = QuantActSiteWrapper(mod, site_id=site_id, scale_field=scale_field, enabled=True)
        set_by_path(unet, path, w)
        wrappers[site_id] = w
    return wrappers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--s0", type=str, required=True, help="Phase D s0.pt")
    ap.add_argument("--calls_raw", type=str, required=True, help="Phase B calls_raw.jsonl for logsnr range")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_name = cfg["run"]["name"]
    out_dir = Path(cfg["run"]["out_dir"]) / run_name
    ensure_dir(out_dir)

    # Phase A reproducibility
    repro = ReproConfig(**cfg["repro"])
    set_reproducibility(repro)
    save_json(out_dir / "config.resolved.json", cfg)
    save_run_metadata(str(out_dir / "run_meta.json"), cfg)

    # Load s0 + site paths
    blob = torch.load(args.s0, map_location="cpu")
    s0_scale_step: Dict[str, float] = blob["s0_scale_step"]
    site_paths: Dict[str, str] = blob["site_paths"]  # site_id -> module path
    site_ids = list(site_paths.keys())

    # logsnr min/max from Phase B traces
    logsnr_min, logsnr_max = _compute_logsnr_range_from_calls_raw(args.calls_raw)
    save_json(out_dir / "logsnr_range.json", {"logsnr_min": logsnr_min, "logsnr_max": logsnr_max})

    # prompts
    prompts = read_text_lines(cfg["data"]["prompts_file"], max_lines=int(cfg["data"].get("max_prompts", 512)))
    if not prompts:
        raise RuntimeError("No prompts loaded for training.")

    # pipeline
    model_id = cfg["model"]["model_id"]
    torch_dtype = _dtype_from_str(cfg["model"]["torch_dtype"])
    device = cfg["model"]["device"]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.unet.eval()

    # scheduler
    solver_name = cfg["solver"]["name"]
    solver_kwargs = cfg["solver"].get("kwargs", {}) or {}
    sched = create_scheduler(solver_name, pipe.scheduler, solver_kwargs)
    pipe.scheduler = DetachingScheduler(sched)  # detach dynamics for training
    pipe.scheduler.set_timesteps(int(cfg["sample"]["num_inference_steps"]), device=device)

    # scale field (learn A only)
    sf_cfg = ScaleFieldConfig(
        r=int(cfg["ctpc"]["r"]),
        delta=float(cfg["ctpc"]["delta"]),
        device=device,
        dtype=torch.float32,  # keep scale params fp32
    )
    scale_field = CTPCScaleField(site_ids, s0_scale_step=s0_scale_step, cfg=sf_cfg).to(device)

    # wrap sites in UNet
    wrappers = _wrap_sites_from_s0(pipe.unet, site_paths, scale_field)

    # replace UNet with distillation wrapper
    max_calls = int(cfg["train"]["max_unet_calls"])
    distill_unet = DistillUNetWrapper(
        base_unet=pipe.unet,
        scheduler=pipe.scheduler,
        wrappers=wrappers,
        logsnr_min=logsnr_min,
        logsnr_max=logsnr_max,
        max_calls=max_calls,
    ).to(device)
    pipe.unet = distill_unet

    # optimizer only for A
    opt = torch.optim.AdamW(
        [scale_field.A],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # training params
    height = int(cfg["sample"]["height"])
    width = int(cfg["sample"]["width"])
    steps = int(cfg["sample"]["num_inference_steps"])
    guidance = float(cfg["sample"]["guidance_scale"])
    n_imgs = int(cfg["sample"]["num_images_per_prompt"])

    lam_lip = float(cfg["train"]["lambda_lip"])
    lip_du = float(cfg["train"]["lip_delta_u"])
    u_batch = int(cfg["train"]["u_batch"])
    grad_accum = int(cfg["train"]["grad_accum"])
    total_updates = int(cfg["train"]["updates"])
    log_every = int(cfg["train"].get("log_every", 10))
    save_every = int(cfg["train"].get("save_every", 200))

    base_seed = int(cfg["repro"]["seed"])
    pbar = tqdm(range(total_updates), desc="phase_f train")

    opt.zero_grad(set_to_none=True)

    for upd in pbar:
        prompt = prompts[upd % len(prompts)]
        seed = base_seed + upd
        g = torch.Generator(device=device).manual_seed(seed)

        distill_unet.reset()

        try:
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
        except EarlyStopSampling:
            pass  # expected

        if distill_unet.loss_accum is None or distill_unet.call_count == 0:
            raise RuntimeError("No UNet calls were captured for distillation loss.")

        rollout_loss = distill_unet.loss_accum / float(distill_unet.call_count)

        # Lipschitz penalty: sample u uniformly in [-1,1]
        u = (2.0 * torch.rand(u_batch, device=device) - 1.0).to(torch.float32)
        lip = scale_field.lip_penalty(u, delta_u=lip_du, reduction="mean")

        loss = rollout_loss + lam_lip * lip
        loss = loss / float(grad_accum)
        loss.backward()

        if (upd + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (upd + 1) % log_every == 0:
            pbar.set_postfix(
                calls=distill_unet.call_count,
                rollout=float(rollout_loss.detach().cpu().item()),
                lip=float(lip.detach().cpu().item()),
                lam_lip=lam_lip,
            )

        if (upd + 1) % save_every == 0:
            ckpt = {
                "A": scale_field.A.detach().cpu(),
                "site_ids": site_ids,
                "s0_scale_step": s0_scale_step,
                "site_paths": site_paths,
                "logsnr_min": logsnr_min,
                "logsnr_max": logsnr_max,
                "cfg": cfg,
                "update": upd + 1,
            }
            torch.save(ckpt, out_dir / f"ctpc_lite_A_step{upd+1}.pt")

    # final save
    ckpt = {
        "A": scale_field.A.detach().cpu(),
        "site_ids": site_ids,
        "s0_scale_step": s0_scale_step,
        "site_paths": site_paths,
        "logsnr_min": logsnr_min,
        "logsnr_max": logsnr_max,
        "cfg": cfg,
        "update": total_updates,
    }
    torch.save(ckpt, out_dir / "ctpc_lite_A_final.pt")
    print(f"[phase_f] wrote {out_dir/'ctpc_lite_A_final.pt'}")


if __name__ == "__main__":
    main()
