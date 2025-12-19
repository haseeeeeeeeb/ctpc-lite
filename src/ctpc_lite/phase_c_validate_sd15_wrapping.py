# ctpc_lite/phase_c_validate_sd15_wrapping.py
from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import torch
from diffusers import StableDiffusionPipeline

from .ctpc_scale_field import CTPCScaleField, ScaleFieldConfig
from .io_utils import save_json
from .logsnr import try_compute_sigma_and_logsnr
from .module_path import get_by_path, set_by_path
from .phase_c_default_sites_sd15 import default_site_ids_sd15, default_site_paths_sd15, make_dummy_s0_scale_step
from .quant_wrappers import QuantActSiteWrapper
from .schedulers import create_scheduler
from .unet_u_patch import patch_unet_forward_with_u


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


def _compute_lam_minmax_from_scheduler(pipe: StableDiffusionPipeline, steps: int) -> Tuple[float, float]:
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    ts = pipe.scheduler.timesteps
    if not torch.is_tensor(ts):
        ts = torch.tensor(list(ts), device=pipe.device)

    vals = []
    for t in ts:
        info = try_compute_sigma_and_logsnr(pipe.scheduler, t)
        lam = info.get("logsnr", None)
        if isinstance(lam, (int, float)) and math.isfinite(float(lam)):
            vals.append(float(lam))

    if not vals:
        # last-resort
        return (-1.0, 1.0)
    return (min(vals), max(vals))


@torch.no_grad()
def _run_once(pipe: StableDiffusionPipeline, prompt: str, steps: int, guidance: float, seed: int):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        output_type="latent",
        generator=g,
    )
    return out.images  # latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--solver", default="kdpm2")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--torch_dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="a photo of a golden retriever wearing sunglasses, shallow depth of field")
    ap.add_argument("--dummy_step", type=float, default=1.0)
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=_dtype_from_str(args.torch_dtype),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    pipe.scheduler = create_scheduler(args.solver, pipe.scheduler, kwargs={})
    lam_min, lam_max = _compute_lam_minmax_from_scheduler(pipe, args.steps)
    patch_unet_forward_with_u(pipe.unet, pipe.scheduler, lam_min=lam_min, lam_max=lam_max)

    # Scale field with dummy priors (Phase D will replace)
    site_ids = default_site_ids_sd15()
    s0 = make_dummy_s0_scale_step(site_ids, step=args.dummy_step)
    sf = CTPCScaleField(site_ids=site_ids, s0_scale_step=s0, cfg=ScaleFieldConfig(r=4, delta=2.0, device=args.device))

    # Wrap sites
    site_paths = default_site_paths_sd15()
    wrappers: Dict[str, QuantActSiteWrapper] = {}
    for sid, pth in site_paths.items():
        mod = get_by_path(pipe.unet, pth)
        w = QuantActSiteWrapper(mod, site_id=sid, scale_field=sf, enabled=True)
        set_by_path(pipe.unet, pth, w)
        wrappers[sid] = w

    # Run quant-enabled vs disabled
    lat_q = _run_once(pipe, args.prompt, args.steps, args.guidance, args.seed)

    for w in wrappers.values():
        w.enabled = False
        w.stats.calls = 0
        w.stats.last_scale = 0.0

    lat_fp = _run_once(pipe, args.prompt, args.steps, args.guidance, args.seed)
    diff = (lat_q - lat_fp).abs().mean().item()

    print("=== Phase C validate (SD1.5 wrapping) ===")
    print(f"solver={args.solver} steps={args.steps} guidance={args.guidance} seed={args.seed}")
    print(f"lam_min={lam_min:.6f} lam_max={lam_max:.6f}")
    print(f"latent mean|diff| (enabled vs disabled) = {diff:.6e}   (EXPECTED: > 0)")

    print("\nwrapper snapshot:")
    for sid, pth in site_paths.items():
        w = get_by_path(pipe.unet, pth)
        assert isinstance(w, QuantActSiteWrapper)
        print(f"  {sid:10s}  path={pth:70s}  calls={w.stats.calls:4d}  last_step={w.stats.last_scale:.6g}")

    save_json("phase_c_validate_out.json", {
        "solver": args.solver,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "latent_mean_abs_diff": diff,
        "site_paths": site_paths,
    })


if __name__ == "__main__":
    main()
