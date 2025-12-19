# ctpc_lite/phase_c_validate_sd15_wrapping.py
from __future__ import annotations

import argparse
from typing import Dict

import torch
from diffusers import StableDiffusionPipeline

from ctpc_lite.phase_c_default_sites_sd15 import default_site_ids_sd15, default_wrap_map_sd15, make_dummy_log_s0
from ctpc_lite.phase_c_patch_unet import get_module_by_name, wrap_modules
from ctpc_lite.phase_c_scale_field import CTPCScaleField, ScaleFieldConfig
from ctpc_lite.phase_c_unet_u_patch import patch_unet_forward_with_u
from ctpc_lite.phase_c_wrappers import QuantActSiteWrapper

from ctpc_lite.schedulers import create_scheduler


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


@torch.no_grad()
def run_once(pipe: StableDiffusionPipeline, prompt: str, steps: int, guidance: float, seed: int):
    device = pipe.device
    g = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        output_type="latent",
        generator=g,
    )
    # diffusers returns latents in out.images when output_type="latent"
    lat = out.images
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--solver", default="kdpm2")
    ap.add_argument("--steps", type=int, default=2)  # keep tiny for validation
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--torch_dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="a photo of a golden retriever wearing sunglasses, shallow depth of field")
    args = ap.parse_args()

    torch_dtype = _dtype_from_str(args.torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    # Scheduler
    pipe.scheduler = create_scheduler(args.solver, pipe.scheduler, kwargs={})
    pipe.scheduler.set_timesteps(args.steps, device=args.device)

    # Phase C: patch UNet forward so u is set each call
    patch_unet_forward_with_u(pipe.unet, pipe.scheduler)

    # Build scale field (dummy priors for Phase C validation)
    site_ids = default_site_ids_sd15()
    log_s0 = make_dummy_log_s0(site_ids, log_s0_value=0.0)  # scale=1
    sf = CTPCScaleField(site_ids, log_s0, ScaleFieldConfig(rank=4, clamp_delta=2.0)).to(args.device)

    # Wrap default sites
    wrap_map = default_wrap_map_sd15()
    wrapped_paths = wrap_modules(pipe.unet, wrap_map, sf, enabled=True)

    # Sanity: confirm wrappers installed
    for pth, sid in wrap_map.items():
        m = get_module_by_name(pipe.unet, pth)
        if not isinstance(m, QuantActSiteWrapper):
            raise RuntimeError(f"Expected QuantActSiteWrapper at '{pth}', found {type(m)}")
    print(f"[phase_c] wrapped {len(wrapped_paths)} modules.")

    # Run quant-enabled
    lat_q = run_once(pipe, args.prompt, steps=args.steps, guidance=args.guidance, seed=args.seed)

    # Print wrapper hit counts / last scales
    print("\n[phase_c] wrapper stats (enabled=True):")
    for pth, sid in wrap_map.items():
        m = get_module_by_name(pipe.unet, pth)
        assert isinstance(m, QuantActSiteWrapper)
        print(f"  {sid:10s}  calls={m.call_count:4d}  last_scale={m.last_scale}")

    # Now disable wrappers and run again (same seed/prompt)
    for pth in wrap_map.keys():
        m = get_module_by_name(pipe.unet, pth)
        assert isinstance(m, QuantActSiteWrapper)
        m.enabled = False
        m.call_count = 0
        m.last_scale = None

    lat_fp = run_once(pipe, args.prompt, steps=args.steps, guidance=args.guidance, seed=args.seed)

    # Compare latents
    diff = (lat_q - lat_fp).abs().mean().item()
    print(f"\n[phase_c] latent mean|diff| (enabled vs disabled) = {diff:.6e}")
    print("[phase_c] EXPECTED: diff > 0 (quantization changes activations and thus latents).")


if __name__ == "__main__":
    main()
