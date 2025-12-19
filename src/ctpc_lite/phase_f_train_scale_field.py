# ctpc_lite/phase_f_train_scale_field.py
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .ctpc_scale_field import CTPCScaleField as CTPCScaleFieldNew
from .ctpc_scale_field import ScaleFieldConfig as ScaleFieldConfigNew
from .io_utils import ensure_dir, load_yaml, read_text_lines, save_json
from .logsnr import try_compute_sigma_and_logsnr
from .phase_a_repro import ReproConfig, save_run_metadata, set_reproducibility
from .phase_c_default_sites_sd15 import default_site_ids_sd15, default_wrap_map_sd15
from .phase_c_patch_unet import get_module_by_name, wrap_modules
from .phase_c_u_context import set_u
from .phase_c_wrappers import QuantActSiteWrapper
from .schedulers import create_scheduler


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


def _make_repro_config(repro_dict: Dict[str, Any]) -> ReproConfig:
    """
    Robustly build ReproConfig even if this repo's ReproConfig schema differs.
    Filters unknown keys based on the __init__ signature.
    """
    repro_dict = repro_dict or {}
    try:
        return ReproConfig(**repro_dict)
    except TypeError:
        sig = inspect.signature(ReproConfig)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in repro_dict.items() if k in allowed}
        return ReproConfig(**filtered)


@torch.no_grad()
def _encode_prompt_embeddings(
    pipe: StableDiffusionPipeline,
    prompt: str,
    device: str,
    num_images_per_prompt: int,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Minimal SD1.5 prompt encoding:
      - returns embeds for CFG if guidance_scale > 1: [2B, T, C]
      - else: [B, T, C]
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_encoder.eval()

    do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0
    batch_size = 1

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder(input_ids)[0]
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    if not do_cfg:
        return prompt_embeds

    uncond_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_embeds = text_encoder(uncond_ids)[0]
    uncond_embeds = uncond_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return torch.cat([uncond_embeds, prompt_embeds], dim=0)


def _init_latents(
    pipe: StableDiffusionPipeline,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    generator: torch.Generator,
    num_images_per_prompt: int,
) -> torch.Tensor:
    # SD1.5 latent shape: (B, 4, H/8, W/8)
    b = int(num_images_per_prompt)
    latents = torch.randn((b, pipe.unet.in_channels, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    sigma0 = getattr(pipe.scheduler, "init_noise_sigma", None)
    if sigma0 is not None:
        latents = latents * float(sigma0)
    return latents


def _get_wrapped_modules(unet: torch.nn.Module, wrap_map: Dict[str, str]) -> List[QuantActSiteWrapper]:
    mods: List[QuantActSiteWrapper] = []
    for pth in wrap_map.keys():
        m = get_module_by_name(unet, pth)
        if not isinstance(m, QuantActSiteWrapper):
            raise RuntimeError(f"Expected QuantActSiteWrapper at '{pth}', found {type(m)}")
        mods.append(m)
    return mods


def _set_wrappers_enabled(wrappers: List[QuantActSiteWrapper], enabled: bool) -> None:
    for m in wrappers:
        m.enabled = enabled


def _compute_u_from_timestep(
    scheduler: Any,
    timestep: Any,
    lam_min: float,
    lam_max: float,
    device: torch.device,
) -> torch.Tensor:
    info = try_compute_sigma_and_logsnr(scheduler, timestep)
    lam = info.get("logsnr", None)
    if lam is None or not isinstance(lam, (float, int)) or not math.isfinite(float(lam)):
        # fallback: neutral
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    lam = float(lam)
    denom = max(1e-12, float(lam_max) - float(lam_min))
    u = 2.0 * (lam - float(lam_min)) / denom - 1.0
    u = max(-1.0, min(1.0, float(u)))
    return torch.tensor(u, device=device, dtype=torch.float32)


def _denoise_latents_manual(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    steps: int,
    guidance_scale: float,
    lam_min: float,
    lam_max: float,
    require_grad: bool,
) -> torch.Tensor:
    """
    Manual denoising loop that supports gradients (unlike pipe(...), which is @torch.no_grad()).
    """
    scheduler = pipe.scheduler
    unet = pipe.unet
    unet.eval()

    do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

    scheduler.set_timesteps(steps, device=latents.device)

    # Ensure timesteps on correct device
    timesteps = scheduler.timesteps
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(list(timesteps), device=latents.device)
    else:
        timesteps = timesteps.to(latents.device)

    # We never want gradients for UNet weights; only for scale field params used inside wrappers.
    for p in unet.parameters():
        p.requires_grad_(False)

    # If require_grad=False, run fully under no_grad to save memory.
    grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with grad_ctx:
        for t in timesteps:
            # CFG duplicates batch
            latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            u_t = _compute_u_from_timestep(scheduler, t, lam_min, lam_max, device=latent_model_input.device)

            with set_u(u_t):
                out = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
                noise_pred = out.sample if hasattr(out, "sample") else out[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)

            step_out = scheduler.step(noise_pred, t, latents)
            latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    return latents


def _load_lam_minmax_from_meta(meta_path: Optional[str]) -> Optional[Tuple[float, float]]:
    if not meta_path:
        return None
    p = Path(meta_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "lam_min" in obj and "lam_max" in obj:
        return float(obj["lam_min"]), float(obj["lam_max"])
    return None


def _fallback_lam_minmax_from_schedule(pipe: StableDiffusionPipeline, steps: int) -> Tuple[float, float]:
    """
    Fallback if you don't provide calls.jsonl.meta.json.
    We compute logsnr on the scheduler timesteps and take min/max.
    """
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    ts = pipe.scheduler.timesteps
    if not torch.is_tensor(ts):
        ts = torch.tensor(list(ts), device=pipe.device)
    vals: List[float] = []
    for t in ts:
        info = try_compute_sigma_and_logsnr(pipe.scheduler, t)
        lam = info.get("logsnr", None)
        if isinstance(lam, (float, int)) and math.isfinite(float(lam)):
            vals.append(float(lam))
    if not vals:
        # last-resort
        return (-1.0, 1.0)
    return (min(vals), max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_name = cfg["run"]["name"]
    out_dir = Path(cfg["run"]["out_dir"]) / run_name
    ensure_dir(out_dir)

    # Repro (robust to schema drift)
    repro = _make_repro_config(cfg.get("repro", {}) or {})
    set_reproducibility(repro)
    save_json(out_dir / "config.resolved.json", cfg)
    save_run_metadata(str(out_dir / "run_meta.json"), cfg)

    # Prompts
    prompts_file = cfg["logging"]["prompts_file"]
    max_prompts = int(cfg["logging"].get("max_prompts", 64))
    prompts = read_text_lines(prompts_file, max_lines=max_prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {prompts_file}")

    # Model
    model_id = cfg["model"]["model_id"]
    torch_dtype = _dtype_from_str(cfg["model"]["torch_dtype"])
    device = cfg["model"]["device"]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Scheduler
    solver_name = cfg["solver"]["name"]
    solver_kwargs = cfg["solver"].get("kwargs", {}) or {}
    pipe.scheduler = create_scheduler(solver_name, pipe.scheduler, solver_kwargs)

    # Train settings
    height = int(cfg["sample"]["height"])
    width = int(cfg["sample"]["width"])
    steps = int(cfg["sample"]["num_inference_steps"])
    guidance = float(cfg["sample"]["guidance_scale"])
    n_imgs = int(cfg["sample"].get("num_images_per_prompt", 1))

    train_cfg = cfg.get("train", {}) or {}
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 5e-2))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    lip_weight = float(train_cfg.get("lip_weight", 0.0))
    lip_delta_u = float(train_cfg.get("lip_delta_u", 0.02))
    u_grid = int(train_cfg.get("lip_u_grid", 64))

    # Priors (Phase D)
    s0_path = (
        cfg.get("phase_f", {}) or {}
    ).get("s0_path", None) or (cfg.get("priors", {}) or {}).get("s0_path", None) or (cfg.get("phase_d", {}) or {}).get("s0_path", None)
    if not s0_path:
        raise RuntimeError("Phase F needs s0_path (Phase D output .pt). Put it in phase_f.s0_path in your YAML.")
    s0_blob = torch.load(s0_path, map_location="cpu")
    s0_scale_step: Dict[str, float] = {k: float(v) for k, v in s0_blob["s0_scale_step"].items()}

    # Build scale field (Phase E version)
    site_ids = default_site_ids_sd15()
    sf_cfg = cfg.get("scale_field", {}) or {}
    r = int(sf_cfg.get("r", 4))
    delta = float(sf_cfg.get("delta", 2.0))
    field = CTPCScaleFieldNew(site_ids=site_ids, s0_scale_step=s0_scale_step, cfg=ScaleFieldConfigNew(r=r, delta=delta, device=device, dtype=torch.float32))

    # Wrap UNet modules with quant sites
    wrap_map = default_wrap_map_sd15()
    wrapped_paths = wrap_modules(pipe.unet, wrap_map, field, enabled=True)
    wrappers = _get_wrapped_modules(pipe.unet, wrap_map)

    save_json(out_dir / "sites.resolved.json", wrap_map)
    print(f"[phase_f] wrapped {len(wrapped_paths)} modules.")

    # u-normalization (prefer Phase B meta if provided)
    meta_path = (cfg.get("phase_f", {}) or {}).get("calls_meta_json", None) or (cfg.get("u_norm", {}) or {}).get("meta_json", None)
    mm = _load_lam_minmax_from_meta(meta_path)
    if mm is None:
        lam_min, lam_max = _fallback_lam_minmax_from_schedule(pipe, steps)
    else:
        lam_min, lam_max = mm
    print(f"[phase_f] u-norm: lam_min={lam_min:.6f}, lam_max={lam_max:.6f}")

    # Optimizer
    opt = torch.optim.Adam([field.A], lr=lr, weight_decay=weight_decay)

    # Deterministic per-prompt
    base_seed = int((cfg.get("repro", {}) or {}).get("seed", 12345))

    # Baseline diff (first prompt)
    prompt0 = prompts[0]
    g0 = torch.Generator(device=device).manual_seed(base_seed)
    lat0 = _init_latents(pipe, height, width, device, torch_dtype, g0, n_imgs)
    embeds0 = _encode_prompt_embeddings(pipe, prompt0, device, n_imgs, guidance)

    _set_wrappers_enabled(wrappers, False)
    lat_fp0 = _denoise_latents_manual(pipe, lat0.clone(), embeds0, steps, guidance, lam_min, lam_max, require_grad=False)

    _set_wrappers_enabled(wrappers, True)
    lat_q0 = _denoise_latents_manual(pipe, lat0.clone(), embeds0, steps, guidance, lam_min, lam_max, require_grad=False)

    base_diff = (lat_q0 - lat_fp0).abs().mean().item()
    print(f"[phase_f] baseline latent mean|diff| (quant vs fp) = {base_diff:.6e}")

    # Training
    global_step = 0
    for ep in range(epochs):
        pbar = tqdm(prompts, desc=f"phase_f: epoch {ep+1}/{epochs}")
        for i, prompt in enumerate(pbar):
            seed_i = base_seed + i
            gen = torch.Generator(device=device).manual_seed(seed_i)

            lat_init = _init_latents(pipe, height, width, device, torch_dtype, gen, n_imgs)
            embeds = _encode_prompt_embeddings(pipe, prompt, device, n_imgs, guidance)

            # Teacher (FP) — no grad
            _set_wrappers_enabled(wrappers, False)
            with torch.no_grad():
                lat_fp = _denoise_latents_manual(
                    pipe, lat_init.clone(), embeds, steps, guidance, lam_min, lam_max, require_grad=False
                )

            # Student (Quant) — WITH grad
            _set_wrappers_enabled(wrappers, True)
            lat_q = _denoise_latents_manual(
                pipe, lat_init.clone(), embeds, steps, guidance, lam_min, lam_max, require_grad=True
            )

            loss = F.mse_loss(lat_q.float(), lat_fp.float())

            if lip_weight > 0.0:
                u = torch.linspace(-1.0, 1.0, steps=u_grid, device=device, dtype=torch.float32)
                loss = loss + float(lip_weight) * field.lip_penalty(u, delta_u=lip_delta_u)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4e}"})

    # Post-train sanity on first prompt
    _set_wrappers_enabled(wrappers, True)
    lat_q1 = _denoise_latents_manual(pipe, lat0.clone(), embeds0, steps, guidance, lam_min, lam_max, require_grad=False)
    post_diff = (lat_q1 - lat_fp0).abs().mean().item()
    print(f"[phase_f] post-train latent mean|diff| (quant vs fp) = {post_diff:.6e}")

    # Save
    out_pt = out_dir / "scale_field.pt"
    torch.save(
        {
            "meta": {
                "model_id": model_id,
                "solver": solver_name,
                "steps": steps,
                "guidance_scale": guidance,
                "height": height,
                "width": width,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "lip_weight": lip_weight,
                "r": r,
                "delta": delta,
                "lam_min": lam_min,
                "lam_max": lam_max,
                "s0_path": str(s0_path),
                "calls_meta_json": meta_path,
            },
            "site_ids": site_ids,
            "wrap_map": wrap_map,
            "state_dict": field.state_dict(),
        },
        out_pt,
    )
    save_json(out_dir / "train_summary.json", {"baseline_diff": base_diff, "post_diff": post_diff})

    print(f"[phase_f] saved {out_pt}")
    print(f"[phase_f] wrote {out_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
