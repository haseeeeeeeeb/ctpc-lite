# ctpc_lite/phase_f_train_ctpc_lite.py
"""
Phase F: Train CTPC scale-field (A) by distilling FP teacher -> INT8-activation-quant student.

Fixes for "PARAM NaN/Inf after step":
- Run UNet in FP32 during Phase F (default) to avoid fp16 backward overflow through quant wrappers.
- Sanitize grads (nan/inf -> 0) BEFORE clipping/step.
- Persist LR backoff (don't reset to initial LR every iteration).
- Log grad stats for debugging.

If you want speed later, try --unet_fp32 0 and --torch_dtype bfloat16 (if supported).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

from .ctpc_context import ctpc_u_context
from .ctpc_scale_field import CTPCScaleField, ScaleFieldConfig
from .io_utils import save_json
from .logsnr import try_compute_sigma_and_logsnr
from .module_path import get_by_path, set_by_path
from .phase_c_default_sites_sd15 import default_site_paths_sd15
from .quant_wrappers import QuantActSiteWrapper
from .schedulers import create_scheduler
from .unet_u_patch import compute_u_from_scheduler


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype '{s}'")


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_lam_minmax_from_scheduler(scheduler, steps: int, device: torch.device) -> Tuple[float, float]:
    scheduler.set_timesteps(steps, device=device)
    ts = scheduler.timesteps
    if not torch.is_tensor(ts):
        ts = torch.tensor(list(ts), device=device)

    vals: List[float] = []
    for t in ts:
        info = try_compute_sigma_and_logsnr(scheduler, t)
        lam = info.get("logsnr", None)
        if isinstance(lam, (int, float)) and math.isfinite(float(lam)):
            vals.append(float(lam))

    if not vals:
        return (-1.0, 1.0)
    return (min(vals), max(vals))


def _get_extra_step_kwargs(scheduler, eta: float = 0.0):
    sig = inspect.signature(scheduler.step)
    kwargs = {}
    if "eta" in sig.parameters:
        kwargs["eta"] = eta
    return kwargs


@torch.no_grad()
def _encode_prompts(pipe: StableDiffusionPipeline, prompts: List[str], device: torch.device):
    tok = pipe.tokenizer
    enc = pipe.text_encoder

    bsz = len(prompts)
    max_len = tok.model_max_length

    text_in = tok(
        prompts,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    uncond_in = tok(
        [""] * bsz,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    text_emb = enc(text_in)[0]
    uncond_emb = enc(uncond_in)[0]
    return uncond_emb, text_emb


def _cfg_noise_pred(unet, latents, t, uncond_emb, text_emb, guidance: float):
    lat_in = torch.cat([latents, latents], dim=0)

    emb = torch.cat([uncond_emb, text_emb], dim=0)
    # Ensure encoder_hidden_states matches UNet/latents dtype
    emb = emb.to(dtype=lat_in.dtype)

    out = unet(lat_in, t, encoder_hidden_states=emb)
    noise = out.sample if hasattr(out, "sample") else out[0]
    noise_uncond, noise_text = noise.chunk(2, dim=0)
    return noise_uncond + guidance * (noise_text - noise_uncond)


def _scale_model_input_if_needed(scheduler, latents, t):
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latents, t)
    return latents


def _load_s0(path: str):
    blob = torch.load(path, map_location="cpu")
    if "site_ids" not in blob:
        raise RuntimeError(f"s0 file missing 'site_ids': {path}")
    if "s0_scale_step" not in blob:
        raise RuntimeError(f"s0 file missing 's0_scale_step': {path}")

    site_ids = list(blob["site_ids"])
    s0_scale_step = blob["s0_scale_step"]

    if isinstance(s0_scale_step, dict):
        fixed = {}
        for k, v in s0_scale_step.items():
            if torch.is_tensor(v):
                fixed[k] = float(v.detach().float().cpu().item())
            else:
                fixed[k] = float(v)
        s0_scale_step = fixed

    site_paths = blob.get("site_paths", None) or default_site_paths_sd15()
    return site_ids, site_paths, s0_scale_step, blob


def _wrap_student_sites(unet, site_paths: Dict[str, str], scale_field: CTPCScaleField) -> Dict[str, QuantActSiteWrapper]:
    wrappers: Dict[str, QuantActSiteWrapper] = {}
    for site_id, path in site_paths.items():
        mod = get_by_path(unet, path)
        w = QuantActSiteWrapper(mod, site_id=site_id, scale_field=scale_field, enabled=True)
        set_by_path(unet, path, w)
        wrappers[site_id] = w
    return wrappers


def _set_wrappers_enabled(wrappers: Dict[str, QuantActSiteWrapper], enabled: bool):
    for w in wrappers.values():
        w.enabled = enabled


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def _is_finite_params(params: List[torch.nn.Parameter]) -> bool:
    for p in params:
        if p is None or p.data is None:
            continue
        if not torch.isfinite(p.data).all().item():
            return False
    return True


def _sanitize_grads(params: List[torch.nn.Parameter], clamp_abs: float = 0.0) -> Tuple[int, float, float]:
    """
    Replace non-finite grad entries with 0. Optionally clamp elementwise.
    Returns: (num_params_with_nonfinite_grad, grad_norm, max_abs_grad)
    """
    nonfinite_params = 0
    max_abs = 0.0

    # nan/inf -> 0 and track max_abs
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all().item():
            nonfinite_params += 1
            p.grad = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            g = p.grad
        if clamp_abs and clamp_abs > 0:
            g.clamp_(-clamp_abs, clamp_abs)
        if g.numel() > 0:
            max_abs = max(max_abs, float(g.detach().abs().max().cpu().item()))

    # grad norm (finite-safe)
    sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float((g.float() ** 2).sum().cpu().item())
    grad_norm = math.sqrt(sq)
    return nonfinite_params, grad_norm, max_abs


@dataclass
class TrainLog:
    it: int
    loss: float
    mse: float
    reg_lip: float
    reg_l2: float
    lr: float
    wall_s: float
    t_window: Tuple[int, int]
    lam_min: float
    lam_max: float
    nonfinite_grads: int
    grad_norm: float
    grad_max_abs: float


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--s0", required=True, help="Path to Phase D s0.pt")
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--solver", default="kdpm2")
    ap.add_argument("--rollout_steps", type=int, default=20)
    ap.add_argument("--loss_window", type=int, default=5)
    ap.add_argument("--window_mode", choices=["first", "random"], default="random")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--eta", type=float, default=0.0)

    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--grad_clip", type=float, default=1.0, help="norm clip")
    ap.add_argument("--grad_clip_value", type=float, default=0.0, help="elementwise clamp on grads after nan_to_num")
    ap.add_argument("--lip_weight", type=float, default=1e-3)
    ap.add_argument("--l2_weight", type=float, default=1e-4)

    ap.add_argument("--param_clip", type=float, default=0.0, help="clamp scale-field params to [-param_clip, param_clip]")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--outdir", default="runs/phase_f_2")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_dtype", default="float16", help="pipeline dtype; UNet may be forced fp32 via --unet_fp32")
    ap.add_argument("--unet_fp32", type=int, default=1, help="1 = run UNet in fp32 during Phase F (recommended)")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--prompt", default=None)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _set_seed(args.seed)

    device = torch.device(args.device)
    pipe_dtype = _dtype_from_str(args.torch_dtype)

    # speed/stability niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load priors (s0) and create scale-field (FP32)
    site_ids, site_paths, s0_scale_step, _ = _load_s0(args.s0)
    sf_cfg = ScaleFieldConfig(r=4, delta=2.0, device=str(device))
    scale_field = CTPCScaleField(site_ids=site_ids, s0_scale_step=s0_scale_step, cfg=sf_cfg).to(device).float()

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=pipe_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.unet.eval()
    pipe.text_encoder.eval()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.eval()

    # Force UNet fp32 (this is the big fix)
    if int(args.unet_fp32) == 1:
        pipe.unet.to(dtype=torch.float32)

    # Scheduler
    pipe.scheduler = create_scheduler(args.solver, pipe.scheduler, kwargs={})
    scheduler = pipe.scheduler

    # Wrap quant sites
    wrappers = _wrap_student_sites(pipe.unet, site_paths=site_paths, scale_field=scale_field)

    # u normalization bounds
    lam_min, lam_max = _compute_lam_minmax_from_scheduler(scheduler, steps=args.rollout_steps, device=device)

    # prompts
    if args.prompt is not None:
        prompt_pool = [args.prompt]
    elif args.prompts_file is not None:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_pool = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not prompt_pool:
            raise RuntimeError(f"prompts_file is empty: {args.prompts_file}")
    else:
        prompt_pool = [
            "a photo of a golden retriever wearing sunglasses, shallow depth of field",
            "cinematic portrait of a woman, rim lighting, 85mm lens, high detail",
            "a watercolor landscape of mountains and a lake at sunrise",
            "a futuristic city skyline at night with neon reflections",
            "macro photograph of a honeybee on a flower, bokeh",
        ]

    # Optimizer
    train_params = [p for p in scale_field.parameters() if p.requires_grad]
    if not train_params:
        raise RuntimeError("No trainable parameters found in CTPCScaleField.")

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    # Persisted LR (so backoff doesn't reset each iteration)
    current_lr = float(args.lr)

    # Save run config
    save_json(os.path.join(args.outdir, "phase_f_config.json"), {
        "args": vars(args),
        "site_ids": site_ids,
        "site_paths": site_paths,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "s0_path": args.s0,
    })

    log_path = os.path.join(args.outdir, "train_log.jsonl")
    t0 = time.time()

    last_good_state = copy.deepcopy(scale_field.state_dict())
    last_good_opt = copy.deepcopy(opt.state_dict())

    for it in range(1, args.iters + 1):
        scheduler.set_timesteps(args.rollout_steps, device=device)
        timesteps = scheduler.timesteps
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(list(timesteps), device=device)

        K = min(int(args.loss_window), int(args.rollout_steps))
        N = int(args.rollout_steps)
        w0 = 0 if args.window_mode == "first" else random.randint(0, max(0, N - K))
        w1 = w0 + K

        batch_prompts = [random.choice(prompt_pool) for _ in range(args.batch_size)]
        with torch.no_grad():
            uncond_emb, text_emb = _encode_prompts(pipe, batch_prompts, device=device)

        unet_dtype = next(pipe.unet.parameters()).dtype
        uncond_emb = uncond_emb.to(dtype=unet_dtype)
        text_emb  = text_emb.to(dtype=unet_dtype)

        # latents always FP32 for scheduler stability
        h, w = args.height // 8, args.width // 8
        latents = torch.randn(
            (args.batch_size, pipe.unet.config.in_channels, h, w),
            device=device,
            dtype=torch.float32,
        )
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * float(scheduler.init_noise_sigma)

        # ensure optimizer LR matches current_lr
        for g in opt.param_groups:
            g["lr"] = current_lr

        opt.zero_grad(set_to_none=True)

        mse_acc = 0.0
        steps_in_loss = 0
        bad_iter = False
        u_in_loss: List[torch.Tensor] = []

        for idx, t in enumerate(timesteps):
            lat_in = _scale_model_input_if_needed(scheduler, latents, t)

            # UNet input dtype should match UNet
            unet_dtype = next(pipe.unet.parameters()).dtype
            lat_in_unet = lat_in.to(dtype=unet_dtype)

            u = compute_u_from_scheduler(scheduler, t, lam_min=lam_min, lam_max=lam_max, device=device).float()

            # Teacher (FP)
            with torch.no_grad():
                _set_wrappers_enabled(wrappers, False)
                eps_t = _cfg_noise_pred(pipe.unet, lat_in_unet, t, uncond_emb, text_emb, guidance=args.guidance)

            # Student (quant + grads)
            _set_wrappers_enabled(wrappers, True)
            with ctpc_u_context(u):
                eps_s = _cfg_noise_pred(pipe.unet, lat_in_unet, t, uncond_emb, text_emb, guidance=args.guidance)

            if (not _is_finite_tensor(eps_s)) or (not _is_finite_tensor(eps_t)):
                bad_iter = True
                break

            if w0 <= idx < w1:
                u_in_loss.append(u.detach().reshape(-1))  # collect u's used in loss window

                mse = F.mse_loss(eps_s.float(), eps_t.float())
                if not torch.isfinite(mse).item():
                    bad_iter = True
                    break
                mse_acc += float(mse.detach().cpu().item())
                steps_in_loss += 1
                mse.backward()

            # advance; keep scheduler in FP32
            with torch.no_grad():
                extra = _get_extra_step_kwargs(scheduler, eta=args.eta)
                latents = scheduler.step(eps_s.detach().float(), t, latents, **extra).prev_sample
                if not _is_finite_tensor(latents):
                    bad_iter = True
                    break
                latents = latents.detach()

        if bad_iter:
            # revert + backoff
            scale_field.load_state_dict(last_good_state)
            opt.load_state_dict(last_good_opt)
            current_lr *= 0.5
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "it": it,
                    "loss": float("nan"),
                    "mse": float("nan"),
                    "reg_lip": 0.0,
                    "reg_l2": 0.0,
                    "lr": current_lr,
                    "wall_s": float(time.time() - t0),
                    "t_window": [int(w0), int(w1)],
                    "lam_min": float(lam_min),
                    "lam_max": float(lam_max),
                    "nonfinite_grads": -1,
                    "grad_norm": float("nan"),
                    "grad_max_abs": float("nan"),
                    "note": "bad_iter_forward_or_latents_nonfinite_reverted_lr_halved",
                }) + "\n")
            print(f"[it {it:5d}] BAD ITER (forward/latents). Reverted, lr -> {current_lr:.2e}")
            continue

        # Regularizers (optional)
        reg_lip = 0.0
        reg_l2 = 0.0

        if args.lip_weight != 0.0 and hasattr(scale_field, "lip_penalty"):
            # Use the actual u's from the rollout window (best match to solver call distribution)
            if len(u_in_loss) > 0:
                u_lip = torch.cat(u_in_loss, dim=0).to(device=device, dtype=torch.float32)
            else:
                # should basically never happen, but keep it safe
                u_lip = torch.empty(16, device=device, dtype=torch.float32).uniform_(-1.0, 1.0)

            # Call lip_penalty(u, ...) with optional du if the method supports it
            sig = inspect.signature(scale_field.lip_penalty)
            if "du" in sig.parameters:
                lp = scale_field.lip_penalty(u_lip, du=0.02)
            else:
                lp = scale_field.lip_penalty(u_lip)

            if torch.is_tensor(lp):
                lp = lp.mean()  # in case it returns per-sample
                reg_lip = float(lp.detach().cpu().item())
                (args.lip_weight * lp).backward()

        if args.l2_weight != 0.0:
            l2 = 0.0
            for p in train_params:
                l2 = l2 + (p.float() ** 2).mean()
            reg_l2 = float(l2.detach().cpu().item())
            (args.l2_weight * l2).backward()

        # SANITIZE grads BEFORE clipping/step
        nonfinite_grads, grad_norm, grad_max_abs = _sanitize_grads(train_params, clamp_abs=float(args.grad_clip_value))

        # If grad norm is non-finite or absurd, skip step and backoff
        if not math.isfinite(grad_norm) or grad_norm > 1e8:
            scale_field.load_state_dict(last_good_state)
            opt.load_state_dict(last_good_opt)
            current_lr *= 0.5
            print(f"[it {it:5d}] GRAD explosion (norm={grad_norm}). Reverted, lr -> {current_lr:.2e}")
            continue

        # Norm clip
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(train_params, float(args.grad_clip))

        # Step
        opt.step()

        # Param clamp (optional)
        if args.param_clip is not None and args.param_clip > 0:
            with torch.no_grad():
                for p in train_params:
                    p.clamp_(-float(args.param_clip), float(args.param_clip))

        # Post-step finiteness check
        if not _is_finite_params(train_params):
            scale_field.load_state_dict(last_good_state)
            opt.load_state_dict(last_good_opt)
            current_lr *= 0.5
            print(f"[it {it:5d}] PARAM NaN/Inf after step. Reverted, lr -> {current_lr:.2e}")
            continue

        # Commit last-good
        last_good_state = copy.deepcopy(scale_field.state_dict())
        last_good_opt = copy.deepcopy(opt.state_dict())

        mse_mean = (mse_acc / max(1, steps_in_loss))
        loss_report = float(mse_mean + args.lip_weight * reg_lip + args.l2_weight * reg_l2)
        wall = time.time() - t0

        rec = TrainLog(
            it=it,
            loss=float(loss_report),
            mse=float(mse_mean),
            reg_lip=float(reg_lip),
            reg_l2=float(reg_l2),
            lr=float(current_lr),
            wall_s=float(wall),
            t_window=(int(w0), int(w1)),
            lam_min=float(lam_min),
            lam_max=float(lam_max),
            nonfinite_grads=int(nonfinite_grads),
            grad_norm=float(grad_norm),
            grad_max_abs=float(grad_max_abs),
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

        if it % 25 == 0 or it == 1:
            print(
                f"[it {it:5d}] mse={rec.mse:.6e}  nf_grads={rec.nonfinite_grads}  "
                f"g_norm={rec.grad_norm:.3e}  g_max={rec.grad_max_abs:.3e}  "
                f"win=[{w0},{w1})  lr={rec.lr:.2e}"
            )

        if (it % args.save_every == 0) or (it == args.iters):
            ckpt = {
                "it": it,
                "scale_field_state": scale_field.state_dict(),
                "opt_state": opt.state_dict(),
                "args": vars(args),
                "lam_min": lam_min,
                "lam_max": lam_max,
                "site_ids": site_ids,
                "site_paths": site_paths,
                "s0_path": args.s0,
                "current_lr": current_lr,
            }
            torch.save(ckpt, os.path.join(args.outdir, f"ckpt_it{it}.pt"))

    save_json(os.path.join(args.outdir, "final_summary.json"), {
        "iters": args.iters,
        "outdir": args.outdir,
        "log_path": log_path,
        "last_ckpt": os.path.join(args.outdir, f"ckpt_it{args.iters}.pt"),
        "note": "If nonfinite_grads stays >0, the issue is inside the quant wrappers' backward; keep --unet_fp32 1 and/or use --grad_clip_value.",
    })
    print(f"\nDone. Logs: {log_path}")


if __name__ == "__main__":
    main()
