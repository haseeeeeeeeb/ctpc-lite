import argparse, json, math, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from diffusers import StableDiffusionPipeline

# schedulers
from diffusers import (
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    KDPM2DiscreteScheduler,
)

# -----------------------------
# Utils: robust scalar extraction
# -----------------------------
def _to_float(x) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

# -----------------------------
# logSNR computation helpers
# -----------------------------
def logsnr_from_alphas_cumprod(scheduler, t_float: float) -> Optional[float]:
    """
    Works when scheduler has alphas_cumprod and timestep corresponds to an index.
    This matches your current logs (VP-style): logSNR = log(alpha_bar) - log(1-alpha_bar).
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        return None

    t_idx = int(round(t_float))
    if abs(t_float - t_idx) > 1e-6:
        return None
    if t_idx < 0 or t_idx >= len(scheduler.alphas_cumprod):
        return None

    alpha_bar = float(scheduler.alphas_cumprod[t_idx].detach().cpu().item())
    eps = 1e-12
    alpha_bar = min(max(alpha_bar, eps), 1.0 - eps)
    return math.log(alpha_bar) - math.log(1.0 - alpha_bar)

def sigma_from_alphas_cumprod_vp(scheduler, t_float: float) -> Optional[float]:
    """
    VP-style sigma used in your log: sigma = sqrt(1 - alpha_bar).
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        return None
    t_idx = int(round(t_float))
    if abs(t_float - t_idx) > 1e-6:
        return None
    if t_idx < 0 or t_idx >= len(scheduler.alphas_cumprod):
        return None
    alpha_bar = float(scheduler.alphas_cumprod[t_idx].detach().cpu().item())
    alpha_bar = min(max(alpha_bar, 0.0), 1.0)
    return math.sqrt(max(1.0 - alpha_bar, 0.0))

# -----------------------------
# Call recorder
# -----------------------------
@dataclass
class CallRecord:
    call_index: int
    t_raw: Any
    t_float: float
    sigma: Optional[float]
    logsnr: Optional[float]
    logsnr_type: str
    notes: str

class UNetCallLogger:
    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.records: List[CallRecord] = []
        self._orig_forward = pipe.unet.forward
        self._call_index = 0

    def _forward_wrapped(self, *args, **kwargs):
        # diffusers UNet forward signature includes `timestep=...`
        t = kwargs.get("timestep", None)
        if t is None and len(args) >= 2:
            t = args[1]  # (sample, timestep, encoder_hidden_states, ...)
        t_float = _to_float(t)

        # Try VP-style from alphas_cumprod (matches SD1.5 common schedulers)
        logsnr = logsnr_from_alphas_cumprod(self.pipe.scheduler, t_float)
        sigma = sigma_from_alphas_cumprod_vp(self.pipe.scheduler, t_float)
        logsnr_type = "vp" if logsnr is not None else "unknown"
        notes = "from alphas_cumprod index" if logsnr is not None else "logsnr unavailable"

        self.records.append(
            CallRecord(
                call_index=self._call_index,
                t_raw=t,
                t_float=t_float,
                sigma=sigma,
                logsnr=logsnr,
                logsnr_type=logsnr_type,
                notes=notes,
            )
        )
        self._call_index += 1
        return self._orig_forward(*args, **kwargs)

    def __enter__(self):
        self.pipe.unet.forward = self._forward_wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        self.pipe.unet.forward = self._orig_forward

# -----------------------------
# Scheduler factory
# -----------------------------
def make_scheduler(name: str, pipe: StableDiffusionPipeline):
    cfg = pipe.scheduler.config
    name = name.lower()

    if name in ["dpmpp_2m", "dpmpp2m", "dpmsolver_multistep"]:
        # This is the one you logged: 1 call/step typically
        return DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++")

    if name in ["pndm"]:
        return PNDMScheduler.from_config(cfg)

    if name in ["kdpm2", "k_dpm_2", "kdpm2_discrete"]:
        return KDPM2DiscreteScheduler.from_config(cfg)

    raise ValueError(f"Unknown scheduler '{name}'")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--scheduler", default="dpmpp_2m", choices=["dpmpp_2m", "pndm", "kdpm2"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--prompts_json", type=str, default="")
    ap.add_argument("--num_prompts", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--out", type=str, default="calls_raw.jsonl")
    args = ap.parse_args()

    # minimal prompts if no file provided
    if args.prompts_json and os.path.exists(args.prompts_json):
        prompts = json.load(open(args.prompts_json, "r", encoding="utf-8"))
    else:
        prompts = [
            "a photo of a golden retriever wearing sunglasses, shallow depth of field",
            "a futuristic city skyline at night, neon lights, cinematic",
        ][: args.num_prompts]

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe.scheduler = make_scheduler(args.scheduler, pipe)
    pipe.set_progress_bar_config(disable=True)

    # Make output deterministic-ish
    torch.backends.cuda.matmul.allow_tf32 = True

    with open(args.out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            seed = args.seed0 + i
            gen = torch.Generator(device="cuda").manual_seed(seed)

            # reset scheduler state each sample
            pipe.scheduler.set_timesteps(args.steps, device="cuda")

            with UNetCallLogger(pipe) as logger:
                _ = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    output_type="latent",   # faster; avoids VAE decode
                )

            payload = {
                "sample_index": i,
                "seed": seed,
                "prompt": prompt,
                "solver": args.scheduler,
                "num_inference_steps": args.steps,
                "calls": [
                    {
                        "call_index": r.call_index,
                        "t_raw": str(r.t_raw) if torch.is_tensor(r.t_raw) else r.t_raw,
                        "t_float": r.t_float,
                        "sigma": r.sigma,
                        "logsnr": r.logsnr,
                        "logsnr_type": r.logsnr_type,
                        "notes": r.notes,
                    }
                    for r in logger.records
                ],
            }
            f.write(json.dumps(payload) + "\n")

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
