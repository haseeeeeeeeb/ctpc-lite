from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .io_utils import ensure_dir, load_yaml, read_text_lines, save_json, save_jsonl
from .phase_a_repro import ReproConfig, save_run_metadata, set_reproducibility
from .phase_b_solver_logger import SolverCallLogger
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


def _schedule_to_list(ts: Any) -> Optional[List[float]]:
    if ts is None:
        return None
    if torch.is_tensor(ts):
        return [float(x) for x in ts.detach().cpu().tolist()]
    try:
        return [float(x) for x in ts]
    except Exception:
        return None


def _infer_unique_call_times(logger_records: List[Dict[str, Any]]) -> List[float]:
    """Infer a schedule from UNet call trace: unique t_float in call order."""
    seen = set()
    uniq = []
    for rec in logger_records:
        t = rec.get("t_float", None)
        if t is None:
            continue
        t = float(t)
        key = round(t, 8)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq


def pick_macro_schedule(all_schedule: List[float], steps: int) -> Dict[str, Any]:
    """
    Define a 'macro' schedule of length `steps` representing the user-visible grid.
    - If all_schedule is expanded with internal stages (KDPM2 pattern: 2*steps-1), macro is every other.
    - Else, macro is an evenly-spaced subset.
    """
    n = len(all_schedule)
    if steps <= 0:
        raise ValueError("steps must be positive")

    if n == steps:
        return {"macro_schedule": list(all_schedule), "rule": "identity"}

    if n == 2 * steps - 1:
        return {"macro_schedule": list(all_schedule[::2]), "rule": "every_other"}

    if steps == 1:
        return {"macro_schedule": [all_schedule[0]], "rule": "single"}

    # evenly spaced indices
    idxs = [round(i * (n - 1) / (steps - 1)) for i in range(steps)]
    idxs = [int(i) for i in idxs]

    # de-dup while preserving order
    seen = set()
    idxs2 = []
    for i in idxs:
        if i not in seen:
            idxs2.append(i)
            seen.add(i)

    # pad if needed
    while len(idxs2) < steps:
        for j in range(len(idxs2) - 1):
            if len(idxs2) >= steps:
                break
            a, b = idxs2[j], idxs2[j + 1]
            mid = (a + b) // 2
            if mid not in seen:
                idxs2.append(mid)
                seen.add(mid)
        idxs2 = sorted(idxs2)
        if len(idxs2) < steps:
            for cand in range(n):
                if cand not in seen:
                    idxs2.append(cand)
                    seen.add(cand)
                if len(idxs2) >= steps:
                    break
            idxs2 = sorted(idxs2)

    idxs2 = idxs2[:steps]
    macro = [all_schedule[i] for i in idxs2]
    return {"macro_schedule": macro, "rule": "evenly_spaced"}


def _maybe_get_schedule(pipe: StableDiffusionPipeline) -> Tuple[Optional[str], Optional[List[float]]]:
    """
    Try multiple common scheduler attributes. Some schedulers don't populate timesteps until after a call.
    """
    sch = pipe.scheduler
    candidates = []
    if hasattr(sch, "timesteps"):
        candidates.append(("timesteps", getattr(sch, "timesteps")))
    if hasattr(sch, "_timesteps"):
        candidates.append(("_timesteps", getattr(sch, "_timesteps")))
    if hasattr(sch, "sigmas"):
        candidates.append(("sigmas", getattr(sch, "sigmas")))
    if hasattr(sch, "_sigmas"):
        candidates.append(("_sigmas", getattr(sch, "_sigmas")))

    for name, ts in candidates:
        ts_list = _schedule_to_list(ts)
        if ts_list is not None and len(ts_list) > 0:
            return name, ts_list
    return None, None


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
    max_prompts = int(cfg["logging"].get("max_prompts", 64))
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

    # Phase B: log UNet calls
    logger = SolverCallLogger(pipe.unet, pipe.scheduler)

    height = int(cfg["sample"]["height"])
    width = int(cfg["sample"]["width"])
    steps = int(cfg["sample"]["num_inference_steps"])
    guidance = float(cfg["sample"]["guidance_scale"])
    n_imgs = int(cfg["sample"]["num_images_per_prompt"])

    # ---- Warmup to force scheduler to materialize timesteps if it only does so at runtime ----
    pipe.scheduler.set_timesteps(steps, device=device)

    base_seed = int(cfg["repro"]["seed"])
    warmup_prompt = prompts[0]
    warmup_seed = base_seed
    warmup_g = torch.Generator(device=device).manual_seed(warmup_seed)

    logger.reset()
    with logger:
        _ = pipe(
            prompt=warmup_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=n_imgs,
            generator=warmup_g,
            output_type="latent",
        )

    # Try to read schedule arrays from scheduler
    sched_src, all_schedule = _maybe_get_schedule(pipe)

    # Fallback: infer schedule from the call trace if scheduler doesn't expose it (PNDM often)
    if all_schedule is None or len(all_schedule) == 0:
        all_schedule = _infer_unique_call_times(logger.records)
        sched_src = "inferred_from_calls_unique"

    macro_info = pick_macro_schedule(all_schedule, steps) if all_schedule else {"macro_schedule": None, "rule": None}
    macro_schedule = macro_info["macro_schedule"]
    macro_rule = macro_info["rule"]

    # Backward compatibility: outer_schedule == macro_schedule
    outer_schedule = macro_schedule

    rows = []
    rows.append(
        {
            "sample_index": 0,
            "seed": warmup_seed,
            "prompt": warmup_prompt,
            "solver": solver_name,
            "num_inference_steps": steps,
            "all_schedule": all_schedule,
            "macro_schedule": macro_schedule,
            "macro_schedule_rule": macro_rule,
            "schedule_source": sched_src,
            "outer_schedule": outer_schedule,
            "calls": logger.records,
        }
    )

    # ---- Remaining prompts ----
    for i, prompt in enumerate(tqdm(prompts[1:], desc="logging calls"), start=1):
        seed_i = base_seed + i
        g = torch.Generator(device=device).manual_seed(seed_i)

        logger.reset()
        with logger:
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

        rows.append(
            {
                "sample_index": i,
                "seed": seed_i,
                "prompt": prompt,
                "solver": solver_name,
                "num_inference_steps": steps,
                "all_schedule": all_schedule,
                "macro_schedule": macro_schedule,
                "macro_schedule_rule": macro_rule,
                "schedule_source": sched_src,
                "outer_schedule": outer_schedule,
                "calls": logger.records,
            }
        )

    raw_path = out_dir / "calls_raw.jsonl"
    save_jsonl(raw_path, rows)

    print(f"[phase_b] wrote {raw_path}")
    print("[phase_b] Saved schedules:")
    print(f"          schedule_source={sched_src}")
    print(f"          all_schedule_len={None if all_schedule is None else len(all_schedule)}")
    print(f"          macro_schedule_len={None if macro_schedule is None else len(macro_schedule)} (rule={macro_rule})")
    print("[phase_b] Next: run postprocess to compute u in [-1,1] from global logsnr range.")
    print(f"          python -m ctpc_lite.cli_postprocess_calls --in_jsonl {raw_path} --out_jsonl {out_dir/'calls.jsonl'}")


if __name__ == "__main__":
    main()
