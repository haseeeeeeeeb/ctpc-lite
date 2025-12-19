from __future__ import annotations

from typing import Any, Dict, Optional, Type

from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler
)

# A minimal mapping. Add more if you need.
SCHEDULER_REGISTRY: dict[str, Type] = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_sde": DPMSolverMultistepScheduler,
    "pndm": PNDMScheduler
}


def create_scheduler(name: str, base_scheduler, kwargs: Optional[Dict[str, Any]] = None):
    """
    Create a scheduler instance from the pipeline's existing scheduler config.

    Important:
    - In diffusers, many DPM++ variants are configured via DPMSolverMultistepScheduler kwargs.
    - We keep this intentionally simple and transparent for Phase B logging.
    """
    name = name.lower()
    kwargs = kwargs or {}

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler '{name}'. Known: {list(SCHEDULER_REGISTRY.keys())}")

    cls = SCHEDULER_REGISTRY[name]

    # Start from base scheduler config, then override with kwargs if supported.
    sch = cls.from_config(base_scheduler.config, **kwargs)

    # Some recommended toggles for DPM++/Karras usage; only apply if attributes exist.
    # (diffusers ignores unsupported keys during from_config in many cases, but not always.)
    return sch
