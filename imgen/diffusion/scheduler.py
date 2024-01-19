# -*- coding: utf-8 -*-
# File: imgen/diffusion/scheduler.py

from typing import Tuple

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    SchedulerMixin,
    UniPCMultistepScheduler,
)

_SD_SCHEDULERS = {
    "DDIMScheduler": DDIMScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
}


def get_sd_scheduler_names() -> Tuple[str]:
    return tuple(_SD_SCHEDULERS.keys())


def get_sd_scheduler(name: str) -> SchedulerMixin:
    if not (sd_scheduler := _SD_SCHEDULERS.get(name, None)):
        raise KeyError(f'Scheduler "{name}" not found')
    return sd_scheduler
