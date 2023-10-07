# -*- coding: utf-8 -*-
# File: base.py

import random
from typing import Any, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from utils.hardware import get_total_mem

from .. import DEVICE
from .model import SDModel, get_sd_model
from .scheduler import SchedulerMixin, get_sd_scheduler


class StableDiffusion_(object):
    POSITIVE_PRESET = "masterpiece, high quality, best quality, absurdres, 4K, 8K, HQ"
    NEGATIVE_PRESET = "simple background, duplicate, low quality, lowest quality, worst quality, bad anatomy, bad proportions, extra limbs, fewer limbs, extra fingers, fewer fingers, lowres, username, artist name, error, watermark, signature, text, extra digits, fewer digits, jpeg artifacts, blurry"
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        model: Optional[SDModel] = None,
        check_nsfw: bool = False,
        positive_preset: Optional[str] = None,
        negative_preset: Optional[str] = None,
        compile_unet: bool = False,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        gpu_id: int = 0,
        scheduler: Optional[SchedulerMixin] = None,
        optimizations: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._pipeline = pipeline
        self._model = model if model else get_sd_model("Stable Diffusion V2.1")
        self._check_nsfw = check_nsfw
        self._positive_preset = positive_preset if positive_preset else self.POSITIVE_PRESET
        self._negative_preset = negative_preset if negative_preset else self.NEGATIVE_PRESET
        self._compile_unet = compile_unet
        self._model_dir = model_dir
        self._device = device.lower() if device else DEVICE
        self._gpu_id = gpu_id
        self._scheduler = scheduler if scheduler else get_sd_scheduler("UniPCMultistepScheduler")
        self._optimizations = optimizations
        self._torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._pipe = self._pipeline.from_pretrained(
            self._model.path,
            torch_dtype=self._torch_dtype,
            scheduler=self._scheduler.from_pretrained(self._model.path, subfolder="scheduler"),
            cache_dir=self._model_dir,
            **kwargs,
        )
        if self._check_nsfw:
            from diffusers.pipelines.stable_diffusion import \
                StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor

            self._pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=self._torch_dtype,
                cache_dir=self._model_dir,
            )
            self._pipe.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=self._torch_dtype,
                cache_dir=self._model_dir,
            )
        if self._device == "cuda":
            self._pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            self._pipe.unet.to(memory_format=torch.channels_last)
            if self._compile_unet and torch.__version__ >= "2.0":
                self._pipe.unet = torch.compile(self._pipe.unet, mode="reduce-overhead", fullgraph=True)
            if isinstance(self._optimizations, list):
                if "enable_vae_slicing" in self._optimizations:
                    self._pipe.enable_vae_slicing()
                if "enable_vae_tiling" in self._optimizations:
                    self._pipe.enable_vae_tiling()
            if torch.__version__ < "2.0":
                self._pipe.enable_xformers_memory_efficient_attention()
        else:
            self._pipe = self._pipe.to(self._device)
            if self._device == "mps" and get_total_mem() < 64 * (1024 ** 3):
                self._pipe.enable_attention_slicing()

    def __call__(self) -> List[Union[Image.Image, np.ndarray]]:
        results = self.pipe(...)
        return results
    
    def get_generator(self, seed: Optional[int] = None):
        return torch.Generator(device=self.device).manual_seed(
            seed if seed or seed == 0 else random.randint(0, (1 << 31) - 1)
        )
    
    @staticmethod
    def get_final_prompt(*prompts: Optional[str]) -> str:
        return ", ".join([p for prompt in prompts if prompt and (p := prompt.strip()) != ""])
    
    def get_positive_prompt(self, prompt: str) -> str:
        return self.get_final_prompt(self.model.prefix, self.positive_preset, prompt)
    
    def get_negative_prompt(self, prompt: Optional[str]) -> str:
        return self.get_final_prompt(self.negative_preset, prompt)
    
    @property
    def pipeline(self) -> DiffusionPipeline:
        return self._pipeline
    
    @property
    def model(self) -> SDModel:
        return self._model
    
    @property
    def check_nsfw(self) -> bool:
        return self._check_nsfw
    
    @property
    def positive_preset(self) -> str:
        return self._positive_preset
    
    @positive_preset.setter
    def positive_preset(self, preset: Optional[str]) -> None:
        self._positive_preset = preset
    
    @property
    def negative_preset(self) -> str:
        return self._negative_preset
    
    @negative_preset.setter
    def negative_preset(self, preset: Optional[str]) -> None:
        self._negative_preset = preset

    @property
    def compile_unet(self) -> bool:
        return self._compile_unet
    
    @property
    def model_dir(self) -> Optional[str]:
        return self._model_dir
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def gpu_id(self) -> int:
        return self._gpu_id
    
    @property
    def scheduler(self) -> SchedulerMixin:
        return self._scheduler
    
    @property
    def optimizations(self) -> Optional[List[str]]:
        return self._optimizations
    
    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe
