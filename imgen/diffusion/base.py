# -*- coding: utf-8 -*-
# File: base.py

import random
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL.Image import Image
from transformers import CLIPImageProcessor

from .. import DEVICE
from .model import SDModel


class StableDiffusion_(object):
    POSITIVE_PRESET = "masterpiece, high quality, absurdres, 4K, 8K, HQ"
    NEGATIVE_PRESET = "simple background, duplicate, low quality, lowest quality, bad anatomy, bad proportions, extra limbs, fewer limbs, extra fingers, fewer fingers, lowres, username, artist name, error, watermark, signature, text, extra digits, fewer digits, jpeg artifacts, blurry"
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        model: SDModel,
        check_nsfw: bool = False,
        positive_preset: Optional[str] = None,
        negative_preset: Optional[str] = None,
        compile: bool = False,
        model_dir: Optional[str] = None,
        device: Optional[str] = None
    ) -> None:
        self._model = model
        self._positive_preset = positive_preset if positive_preset else self.POSITIVE_PRESET
        self._negative_preset = negative_preset if negative_preset else self.NEGATIVE_PRESET
        self._device = device if device else DEVICE
        self._torch_dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._pipe = pipeline.from_pretrained(
            model.path,
            torch_dtype=self._torch_dtype,
            cache_dir=model_dir,
        )
        if check_nsfw:
            self._pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=self._torch_dtype,
                cache_dir=model_dir,
            )
            self._pipe.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=self._torch_dtype,
                cache_dir=model_dir,
            )
        if self._device != "cpu":
            if compile and torch.__version__ >= "2.0":
                self._pipe.unet = torch.compile(self._pipe.unet, mode="reduce-overhead", fullgraph=True)
            self._pipe = self._pipe.to(self._device)
            # self._pipe.enable_model_cpu_offload()
            # self._pipe.enable_vae_slicing()
            # self._pipe.enable_vae_tiling()
            if torch.__version__ < "2.0":
                self._pipe.enable_xformers_memory_efficient_attention()

    def __call__(self) -> List[Image]:
        results = self.pipe(...)
        return results
    
    def get_generator(self, seed: Optional[int] = None):
        return torch.Generator(device=self.device).manual_seed(
            seed if seed or seed == 0 else random.randint(0, (1 << 31) - 1)
        )
    
    @staticmethod
    def get_final_prompt(*prompts: Optional[str]) -> str:
        output = []
        for prompt in prompts:
            if prompt and prompt.strip() != "":
                output.extend([kw.strip() for kw in prompt.split(",")])
        return ", ".join(output)
    
    def get_positive_prompt(self, prompt: str) -> str:
        return self.get_final_prompt(self.model.prefix, self.positive_preset, prompt)
    
    def get_negative_prompt(self, prompt: Optional[str]) -> str:
        return self.get_final_prompt(self.negative_preset, prompt)
    
    @property
    def model(self) -> SDModel:
        return self._model
    
    @property
    def positive_preset(self) -> str:
        return self._positive_preset
    
    @property
    def negative_preset(self) -> str:
        return self._negative_preset
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe
