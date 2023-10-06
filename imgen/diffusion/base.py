# -*- coding: utf-8 -*-
# File: base.py

import random
from typing import Dict, List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL.Image import Image
from transformers import CLIPFeatureExtractor

from .. import DEVICE
from .model import SD_MODELS, SDModel


class StableDiffusion_(object):
    positive_preset = "masterpiece, high quality, absurdres, 4K, 8K, HQ"
    negative_preset = "simple background, duplicate, low quality, lowest quality, bad anatomy, bad proportions, extra limbs, fewer limbs, extra fingers, fewer fingers, lowres, username, artist name, error, watermark, signature, text, extra digits, fewer digits, jpeg artifacts, blurry"
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        model: SDModel,
        check_nsfw: bool = False,
        model_dir: Optional[str] = "./models",
        device: Optional[str] = None
    ) -> None:
        self._model = model
        self._device = device if device else DEVICE
        self._pipe = pipeline.from_pretrained(
            model.path,
            torch_dtype=torch.float16,
            cache_dir=model_dir,
        )
        if check_nsfw:
            self._pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=torch.float16,
                cache_dir=model_dir,
            )
            self._pipe.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float16,
                cache_dir=model_dir,
            )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)
            self._pipe.enable_xformers_memory_efficient_attention()

    def __call__(self) -> List[Image]:
        results = self.pipe(...)
        return results

    @classmethod
    def models(cls) -> Dict[str, SDModel]:
        return SD_MODELS
    
    def get_generator(self, seed: Optional[int] = None):
        return torch.Generator(device=self.device).manual_seed(
            seed if seed or seed == 0 else random.randint(0, (1 << 31) - 1)
        )
    
    @staticmethod
    def get_final_prompt(*prompts: Optional[str]) -> str:
        output = []
        for prompt in [prompt for prompt in prompts if prompt and prompt.strip() != ""]:
            output.extend(list(map(lambda x: x.strip(), prompt.split(","))))
        return ", ".join(output)
    
    def get_positive_prompt(self, prompt: str) -> str:
        return self.get_final_prompt(self.model.prefix, self.positive_preset, prompt)
    
    def get_negative_prompt(self, prompt: Optional[str]) -> str:
        return self.get_final_prompt(self.negative_preset, prompt)
    
    @property
    def model(self) -> SDModel:
        return self._model
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe
