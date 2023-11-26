# -*- coding: utf-8 -*-
# File: base.py

import os
import random
from typing import Any, List, Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from utils.date_time import get_datetime
from utils.file_ops import create_dir
from utils.hardware import get_total_mem
from utils.types_ import PathLike

from .. import DEVICE
from .model import SDModel, get_sd_model
from .scheduler import SchedulerMixin, get_sd_scheduler


class StableDiffusion_(object):
    POSITIVE_PRESET = "(((masterpiece))), (((best quality))), ((ultra-detailed)), ((8k))"
    NEGATIVE_PRESET = "lowres, worst quality, low quality, standard quality, error, jpeg artifacts, blurry, username, signature, watermark, text"
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        *,
        model: Optional[SDModel] = None,
        check_nsfw: bool = False,
        positive_preset: Optional[str] = None,
        negative_preset: Optional[str] = None,
        compile_unet: bool = False,
        model_dir: Optional[PathLike] = None,
        device: Optional[str] = None,
        gpu_id: int = 0,
        scheduler: Optional[SchedulerMixin] = None,
        custom_pipeline: Optional[str] = "lpw_stable_diffusion",  # https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion
        optimizations: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._pipeline = pipeline
        self._model = model if model is not None else get_sd_model("SD V2.1")
        self._check_nsfw = check_nsfw
        self._positive_preset = positive_preset if positive_preset is not None else self.POSITIVE_PRESET
        self._negative_preset = negative_preset if negative_preset is not None else self.NEGATIVE_PRESET
        self._compile_unet = compile_unet
        self._model_dir = model_dir
        self._device = device.lower() if device is not None else DEVICE
        self._gpu_id = gpu_id
        self._scheduler = scheduler if scheduler is not None else get_sd_scheduler("UniPCMultistepScheduler")
        self._custom_pipeline = custom_pipeline
        self._optimizations = optimizations
        self._torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._pipe = self._pipeline.from_pretrained(
            self._model.path,
            torch_dtype=self._torch_dtype,
            scheduler=self._scheduler.from_pretrained(self._model.path, subfolder="scheduler"),
            custom_pipeline=self._custom_pipeline,
            cache_dir=self._model_dir,
            **kwargs,
        )

        # Filter out NSFW images
        if self._check_nsfw:
            from diffusers.pipelines.stable_diffusion import \
                StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor

            # Add more special prompts to further reduce the chance of generating NSFW images
            self._positive_preset = self.combine_prompts(self._positive_preset, "(family friendly:0.85)")
            self._negative_preset = self.combine_prompts("((nsfw))", "((nude))", self._negative_preset)

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

        # Enable optimizations
        if self._device.startswith("cuda"):
            if self._compile_unet and torch.__version__ >= "2.0":
                self._pipe = self._pipe.to(self._device)
                self._pipe.unet = torch.compile(self._pipe.unet, mode="reduce-overhead", fullgraph=True)
            else:
                self._pipe.enable_model_cpu_offload(gpu_id=gpu_id)

            if isinstance(self._optimizations, list):
                if "enable_vae_slicing" in self._optimizations:
                    self._pipe.enable_vae_slicing()
                if "enable_vae_tiling" in self._optimizations:
                    self._pipe.enable_vae_tiling()

            if torch.__version__ < "2.0":
                self._pipe.enable_xformers_memory_efficient_attention()
        else:
            self._pipe = self._pipe.to(self._device)
            if self._device == "mps" and get_total_mem() < 64 * (1024 ** 3):  # MPS with less than 64 GB RAM
                self._pipe.enable_attention_slicing()

    def __call__(
        self,
        *,
        output_dir: Optional[PathLike] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        results = self.pipe(**kwargs).images
        self.save_imgs(results, output_dir=output_dir)

        return results
    
    def get_generator(self, seed: Optional[int] = None) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed if seed is not None else random.randint(0, (1 << 31) - 1)
        )
    
    @staticmethod
    def open_img(img_path: PathLike) -> Image.Image:
        return Image.open(img_path).convert("RGB")
    
    @staticmethod
    def combine_prompts(*prompts: Optional[str]) -> str:
        return ", ".join([p for prompt in prompts if prompt is not None and (p := prompt.strip()) != ""])
    
    def get_positive_prompt(self, prompt: str) -> str:
        return self.combine_prompts(self.model.prefix, self.positive_preset, prompt)
    
    def get_negative_prompt(self, prompt: Optional[str]) -> str:
        return self.combine_prompts(self.negative_preset, prompt)
    
    @staticmethod
    def save_imgs(
        imgs: List[Image.Image],
        output_dir: Optional[PathLike] = None,
    ) -> None:
        if output_dir is not None:
            create_dir(output_dir)
            for i, img in enumerate(imgs, start=1):
                img.save(os.path.join(output_dir, f"output_{get_datetime()}_{i:0{len(imgs)}d}.png"))
    
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
    def custom_pipeline(self) -> str:
        return self._custom_pipeline
    
    @property
    def optimizations(self) -> Optional[List[str]]:
        return self._optimizations
    
    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe
