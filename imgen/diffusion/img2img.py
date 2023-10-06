# -*- coding: utf-8 -*-
# File: img2img.py

from typing import Any, List, Optional

from diffusers import StableDiffusionImg2ImgPipeline
from PIL.Image import Image

from .base import StableDiffusion_
from .model import SDModel


class SDText2Image(StableDiffusion_):
    def __init__(
        self,
        model: SDModel,
        check_nsfw: bool = False,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            StableDiffusionImg2ImgPipeline,
            model,
            check_nsfw=check_nsfw,
            model_dir=model_dir,
        )
    
    def __call__(
            self,
            img: Image,
            prompt: str,
            neg_prompt: Optional[str] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            n_images: int = 1,
            n_steps: int = 50,
            strength: float = 0.8,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> List[Image]:
        results = self.pipe(
            image=img,
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(neg_prompt),
            width=width,
            height=height,
            num_images_per_prompt=n_images,
            num_inference_steps=n_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        ).images
        return results
