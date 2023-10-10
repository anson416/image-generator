# -*- coding: utf-8 -*-
# File: txt2img.py

from typing import Any, List, Optional

from diffusers import StableDiffusionPipeline
from PIL import Image
from utils.types import Pathlike

from .base import StableDiffusion_


class SDText2Image(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            StableDiffusionPipeline,
            **kwargs,
        )

    def __call__(
        self,
        *,
        prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        n_imgs: int = 1,
        output_dir: Optional[Pathlike] = None,
        n_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        return super().__call__(
            output_dir=output_dir,
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(neg_prompt),
            width=width,
            height=height,
            num_images_per_prompt=n_imgs,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )
