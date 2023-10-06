# -*- coding: utf-8 -*-
# File: image2image.py

from typing import List, Optional
from PIL.Image import Image

from diffusers import StableDiffusionImg2ImgPipeline

from .base import StableDiffusion_
from .model import SDModel

class SDText2Image(StableDiffusion_):
    def __init__(
        self,
        model: SDModel,
        check_nsfw: bool = False,
        model_dir: Optional[str] = "./models",
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
            seed: Optional[int] = None,
        ) -> List[Image]:
        results = self.pipe(
            image=img,
            prompt=prompt,
            negative_prompt="nude",
            width=None,
            height=None,
            num_images_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=self.get_generator(seed=seed),
        )
        return results
