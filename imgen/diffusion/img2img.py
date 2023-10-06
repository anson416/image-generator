# -*- coding: utf-8 -*-
# File: img2img.py

from typing import Any, List, Optional, Union

import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from .base import StableDiffusion_
from .model import SDModel


class SDImage2Image(StableDiffusion_):
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
            prompt: str,
            img_path: Optional[str] = None,
            img: Optional[Union[Image.Image, np.ndarray]] = None,
            neg_prompt: Optional[str] = None,
            n_images: int = 1,
            n_steps: int = 50,
            strength: float = 0.8,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> List[Image.Image]:
        assert img_path or img, "img_path and img cannot be both None"

        results = self.pipe(
            image=Image.open(img_path).convert("RGB") if img_path else img,
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(neg_prompt),
            num_images_per_prompt=n_images,
            num_inference_steps=n_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        ).images
        return results
