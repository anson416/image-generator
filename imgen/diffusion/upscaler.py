# -*- coding: utf-8 -*-
# File: upscaler.py

from typing import Any, List, Optional, Union

import numpy as np
from diffusers import StableDiffusionLatentUpscalePipeline
from PIL import Image
from utils.types import Pathlike

from .base import StableDiffusion_
from .model import get_sd_model


class SDLatentUpscaler(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            StableDiffusionLatentUpscalePipeline,
            model=get_sd_model("SD x2 Latent Upscaler"),
            check_nsfw=False,
            **kwargs,
        )

    def __call__(
        self,
        *,
        img_path: Optional[Union[Pathlike, List[Pathlike]]] = None,
        img: Optional[Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]] = None,
        prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
        n_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        assert img_path or img, "img_path and img cannot be both None"

        if img_path:
            if not isinstance(img_path, list):
                img_path = [img_path]
            image = [Image.open(p).convert("RGB") for p in img_path]
        else:
            image = [img] if not isinstance(img, list) else img

        return super().__call__(
            output_dir=output_dir,
            image=image,
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(neg_prompt),
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )
