# -*- coding: utf-8 -*-
# File: imgen/diffusion/upscaler.py

from typing import Any, Optional

import numpy as np
from PIL import Image
from utils.types_ import PathLike

from .base import StableDiffusion_
from .model import get_sd_model


class SDx2ImageUpscaler(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        from diffusers import StableDiffusionLatentUpscalePipeline

        super().__init__(
            StableDiffusionLatentUpscalePipeline,
            model=get_sd_model("Stable Diffusion x2 Latent Upscaler"),
            **kwargs,
        )
        self.initialize()

    def __call__(
        self,
        *,
        img: Optional[Image.Image | np.ndarray] = None,
        img_path: Optional[PathLike] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_steps: int = 50,
        guidance_scale: float = 7.5,
        output_dir: Optional[PathLike] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        assert (
            img is not None or img_path is not None
        ), "`img` and `img_path` cannot be both None."
        return super().__call__(
            output_dir=output_dir,
            image=img if img is not None else self.load_img(img_path),
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(negative_prompt),
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )


class SDx4ImageUpscaler(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        from diffusers import StableDiffusionUpscalePipeline

        super().__init__(
            StableDiffusionUpscalePipeline,
            model=get_sd_model("Stable Diffusion x4 Upscaler"),
            **kwargs,
        )
        self.initialize()

    def __call__(
        self,
        *,
        img_path: Optional[PathLike] = None,
        img: Optional[Image.Image | np.ndarray] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_imgs: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 7.5,
        output_dir: Optional[PathLike] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        assert (
            img is not None or img_path is not None
        ), "`img` and `img_path` cannot be both None."
        return super().__call__(
            output_dir=output_dir,
            image=img if img is not None else self.load_img(img_path),
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(negative_prompt),
            num_images_per_prompt=n_imgs,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )
