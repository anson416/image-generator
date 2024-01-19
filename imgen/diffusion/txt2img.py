# -*- coding: utf-8 -*-
# File: imgen/diffusion/txt2img.py

from typing import Any, Optional

from PIL import Image
from utils.types_ import PathLike

from .base import StableDiffusion_


class SDText2Image(StableDiffusion_):
    """
    A subclass of `StableDiffusion_`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize an instance of `SDText2Image` for generating images from
        text prompts.

        Args:
            model (Optional[SDModel], optional): Instance of `SDModel` from
                model.py. Can be obtained using `get_sd_model(name)` from
                model.py, where `name` is one of the model names from
                `get_sd_model_names()`. Defaults to
                `get_sd_model("Dreamlike Photoreal 2.0")`.
            check_nsfw (bool, optional): Enable a safety checker to prevent
                NSFW (not safe for work) images. Defaults to False.
            compile_unet (bool, optional): Compile UNet for an additonal
                speed-up. Though, this is not suitable for all cases. Defaults
                to False.
            model_dir (Optional[PathLike], optional): Directory to which
                downloaded models will be saved. Defaults to ~/.cache.
            device (Optional[str], optional): Device on which Stable Diffusion
                will run. Defaults to "cuda" if a GPU is available otherwise
                "cpu".
            gpu_id (int, optional): ID of GPU that shall be used in inference.
                Defaults to 0.
            scheduler (Optional[SchedulerMixin], optional): A sampler to
                denoise the encoded image latents. Can be obtained using
                `get_sd_scheduler(name)` from scheduler.py, where `name` is one
                of the scheduler names from `get_sd_scheduler_names()`.
                Defaults to `get_sd_scheduler("UniPCMultistepScheduler")`.
            optimizations (Optional[list[str]], optional): A list of
                optimization methods, e.g. "enable_vae_slicing" and
                "enable_vae_tiling". For more information, visit
                https://huggingface.co/docs/diffusers/optimization/memory.
                Defaults to None.
        """

        from diffusers import StableDiffusionPipeline

        super().__init__(
            StableDiffusionPipeline,
            **kwargs,
        )
        self.initialize(
            custom_pipeline=f"lpw_stable_diffusion{'_xl' if 'Stable Diffusion XL' in self.model.name else ''}",
        )

    def __call__(
        self,
        *,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        n_imgs: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 7.5,
        output_dir: Optional[PathLike] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompt (Optional[str], optional): Prompt to guide image generation.
                Defaults to None.
            negative_prompt (Optional[str], optional): Prompt to guide what to
                not include in image generation. Defaults to None.
            width (Optional[int], optional): Width (in pixels) of the generated
                images. Defaults to the default settings of the model.
            height (Optional[int], optional): Height (in pixels) of the
                generated images. Defaults to the default settings of the
                model.
            n_imgs (int, optional): Number of images to generate. Defaults to 1.
            n_steps (int, optional): Number of denoising steps. More denoising
                steps usually lead to a higher quality image at the expense of
                slower inference. Defaults to 50.
            guidance_scale (float, optional): A higher value encourages the
                model to generate images closely linked to `prompt` at the
                expense of lower image quality. Enabled when
                `guidance_scale` > 1. Defaults to 7.5.
            output_dir (Optional[PathLike], optional): Directory to which
                generated images will be saved. None means the generated images
                will not be saved. Defaults to None.
            seed (Optional[int], optional): Seed for the random generator to
                produce deterministic results. If None, it is automatically set
                to a random integer between 0 and 2147483647. Defaults to None.

        Returns:
            list[Image.Image]: List of generated images. Could be replaced by
                black images if `check_nsfw` is True and NSFW content is found.
        """

        return super().__call__(
            output_dir=output_dir,
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(negative_prompt),
            width=width,
            height=height,
            num_images_per_prompt=n_imgs,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )
