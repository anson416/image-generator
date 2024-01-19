# -*- coding: utf-8 -*-
# File: imgen/diffusion/img2img.py

from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
from utils.date_time import get_datetime
from utils.types_ import PathLike

from .base import StableDiffusion_


class SDImage2Image(StableDiffusion_):
    """
    A subclass of `StableDiffusion_`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize an instance of `SDImage2Image` for generating images from
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

        from diffusers import StableDiffusionImg2ImgPipeline

        super().__init__(
            StableDiffusionImg2ImgPipeline,
            **kwargs,
        )
        self.initialize(
            custom_pipeline=f"lpw_stable_diffusion{'_xl' if 'Stable Diffusion XL' in self.model.name else ''}",
        )

    def __call__(
        self,
        *,
        img: Optional[Image.Image | np.ndarray] = None,
        img_path: Optional[PathLike] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_imgs: int = 1,
        n_steps: int = 50,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        output_dir: Optional[PathLike] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """
        Generate images from a base image and text prompts.

        Args:
            img (Optional[Image.Image | np.ndarray], optional): Loaded image.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            img_path (Optional[PathLike], optional): Path to an image file.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            prompt (Optional[str], optional): Prompt to guide image generation.
                Defaults to None.
            negative_prompt (Optional[str], optional): Prompt to guide what to
                not include in image generation. Defaults to None.
            n_imgs (int, optional): Number of images to generate. Defaults to 1.
            n_steps (int, optional): Number of denoising steps. More denoising
                steps usually lead to a higher quality image at the expense of
                slower inference. Defaults to 50.
            strength (float, optional): Control extent to transform the
                original image. Must be between 0 and 1. A higher value means
                more noise is added to the original image. The number of
                denoising steps depends on `strength`. A value of 1 essentially
                ignores the image. Defaults to 0.8.
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
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )


class ControlSDImage2Image(StableDiffusion_):
    """
    A subclass of `StableDiffusion_`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize an instance of `ControlSDImage2Image` for generating images
        from text prompts.

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

        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

        super().__init__(
            StableDiffusionControlNetImg2ImgPipeline,
            **kwargs,
        )
        self.initialize(
            controlnet=ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self.torch_dtype,
                cache_dir=self.model_dir,
            ),
        )

    def __call__(
        self,
        *,
        img: Optional[Image.Image | np.ndarray] = None,
        img_path: Optional[PathLike] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_imgs: int = 1,
        n_steps: int = 50,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        lower_threshold: float = 100,
        upper_threshold: float = 200,
        output_dir: Optional[PathLike] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """
        Generate controlled images (by using edges) from a base image and text
        prompts.

        Args:
            img (Optional[Image.Image | np.ndarray], optional): Loaded image.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            img_path (Optional[PathLike], optional): Path to an image file.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            prompt (Optional[str], optional): Prompt to guide image generation.
                Defaults to None.
            negative_prompt (Optional[str], optional): Prompt to guide what to
                not include in image generation. Defaults to None.
            n_imgs (int, optional): Number of images to generate. Defaults to 1.
            n_steps (int, optional): Number of denoising steps. More denoising
                steps usually lead to a higher quality image at the expense of
                slower inference. Defaults to 50.
            strength (float, optional): Control extent to transform the
                original image. Must be between 0 and 1. A higher value means
                more noise is added to the original image. The number of
                denoising steps depends on `strength`. A value of 1 essentially
                ignores the image. Defaults to 0.8.
            guidance_scale (float, optional): A higher value encourages the
                model to generate images closely linked to `prompt` at the
                expense of lower image quality. Enabled when
                `guidance_scale` > 1. Defaults to 7.5.
            lower_threshold (float, optional): Any edges with intensity
                gradient lower than `lower_threshold` are sure to be non-edges.
                Defaults to 100.
            upper_threshold (float, optional): Any edges with intensity
                gradient higher than `upper_threshold` are sure to be edges.
                Defaults to 200.
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

        assert (
            img is not None or img_path is not None
        ), "`img` and `img_path` cannot be both None."
        image = img if img is not None else self.load_img(img_path)
        return super().__call__(
            output_dir=output_dir,
            image=image,
            control_image=self.get_canny_img(
                img=image,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            ),
            prompt=self.get_positive_prompt(prompt),
            negative_prompt=self.get_negative_prompt(negative_prompt),
            num_images_per_prompt=n_imgs,
            num_inference_steps=n_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self.get_generator(seed=seed),
            **kwargs,
        )

    @staticmethod
    def get_canny_img(
        img: Optional[Image.Image | np.ndarray] = None,
        img_path: Optional[PathLike] = None,
        lower_threshold: float = 100,
        upper_threshold: float = 200,
    ) -> Image.Image:
        """
        Detect edges in an image.

        Args:
            img (Optional[Image.Image | np.ndarray], optional): Loaded image.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            img_path (Optional[PathLike], optional): Path to an image file.
                `img` (primary) and `img_path` cannot be both None. Defaults to
                None.
            lower_threshold (float, optional): Any edges with intensity
                gradient lower than `lower_threshold` are sure to be non-edges.
                Defaults to 100.
            upper_threshold (float, optional): Any edges with intensity
                gradient higher than `upper_threshold` are sure to be edges.
                Defaults to 200.

        Returns:
            Image.Image: Edges in the original image.
        """

        assert (
            img is not None or img_path is not None
        ), "`img` and `img_path` cannot be both None."

        canny_img = img if img is not None else StableDiffusion_.load_img(img_path)
        canny_img = np.array(canny_img)
        canny_img = cv2.Canny(canny_img, lower_threshold, upper_threshold)[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)

        return Image.fromarray(canny_img)


def video2video(
    pipe: SDImage2Image,
    video_path: PathLike,
    output_path: Optional[PathLike] = None,
    **kwargs: Any,
) -> None:
    output_path = (
        output_path if output_path is not None else f"./output_{get_datetime()}.mp4"
    )
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    video_writer = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not video_writer:
            video_writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.size
            )
        video_writer.write(
            np.array(
                pipe(
                    img=frame,
                    n_imgs=1,
                    output_dir=None,
                    **kwargs,
                )[0]
            )[:, :, ::-1]
        )

    video_writer.release()
    video.release()
    cv2.destroyAllWindows()
