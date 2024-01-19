# -*- coding: utf-8 -*-
# File: imgen/diffusion/base.py

import os
import random
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from utils.date_time import get_datetime
from utils.file_ops import create_dir
from utils.hardware import get_total_mem
from utils.types_ import PathLike

from . import DEVICE
from .model import SDModel, get_sd_model
from .scheduler import SchedulerMixin, get_sd_scheduler


class StableDiffusion_(object):
    """
    A superclass for subclasses responsible to generate images using Stable
    Diffusion.
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        *,
        model: Optional[SDModel] = None,
        check_nsfw: bool = False,
        compile_unet: bool = False,
        model_dir: Optional[PathLike] = None,
        device: Optional[str] = None,
        gpu_id: int = 0,
        scheduler: Optional[SchedulerMixin] = None,
        optimizations: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize an instance of `StableDiffusion_`. Should be used as a
        superclass. A custom pipeline called "lpw_stable_diffusion" is used.
        For more information, visit https://github.com/huggingface/diffusers/tree/main/examples/community#long-prompt-weighting-stable-diffusion.

        Args:
            pipeline (DiffusionPipeline): A pipeline extended from
                DiffusionPipeline, e.g. `StableDiffusionPipeline` or
                `StableDiffusionImg2ImgPipeline`.
            model (Optional[SDModel], optional): Instance of `SDModel` from
                model.py. Can be obtained using `get_sd_model(name)` from
                model.py, where `name` is one of the model names from
                `get_sd_model_names()`. Defaults to
                `get_sd_model("Openjourney")`.
            check_nsfw (bool, optional): Enable a safety checker to filter out
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
                Defaults to `get_sd_scheduler("DPMSolverMultistepScheduler")`.
            optimizations (Optional[list[str]], optional): A list of
                optimization methods, e.g. "enable_vae_slicing" and
                "enable_vae_tiling". For more information, visit
                https://huggingface.co/docs/diffusers/optimization/memory.
                Defaults to None.
        """

        self._pipeline = pipeline
        self._model = model if model is not None else get_sd_model("Openjourney")
        self._check_nsfw = check_nsfw
        self._compile_unet = compile_unet
        self._model_dir = model_dir
        self._device = torch.device(device) if device is not None else DEVICE
        self._gpu_id = gpu_id
        self._scheduler = (
            scheduler
            if scheduler is not None
            else get_sd_scheduler("DPMSolverMultistepScheduler")
        )
        self._optimizations = optimizations
        self._torch_dtype = (
            torch.float16 if self._device.type == "cuda" else torch.float32
        )
        self._pipe: Optional[DiffusionPipeline] = None

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize diffusion pipeline, safety checker and optimizations if necessary. Must be
        called by each subclass (preferably in the `__init__()` method).

        Args:
            **kwargs (Any, optional): Keyword arguments (except `torch_dtype`,
                `scheduler`, and `cache_dir`) for instantiating a diffusion
                pipeline from pretrained weights.
        """

        self._pipe = self.pipeline.from_pretrained(
            self.model.path,
            torch_dtype=self._torch_dtype,
            scheduler=self.scheduler.from_pretrained(
                self.model.path, subfolder="scheduler"
            ),
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.model_dir,
            **kwargs,
        )

        # Filter out NSFW images
        if self.check_nsfw:
            from diffusers.pipelines.stable_diffusion import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPImageProcessor

            self._pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=self._torch_dtype,
                cache_dir=self.model_dir,
            )
            self._pipe.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=self._torch_dtype,
                cache_dir=self.model_dir,
            )
            self._pipe.requires_safety_checker = True

        # Enable optimizations
        self._pipe = self._pipe.to(self.device)
        if self.device.type == "cuda":
            if self.compile_unet and torch.__version__ >= "2.0":
                self._pipe.unet = torch.compile(
                    self._pipe.unet, mode="reduce-overhead", fullgraph=True
                )
            else:
                self._pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)
            if isinstance(self.optimizations, list):
                for opt in self.optimizations:
                    getattr(self._pipe, opt)()
            if torch.__version__ < "2.0":
                self._pipe.enable_xformers_memory_efficient_attention()
        else:
            if self.device.type == "mps" and get_total_mem() < 64 * (
                1024**3
            ):  # MPS with less than 64 GB RAM
                self._pipe.enable_attention_slicing()

    def __call__(
        self,
        *,
        output_dir: Optional[PathLike] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """
        Generate images using the diffusion pipeline.

        Args:
            output_dir (Optional[PathLike], optional): Directory to which
                generated images will be saved. None means the generated images
                will not be saved. Defaults to None.
            **kwargs (Any, optional): Keyword-only arguments passed to the
                diffusion pipeline.

        Returns:
            list[Image.Image]: List of generated images. Could be replaced by
                black images if NSFW content is found.
        """

        results = self._pipe(**kwargs).images
        self.save_imgs(results, output_dir=output_dir)
        return results

    def get_generator(self, seed: Optional[int] = None) -> torch.Generator:
        """
        Return a (seeded) random generator for PyTorch.

        Args:
            seed (Optional[int], optional): Seed for the random generator to
                produce deterministic results. If None, it is automatically set
                to a random integer between 0 and 2147483647. Defaults to None.

        Returns:
            torch.Generator: Random number generator.
        """

        return torch.Generator(device=self.device).manual_seed(
            seed if seed is not None else random.randint(0, (1 << 31) - 1),
        )

    @staticmethod
    def load_img(img_path: PathLike) -> Image.Image:
        """
        Load an image file.

        Args:
            img_path (Optional[PathLike], optional): Path to an image file.

        Returns:
            Image.Image: Loaded image.
        """

        return Image.open(img_path).convert("RGB")

    @staticmethod
    def combine_prompts(*prompts: Optional[str]) -> str:
        """
        Combine comma-separated prompts into a single prompt.

        Returns:
            str: Combined prompt.
        """

        return ", ".join(
            [
                p
                for prompt in prompts
                if prompt is not None and (p := prompt.strip()) != ""
            ]
        )

    def get_positive_prompt(self, prompt: str) -> str:
        """
        Return the final positive prompt.

        Args:
            prompt (str): Per-image positive prompt.

        Returns:
            str: Final positive prompt.
        """

        return self.combine_prompts(self.model.positive_prefix, prompt)

    def get_negative_prompt(self, prompt: str) -> str:
        """
        Return the final negative prompt.

        Args:
            prompt (str): Per-image negative prompt.

        Returns:
            str: Final negative prompt.
        """

        return self.combine_prompts(self.model.negative_prefix, prompt)

    @staticmethod
    def save_imgs(
        imgs: list[Image.Image],
        output_dir: Optional[PathLike] = None,
    ) -> None:
        """
        Save images.

        Args:
            imgs (list[Image.Image]): List of images.
            output_dir (Optional[PathLike], optional): Directory to which
                images will be saved. If None, the images will not be saved.
                Defaults to None.
        """

        if output_dir is None:
            return

        create_dir(output_dir, exist_ok=True)
        dt = get_datetime()
        for i, img in enumerate(imgs, start=1):
            img.save(os.path.join(output_dir, f"output_{dt}_{i:0{len(imgs)}d}.png"))

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
    def compile_unet(self) -> bool:
        return self._compile_unet

    @property
    def model_dir(self) -> Optional[str]:
        return self._model_dir

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def gpu_id(self) -> int:
        return self._gpu_id

    @property
    def scheduler(self) -> SchedulerMixin:
        return self._scheduler

    @property
    def optimizations(self) -> Optional[list[str]]:
        return self._optimizations

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._torch_dtype
