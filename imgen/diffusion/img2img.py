# -*- coding: utf-8 -*-
# File: img2img.py

from typing import Any, List, Optional, Union

import cv2
import numpy as np
from diffusers import (ControlNetModel,
                       StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionImg2ImgPipeline)
from PIL import Image
from utils.date_time import get_datetime
from utils.types_ import PathLike

from .base import StableDiffusion_


class SDImage2Image(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            StableDiffusionImg2ImgPipeline,
            **kwargs,
        )
        self.initialize()
    
    def __call__(
        self,
        *,
        img: Optional[Union[Image.Image, np.ndarray]] = None,
        img_path: Optional[PathLike] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_imgs: int = 1,
        output_dir: Optional[PathLike] = None,
        n_steps: int = 50,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        return super().__call__(
            output_dir=output_dir,
            image=self.load_img(img=img, img_path=img_path),
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
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            StableDiffusionControlNetImg2ImgPipeline,
            **kwargs,
        )
        self.pipe.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=self.torch_dtype,
            cache_dir=self.model_dir,
        )
        self.initialize()

    def __call__(
        self,
        *,
        img: Optional[Union[Image.Image, np.ndarray]] = None,
        img_path: Optional[PathLike] = None,
        lower_threshold: float = 100,
        upper_threshold: float = 200,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n_imgs: int = 1,
        output_dir: Optional[PathLike] = None,
        n_steps: int = 50,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        image = self.load_img(img=img, img_path=img_path)
        
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
        img: Optional[Union[Image.Image, np.ndarray]] = None,
        img_path: Optional[PathLike] = None,
        lower_threshold: float = 100,
        upper_threshold: float = 200,
    ) -> Image.Image:
        canny_img = StableDiffusion_.load_img(img=img, img_path=img_path)
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
    output_path = output_path if output_path else f"./output_{get_datetime()}.mp4"
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS if int(cv2.__version__.split(".")[0]) < 3 else cv2.CAP_PROP_FPS)

    video_writer = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not video_writer:
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.size)
        video_writer.write(np.array(pipe(
            img=frame,
            n_imgs=1,
            output_dir=None,
            **kwargs,
        )[0])[:, :, ::-1])

    video_writer.release()
    video.release()
    cv2.destroyAllWindows()
