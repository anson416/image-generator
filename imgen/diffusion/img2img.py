# -*- coding: utf-8 -*-
# File: img2img.py

import os
from typing import Any, List, Optional, Union

import cv2
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from utils.date_time import get_datetime
from utils.file_ops import create_dir

from .base import StableDiffusion_


class SDImage2Image(StableDiffusion_):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(StableDiffusionImg2ImgPipeline, **kwargs)
    
    def __call__(
            self,
            prompt: Optional[str] = None,
            img_path: Optional[str] = None,
            img: Optional[Union[Image.Image, np.ndarray]] = None,
            neg_prompt: Optional[str] = None,
            n_images: int = 1,
            output_dir: Optional[str] = None,
            n_steps: int = 50,
            strength: float = 0.8,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            output_type: str = "pil",
            **kwargs: Any,
        ) -> List[Union[Image.Image, np.ndarray]]:
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
            output_type=output_type,
            **kwargs,
        ).images

        if output_dir:
            create_dir(output_dir)
            for i, result in enumerate(results, start=1):
                result.save(os.path.join(output_dir, f"output_{i}.png"))

        return results


def video2video(
    pipe: SDImage2Image,
    video_path: str,
    output_path: Optional[str] = None
) -> None:
    output_path = output_path if output_path else f"./output_{get_datetime(r'%Y%m%d', r'%H%M%S', '')}.mp4"
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
        video_writer.write(np.array(pipe(img=frame)[0])[:, :, ::-1])

    video_writer.release()
    video.release()
    cv2.destroyAllWindows()
