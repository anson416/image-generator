# -*- coding: utf-8 -*-
# File: imgen/gan/realesrgan/inference.py

import os
from typing import Literal, Optional, Union

import cv2
import cv2.typing
from utils.date_time import get_datetime
from utils.file_ops import create_dir

from . import DEVICE, WEIGHTS_PATH, RealESRGANer
from .archs.srvgg_arch import SRVGGNetCompact
from .basicsr.archs.rrdbnet_arch import RRDBNet


class RealESRGAN(object):
    def __init__(
        self,
        model_name: Literal[
            "realesr-general-x4v3",
            "RealESRGAN_x4plus",
        ] = "realesr-general-x4v3",
        denoise_strength: float = 1,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: Optional[bool] = None,
        gpu_id: int = 0,
    ) -> None:
        """
        Create an instance of `RealESRGAN`.

        Args:
            model_name (str): Real-ESRGAN model name. Defaults to
                "realesr-general-x4v3".
            denoise_strength (float, optional): A value between 0 and 1
                controlling the level of denoising. 0 for weak denoising, 1 for
                strong denoising. Used only when `model_name` equals
                "realesr-general-x4v3". Defaults to 1.
            tile (int, optional): If not 0, first crop
                the input image into `tile` tiles, and then process each of
                them. When using CUDA, a high value may cause out of memory
                (OOM). Defaults to 0.
            tile_pad (int, optional): Pad size for each tile to prevent border
                artifacts. Defaults to 10.
            pre_pad (int, optional): Pad size for the input image to prevent
                border artifacts. Defaults to 10.
            half (Optional[bool], optional): Use half precision (FP16) to speed
                up inference. May not be suitable when using CPU. Defaults to
                `torch.cuda.is_available()`.
            gpu_id (int, optional): Device ID of GPU to use. Defaults to 0.
        """

        assert 0 <= denoise_strength <= 1, (
            f"{denoise_strength} < 0 or {denoise_strength} > 1. "
            "`denoise_strength` must be a number between 0 and 1 inclusively."
        )
        assert tile >= 0, f"{tile} < 0. `tile` must be a non-negative integer."
        assert (
            tile_pad >= 0
        ), f"{tile_pad} < 0. `tile_pad` must be a non-negative integer."
        assert pre_pad >= 0, f"{pre_pad} < 0. `pre_pad` must be a non-negative integer."
        assert gpu_id >= 0, f"{gpu_id} < 0. `gpu_id` must be a non-negative integer."

        self._model_name = model_name
        self._denoise_strength = denoise_strength
        self._tile = tile
        self._tile_pad = tile_pad
        self._pre_pad = pre_pad
        self._half = half if half is not None else DEVICE.type == "cuda"
        self._gpu_id = gpu_id

        self._upsampler: Optional[RealESRGANer] = None
        self._initialize()

    def _initialize(self) -> None:
        # Initialize model
        if self.model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            net_scale = 4
        elif self.model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            net_scale = 4

        # Determine model path
        model_path: Union[str, list[str]] = os.path.join(
            WEIGHTS_PATH, f"{self.model_name}.pth"
        )
        dni_weight: Optional[list[float]] = None
        if self.model_name == "realesr-general-x4v3" and self.denoise_strength != 1:
            wdn_model_path = os.path.join(WEIGHTS_PATH, "realesr-general-wdn-x4v3.pth")
            model_path = [model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        # Initialize upsampler
        self._upsampler = RealESRGANer(
            scale=net_scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half,
            gpu_id=self.gpu_id,
        )

    def __call__(
        self,
        img: cv2.typing.MatLike,
        out_scale: int = 4,
        output_dir: Optional[str] = None,
        img_name: Optional[str] = None,
        suffix: Optional[str] = None,
        ext: str = "png",
    ) -> Optional[cv2.typing.MatLike]:
        """
        Upsample an image.

        Args:
            img (cv2.typing.MatLike): Target image.
            out_scale (int, optional): Final upsampling scale of `img`.
                Defaults to 4.
            output_dir (Optional[str], optional): If not None, save the
                upsampled image to `output_dir`. Defaults to None.
            img_name (Optional[str], optional): Name of the upsampled image.
                Used only when `output_dir` is not None. Defaults to None.
            suffix (Optional[str], optional): Suffix to put after `img_name`.
                Used only when `output_dir` is not None. Defaults to None.
            ext (str, optional): File extension of the upsampled image. Used
                only when `output_dir` is not None. Defaults to "png".

        Returns:
            Optional[cv2.typing.MatLike]: Upsampled image.
        """

        output: cv2.typing.MatLike = self._upsampler.enhance(img, outscale=out_scale)[0]

        if output_dir is not None:
            # Create output directory
            create_dir(output_dir, exist_ok=True)

            # Set output image name
            if img_name is None:
                img_name = "output"
            if suffix is None:
                suffix = get_datetime()

            # Save image
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_{suffix}.{ext}"), output)

        return output

    @staticmethod
    def load_img(path: str) -> cv2.typing.MatLike:
        """
        Load an image.

        Args:
            path (str): Path to target image.

        Returns:
            cv2.typing.MatLike: Loaded image.
        """

        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def denoise_strength(self) -> float:
        return self._denoise_strength

    @property
    def tile(self) -> int:
        return self._tile

    @property
    def tile_pad(self) -> int:
        return self._tile_pad

    @property
    def pre_pad(self) -> int:
        return self._pre_pad

    @property
    def half(self) -> bool:
        return self._half

    @property
    def gpu_id(self) -> int:
        return self._gpu_id
