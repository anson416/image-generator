# -*- coding: utf-8 -*-
# File: imgen/diffusion/model.py

from typing import Optional

from utils.types_ import PathLike


class SDModel(object):
    def __init__(
        self,
        name: str,
        path: PathLike,
        positive_prefix: Optional[str] = None,
        negative_prefix: Optional[str] = None,
        website: Optional[str] = None,
    ) -> None:
        self._name = name
        self._path = path
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._website = website

    def __repr__(self) -> str:
        return f"SDModel({self.name}, {self.path}, {self.website})"

    def __str__(self) -> str:
        return f"SDModel({self.name}, {self.path}, {self.website})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def positive_prefix(self) -> str:
        return self._positive_prefix

    @property
    def negative_prefix(self) -> str:
        return self._negative_prefix

    @property
    def website(self) -> str:
        return self._website


_SD_MODELS = {
    "AbyssOrangeMix3A1B": SDModel(
        "AbyssOrangeMix3A1B",
        "stablediffusionapi/abyssorangemix3a1b",
        "",
        "",
        "https://huggingface.co/stablediffusionapi/abyssorangemix3a1b",
    ),
    "Anything v4.0": SDModel(
        "Anything v4.0",
        "xyn-ai/anything-v4.0",
        "",
        "",
        "https://huggingface.co/xyn-ai/anything-v4.0",
    ),
    "Arcane Diffusion": SDModel(
        "Arcane Diffusion",
        "nitrosocke/Arcane-Diffusion",
        "arcane style",
        "",
        "https://huggingface.co/nitrosocke/Arcane-Diffusion",
    ),
    "CetusMix": SDModel(
        "CetusMix",
        "stablediffusionapi/cetusmix",
        "",
        "",
        "https://huggingface.co/stablediffusionapi/cetusmix",
    ),
    "Dreamlike Anime 1.0": SDModel(
        "Dreamlike Anime 1.0",
        "dreamlike-art/dreamlike-anime-1.0",
        "photo anime",
        "",
        "https://huggingface.co/dreamlike-art/dreamlike-anime-1.0",
    ),
    "Dreamlike Diffusion 1.0": SDModel(
        "Dreamlike Diffusion 1.0",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "dreamlikeart",
        "",
        "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0",
    ),
    "Dreamlike Photoreal 2.0": SDModel(
        "Dreamlike Photoreal 2.0",
        "dreamlike-art/dreamlike-photoreal-2.0",
        "photo",
        "",
        "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0",
    ),
    "Ghibli Diffusion": SDModel(
        "Ghibli Diffusion",
        "nitrosocke/Ghibli-Diffusion",
        "ghibli style",
        "https://huggingface.co/nitrosocke/Ghibli-Diffusion",
    ),
    "MeinaMix v11": SDModel(
        "MeinaMix v11",
        "Meina/MeinaMix_V11",
        "",
        "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic)",
        "https://huggingface.co/Meina/MeinaMix_V11",
    ),
    "Openjourney": SDModel(
        "Openjourney",
        "prompthero/openjourney",
        "mdjrny-v4 style",
        "",
        "https://huggingface.co/prompthero/openjourney",
    ),
    "Openjourney v4": SDModel(
        "Openjourney v4",
        "prompthero/openjourney-v4",
        "",
        "",
        "https://huggingface.co/prompthero/openjourney-v4",
    ),
    "Stable Diffusion v1.5": SDModel(
        "Stable Diffusion v1.5",
        "runwayml/stable-diffusion-v1-5",
        "",
        "",
        "https://huggingface.co/runwayml/stable-diffusion-v1-5",
    ),
    "Stable Diffusion v2": SDModel(
        "Stable Diffusion v2",
        "stabilityai/stable-diffusion-2",
        "",
        "",
        "https://huggingface.co/stabilityai/stable-diffusion-2",
    ),
    "Stable Diffusion v2-1": SDModel(
        "Stable Diffusion v2-1",
        "stabilityai/stable-diffusion-2-1",
        "",
        "",
        "https://huggingface.co/stabilityai/stable-diffusion-2-1",
    ),
    "Stable Diffusion x2 Latent Upscaler": SDModel(
        "Stable Diffusion x2 Latent Upscaler",
        "stabilityai/sd-x2-latent-upscaler",
        "",
        "",
        "https://huggingface.co/stabilityai/sd-x2-latent-upscaler",
    ),
    "Stable Diffusion x4 Upscaler": SDModel(
        "Stable Diffusion x4 Upscaler",
        "stabilityai/stable-diffusion-x4-upscaler",
        "",
        "",
        "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
    ),
    "Stable Diffusion XL Base V1.0": SDModel(
        "Stable Diffusion XL Base V1.0",
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        "",
        "",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    ),
    "Stable Diffusion XL Refiner V1.0": SDModel(
        "Stable Diffusion XL Refiner V1.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "",
        "",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0",
    ),
    "Waifu Diffusion": SDModel(
        "Waifu Diffusion",
        "hakurei/waifu-diffusion",
        "",
        "",
        "https://huggingface.co/hakurei/waifu-diffusion",
    ),
}


def get_sd_models() -> dict[str, SDModel]:
    return _SD_MODELS


def get_sd_model_names() -> tuple[str]:
    return tuple(_SD_MODELS.keys())


def get_sd_model(name: str) -> SDModel:
    if (sd_model := _SD_MODELS.get(name)) is None:
        raise KeyError(f'Model "{name}" not found.')
    return sd_model
