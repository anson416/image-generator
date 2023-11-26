# -*- coding: utf-8 -*-
# File: model.py

from typing import Dict, Optional, Tuple

from utils.types_ import PathLike


class SDModel(object):
    def __init__(
        self,
        name: str,
        path: PathLike,
        prefix: Optional[str] = None,
        website: Optional[str] = None,
    ) -> None:
        self._name = name
        self._path = path
        self._prefix = prefix
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
    def prefix(self) -> str:
        return self._prefix
    
    @property
    def website(self) -> str:
        return self._website
    

_SD_MODELS = {
    "Anything V4.0": SDModel("Anything V4.0", "xyn-ai/anything-v4.0", "", "https://huggingface.co/xyn-ai/anything-v4.0"),
    "Anything V4.5": SDModel("Anything V4.5", "Airic/Anything-V4.5", "", "https://huggingface.co/Airic/Anything-V4.5"),
    "Arcane Diffusion": SDModel("Arcane Diffusion", "nitrosocke/Arcane-Diffusion", "arcane style", "https://huggingface.co/nitrosocke/Arcane-Diffusion"),
    "Dreamlike Anime 1.0": SDModel("Dreamlike Anime 1.0", "dreamlike-art/dreamlike-anime-1.0", "photo anime", "https://huggingface.co/dreamlike-art/dreamlike-anime-1.0"),
    "Dreamlike Diffusion 1.0": SDModel("Dreamlike Diffusion 1.0", "dreamlike-art/dreamlike-diffusion-1.0", "dreamlikeart", "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0"),
    "Dreamlike Photoreal 2.0": SDModel("Dreamlike Photoreal 2.0", "dreamlike-art/dreamlike-photoreal-2.0", "photo", "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0"),
    "Ghibli Diffusion": SDModel("Ghibli Diffusion", "nitrosocke/Ghibli-Diffusion", "ghibli style", "https://huggingface.co/nitrosocke/Ghibli-Diffusion"),
    "Openjourney": SDModel("Openjourney", "prompthero/openjourney", "mdjrny-v4 style", "https://huggingface.co/prompthero/openjourney"),
    "SD V1.5": SDModel("Stable Diffusion V1.5", "runwayml/stable-diffusion-v1-5", "", "https://huggingface.co/runwayml/stable-diffusion-v1-5"),
    "SD V2.0": SDModel("Stable Diffusion V2.0", "stabilityai/stable-diffusion-2", "", "https://huggingface.co/stabilityai/stable-diffusion-2"),
    "SD V2.1": SDModel("Stable Diffusion V2.1", "stabilityai/stable-diffusion-2-1", "", "https://huggingface.co/stabilityai/stable-diffusion-2-1"),
    "SD x2 Latent Upscaler": SDModel("Stable Diffusion x2 Latent Upscaler", "stabilityai/sd-x2-latent-upscaler", "", "https://huggingface.co/stabilityai/sd-x2-latent-upscaler"),
    "SD x4 Upscaler": SDModel("Stable Diffusion x4 Upscaler", "stabilityai/stable-diffusion-x4-upscaler", "", "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"),
    "SDXL Base V1.0": SDModel("Stable Diffusion XL Base V1.0", "stabilityai/stable-diffusion-xl-base-1.0", "", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"),
    "Waifu Diffusion": SDModel("Waifu Diffusion", "hakurei/waifu-diffusion", "", "https://huggingface.co/hakurei/waifu-diffusion"),
}


def get_sd_models() -> Dict[str, SDModel]:
    return _SD_MODELS


def get_sd_model_names() -> Tuple[str]:
    return tuple(_SD_MODELS.keys())


def get_sd_model(name: str) -> SDModel:
    if not (sd_model := _SD_MODELS.get(name, None)):
        raise KeyError(f"Model \"{name}\" not found")
    return sd_model
