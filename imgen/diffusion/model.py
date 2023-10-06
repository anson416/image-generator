# -*- coding: utf-8 -*-
# File: model.py

from typing import Optional


class SDModel(object):
    def __init__(
        self,
        name: str,
        path: str,
        prefix: Optional[str] = "",
        website: Optional[str] = "",
    ) -> None:
        self._name = name
        self._path = path
        self._prefix = prefix if prefix else ""
        self._website = website if website else ""

    def __repr__(self) -> str:
        return f"SDModel({self.name}, {self.website})"
    
    def __str__(self) -> str:
        return f"SDModel({self.name}, {self.website})"

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
    

SD_MODELS = {
    "Anything V4.0": SDModel("Anything V4.0", "andite/anything-v4.0", "", "https://huggingface.co/xyn-ai/anything-v4.0"),
    "Anything V4.5": SDModel("Anything V4.5", "andite/anything-v4.5", "", "https://huggingface.co/Airic/Anything-V4.5"),
    "Arcane Diffusion": SDModel("Arcane Diffusion", "nitrosocke/Arcane-Diffusion", "arcane style", "https://huggingface.co/nitrosocke/Arcane-Diffusion"),
    "Dreamlike Anime 1.0": SDModel("Dreamlike Anime 1.0", "dreamlike-art/dreamlike-anime-1.0", "photo anime", "https://huggingface.co/dreamlike-art/dreamlike-anime-1.0"),
    "Dreamlike Diffusion 1.0": SDModel("Dreamlike Diffusion 1.0", "dreamlike-art/dreamlike-diffusion-1.0", "dreamlikeart", "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0"),
    "Dreamlike Photoreal 2.0": SDModel("Dreamlike Photoreal 2.0", "dreamlike-art/dreamlike-photoreal-2.0", "photo", "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0"),
    "Waifu Diffusion": SDModel("Waifu Diffusion", "hakurei/waifu-diffusion", "", "https://huggingface.co/hakurei/waifu-diffusion")
}
