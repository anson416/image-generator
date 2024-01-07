# -*- coding: utf-8 -*-
# File: __init__.py

"""
Image generation tool.
"""

import torch

__version__ = "0.5.2"
__author__ = "Anson Lam"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
