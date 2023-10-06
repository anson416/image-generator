# -*- coding: utf-8 -*-
# File: __init__.py

"""
Image generation tool.
"""

import torch

__version__ = "0.1.7"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
