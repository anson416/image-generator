# -*- coding: utf-8 -*-
# File: imgen/__init__.py

"""
Image generation tool.
"""

import torch

__version__ = "0.7.1"
__author__ = "Anson Lam"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
