# -*- coding: utf-8 -*-
# File: __init__.py

"""
Image generation tool.
"""

import constants
import torch

__version__ = constants.VERSION
__author__ = constants.AUTHOR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
