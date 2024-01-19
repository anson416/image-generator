# -*- coding: utf-8 -*-
# File: imgen/gan/realesrgan/__init__.py

import os

from .. import DEVICE
from .utils import *

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
