# -*- coding: utf-8 -*-
# File: setup.py

from pathlib import Path

from setuptools import find_packages, setup

NAME = "imgen"
DESCRIPTION = "Image generation tool."
EMAIL = "lamyiufung2003@gmail.com"
PYTHON_VERSION = ">=3.10.0"
URL = "https://github.com/anson416/image-generator"

with (Path(NAME) / "__init__.py").open() as f:
    for line in f.read().splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip(" '\"")
        if line.startswith("__author__"):
            author = line.split("=")[-1].strip(" '\"")

with Path("./README.md").open() as f:
    long_description = f.read()

with Path("./requirements.txt").open() as f:
    install_requires = f.read().splitlines()

setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=EMAIL,
    python_requires=PYTHON_VERSION,
    url=URL,
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
