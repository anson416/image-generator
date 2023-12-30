# -*- coding: utf-8 -*-
# File: setup.py

from setuptools import find_packages, setup

from imgen import constants

NAME = "imgen"
DESCRIPTION = "Image generation tool."
EMAIL = "lamyiufung2003@gmail.com"
PYTHON_VERSION = ">=3.10.0"
URL = "https://github.com/anson416/image-generator"

with open("README.md", "r") as f:
    long_description = f.read()

with open("./requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name=NAME,
    version=constants.VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=constants.AUTHOR,
    author_email=EMAIL,
    python_requires=PYTHON_VERSION,
    url=URL,
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
