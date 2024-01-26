"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ilmot",
    version="0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "Pillow",
        "plyfile",
        "psutil",
        "pycocotools",
        "pydantic",
        "pytoml",
        "PyYAML",
        "requests",
        "scalabel",
        "torch",
        "torchvision",
        "tqdm",
        "devtools",
    ],
)
