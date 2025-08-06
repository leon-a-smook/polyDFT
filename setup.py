# setup.py
from setuptools import setup, find_packages

setup(
    name="dft",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib",
        "pyyaml",
        "numpy"
    ],
    author="Leon A. Smook",
    description="Polymer DFT simulation package",
)