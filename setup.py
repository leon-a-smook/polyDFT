# setup.py
from setuptools import setup, find_packages

setup(
    name="polymer_dft",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib",
        "pyyaml",
    ],
    author="Your Name",
    description="Polymer DFT simulation package",
)