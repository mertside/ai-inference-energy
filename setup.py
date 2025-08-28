"""
Setup script for AI Inference Energy Profiling Framework.

This setup.py file allows the framework to be installed as a proper Python package,
making imports and usage more convenient.

Installation:
    pip install -e .  # Development install
    pip install .     # Regular install
"""

import pathlib

from setuptools import find_packages, setup

# Read the README file
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

# Read requirements
REQUIREMENTS = (HERE / "requirements.txt").read_text().strip().split("\n")

setup(
    name="ai-inference-energy",
    version="2.1.0",
    description="Energy profiling framework for AI inference workloads on NVIDIA GPUs",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Mert Side",
    author_email="mert.side@ttu.edu",
    url="https://github.com/mertside/ai-inference-energy",
    license="MIT",
    packages=find_packages(include=["examples", "examples.*", "*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
    ],
    keywords=[
        "ai",
        "inference",
        "energy",
        "profiling",
        "gpu",
        "nvidia",
        "dvfs",
        "llama",
        "stable-diffusion",
        "power",
        "frequency",
        "scaling",
    ],
    entry_points={
        "console_scripts": [
            "ai-energy-profile=sample-collection-scripts.profile:main",
            "ai-energy-launch=examples.example_usage:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/mertside/ai-inference-energy/issues",
        "Source": "https://github.com/mertside/ai-inference-energy",
        "Documentation": "https://github.com/mertside/ai-inference-energy/wiki",
    },
)
