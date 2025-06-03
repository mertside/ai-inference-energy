# AI Inference Energy

This repository contains the code, scripts, and datasets used in our study on **energy-efficient GPU frequency selection for AI inference workloads**. Building on prior work in analytical and machine learning-based DVFS (Dynamic Voltage and Frequency Scaling), this project investigates how modern AI inference tasks—such as large language models (LLMs), diffusion-based image generation, and retrieval-augmented generation (RAG)—respond to core frequency scaling on **NVIDIA A100 and H100 GPUs**.

## Project Overview

As AI workloads grow in complexity and energy demand, static frequency settings on GPUs often result in sub-optimal trade-offs between performance and power. In this work, we extend prior DVFS optimization frameworks to emerging AI inference scenarios and evaluate their effectiveness using open-source models, including:

- [LLaMA](https://github.com/meta-llama/llama): Text generation via transformer-based LLMs.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion): Latent diffusion model for image generation.
- [ICEAGE](https://...): Retrieval-augmented inference pipeline for scientific data.

## Goals

- Profile GPU power, utilization, and performance across DVFS settings.
- Adapt analytic and ML-based frequency prediction models from prior HPC studies.
- Evaluate energy savings and throughput impact on modern inference workloads.
- Provide reproducible benchmarks and analysis for A100 and H100 platforms.

## Repository Structure

```bash
ai-inference-energy/
├── 
├── 
├── 
├── 
├── 
└── README.md            # Project overview and usage
