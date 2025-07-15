# LLaMA Environment Setup for Energy Profiling Research

This directory contains conda environment configurations for different cluster setups to run LLaMA text generation workloads.

## Available Environments

- **conda_env_h100.yml**: Optimized for H100 GPUs with latest CUDA and transformer versions
- **conda_env_a100.yml**: Optimized for A100 GPUs with stable CUDA 11.8
- **conda_env_v100.yml**: Compatible with V100 GPUs using CUDA 11.7
- **conda_env_generic.yml**: Generic environment for development and testing

## Quick Setup

```bash
# For H100 clusters
conda env create -f conda_env_h100.yml
conda activate llama-h100

# For A100 clusters  
conda env create -f conda_env_a100.yml
conda activate llama-a100

# For V100 clusters
conda env create -f conda_env_v100.yml
conda activate llama-v100
```

## Authentication

Before running LLaMA models, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. Ensure you have access to the LLaMA models you plan to use.

## Verification

Test your environment setup:

```bash
cd ../
python LlamaViaHF.py --model llama-7b --prompt "Hello world" --max-tokens 10 --metrics
```

## GPU Memory Requirements

| Model | Minimum VRAM | Recommended VRAM |
|-------|-------------|------------------|
| LLaMA-7B | 14GB | 16GB |
| LLaMA-13B | 26GB | 32GB |
| LLaMA-30B | 60GB | 80GB |
| LLaMA-65B | 120GB | 160GB |
| Llama-2-7B | 14GB | 16GB |
| Llama-2-13B | 26GB | 32GB |
| Llama-2-70B | 140GB | 200GB |

Use model sharding for larger models on multi-GPU setups.
