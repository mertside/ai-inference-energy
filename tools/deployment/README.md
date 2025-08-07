# Deployment Tools

This directory contains production-ready interfaces for optimal GPU frequency deployment.

## Scripts Overview

### Production Interface

**`deployment_interface.py`**
- **Purpose**: Simple deployment interface for production use
- **Features**: 
  - Easy-to-use API for any GPU-application combination
  - Real data-based frequencies for H100
  - Conservative estimates for A100/V100
  - Command generation for nvidia-smi
- **Usage**:
  ```python
  from deployment_interface import get_optimal_frequency_simple
  
  # Get optimal frequency
  freq = get_optimal_frequency_simple('H100', 'llama')  # Returns 990 MHz
  
  # Get nvidia-smi command
  cmd = get_frequency_command('H100', 'llama')
  ```

## Supported Configurations

### H100 (Real Data Validated)
- **Llama**: 990 MHz (21.5% energy savings, 13.5% performance impact)
- **ViT**: 675 MHz (40.0% energy savings, 14.1% performance impact) 
- **Stable Diffusion**: 1320 MHz (18.1% energy savings, 17.6% performance impact)
- **Whisper**: 1500 MHz (19.5% energy savings, 18.0% performance impact)

### A100 (Conservative Estimates)
- **Llama**: 1200 MHz (15.0% energy savings)
- **ViT**: 1050 MHz (18.0% energy savings)
- **Stable Diffusion**: 1100 MHz (20.0% energy savings)
- **Whisper**: 1150 MHz (16.0% energy savings)

### V100 (Conservative Estimates)  
- **Llama**: 1100 MHz (15.0% energy savings)
- **ViT**: 1050 MHz (18.0% energy savings)
- **Stable Diffusion**: 1000 MHz (22.0% energy savings)
- **Whisper**: 1080 MHz (17.0% energy savings)

## API Reference

### Functions

**`get_optimal_frequency_simple(gpu, application)`**
- Returns optimal frequency in MHz for given GPU-application pair
- Returns None if combination not supported

**`get_frequency_command(gpu, application)`**
- Returns complete nvidia-smi command string
- Ready to execute for setting optimal frequency

**`get_energy_savings_info(gpu, application)`**
- Returns energy savings and performance impact information
- Includes data source reliability indicator

## Production Deployment

This interface is designed for production deployment with:
- Safety-first approach (conservative estimates for unvalidated data)
- Simple API for integration
- Clear data source attribution
- Ready-to-use nvidia-smi commands
