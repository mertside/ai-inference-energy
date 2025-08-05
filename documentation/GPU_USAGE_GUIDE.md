# GPU Usage Guide

This guide outlines basic GPU setup checks and usage patterns for running inference energy profiling.

## GPU Setup

1. **Verify drivers and tools**
   ```bash
   nvidia-smi            # confirm GPU visibility
   dcgmi discovery --list  # optional: check DCGM interface
   ```
2. **Confirm Python environment** – ensure required packages are installed.

## Running Profiling Scripts

Use the launch script inside `sample-collection-scripts` to run workloads. The script automatically detects the GPU type and chooses an appropriate profiling tool.

```bash
cd sample-collection-scripts
./launch_v2.sh --gpu-type A100 --app-name LSTM --profiling-mode baseline
```

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for more command examples and [SUBMIT_JOBS_README.md](SUBMIT_JOBS_README.md) for batch submission instructions.

## Troubleshooting

If profiling tools are missing or GPUs are not visible:

- Reinstall NVIDIA drivers or ensure you are inside a GPU-enabled environment.
- Run `sudo nvidia-smi --gpu-reset` to reset a stuck GPU.
- Consult the repository [README](../README.md) for additional tips.
