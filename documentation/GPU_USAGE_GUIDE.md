# GPU Usage Guide

This guide summarizes GPU setup and usage for the AI Inference Energy profiling framework. It covers verifying GPU availability, launching jobs, and basic troubleshooting for NVIDIA A100, V100 and H100 devices.

## GPU Setup

1. **Verify GPU visibility**
   ```bash
   nvidia-smi
   ```
2. **Check optional DCGM tools**
   ```bash
   dcgmi discovery --list
   ```
3. **Configure environments** – see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for CLI examples and environment tips.

## Running the Framework

Launch profiling from the `sample-collection-scripts` directory:

```bash
cd ../sample-collection-scripts
./launch_v2.sh --gpu-type A100 --profiling-mode baseline
```

### GPU Notes

- **A100** – use `--gpu-type A100`; requires HPCC toreador nodes.
- **V100** – use `--gpu-type V100` for matador nodes.
- **H100** – use `--gpu-type H100` for REPACSS systems.

## Troubleshooting

```bash
# Reset GPU if necessary
sudo nvidia-smi --gpu-reset

# Test configuration
./launch_v2.sh --help
```

## Related Documentation

- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
- [SUBMIT_JOBS_README.md](SUBMIT_JOBS_README.md)
- [../sample-collection-scripts/README.md](../sample-collection-scripts/README.md)
