# Quick Fix Guide for Common Issues

## Issue 1: Python 3.6 Dataclass Compatibility ✅ FIXED
**Problem:** `ValueError: mutable default <class 'list'> for field A100_CORE_FREQUENCIES is not allowed`

**Solution:** Updated `config.py` to use regular classes instead of dataclasses for Python 3.6 compatibility.

## Issue 2: Python 3.6 subprocess.run Compatibility ✅ FIXED
**Problem:** `TypeError: unexpected keyword argument 'capture_output'`

**Cause:** The `capture_output` parameter was added in Python 3.7, but the cluster uses Python 3.6.

**Solution:** Replaced all `capture_output=True` with `stdout=subprocess.PIPE, stderr=subprocess.PIPE` in:
- `sample-collection-scripts/profile.py`
- `sample-collection-scripts/profile_smi.py`

## Issue 3: GPU Type Mismatch ⚠️ NEEDS ATTENTION
**Problem:** You're running V100 configuration but have an A100 GPU

**From your logs:**
```
[INFO] Detected GPU: NVIDIA A100-PCIE-40GB
[WARNING] GPU type set to V100 but detected GPU doesn't appear to be V100
```

**Solutions:**

### Option 1: Use A100 Configuration (Recommended)
Change your submit script to use A100 instead of V100:

```bash
# In submit_job_v100_baseline.sh, change:
LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 2 --sleep-interval 1"

# To:
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 2 --sleep-interval 1"
```

### Option 2: Create A100 Baseline Script
Create a new script `submit_job_a100_baseline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=AI_ENERGY_A100_BASELINE
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=01:00:00

set -euo pipefail

readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# A100 baseline configuration
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 2 --sleep-interval 1"

# ... rest of script same as V100 version
```

### Option 3: Direct Command Line
```bash
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 2
```

## Expected Behavior After Fix

Once you use the correct GPU type, you should see:
```
[INFO] Detected GPU: NVIDIA A100-PCIE-40GB
[INFO] GPU Type: A100 (GA100)  # ✓ Matches
[INFO] Memory frequency: 1215MHz  # ✓ A100 specs
[INFO] Default core frequency: 1410MHz  # ✓ A100 specs
```

## Quick Test

To test the fix without running a full job:
```bash
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1 --help
```

This should show you the configuration without errors.
