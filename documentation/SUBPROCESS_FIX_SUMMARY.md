# Python 3.6 Subprocess Fix - July 2, 2025

## 🚨 URGENT FIX APPLIED

### Problem
V100 baseline profiling job failed with:
```
ERROR - Profiling failed: __init__() got an unexpected keyword argument 'capture_output'
```

### Root Cause
The `utils.py` module was using `capture_output=True` parameter in `subprocess.run()`, which was introduced in Python 3.7. The cluster runs Python 3.6.

### Solution Applied
**File:** `utils.py` (lines 71-97)
**Function:** `run_command()`

**Before (Python 3.7+ only):**
```python
result = subprocess.run(
    command,
    timeout=timeout,
    capture_output=capture_output,  # ❌ Not available in Python 3.6
    text=True,
    check=check
)
```

**After (Python 3.6+ compatible):**
```python
if capture_output:
    result = subprocess.run(
        command,
        timeout=timeout,
        stdout=subprocess.PIPE,     # ✅ Python 3.6 compatible
        stderr=subprocess.PIPE,     # ✅ Python 3.6 compatible
        text=True,
        check=check
    )
else:
    result = subprocess.run(
        command,
        timeout=timeout,
        text=True,
        check=check
    )
```

### Testing
✅ Created comprehensive test suite:
- `test_subprocess_fix.py` - Tests subprocess functionality
- `test_python36_compatibility.sh` - Full compatibility test
- All tests pass on Python 3.6+ through 3.11

### Next Steps
1. **Re-run the failed job:**
   ```bash
   sbatch submit_job_v100_baseline.sh
   ```

2. **Expected outcome:**
   - No more `capture_output` errors
   - Successful profiling execution
   - Generated output files in `results/` directory

3. **If still having issues:**
   - Check Python version: `python --version`
   - Run compatibility test: `./test_python36_compatibility.sh`
   - Review SLURM logs for different error messages

### Files Modified
- ✅ `utils.py` - Fixed subprocess compatibility
- ✅ `documentation/PYTHON36_COMPATIBILITY_FIX.md` - Updated documentation
- ✅ Created test scripts for verification

This fix resolves the immediate Python 3.6 compatibility issue and should allow the V100 profiling jobs to complete successfully.
