# Python 3.6 `text` Parameter Fix - July 2, 2025 (12:17 PM)

## 🚨 SECOND URGENT FIX APPLIED

### Problem
After fixing the `capture_output` issue, the V100 baseline profiling job failed again with:
```
ERROR - Profiling failed: __init__() got an unexpected keyword argument 'text'
```

### Root Cause
The `text=True` parameter in `subprocess.run()` was also introduced in Python 3.7. The cluster runs Python 3.6, which uses `universal_newlines=True` instead.

### Solution Applied
**File:** `utils.py` (lines 95-116)
**Function:** `run_command()`

**Before (Python 3.7+ only):**
```python
result = subprocess.run(
    command,
    timeout=timeout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,                  # ❌ Not available in Python 3.6
    check=check
)
```

**After (Python 3.6+ compatible):**
```python
result = subprocess.run(
    command,
    timeout=timeout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,    # ✅ Python 3.6 compatible
    check=check
)
```

### Python Version Compatibility

| Parameter | Python 3.6 | Python 3.7+ | Status |
|-----------|-------------|--------------|--------|
| `capture_output=True` | ❌ Not available | ✅ Available | ✅ Fixed (use stdout/stderr) |
| `text=True` | ❌ Not available | ✅ Available | ✅ Fixed (use universal_newlines) |
| `universal_newlines=True` | ✅ Available | ✅ Available | ✅ Compatible |

### Testing
✅ Local test passes:
```bash
cd tests && python test_subprocess_fix.py
✓ All subprocess tests passed!
```

### Next Steps
1. **Re-run the failed job:**
   ```bash
   sbatch submit_job_v100_baseline.sh
   ```

2. **Expected outcome:**
   - No more `text` parameter errors
   - No more `capture_output` parameter errors
   - Successful profiling execution
   - Generated output files in `results/` directory

3. **If still having issues:**
   - Check for other Python 3.7+ features being used
   - Review the specific error message
   - Run compatibility test: `./tests/test_python36_compatibility.sh`

### Files Modified
- ✅ `utils.py` - Fixed both `capture_output` and `text` parameter compatibility
- ✅ `documentation/PYTHON36_COMPATIBILITY_FIX.md` - Updated with both fixes

### Impact
This completes the Python 3.6 compatibility fixes for subprocess calls. The framework should now work correctly on any Python 3.6+ environment without subprocess-related errors.

## Summary of All Python 3.6 Fixes

1. **First Issue:** `capture_output` parameter → Fixed with `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
2. **Second Issue:** `text` parameter → Fixed with `universal_newlines=True`
3. **Result:** Full Python 3.6+ compatibility for subprocess operations

The V100 profiling jobs should now complete successfully without Python compatibility errors.
