# Migration Guide: Legacy to Refactored Framework

This guide helps you migrate from the original monolithic scripts to the new modular framework.

## Overview of Changes

### Architecture Transformation
```
OLD (Legacy):                    NEW (Refactored):
├── launch.sh (1200+ lines)     ├── launch_v2.sh (200 lines)
├── profile.py                  ├── lib/
├── control.sh                  │   ├── common.sh
├── submit_job_*.sh             │   ├── gpu_config.sh
└── ...                         │   ├── profiling.sh
                                 │   └── args_parser.sh
                                 ├── config/
                                 │   └── defaults.sh
                                 └── legacy/ (preserved)
```

## Immediate Benefits

✅ **Preserved Functionality**: All existing features work exactly the same  
✅ **Enhanced Reliability**: Better error handling and validation  
✅ **Improved Usability**: Comprehensive help and debug capabilities  
✅ **Future-Proof**: Modular design enables easy extensions  

## Step-by-Step Migration

### Phase 1: Test New Framework (Safe)
```bash
# Test with existing workflow
./launch_v2.sh --help

# Run a quick baseline test
./launch_v2.sh --profiling-mode baseline --num-runs 1 --debug

# Compare with legacy
./launch.sh # Your existing command
./launch_v2.sh # Equivalent new command
```

### Phase 2: Update Job Scripts (Optional)
The new framework can be integrated into existing SLURM job scripts:

```bash
# In submit_job_h100.sh (example update)
# OLD:
# ./launch.sh

# NEW:
./launch_v2.sh --gpu-type H100 --profiling-mode dvfs
```

### Phase 3: Leverage New Features
```bash
# Use enhanced debugging
./launch_v2.sh --debug

# Use smart GPU detection  
./launch_v2.sh # Automatically detects GPU type

# Use comprehensive help
./launch_v2.sh --help
```

## Command Equivalence Guide

| Legacy Command | New Framework Equivalent |
|----------------|--------------------------|
| `./launch.sh` | `./launch_v2.sh` |
| Edit script for GPU type | `--gpu-type A100/V100/H100` |
| Edit script for tool | `--profiling-tool dcgmi/nvidia-smi` |
| Edit script for runs | `--num-runs N` |
| Edit script for app | `--app-executable path --app-params "args"` |

## Configuration Migration

### Old Method (Edit Script)
```bash
# Edit launch.sh
DEFAULT_GPU_TYPE="V100"
DEFAULT_NUM_RUNS=5
```

### New Method (Command Line)
```bash
./launch_v2.sh --gpu-type V100 --num-runs 5
```

### New Method (Config File)
```bash
# Create config/user_config.sh
DEFAULT_GPU_TYPE="V100"
DEFAULT_NUM_RUNS=5
```

## Feature Mapping

### Error Handling
| Legacy | New Framework |
|--------|---------------|
| Basic error messages | Detailed error descriptions with solutions |
| Manual debugging | `--debug` flag with comprehensive logging |
| Hard to trace issues | Modular libraries for isolated testing |

### User Experience
| Legacy | New Framework |
|--------|---------------|
| Edit script to configure | Command-line arguments |
| Read code for options | `--help` with examples |
| Manual validation | Automatic input validation |
| No progress indication | Progress bars and status updates |

### Maintenance
| Legacy | New Framework |
|--------|---------------|
| Monolithic 1200+ lines | Modular libraries (~200 lines each) |
| Mixed concerns | Separated functionality |
| Hard to test | Individual library testing |
| Copy-paste changes | Reusable functions |

## Compatibility Matrix

| Feature | Legacy | v2.0 | Notes |
|---------|--------|------|-------|
| GPU Types | A100, V100, H100 | ✅ Same | Auto-detection added |
| Profiling Tools | dcgmi, nvidia-smi | ✅ Same | Better fallback logic |
| DVFS Mode | ✅ | ✅ Same | Enhanced progress tracking |
| Baseline Mode | ✅ | ✅ Same | Improved validation |
| Output Format | CSV + logs | ✅ Same | Added summary generation |
| SLURM Integration | ✅ | ✅ Same | Compatible job scripts |

## Risk Assessment

### Low Risk ✅
- **Testing**: Both frameworks can run side-by-side
- **Rollback**: Legacy scripts preserved and functional
- **Compatibility**: Same output format and file structure
- **Integration**: Existing job submission scripts unchanged

### Zero Risk 🛡️
- **Data**: No changes to existing results or data
- **Dependencies**: Same external tool requirements
- **Environment**: No changes to conda environments or modules

## Troubleshooting Migration

### Issue: "Library not found"
```bash
# Check library directory exists
ls -la lib/

# If missing, ensure all library files are created
# Re-run the refactoring process
```

### Issue: "Command not working"
```bash
# Check permissions
chmod +x launch_v2.sh

# Test individual libraries
source lib/common.sh && log_info "Test successful"
```

### Issue: "Different results"
```bash
# Enable debug mode for detailed comparison
./launch_v2.sh --debug

# Check configuration differences
./launch_v2.sh --version
```

## Validation Checklist

Before full migration, verify:

- [ ] `./launch_v2.sh --help` shows comprehensive help
- [ ] `./launch_v2.sh --version` shows version information  
- [ ] `./launch_v2.sh --debug` enables debug logging
- [ ] Invalid arguments show helpful error messages
- [ ] GPU auto-detection works (if nvidia-smi available)
- [ ] Profiling tool fallback works correctly
- [ ] All original functionality accessible via new interface

## Gradual Migration Strategy

### Week 1: Testing Phase
- Test new framework with baseline experiments
- Compare outputs with legacy framework
- Familiarize team with new command-line interface

### Week 2: Parallel Operation  
- Use new framework for new experiments
- Keep legacy framework for critical runs
- Update documentation and training materials

### Week 3: Feature Adoption
- Leverage new debugging capabilities
- Use enhanced error handling for troubleshooting
- Start using advanced command-line options

### Week 4: Full Migration
- Update job submission scripts to use new framework
- Archive legacy scripts (keep for reference)
- Train users on new capabilities

## Rollback Plan

If issues arise:

1. **Immediate**: Use legacy scripts (unchanged and functional)
2. **Investigation**: Enable debug mode (`--debug`) for diagnostics  
3. **Isolation**: Test individual libraries to identify issues
4. **Resolution**: Fix specific modules without affecting others
5. **Validation**: Re-test with known working configurations

## Support Resources

### Self-Help
- Comprehensive `--help` system
- Debug output with `--debug` flag
- Modular testing of individual libraries

### Documentation
- `README_v2.md` - Complete new framework documentation
- Library documentation in each `.sh` file
- Inline help and examples

### Escalation Path
1. Check debug output for specific error messages
2. Test with legacy framework to isolate framework vs environment issues
3. Review library-specific documentation
4. Test individual library components

The refactored framework is designed for zero-risk migration with significant long-term benefits for maintainability and usability.
