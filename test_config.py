#!/usr/bin/env python3
"""
Quick test script to check if config.py loads correctly.
"""

try:
    from config import profiling_config, gpu_config, model_config, system_config
    print("✓ Config module loaded successfully")
    print(f"  - DCGMI fields count: {len(profiling_config.DCGMI_FIELDS)}")
    print(f"  - A100 frequencies count: {len(gpu_config.A100_CORE_FREQUENCIES)}")
    print(f"  - Default interval: {profiling_config.DEFAULT_INTERVAL_MS}ms")
    print(f"  - LLaMA model: {model_config.LLAMA_MODEL_NAME}")
except Exception as e:
    print(f"✗ Config module failed to load: {e}")
    import traceback
    traceback.print_exc()
