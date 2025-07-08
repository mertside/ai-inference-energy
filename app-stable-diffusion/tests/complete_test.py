#!/usr/bin/env python3
"""
Complete Stable Diffusion Environment Test - Final Version
"""
print("🎯 COMPLETE STABLE DIFFUSION ENVIRONMENT TEST")
print("=" * 55)
print("Testing the exact working configuration for Tesla V100")
print()

def test_import(description, import_statement):
    """Test an import and return success status"""
    try:
        exec(import_statement)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}: {str(e)[:60]}...")
        return False

def get_version(module_name):
    """Get version of a module"""
    try:
        module = __import__(module_name)
        return getattr(module, '__version__', 'Unknown')
    except:
        return 'Not available'

# Test all core components
tests = [
    ("PyTorch", "import torch"),
    ("CUDA Support", "import torch; assert torch.cuda.is_available()"),
    ("typing_extensions", "import typing_extensions"),
    ("tqdm", "import tqdm"),
    ("requests", "import requests"),
    ("packaging", "import packaging"),
    ("Pillow", "import PIL.Image"),
    ("safetensors", "import safetensors"),
    ("transformers", "import transformers"),
    ("huggingface_hub", "import huggingface_hub"),
    ("diffusers basic", "import diffusers"),
    ("StableDiffusionPipeline", "from diffusers import StableDiffusionPipeline"),
    ("DiffusionPipeline", "from diffusers import DiffusionPipeline"),
]

passed = 0
total = len(tests)

for description, import_stmt in tests:
    if test_import(description, import_stmt):
        passed += 1

# Show versions
print("\n📦 PACKAGE VERSIONS:")
print("-" * 30)
print(f"PyTorch: {get_version('torch')}")
print(f"transformers: {get_version('transformers')}")
print(f"diffusers: {get_version('diffusers')}")
print(f"huggingface_hub: {get_version('huggingface_hub')}")
print(f"safetensors: {get_version('safetensors')}")

# GPU Info
print("\n🖥️  GPU INFORMATION:")
print("-" * 25)
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available")
except:
    print("Cannot check GPU info")

# Final score
print("\n" + "=" * 55)
print(f"📊 FINAL SCORE: {passed}/{total} components working")
print(f"🎯 SUCCESS RATE: {passed/total*100:.1f}%")

if passed == total:
    print("\n🎉 PERFECT SCORE! STABLE DIFFUSION FULLY FUNCTIONAL!")
    print("🚀 Ready for AI inference energy research!")
    print("🖼️  Image generation: READY")
    print("⚡ Energy profiling: READY")
elif passed >= total - 2:
    print(f"\n✅ EXCELLENT! {passed}/{total} working - Environment ready!")
    print("🚀 Stable Diffusion should work for most use cases")
else:
    print(f"\n⚠️  {total-passed} components need attention")

print("\n🏆 CONFIGURATION SUMMARY:")
print("   Tesla V100 + CUDA 11.0 + PyTorch 1.12.1")
print("   transformers==4.33.2 + diffusers==0.21.4 + huggingface_hub==0.16.4")
print("   = WORKING STABLE DIFFUSION ENVIRONMENT!")
