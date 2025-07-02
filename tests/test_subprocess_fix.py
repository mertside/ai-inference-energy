#!/usr/bin/env python3
"""
Quick test to verify Python 3.6 subprocess compatibility fix.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_subprocess_compatibility():
    """Test the fixed subprocess functionality."""
    print(f"Testing subprocess compatibility on Python {sys.version}")
    
    try:
        from utils import run_command
        print("✓ Successfully imported run_command from utils")
        
        # Test simple command with output capture
        result = run_command(['echo', 'Hello World'], capture_output=True)
        print(f"✓ Command executed successfully: {result.stdout.strip()}")
        
        # Test command without output capture
        result = run_command(['echo', 'No capture'], capture_output=False)
        print("✓ Command executed without capture")
        
        print("✓ All subprocess tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Subprocess test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_subprocess_compatibility()
    sys.exit(0 if success else 1)
