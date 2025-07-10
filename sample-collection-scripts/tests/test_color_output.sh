#!/bin/bash
#
# Simple Color Output Test Script
#

echo "=== Testing Color Output Functionality ==="

# Test 1: Help with colors enabled
echo "Test 1: Help with colors enabled (normal terminal)"
./launch_v2.sh --help | head -5
echo ""

# Test 2: Help with colors disabled via NO_COLOR
echo "Test 2: Help with colors disabled via NO_COLOR"
NO_COLOR=1 ./launch_v2.sh --help | head -5
echo ""

# Test 3: Help with colors disabled via DISABLE_COLORS
echo "Test 3: Help with colors disabled via DISABLE_COLORS"
DISABLE_COLORS=1 ./launch_v2.sh --help | head -5
echo ""

# Test 4: Help with colors disabled via TERM=dumb
echo "Test 4: Help with colors disabled via TERM=dumb"
TERM=dumb ./launch_v2.sh --help | head -5
echo ""

echo "=== Color Output Test Complete ==="
echo ""
echo "All tests passed if no raw escape codes (like \\033[) were visible in the output above."
