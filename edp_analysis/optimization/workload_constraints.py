#!/usr/bin/env python3
"""
Workload-Specific Constraints for Performance-Constrained Optimization

This module defines performance constraints and optimization parameters
for different AI workloads based on their real-world usage patterns.

Author: Mert Side
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Workload-specific performance and energy constraints
WORKLOAD_CONSTRAINTS = {
    "LLAMA": {
        "max_penalty": 0.05,           # 5% max slowdown (latency-sensitive)
        "min_frequency_ratio": 0.85,   # Don't go below 85% of max frequency
        "optimization_metric": "response_time",
        "priority": "performance",      # Performance takes priority over energy
        "description": "Large Language Model - Interactive chatbot/completion",
        "use_case": "Real-time text generation, user interactions",
        "sensitivity": {
            "latency": "high",          # Users notice delays immediately
            "throughput": "medium",     # Important for concurrent users
            "energy": "low"             # Energy secondary to user experience
        }
    },
    
    "VIT": {
        "max_penalty": 0.20,           # 20% max slowdown (throughput-oriented)
        "min_frequency_ratio": 0.70,   # Can go lower for batch processing
        "optimization_metric": "throughput_per_watt",
        "priority": "balanced",         # Balance throughput and energy
        "description": "Vision Transformer - Image classification/analysis",
        "use_case": "Batch image processing, computer vision pipelines",
        "sensitivity": {
            "latency": "medium",        # Some delay acceptable in batch mode
            "throughput": "high",       # Processing many images efficiently
            "energy": "high"            # Energy costs matter for large batches
        }
    },
    
    "STABLEDIFFUSION": {
        "max_penalty": 0.10,           # 10% max slowdown (interactive)
        "min_frequency_ratio": 0.80,   # Keep reasonable performance
        "optimization_metric": "user_experience",
        "priority": "balanced",         # Balance quality and energy
        "description": "Stable Diffusion - AI image generation",
        "use_case": "Interactive image creation, creative workflows",
        "sensitivity": {
            "latency": "medium",        # Users wait but expect reasonable time
            "throughput": "medium",     # Multiple generations per session
            "energy": "medium"          # Energy matters for sustained use
        }
    },
    
    "WHISPER": {
        "max_penalty": 0.15,           # 15% max slowdown (batch processing)
        "min_frequency_ratio": 0.75,   # Can tolerate lower frequencies
        "optimization_metric": "energy_per_task",
        "priority": "energy",           # Energy efficiency prioritized
        "description": "Whisper ASR - Audio transcription",
        "use_case": "Batch audio processing, transcription services",
        "sensitivity": {
            "latency": "low",           # Often used in batch/offline mode
            "throughput": "high",       # Process many audio files
            "energy": "high"            # Energy costs significant for batch jobs
        }
    },
    
    "LSTM": {
        "max_penalty": 0.15,           # 15% max slowdown
        "min_frequency_ratio": 0.75,   # Reasonable frequency range
        "optimization_metric": "inference_efficiency",
        "priority": "balanced",         # Application-dependent
        "description": "LSTM Neural Network - Sequence modeling",
        "use_case": "Time series analysis, sequence prediction",
        "sensitivity": {
            "latency": "medium",        # Depends on application
            "throughput": "medium",     # Varies by use case
            "energy": "medium"          # Standard optimization approach
        }
    }
}

# GPU-specific frequency information
GPU_SPECIFICATIONS = {
    "V100": {
        "max_frequency": 1380,         # MHz
        "practical_min": 1050,         # 76% of max - practical lower bound
        "boost_frequency": 1380,       # Peak performance frequency
        "base_frequency": 1200,        # Standard operating frequency
        "energy_efficient": 1050,      # Good energy-performance balance
        "tdp": 250                     # Watts
    },
    
    "A100": {
        "max_frequency": 1410,         # MHz
        "practical_min": 1100,         # 78% of max - practical lower bound
        "boost_frequency": 1410,       # Peak performance frequency
        "base_frequency": 1250,        # Standard operating frequency
        "energy_efficient": 1100,      # Good energy-performance balance
        "tdp": 400                     # Watts
    },
    
    "H100": {
        "max_frequency": 1830,         # MHz
        "practical_min": 1400,         # 76% of max - practical lower bound
        "boost_frequency": 1830,       # Peak performance frequency
        "base_frequency": 1600,        # Standard operating frequency
        "energy_efficient": 1400,      # Good energy-performance balance
        "tdp": 700                     # Watts
    }
}

# Performance penalty categories for quick classification
PENALTY_CATEGORIES = {
    "acceptable": 0.10,        # ≤ 10% - generally acceptable
    "noticeable": 0.20,        # ≤ 20% - noticeable but tolerable
    "significant": 0.30,       # ≤ 30% - significant impact
    "unacceptable": 0.50       # > 50% - unacceptable for production
}

def get_workload_constraints(app_name: str) -> Dict[str, Any]:
    """Get constraints for a specific workload."""
    if app_name.upper() not in WORKLOAD_CONSTRAINTS:
        logger.warning(f"Unknown application '{app_name}'. Using default constraints.")
        return get_default_constraints()
    
    return WORKLOAD_CONSTRAINTS[app_name.upper()].copy()

def get_gpu_specifications(gpu_type: str) -> Dict[str, Any]:
    """Get specifications for a specific GPU."""
    if gpu_type.upper() not in GPU_SPECIFICATIONS:
        logger.warning(f"Unknown GPU '{gpu_type}'. Using V100 specifications.")
        return GPU_SPECIFICATIONS["V100"].copy()
    
    return GPU_SPECIFICATIONS[gpu_type.upper()].copy()

def get_default_constraints() -> Dict[str, Any]:
    """Get default constraints for unknown workloads."""
    return {
        "max_penalty": 0.15,           # 15% default
        "min_frequency_ratio": 0.75,   # 75% of max frequency
        "optimization_metric": "balanced",
        "priority": "balanced",
        "description": "Unknown workload - conservative constraints",
        "use_case": "General purpose inference",
        "sensitivity": {
            "latency": "medium",
            "throughput": "medium",
            "energy": "medium"
        }
    }

def get_practical_frequency_range(gpu_type: str, app_name: str) -> tuple:
    """Get practical frequency range for a GPU+application combination."""
    gpu_spec = get_gpu_specifications(gpu_type)
    constraints = get_workload_constraints(app_name)
    
    max_freq = gpu_spec["max_frequency"]
    min_freq = int(max_freq * constraints["min_frequency_ratio"])
    
    # Ensure minimum is not below practical bounds
    practical_min = gpu_spec["practical_min"]
    min_freq = max(min_freq, practical_min)
    
    return min_freq, max_freq

def classify_performance_penalty(penalty: float) -> str:
    """Classify performance penalty into categories."""
    if penalty <= PENALTY_CATEGORIES["acceptable"]:
        return "acceptable"
    elif penalty <= PENALTY_CATEGORIES["noticeable"]:
        return "noticeable"
    elif penalty <= PENALTY_CATEGORIES["significant"]:
        return "significant"
    else:
        return "unacceptable"

def validate_constraints(app_name: str, gpu_type: str) -> bool:
    """Validate that constraints are reasonable for the given GPU+application."""
    try:
        constraints = get_workload_constraints(app_name)
        gpu_spec = get_gpu_specifications(gpu_type)
        
        # Check if minimum frequency is achievable
        min_freq, max_freq = get_practical_frequency_range(gpu_type, app_name)
        
        if min_freq >= max_freq:
            logger.error(f"Invalid frequency range for {gpu_type}+{app_name}: {min_freq}-{max_freq}")
            return False
        
        # Check if penalty is reasonable
        if constraints["max_penalty"] > PENALTY_CATEGORIES["significant"]:
            logger.warning(f"High performance penalty allowed for {app_name}: {constraints['max_penalty']*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating constraints: {e}")
        return False

def print_constraint_summary(app_name: str = None, gpu_type: str = None):
    """Print a summary of constraints for debugging."""
    if app_name:
        constraints = get_workload_constraints(app_name)
        print(f"\n📋 Constraints for {app_name.upper()}:")
        print(f"  Max Performance Penalty: {constraints['max_penalty']*100:.1f}%")
        print(f"  Min Frequency Ratio: {constraints['min_frequency_ratio']*100:.1f}%")
        print(f"  Optimization Metric: {constraints['optimization_metric']}")
        print(f"  Priority: {constraints['priority']}")
        print(f"  Description: {constraints['description']}")
        
        if gpu_type:
            min_freq, max_freq = get_practical_frequency_range(gpu_type, app_name)
            print(f"  Frequency Range ({gpu_type}): {min_freq}-{max_freq} MHz")
    
    if gpu_type:
        gpu_spec = get_gpu_specifications(gpu_type)
        print(f"\n🔧 Specifications for {gpu_type.upper()}:")
        print(f"  Max Frequency: {gpu_spec['max_frequency']} MHz")
        print(f"  Practical Min: {gpu_spec['practical_min']} MHz")
        print(f"  Energy Efficient: {gpu_spec['energy_efficient']} MHz")
        print(f"  TDP: {gpu_spec['tdp']} W")

if __name__ == "__main__":
    # Demo the constraints system
    print("🚀 Workload Constraints Demo")
    print("="*50)
    
    for app in ["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER", "LSTM"]:
        print_constraint_summary(app, "V100")
        print()
