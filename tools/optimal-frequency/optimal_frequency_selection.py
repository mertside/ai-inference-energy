#!/usr/bin/env python3
"""
Real-Time Optimal Frequency Selection for AI Inference.

This script implements real-time optimal frequency selection for AI inference
workloads using trained models and your proven single-run prediction approach.
It integrates with your existing framework to provide production-ready optimal
frequency selection with performance constraints.

Features:
- Single-run prediction (run at max frequency once)
- Performance-constrained optimization (≤5% degradation)
- Cross-architecture support (V100, A100, H100)
- Integration with existing launch_v2.sh framework
- EDP/ED2P optimization methods
- Real-time frequency switching

Requirements:
- Trained power and performance models
- GPU DVFS control capabilities
- Python 3.8+ with your existing dependencies

Author: Mert Side
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from config import profiling_config
    from utils import setup_logging, run_command
except ImportError:
    # Fallback configuration
    class ProfilingConfig:
        DEFAULT_INTERVAL_MS = 50
        TEMP_OUTPUT_FILE = "optimal_freq_profile.csv"
    
    profiling_config = ProfilingConfig()
    
    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)
    
    def run_command(cmd, *args, **kwargs):
        return subprocess.run(cmd, *args, **kwargs)


class OptimalFrequencyController:
    """
    Real-time optimal frequency controller for AI inference workloads.
    
    Implements your proven methodology for production deployment:
    1. Run workload once at maximum frequency
    2. Extract features during execution
    3. Predict optimal frequency using trained models
    4. Apply performance constraints
    5. Switch to optimal frequency for subsequent runs
    """

    def __init__(self, models_dir: str, gpu_type: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the optimal frequency controller.
        
        Args:
            models_dir: Directory containing trained models
            gpu_type: GPU architecture (v100, a100, h100)
            logger: Optional logger instance
        """
        self.models_dir = Path(models_dir)
        self.gpu_type = gpu_type.lower()
        self.logger = logger or setup_logging()
        
        # Load trained models
        self.power_model = None
        self.performance_model = None
        self.load_models()
        
        # GPU configuration
        self.gpu_configs = {
            'v100': {
                'max_frequency': 1530,
                'available_frequencies': [405, 540, 675, 810, 945, 1080, 1215, 1350, 1485, 1530],
                'memory_frequency': 877  # Fixed for V100
            },
            'a100': {
                'max_frequency': 1410, 
                'available_frequencies': [210, 510, 810, 1110, 1260, 1410],
                'memory_frequency': 1215  # Fixed for A100
            },
            'h100': {
                'max_frequency': 1980,
                'available_frequencies': [210, 510, 810, 1110, 1410, 1710, 1980],
                'memory_frequency': 2619  # Fixed for H100
            }
        }
        
        if self.gpu_type not in self.gpu_configs:
            raise ValueError(f"Unsupported GPU type: {self.gpu_type}")

    def load_models(self) -> None:
        """Load trained power and performance models."""
        power_model_path = self.models_dir / 'ai_power_model.pkl'
        performance_model_path = self.models_dir / 'ai_performance_model.pkl'
        
        if not power_model_path.exists():
            raise FileNotFoundError(f"Power model not found: {power_model_path}")
        if not performance_model_path.exists():
            raise FileNotFoundError(f"Performance model not found: {performance_model_path}")
        
        # Load models (assuming they follow the structure from power_modeling.py)
        with open(power_model_path, 'rb') as f:
            power_model_data = pickle.load(f)
            self.power_model = power_model_data['model']
            self.power_feature_names = power_model_data['feature_names']
        
        with open(performance_model_path, 'rb') as f:
            performance_model_data = pickle.load(f)
            self.performance_model = performance_model_data['model']
            self.performance_feature_names = performance_model_data['feature_names']
        
        self.logger.info(f"Loaded models for {self.gpu_type.upper()} optimal frequency selection")

    def set_gpu_frequency(self, frequency: int) -> bool:
        """
        Set GPU frequency using nvidia-smi.
        
        Args:
            frequency: Target frequency in MHz
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set application clock (persistent across reboots)
            cmd = [
                "nvidia-smi", "-pm", "ENABLED",  # Enable persistence mode
                "-ac", f"{self.gpu_configs[self.gpu_type]['memory_frequency']},{frequency}"
            ]
            
            result = run_command(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Set GPU frequency to {frequency} MHz")
                return True
            else:
                self.logger.error(f"Failed to set GPU frequency: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting GPU frequency: {e}")
            return False

    def reset_gpu_frequency(self) -> bool:
        """Reset GPU to default frequency settings."""
        try:
            # Reset to auto boost
            cmd = ["nvidia-smi", "-rac"]
            result = run_command(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Reset GPU to default frequency settings")
                return True
            else:
                self.logger.error(f"Failed to reset GPU frequency: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error resetting GPU frequency: {e}")
            return False

    def profile_at_max_frequency(self, app_command: str, workload: str) -> Tuple[Dict[str, float], float]:
        """
        Profile application at maximum frequency to extract features.
        
        Your proven approach: run once at max frequency to get all needed features.
        
        Args:
            app_command: Application command to profile
            workload: Workload name for feature extraction
            
        Returns:
            Tuple of (extracted_features, execution_time)
        """
        max_freq = self.gpu_configs[self.gpu_type]['max_frequency']
        
        # Set to maximum frequency
        if not self.set_gpu_frequency(max_freq):
            raise RuntimeError("Failed to set maximum frequency for profiling")
        
        # Wait for frequency to stabilize
        time.sleep(2)
        
        # Profile using your existing infrastructure
        profile_output = profiling_config.TEMP_OUTPUT_FILE
        
        try:
            # Use your existing profiling script
            profile_cmd = [
                sys.executable,
                str(Path(__file__).parent / "profile.py"),
                "-o", profile_output,
                "-i", str(profiling_config.DEFAULT_INTERVAL_MS),
                *app_command.split()
            ]
            
            start_time = time.time()
            result = run_command(profile_cmd, capture_output=True, text=True)
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                raise RuntimeError(f"Profiling failed: {result.stderr}")
            
            # Extract features from profiling data
            features = self.extract_features_from_profile(profile_output, workload)
            
            return features, execution_time
            
        finally:
            # Clean up profile file
            if os.path.exists(profile_output):
                os.remove(profile_output)

    def extract_features_from_profile(self, profile_file: str, workload: str) -> Dict[str, float]:
        """
        Extract AI inference features from profiling data.
        
        Args:
            profile_file: Path to profiling CSV file
            workload: Workload name for workload-specific features
            
        Returns:
            Dictionary of extracted features
        """
        try:
            import pandas as pd
            
            # Load profiling data (assumes DCGMI format from your framework)
            df = pd.read_csv(profile_file, header=None)
            
            # Map columns based on your DCGMI field configuration
            # (This should match the field mapping from aggregate_results.py)
            dcgmi_mapping = {
                2: 'power_usage',     # Field 155 
                6: 'gpu_util',        # Field 203
                7: 'mem_copy_util',   # Field 204
                11: 'sm_clock',       # Field 100
                19: 'dram_active',    # Field 1005 - Key feature
                21: 'sm_active',      # Field 1002 - Maps to fp_active
                22: 'sm_occupancy',   # Field 1003
                23: 'tensor_active',  # Field 1004 - AI-specific
                26: 'fp32_active',    # Field 1007
                27: 'fp16_active'     # Field 1008 - AI-specific
            }
            
            # Extract core features (your proven approach)
            features = {}
            
            for col_idx, field_name in dcgmi_mapping.items():
                if col_idx < len(df.columns):
                    values = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna()
                    if not values.empty:
                        features[field_name] = values.mean()
            
            # Map to your proven feature names
            if 'sm_active' in features:
                features['fp_active'] = features['sm_active']
            if 'sm_clock' in features:
                features['sm_app_clock'] = features['sm_clock']
            
            # Add architectural factor
            arch_factors = {'v100': 80/56, 'a100': 108/56, 'h100': 132/56}
            features['architecture_factor'] = arch_factors[self.gpu_type]
            
            # Add AI workload-specific features
            self.add_workload_specific_features(features, workload)
            
            self.logger.debug(f"Extracted features: {list(features.keys())}")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return {}

    def add_workload_specific_features(self, features: Dict[str, float], workload: str) -> None:
        """Add workload-specific features based on workload type."""
        
        if workload == 'llama':
            # LLaMA-specific features
            if 'tensor_active' in features:
                features['tensor_core_util'] = features['tensor_active']
            if 'fp16_active' in features:
                features['mixed_precision_ratio'] = features['fp16_active']
            if 'sm_occupancy' in features:
                features['sequence_processing_ratio'] = features['sm_occupancy']
                
        elif workload == 'stablediffusion':
            # Stable Diffusion-specific features
            if 'tensor_active' in features:
                features['attention_compute_ratio'] = features['tensor_active']
            if 'gpu_util' in features:
                features['denoising_compute_ratio'] = features['gpu_util']
                
        elif workload == 'vit':
            # Vision Transformer-specific features
            if 'tensor_active' in features:
                features['attention_compute_ratio'] = features['tensor_active']
            if 'sm_occupancy' in features:
                features['patch_processing_efficiency'] = features['sm_occupancy']
                
        elif workload == 'whisper':
            # Whisper-specific features
            if 'tensor_active' in features:
                features['encoder_decoder_ratio'] = features['tensor_active']
            if 'dram_active' in features:
                features['beam_search_memory'] = features['dram_active']
        
        # Add workload indicators (one-hot encoding)
        for wl in ['llama', 'stablediffusion', 'vit', 'whisper']:
            features[f'workload_{wl}'] = 1.0 if workload == wl else 0.0

    def predict_power(self, features: Dict[str, float]) -> float:
        """Predict power consumption using trained model."""
        try:
            import numpy as np
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.power_feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                elif 'interaction' in feature_name:
                    # Calculate interaction terms
                    if feature_name == 'freq_fp_interaction':
                        val = features.get('sm_app_clock', 0) * features.get('fp_active', 0)
                    elif feature_name == 'freq_dram_interaction':
                        val = features.get('sm_app_clock', 0) * features.get('dram_active', 0)
                    else:
                        val = 0.0
                    feature_vector.append(val)
                else:
                    feature_vector.append(0.0)
            
            X = np.array(feature_vector).reshape(1, -1)
            return self.power_model.predict(X)[0]
            
        except Exception as e:
            self.logger.error(f"Power prediction failed: {e}")
            return 0.0

    def predict_performance(self, features: Dict[str, float]) -> float:
        """Predict execution time using trained model."""
        try:
            import numpy as np
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.performance_feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)
            
            X = np.array(feature_vector).reshape(1, -1)
            return self.performance_model.predict(X)[0]
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return 0.0

    def find_optimal_frequency(self, features: Dict[str, float], 
                              baseline_time: float,
                              performance_constraint_pct: float = 5.0) -> Tuple[int, Dict[str, Any]]:
        """
        Find optimal frequency using your proven EDP approach with constraints.
        
        Args:
            features: Features extracted from max frequency run
            baseline_time: Baseline execution time (max frequency)
            performance_constraint_pct: Performance degradation constraint
            
        Returns:
            Tuple of (optimal_frequency, optimization_details)
        """
        available_frequencies = self.gpu_configs[self.gpu_type]['available_frequencies']
        max_acceptable_time = baseline_time * (1 + performance_constraint_pct / 100)
        
        predictions = []
        
        for freq in available_frequencies:
            # Update features with current frequency
            freq_features = features.copy()
            freq_features['sm_app_clock'] = freq
            
            # Update interaction terms
            if 'fp_active' in features:
                freq_features['freq_fp_interaction'] = freq * features['fp_active']
            if 'dram_active' in features:
                freq_features['freq_dram_interaction'] = freq * features['dram_active']
            
            # Predict power and performance
            predicted_power = self.predict_power(freq_features)
            predicted_time = self.predict_performance(freq_features)
            
            # Calculate energy and EDP
            predicted_energy = predicted_power * predicted_time
            edp = predicted_energy * predicted_time
            
            predictions.append({
                'frequency': freq,
                'power': predicted_power,
                'execution_time': predicted_time,
                'energy': predicted_energy,
                'edp': edp,
                'meets_constraint': predicted_time <= max_acceptable_time
            })
        
        # Filter frequencies meeting performance constraint
        valid_predictions = [p for p in predictions if p['meets_constraint']]
        
        if not valid_predictions:
            # No frequency meets constraint, return maximum frequency
            max_freq = max(available_frequencies)
            self.logger.warning(f"No frequency meets {performance_constraint_pct}% constraint")
            return max_freq, {'error': 'No frequency meets performance constraint'}
        
        # Find optimal frequency (minimum EDP among valid frequencies)
        optimal = min(valid_predictions, key=lambda x: x['edp'])
        
        # Calculate optimization details
        baseline = next(p for p in predictions if p['frequency'] == max(available_frequencies))
        
        details = {
            'optimal_frequency': optimal['frequency'],
            'predicted_power': optimal['power'],
            'predicted_time': optimal['execution_time'],
            'predicted_energy': optimal['energy'],
            'predicted_edp': optimal['edp'],
            'baseline_energy': baseline['energy'],
            'energy_savings_pct': (1 - optimal['energy'] / baseline['energy']) * 100,
            'performance_impact_pct': (optimal['execution_time'] / baseline['execution_time'] - 1) * 100,
            'constraint_satisfied': optimal['meets_constraint'],
            'valid_frequencies': len(valid_predictions),
            'total_frequencies': len(predictions)
        }
        
        return optimal['frequency'], details

    def run_optimal_frequency_selection(self, app_command: str, workload: str,
                                       performance_constraint_pct: float = 5.0,
                                       apply_frequency: bool = True) -> Dict[str, Any]:
        """
        Complete optimal frequency selection workflow.
        
        Your proven approach:
        1. Profile at max frequency
        2. Extract features
        3. Predict optimal frequency
        4. Apply frequency setting
        
        Args:
            app_command: Application command to optimize
            workload: Workload name
            performance_constraint_pct: Performance constraint percentage
            apply_frequency: Whether to actually apply the optimal frequency
            
        Returns:
            Optimization results dictionary
        """
        self.logger.info(f"Starting optimal frequency selection for {workload}")
        
        try:
            # Step 1: Profile at maximum frequency
            self.logger.info("Step 1: Profiling at maximum frequency")
            features, baseline_time = self.profile_at_max_frequency(app_command, workload)
            
            if not features:
                raise RuntimeError("Failed to extract features from profiling")
            
            # Step 2: Predict optimal frequency
            self.logger.info("Step 2: Predicting optimal frequency")
            optimal_freq, details = self.find_optimal_frequency(
                features, baseline_time, performance_constraint_pct
            )
            
            # Step 3: Apply optimal frequency
            if apply_frequency and 'error' not in details:
                self.logger.info(f"Step 3: Applying optimal frequency ({optimal_freq} MHz)")
                if self.set_gpu_frequency(optimal_freq):
                    details['frequency_applied'] = True
                else:
                    details['frequency_applied'] = False
                    self.logger.warning("Failed to apply optimal frequency")
            else:
                details['frequency_applied'] = False
            
            # Compile results
            results = {
                'workload': workload,
                'gpu_type': self.gpu_type,
                'baseline_time': baseline_time,
                'extracted_features': features,
                'optimization_details': details,
                'timestamp': time.time()
            }
            
            # Log summary
            if 'error' not in details:
                self.logger.info(
                    f"Optimal frequency selection completed:"
                    f"\n  Optimal frequency: {optimal_freq} MHz"
                    f"\n  Expected energy savings: {details['energy_savings_pct']:.1f}%"
                    f"\n  Expected performance impact: {details['performance_impact_pct']:.1f}%"
                )
            else:
                self.logger.warning(f"Optimization failed: {details['error']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimal frequency selection failed: {e}")
            return {'error': str(e)}
        
        finally:
            # Always reset frequency on exit (unless explicitly applied)
            if not apply_frequency:
                self.reset_gpu_frequency()


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Real-time optimal frequency selection for AI inference workloads"
    )
    parser.add_argument(
        "-m", "--models",
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "-g", "--gpu",
        choices=['v100', 'a100', 'h100'],
        required=True,
        help="GPU architecture"
    )
    parser.add_argument(
        "-w", "--workload",
        choices=['llama', 'stablediffusion', 'vit', 'whisper'],
        required=True,
        help="AI workload type"
    )
    parser.add_argument(
        "-c", "--command",
        required=True,
        help="Application command to optimize"
    )
    parser.add_argument(
        "--constraint",
        type=float,
        default=5.0,
        help="Performance degradation constraint percentage (default: 5.0)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the optimal frequency setting"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Initialize controller
        controller = OptimalFrequencyController(
            models_dir=args.models,
            gpu_type=args.gpu,
            logger=logger
        )
        
        # Run optimal frequency selection
        results = controller.run_optimal_frequency_selection(
            app_command=args.command,
            workload=args.workload,
            performance_constraint_pct=args.constraint,
            apply_frequency=args.apply
        )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {args.output}")
        
        # Display summary
        if 'error' not in results:
            details = results['optimization_details']
            print("=" * 60)
            print("OPTIMAL FREQUENCY SELECTION RESULTS")
            print("=" * 60)
            print(f"Workload: {args.workload.upper()}")
            print(f"GPU: {args.gpu.upper()}")
            print(f"Optimal frequency: {details['optimal_frequency']} MHz")
            print(f"Expected energy savings: {details['energy_savings_pct']:.1f}%")
            print(f"Expected performance impact: {details['performance_impact_pct']:.1f}%")
            print(f"Constraint satisfied: {details['constraint_satisfied']}")
            print(f"Frequency applied: {details['frequency_applied']}")
        else:
            print(f"Optimization failed: {results['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimal frequency selection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
