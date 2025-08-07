#!/usr/bin/env python3
"""
Optimal Frequency Selection for AI Inference Workloads

This script implements optimal frequency selection using your proven FGCS/ICPP methodology
extended for AI inference workloads with performance constraints.

Key features:
1. Performance-constrained optimization (≤5% degradation)
2. Cross-GPU model portability (V100, A100, H100)
3. AI workload-specific modeling
4. EDP/ED2P optimization with constraints
5. Single-run prediction capability

Author: Mert Side
Based on FGCS 2023 and ICPP 2023 research
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Model training will be disabled.")
    SKLEARN_AVAILABLE = False

class OptimalFrequencySelector:
    """
    Implements optimal frequency selection for AI inference workloads
    using proven FGCS/ICPP methodology with performance constraints
    """
    
    def __init__(self, performance_constraint_pct: float = 5.0):
        self.performance_constraint_pct = performance_constraint_pct
        self.power_model = None
        self.performance_model = None
        self.baselines = {}
        self.training_data = None
        self.model_metadata = {}
        
        # GPU architectural scaling factors (from your FGCS paper)
        self.gpu_scaling_factors = {
            'V100': 80/56,   # Your proven scaling approach
            'A100': 108/56,  # Ampere vs baseline
            'H100': 132/56   # Hopper vs baseline
        }
        
        logger.info(f"Initialized optimal frequency selector with {performance_constraint_pct}% constraint")
        
    def load_aggregated_data(self, data_file: str, baselines_file: str) -> bool:
        """Load aggregated data and baselines"""
        try:
            # Load main dataset
            self.training_data = pd.read_csv(data_file)
            logger.info(f"Loaded {len(self.training_data)} data points from {data_file}")
            
            # Load baselines
            with open(baselines_file, 'r') as f:
                self.baselines = json.load(f)
            logger.info(f"Loaded baselines for {len(self.baselines)} GPU-workload combinations")
            
            # Validate data
            required_columns = ['gpu', 'workload', 'frequency_mhz', 'duration_seconds', 
                              'fp_active', 'dram_active', 'sm_app_clock', 'avg_power_watts']
            missing_columns = [col for col in required_columns if col not in self.training_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling using your proven approach"""
        features_df = df.copy()
        
        # Your proven core features
        feature_columns = ['fp_active', 'dram_active', 'sm_app_clock']
        
        # Add architectural scaling factor (your FGCS approach)
        features_df['arch_factor'] = features_df['gpu'].map(self.gpu_scaling_factors).fillna(1.0)
        feature_columns.append('arch_factor')
        
        # Add AI-specific features if available
        ai_features = ['tensor_core_utilization', 'memory_bandwidth_utilization', 
                      'gpu_utilization', 'mixed_precision_ratio']
        for feature in ai_features:
            if feature in features_df.columns:
                feature_columns.append(feature)
                
        # Add workload encoding (for workload-specific patterns)
        workload_dummies = pd.get_dummies(features_df['workload'], prefix='workload')
        features_df = pd.concat([features_df, workload_dummies], axis=1)
        feature_columns.extend(workload_dummies.columns.tolist())
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        return features_df[feature_columns]
        
    def train_power_model(self, df: pd.DataFrame) -> Optional[object]:
        """Train power model using your proven linear approach"""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for model training")
            return None
            
        logger.info("Training power model (linear regression - your proven approach)")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['avg_power_watts']  # Target: average power
        
        # Your proven linear model approach
        model = LinearRegression()
        
        # Train-test split for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Power model performance:")
        logger.info(f"  Mean Absolute Error: {mae:.2f}W")
        logger.info(f"  R² Score: {r2:.3f}")
        
        # Cross-validation (your rigorous validation approach)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        logger.info(f"  Cross-validation R² (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Store model metadata
        self.model_metadata['power_model'] = {
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return model
        
    def train_performance_model(self, df: pd.DataFrame) -> Optional[object]:
        """Train performance model using your proven polynomial approach"""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for model training")
            return None
            
        logger.info("Training performance model (polynomial regression - your proven approach)")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['duration_seconds']  # Target: execution time
        
        # Your proven polynomial approach (2nd degree)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Performance model performance:")
        logger.info(f"  Mean Absolute Error: {mae:.2f}s")
        logger.info(f"  R² Score: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        logger.info(f"  Cross-validation R² (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Store model metadata
        self.model_metadata['performance_model'] = {
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return model
        
    def train_models(self) -> bool:
        """Train both power and performance models"""
        if self.training_data is None:
            logger.error("No training data loaded")
            return False
            
        # Filter out failed runs and outliers
        clean_data = self.training_data[
            (self.training_data['avg_power_watts'] > 0) &
            (self.training_data['duration_seconds'] > 0) &
            (self.training_data['avg_power_watts'] < 1000) &  # Reasonable power range
            (self.training_data['duration_seconds'] < 3600)   # Reasonable duration range
        ].copy()
        
        logger.info(f"Training models on {len(clean_data)} clean data points")
        
        # Train power model
        self.power_model = self.train_power_model(clean_data)
        if self.power_model is None:
            return False
            
        # Train performance model
        self.performance_model = self.train_performance_model(clean_data)
        if self.performance_model is None:
            return False
            
        logger.info("Model training completed successfully")
        return True
        
    def predict_metrics(self, features: Dict[str, float], gpu: str, workload: str) -> Dict[str, float]:
        """Predict power and performance for given features"""
        if self.power_model is None or self.performance_model is None:
            logger.error("Models not trained")
            return {}
            
        # Create feature vector
        feature_dict = features.copy()
        feature_dict['arch_factor'] = self.gpu_scaling_factors.get(gpu, 1.0)
        
        # Add workload encoding
        for wl in ['llama', 'stablediffusion', 'vit', 'whisper']:
            feature_dict[f'workload_{wl}'] = 1.0 if wl == workload else 0.0
            
        # Create DataFrame with correct column order
        feature_row = pd.DataFrame([feature_dict])
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in feature_row.columns:
                feature_row[col] = 0.0
                
        feature_vector = feature_row[self.feature_columns]
        
        # Predict
        predicted_power = self.power_model.predict(feature_vector)[0]
        predicted_time = self.performance_model.predict(feature_vector)[0]
        
        # Calculate derived metrics
        predicted_energy = predicted_power * predicted_time
        predicted_edp = predicted_energy * predicted_time
        predicted_ed2p = predicted_energy * (predicted_time ** 2)
        
        return {
            'predicted_power_watts': predicted_power,
            'predicted_time_seconds': predicted_time,
            'predicted_energy_joules': predicted_energy,
            'predicted_edp': predicted_edp,
            'predicted_ed2p': predicted_ed2p
        }
        
    def find_optimal_frequency_constrained(self, gpu: str, workload: str, 
                                         baseline_features: Dict[str, float],
                                         available_frequencies: List[int]) -> Dict:
        """Find optimal frequency with performance constraint (your approach)"""
        baseline_key = f"{gpu}_{workload}"
        
        if baseline_key not in self.baselines:
            logger.error(f"No baseline found for {gpu} {workload}")
            return {}
            
        baseline = self.baselines[baseline_key]
        max_acceptable_time = baseline['max_acceptable_time_seconds']
        
        logger.info(f"Finding optimal frequency for {gpu} {workload}")
        logger.info(f"Performance constraint: ≤{self.performance_constraint_pct}% degradation")
        logger.info(f"Max acceptable time: {max_acceptable_time:.2f}s")
        
        # Evaluate all frequencies
        frequency_results = []
        
        for frequency in available_frequencies:
            # Update features with new frequency
            features = baseline_features.copy()
            features['sm_app_clock'] = frequency
            
            # Predict metrics
            predictions = self.predict_metrics(features, gpu, workload)
            
            predicted_time = predictions['predicted_time_seconds']
            predicted_energy = predictions['predicted_energy_joules']
            predicted_power = predictions['predicted_power_watts']
            
            # Check performance constraint
            performance_degradation = (predicted_time / baseline['baseline_time_seconds'] - 1) * 100
            meets_constraint = predicted_time <= max_acceptable_time
            
            # Calculate energy savings vs baseline
            energy_savings = (baseline['baseline_energy_joules'] - predicted_energy) / baseline['baseline_energy_joules'] * 100
            
            frequency_results.append({
                'frequency_mhz': frequency,
                'predicted_power_watts': predicted_power,
                'predicted_time_seconds': predicted_time,
                'predicted_energy_joules': predicted_energy,
                'predicted_edp': predictions['predicted_edp'],
                'performance_degradation_pct': performance_degradation,
                'energy_savings_pct': energy_savings,
                'meets_constraint': meets_constraint
            })
            
        # Filter frequencies that meet performance constraint
        valid_frequencies = [f for f in frequency_results if f['meets_constraint']]
        
        if not valid_frequencies:
            logger.warning(f"No frequency meets {self.performance_constraint_pct}% constraint for {gpu} {workload}")
            # Return maximum frequency as fallback
            max_freq_result = max(frequency_results, key=lambda x: x['frequency_mhz'])
            max_freq_result['optimal_reason'] = 'fallback_max_frequency'
            return max_freq_result
            
        # Select frequency with minimum energy among valid options
        optimal_result = min(valid_frequencies, key=lambda x: x['predicted_energy_joules'])
        optimal_result['optimal_reason'] = 'minimum_energy_with_constraint'
        
        # Log results
        logger.info(f"Optimal frequency: {optimal_result['frequency_mhz']} MHz")
        logger.info(f"  Expected energy savings: {optimal_result['energy_savings_pct']:.1f}%")
        logger.info(f"  Performance impact: {optimal_result['performance_degradation_pct']:.1f}%")
        logger.info(f"  Constraint satisfied: {optimal_result['meets_constraint']}")
        
        return optimal_result
        
    def get_gpu_frequency_ranges(self) -> Dict[str, List[int]]:
        """Get available frequency ranges for each GPU (from your data)"""
        if self.training_data is None:
            logger.error("No training data available")
            return {}
            
        frequency_ranges = {}
        
        for gpu in self.training_data['gpu'].unique():
            gpu_data = self.training_data[self.training_data['gpu'] == gpu]
            frequencies = sorted(gpu_data['frequency_mhz'].unique())
            frequency_ranges[gpu] = frequencies
            logger.info(f"{gpu} frequencies: {len(frequencies)} ({frequencies[0]}-{frequencies[-1]} MHz)")
            
        return frequency_ranges
        
    def extract_baseline_features(self, gpu: str, workload: str) -> Optional[Dict[str, float]]:
        """Extract baseline features from maximum frequency run"""
        if self.training_data is None:
            return None
            
        # Find maximum frequency data for this GPU-workload combination
        subset = self.training_data[
            (self.training_data['gpu'] == gpu) & 
            (self.training_data['workload'] == workload)
        ]
        
        if subset.empty:
            return None
            
        max_freq = subset['frequency_mhz'].max()
        max_freq_data = subset[subset['frequency_mhz'] == max_freq]
        
        if max_freq_data.empty:
            return None
            
        # Extract your proven features (mean across runs at max frequency)
        features = {
            'fp_active': max_freq_data['fp_active'].mean(),
            'dram_active': max_freq_data['dram_active'].mean(),
            'sm_app_clock': max_freq,  # Will be updated during frequency sweep
        }
        
        # Add AI-specific features if available
        ai_features = ['tensor_core_utilization', 'memory_bandwidth_utilization', 
                      'gpu_utilization', 'mixed_precision_ratio']
        for feature in ai_features:
            if feature in max_freq_data.columns:
                features[feature] = max_freq_data[feature].mean()
                
        return features
        
    def run_optimal_frequency_analysis(self, output_dir: str = "optimal_frequency_results"):
        """Run complete optimal frequency analysis for all GPU-workload combinations"""
        if not self.power_model or not self.performance_model:
            logger.error("Models not trained")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        frequency_ranges = self.get_gpu_frequency_ranges()
        results = []
        
        logger.info("Running optimal frequency analysis for all combinations...")
        
        for gpu in ['V100', 'A100', 'H100']:
            for workload in ['llama', 'stablediffusion', 'vit', 'whisper']:
                baseline_key = f"{gpu}_{workload}"
                
                if baseline_key not in self.baselines:
                    logger.warning(f"No baseline for {gpu} {workload}")
                    continue
                    
                if gpu not in frequency_ranges:
                    logger.warning(f"No frequency data for {gpu}")
                    continue
                    
                # Extract baseline features (your single-run approach)
                baseline_features = self.extract_baseline_features(gpu, workload)
                if baseline_features is None:
                    logger.warning(f"Could not extract baseline features for {gpu} {workload}")
                    continue
                    
                # Find optimal frequency
                optimal_result = self.find_optimal_frequency_constrained(
                    gpu, workload, baseline_features, frequency_ranges[gpu]
                )
                
                if optimal_result:
                    optimal_result.update({
                        'gpu': gpu,
                        'workload': workload,
                        'baseline_features': baseline_features
                    })
                    results.append(optimal_result)
                    
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"optimal_frequencies_{timestamp}.csv"
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"Optimal frequency analysis completed: {results_file}")
        logger.info(f"Generated {len(results)} optimal frequency recommendations")
        
        # Generate summary
        self.generate_analysis_summary(results, output_path, timestamp)
        
        return results_df
        
    def generate_analysis_summary(self, results: List[Dict], output_path: Path, timestamp: str):
        """Generate comprehensive analysis summary"""
        summary_file = output_path / f"analysis_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Optimal Frequency Selection Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Performance constraint: ≤{self.performance_constraint_pct}% degradation\n\n")
            
            # Overall statistics
            if results:
                energy_savings = [r['energy_savings_pct'] for r in results if 'energy_savings_pct' in r]
                performance_impacts = [r['performance_degradation_pct'] for r in results if 'performance_degradation_pct' in r]
                
                f.write(f"Overall Results ({len(results)} combinations):\n")
                f.write(f"  Average energy savings: {np.mean(energy_savings):.1f}%\n")
                f.write(f"  Energy savings range: {np.min(energy_savings):.1f}% to {np.max(energy_savings):.1f}%\n")
                f.write(f"  Average performance impact: {np.mean(performance_impacts):.1f}%\n")
                f.write(f"  Performance impact range: {np.min(performance_impacts):.1f}% to {np.max(performance_impacts):.1f}%\n\n")
            
            # Model performance
            f.write("Model Performance:\n")
            if 'power_model' in self.model_metadata:
                pm = self.model_metadata['power_model']
                f.write(f"  Power model R²: {pm['r2']:.3f} (CV: {pm['cv_r2_mean']:.3f}±{pm['cv_r2_std']:.3f})\n")
            if 'performance_model' in self.model_metadata:
                tm = self.model_metadata['performance_model']
                f.write(f"  Performance model R²: {tm['r2']:.3f} (CV: {tm['cv_r2_mean']:.3f}±{tm['cv_r2_std']:.3f})\n")
            f.write("\n")
            
            # Per-GPU analysis
            for gpu in ['V100', 'A100', 'H100']:
                gpu_results = [r for r in results if r.get('gpu') == gpu]
                if gpu_results:
                    f.write(f"{gpu} Results:\n")
                    gpu_energy_savings = [r['energy_savings_pct'] for r in gpu_results]
                    f.write(f"  Average energy savings: {np.mean(gpu_energy_savings):.1f}%\n")
                    
                    for result in gpu_results:
                        f.write(f"    {result['workload']}: {result['frequency_mhz']}MHz, "
                               f"{result['energy_savings_pct']:.1f}% energy, "
                               f"{result['performance_degradation_pct']:.1f}% perf impact\n")
                    f.write("\n")
                    
        logger.info(f"Analysis summary saved: {summary_file}")
        
    def save_models(self, output_dir: str = "models"):
        """Save trained models for deployment"""
        if not self.power_model or not self.performance_model:
            logger.error("No models to save")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        power_model_file = output_path / f"power_model_{timestamp}.pkl"
        performance_model_file = output_path / f"performance_model_{timestamp}.pkl"
        metadata_file = output_path / f"model_metadata_{timestamp}.json"
        
        with open(power_model_file, 'wb') as f:
            pickle.dump(self.power_model, f)
            
        with open(performance_model_file, 'wb') as f:
            pickle.dump(self.performance_model, f)
            
        # Save metadata including feature columns and baselines
        metadata = {
            'feature_columns': self.feature_columns,
            'baselines': self.baselines,
            'gpu_scaling_factors': self.gpu_scaling_factors,
            'performance_constraint_pct': self.performance_constraint_pct,
            'model_metadata': self.model_metadata,
            'timestamp': timestamp
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Models saved:")
        logger.info(f"  Power model: {power_model_file}")
        logger.info(f"  Performance model: {performance_model_file}")
        logger.info(f"  Metadata: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="Optimal frequency selection for AI inference")
    parser.add_argument("--data-file", required=True,
                       help="Aggregated data CSV file")
    parser.add_argument("--baselines-file", required=True,
                       help="Performance baselines JSON file")
    parser.add_argument("--constraint-pct", type=float, default=5.0,
                       help="Performance degradation constraint percentage")
    parser.add_argument("--output-dir", default="optimal_frequency_results",
                       help="Output directory for results")
    parser.add_argument("--save-models", action="store_true",
                       help="Save trained models")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for this script")
        return 1
        
    # Initialize selector
    selector = OptimalFrequencySelector(args.constraint_pct)
    
    # Load data
    if not selector.load_aggregated_data(args.data_file, args.baselines_file):
        return 1
        
    # Train models
    if not selector.train_models():
        return 1
        
    # Run analysis
    results_df = selector.run_optimal_frequency_analysis(args.output_dir)
    
    # Save models if requested
    if args.save_models:
        selector.save_models(os.path.join(args.output_dir, "models"))
        
    logger.info("Optimal frequency selection analysis completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())
