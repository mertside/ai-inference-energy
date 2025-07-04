"""
Power Modeling Framework

This module provides comprehensive power modeling capabilities for GPU energy profiling,
based on the FGCS 2023 methodology and enhanced with additional ML algorithms.

Main Components:
- FGCSPowerModelingFramework: High-level interface for complete power modeling pipeline
- Model implementations: Polynomial, Random Forest, XGBoost, and FGCS-original models
- EDP optimization: Energy-delay product optimization algorithms
- Performance metrics: Comprehensive model evaluation and performance calculation

Usage:
    from power_modeling import FGCSPowerModelingFramework, analyze_application
    
    # Quick analysis
    results = analyze_application("profiling_data.csv", app_name="MyApp")
    
    # Full framework
    framework = FGCSPowerModelingFramework()
    results = framework.analyze_from_file("profiling_data.csv")
"""

# Import core framework
from .fgcs_integration import FGCSPowerModelingFramework, analyze_application

# Import model factories and key model classes
from .models.model_factory import FGCSModelFactory, ModelPipeline
from .models.fgcs_models import (
    FGCSPowerModel, 
    PolynomialPowerModel, 
    PerformanceMetricsCalculator
)
from .models.ensemble_models import (
    EnhancedRandomForestModel,
    XGBoostPowerModel,
    ModelEvaluator
)

# Import preprocessing utilities
from .feature_engineering.preprocessing import DataPreprocessor

# Import validation utilities
from .validation import (
    ModelValidationMetrics,
    CrossValidationAnalyzer,
    PowerModelValidator,
    generate_validation_report
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Mert Side"

# Define what gets exported when using "from power_modeling import *"
__all__ = [
    # High-level interface
    'FGCSPowerModelingFramework',
    'analyze_application',
    
    # Model factories
    'FGCSModelFactory',
    'ModelPipeline',
    
    # Core models
    'FGCSPowerModel',
    'PolynomialPowerModel',
    'EnhancedRandomForestModel',
    'XGBoostPowerModel',
    
    # Utilities
    'PerformanceMetricsCalculator',
    'ModelEvaluator',
    'DataPreprocessor',
    
    # Validation
    'ModelValidationMetrics',
    'CrossValidationAnalyzer',
    'PowerModelValidator',
    'generate_validation_report',
    
    # Metadata
    '__version__',
    '__author__'
]