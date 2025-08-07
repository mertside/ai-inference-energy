# Optimal Frequency Selection Tools

This directory contains various implementations and approaches for optimal GPU frequency selection.

## Scripts Overview

### Production-Ready Selectors

**`comprehensive_optimal_selector.py`**
- **Purpose**: Comprehensive optimal frequency selection system
- **Features**: Complete analysis with validation and export capabilities
- **Usage**: `python comprehensive_optimal_selector.py`

**`production_optimal_selector.py`**
- **Purpose**: Production-ready optimal frequency selector
- **Features**: Robust, validated selection for production deployment
- **Usage**: `python production_optimal_selector.py`

### Real Data-Based Selectors

**`real_data_optimal_selector.py`**
- **Purpose**: Real experimental data-based optimal frequency selector
- **Features**: Uses actual GPU profiling measurements
- **Usage**: `python real_data_optimal_selector.py`

**`real_data_optimal_frequency.py`**
- **Purpose**: Real data optimal frequency utilities
- **Features**: Utility functions for real data processing
- **Usage**: Import as module or run standalone

**`corrected_real_optimal.py`**
- **Purpose**: Corrected version of real optimal frequency selection
- **Features**: Fixed duplicate handling and improved analysis
- **Usage**: `python corrected_real_optimal.py`

### Simplified Versions

**`simple_real_optimal.py`**
- **Purpose**: Simplified real optimal frequency selection
- **Features**: Streamlined approach for quick results
- **Usage**: `python simple_real_optimal.py`

**`fixed_real_optimal.py`**
- **Purpose**: Fixed version addressing specific issues
- **Features**: Bug fixes and improvements over earlier versions
- **Usage**: `python fixed_real_optimal.py`

### Core Infrastructure

**`optimal_frequency_analysis.py`**
- **Purpose**: Core frequency analysis utilities
- **Features**: Base analysis functions and metrics

**`optimal_frequency_selection.py`**
- **Purpose**: Core frequency selection algorithms
- **Features**: Selection algorithms and optimization methods

**`optimal_frequency_selector.py`**
- **Purpose**: Main frequency selector implementation
- **Features**: Primary selector interface

## Selection Methodology

All selectors use Energy-Delay Product (EDP) optimization with constraints:
- **Objective**: Minimize EDP = Energy × Execution_Time
- **Constraint**: Performance degradation ≤ 5%
- **Data Source**: Real GPU profiling measurements where available

## Algorithm Evolution

1. **Heuristic-based** → **Real data-driven**
2. **Simple analysis** → **Comprehensive validation**
3. **Single GPU focus** → **Multi-GPU support**
4. **Basic metrics** → **Advanced EDP optimization**

## Usage Patterns

- **Development**: Use `simple_real_optimal.py` for quick testing
- **Analysis**: Use `comprehensive_optimal_selector.py` for detailed insights
- **Production**: Use `production_optimal_selector.py` for deployment
- **Research**: Use `real_data_optimal_selector.py` for experimental analysis

## Output Consistency

All selectors provide:
- Optimal frequency (MHz)
- Energy savings (%)
- Performance impact (%)
- Data source confidence level
- Validation status
