# ML-Based Frequency Prediction Implementation Plan

Revised Plan Addendum (v1.1)
--------------------------------
This addendum refines the step-by-step plan to align with our implemented profiling and analysis methodology. It introduces robust data ingestion, consistent label generation via the optimizer, probe policies for single/few-run features, frequency snapping, and concrete evaluation criteria. The sections below supersede earlier steps where they conflict.

High-level goals
- Train on full DVFS sweeps (warm-run averages, outlier-filtered) to learn ground-truth EDP-optimal frequencies (and optionally ED²P).
- Predict optimal frequency from one (or a few) short profiling runs, while honoring a performance threshold relative to the fastest execution (default 5%).

Files to add under `tools/ml_prediction/`
- `profile_reader.py`: Robust DCGMI/nvidia-smi parser, warm-run aggregation, IQR outlier filtering.
- `label_builder.py`: Wraps `tools/analysis/edp_optimizer.py` to produce labels JSON aligned with the performance threshold.
- `feature_extractor.py`: Feature computation from aggregated runs (stats, ratios, trends, context).
- `dataset_builder.py`: Assembles training dataset from probe runs and labels; supports probe policies.
- `models/random_forest_predictor.py`: Baseline classifier with frequency snapping and confidence.
- (Later) `api/` and `cli/` for deployment.

Probe policies (how features are created at inference and training)
- `max-only` (default): Use the maximum supported frequency run as the probe (fast and high-SNR).
- `tri-point`: Use three runs (e.g., max, ~0.7×max, ~0.5×max). Concatenate/aggregate features and include a `probe_policy` flag.

Frequency snapping and ranges
- Always snap predictions to a legal frequency from `hardware.gpu_info.GPUSpecifications(...).get_available_frequencies()`.
- Do not hardcode min frequencies; read per-GPU ranges from the HAL.

Updated phases and tasks
1) Data Extraction and Labeling (Week 1–2)
   - Implement `profile_reader.py`:
     - Parse DCGMI whitespace tables reliably; map field IDs/names to canonical columns (POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA, etc.).
     - Aggregate warm runs only (exclude first run at each frequency) and apply IQR outlier filtering.
     - Provide per-frequency aggregates: avg_power, avg_timing, energy (avg_power × avg_time), plus basic dispersion metrics.
   - Implement `label_builder.py`:
     - Call `edp_optimizer.py` to compute EDP/ED²P optimal frequencies using the same performance threshold (default 5%).
     - Export `labels.json` with `{gpu, workload, performance_threshold, optimal_frequency_edp, optimal_frequency_ed2p, fastest_frequency, metrics...}`.

2) Dataset Building (Week 2–3)
   - Implement `feature_extractor.py`:
     - Stats: mean/std/p95/min/max for key metrics; simple trends (slopes) for POWER/TMPTR/GPUTL.
     - Ratios: MCUTL/GPUTL, GPUTL/POWER, TENSOR/SMACT (when available).
     - Normalized clocks: SMCLK/max, MMCLK/max; include `gpu_type`, `sampling_interval_ms`, `power_limit` if available.
   - Implement `dataset_builder.py`:
     - For each `results_*` dir, load label from `labels.json` and build features from probe runs per chosen policy (no leakage from the rest of the sweep).
     - Save parquet/CSV with features + `{gpu, workload, probe_policy, label_edp, label_ed2p, performance_threshold}`.

3) Baseline Modeling (Week 3–4)
   - Random Forest classifier over legal frequency bins per GPU.
   - Provide `.predict_single()` that snaps to valid clocks and returns a confidence score (class probability or ensemble agreement).
   - Optionally add a regression head or a candidate-set EDP(Time/Energy) predictor that selects argmin under the constraint.

4) Evaluation (Week 4)
   - Metrics: (a) snapped frequency error (MHz), (b) EDP gap vs optimal (%), (c) energy savings vs max (%), (d) performance delta vs fastest (%).
   - Splits: cross-workload (hold-out apps), cross-GPU (hold-out GPU family), plus random split for sanity.

5) Few-shot Extension (Week 5)
   - Add `tri-point` policy at inference when confidence is low: probe two additional frequencies around the predicted optimum, recompute features, re-predict.

6) Advanced Models (Week 6–8)
   - Gradient-boosted trees; multitask (EDP + ED²P); curve modeling to predict EDP(Time/Energy) on candidates and pick argmin; ensembles with meta-learner.

7) Packaging (Week 9–10)
   - REST API and CLI wrappers for prediction; include confidence-based guardrails and optional few-shot fallback.

Acceptance criteria per phase
- P1: Reader returns consistent per-frequency aggregates; labels identical to `edp_optimizer` output for the same threshold.
- P2: Dataset export reproducible; schema documented; basic stats produced.
- P3–4: Baseline RF model achieves small EDP gap and keeps performance within threshold on held-out workloads/GPUs.
- P5–7: Confidence-driven few-shot probing improves low-confidence cases; API/CLI deliver end-to-end prediction.

Notes on revisions vs original text
- Replace naive CSV reads with a robust DCGMI parser and whitespace handling; handle `N/A` rows and header lines.
- Align min/max/available frequencies with `hardware/gpu_info.py`; remove hardcoded `min_frequency=210`.
- Ensure warm-run averaging and IQR outlier filtering match the visualization code path.

---

Progress Update (v1.2) and Next Actions
----------------------------------------
Completed
- CLI: `build_labels.py` generates `labels.json` using `edp_optimizer.py` with consistent performance thresholding.
- CLI: `build_dataset.py` supports probe policies `max-only`, `tri-point`, and `all-freq`.
- Baseline: `train_baseline.py` (RandomForest) with frequency snapping and quick error metrics.
- Features added in lightweight path: `power_mean`, `duration_seconds`, `energy_estimate_j`, `probe_frequency_mhz`, `probe_freq_ratio`, plus context.

Findings
- `max-only` dataset (12 rows total) is too small and yields poor results (≈352.5 MHz median error; 0% within 60 MHz). Expected due to dataset size and minimal features.
- `all-freq` dataset increases sample count substantially and should be used for baseline training.

Immediate Next Steps
- Build `all-freq` dataset and retrain to establish a stronger baseline.
- Replace lightweight features with `feature_extractor.py` features (POWER/GPUTL/MCUTL/SMCLK/TMPTR/SMACT/DRAMA + trends/ratios) for accuracy.
- Add evaluation script with cross‑workload and cross‑GPU splits and EDP gap metrics.

Targets
- Baseline on `all-freq`: within‑60 MHz accuracy > 60%; improved further with richer features.
- EDP gap within 10–15% of optimal on average for ≥2/3 workloads.

Updated TODO Checklist
- [x] Labels CLI and JSON export
- [x] Dataset builder CLI with `max-only`, `tri-point`, `all-freq`; include probe frequency features
- [x] Baseline RF training script with snapping and quick metrics
- [ ] Integrate `feature_extractor.py` into dataset builder (richer features)
- [ ] Evaluation script with cross‑workload/GPU splits + EDP gap
- [ ] Few‑shot inference gating (tri‑point) based on confidence
- [ ] Advanced models (XGB/NN/ensembles) and ablations


## 🎯 Project Overview

We would like to develop a machine learning model that can take one (or a few) profiled runs of any (seen or unseen) AI inference application and determine what would be the optimal frequency for that. Our features could be the profiling metrics we gathered such as GPUTL, and MCUTL and our target would be to predict the Optimal Frequency using EDP (Time and Power).

**Goal**: Develop ML models that predict optimal GPU frequencies for AI inference workloads using minimal profiling data.

**Input**: Single baseline profiling run (2-3 minutes)
**Output**: Predicted optimal frequency for EDP minimization
**Current State**: Full DVFS sweep (2+ hours) → Target: Single prediction (seconds)

---

## 📋 Phase 1: Data Pipeline Development (Week 1-2)

### 1.1 Feature Engineering Pipeline

**File**: `tools/ml_prediction/feature_extractor.py`

```python
class ProfileFeatureExtractor:
    """Extract ML features from profiling data"""

    def __init__(self):
        self.feature_categories = {
            'utilization': ['GPUTL', 'MCUTL', 'SMACT'],
            'memory': ['DRAMA', 'PCUTL'],
            'compute': ['TENSOR', 'SMACT'],
            'thermal': ['GTEMP'],
            'power': ['POWER']
        }

    def extract_features(self, profile_df: pd.DataFrame,
                        metadata: dict) -> dict:
        """Extract comprehensive feature set"""
        features = {}

        # 1. Statistical features for each metric
        for category, metrics in self.feature_categories.items():
            for metric in metrics:
                if metric in profile_df.columns:
                    prefix = f"{metric.lower()}"
                    features[f"{prefix}_mean"] = profile_df[metric].mean()
                    features[f"{prefix}_std"] = profile_df[metric].std()
                    features[f"{prefix}_max"] = profile_df[metric].max()
                    features[f"{prefix}_min"] = profile_df[metric].min()
                    features[f"{prefix}_p95"] = profile_df[metric].quantile(0.95)

        # 2. Derived ratio features
        features['mem_to_gpu_ratio'] = (
            profile_df['MCUTL'].mean() / max(profile_df['GPUTL'].mean(), 1)
        )
        features['power_efficiency'] = (
            profile_df['GPUTL'].mean() / max(profile_df['POWER'].mean(), 1)
        )
        features['compute_intensity'] = (
            profile_df['TENSOR'].mean() / max(profile_df['SMACT'].mean(), 1)
        )

        # 3. Temporal patterns
        features['util_stability'] = 1 / (1 + profile_df['GPUTL'].std())
        features['power_trend'] = self._calculate_trend(profile_df['POWER'])
        features['temp_rise_rate'] = self._calculate_temp_rise(profile_df['GTEMP'])

        # 4. Hardware context
        features['gpu_type'] = metadata['gpu_type']
        features['max_frequency'] = metadata['max_frequency']
        features['min_frequency'] = metadata['min_frequency']

        # 5. Workload characteristics
        features['app_type'] = metadata['application']
        features['batch_size'] = metadata.get('batch_size', 1)

        return features

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend coefficient"""
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]

    def _calculate_temp_rise(self, temp_series: pd.Series) -> float:
        """Calculate temperature rise rate (°C/minute)"""
        if len(temp_series) < 2:
            return 0.0
        duration_minutes = len(temp_series) / 60  # Assuming 1-second intervals
        return (temp_series.iloc[-1] - temp_series.iloc[0]) / duration_minutes
```

**Tasks**:
- [ ] Implement feature extractor class
- [ ] Add unit tests for feature extraction
- [ ] Create feature documentation
- [ ] Validate features on existing data

### 1.2 Training Dataset Creation

**File**: `tools/ml_prediction/dataset_builder.py`

```python
class FrequencyPredictionDataset:
    """Build training dataset from existing DVFS experiments"""

    def __init__(self, results_base_dir: Path):
        self.results_dir = results_base_dir
        self.feature_extractor = ProfileFeatureExtractor()
        self.edp_optimizer = EDPOptimizer()

    def create_training_dataset(self) -> pd.DataFrame:
        """Create comprehensive training dataset"""
        training_samples = []

        # Scan all experiment results
        for gpu_dir in self.results_dir.glob("results_*"):
            gpu_type, app_name = self._parse_directory_name(gpu_dir)

            print(f"Processing {gpu_type} - {app_name}")

            # 1. Get baseline features (from lowest frequency run)
            baseline_features = self._extract_baseline_features(gpu_dir, gpu_type, app_name)

            # 2. Get optimal frequency from EDP analysis
            optimal_freq, edp_data = self._get_optimal_frequency(gpu_dir)

            # 3. Get performance metrics at optimal frequency
            performance_data = self._get_performance_at_frequency(gpu_dir, optimal_freq)

            if baseline_features and optimal_freq:
                sample = {
                    **baseline_features,
                    'optimal_frequency': optimal_freq,
                    'energy_savings': edp_data['energy_savings'],
                    'performance_degradation': edp_data['perf_degradation'],
                    'edp_improvement': edp_data['edp_improvement']
                }
                training_samples.append(sample)

        return pd.DataFrame(training_samples)

    def _extract_baseline_features(self, experiment_dir: Path,
                                 gpu_type: str, app_name: str) -> dict:
        """Extract features from baseline (max frequency) run"""
        # Find the highest frequency run as baseline
        freq_dirs = list(experiment_dir.glob("run_*_freq_*"))
        if not freq_dirs:
            return None

        # Get the maximum frequency run
        max_freq_dir = max(freq_dirs,
                          key=lambda x: int(x.name.split('_freq_')[1]))

        profile_file = max_freq_dir / "dcgmi_profile.csv"
        if not profile_file.exists():
            return None

        # Load profiling data
        profile_df = pd.read_csv(profile_file)

        # Metadata for feature extraction
        metadata = {
            'gpu_type': gpu_type,
            'application': app_name,
            'max_frequency': int(max_freq_dir.name.split('_freq_')[1]),
            'min_frequency': 210,  # Common minimum
        }

        return self.feature_extractor.extract_features(profile_df, metadata)
```

**Tasks**:
- [ ] Implement dataset builder
- [ ] Create data validation checks
- [ ] Add progress tracking for large datasets
- [ ] Generate dataset statistics report

### 1.3 Data Validation Pipeline

**File**: `tools/ml_prediction/data_validator.py`

```python
class DataValidator:
    """Validate and clean training dataset"""

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""

        print(f"Initial dataset size: {len(df)} samples")

        # 1. Remove samples with missing critical features
        critical_features = ['gputl_mean', 'power_mean', 'optimal_frequency']
        df_clean = df.dropna(subset=critical_features)
        print(f"After removing missing values: {len(df_clean)} samples")

        # 2. Remove outliers using IQR method
        df_clean = self._remove_outliers(df_clean)
        print(f"After outlier removal: {len(df_clean)} samples")

        # 3. Validate frequency ranges
        df_clean = self._validate_frequencies(df_clean)
        print(f"After frequency validation: {len(df_clean)} samples")

        # 4. Check feature distributions
        self._generate_data_report(df_clean)

        return df_clean

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method for numerical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df
```

**Tasks**:
- [ ] Implement data validation pipeline
- [ ] Create automated data quality reports
- [ ] Add outlier detection methods
- [ ] Generate feature correlation analysis

---

## 📊 Phase 2: Baseline Model Development (Week 3-4)

### 2.1 Random Forest Implementation

**File**: `tools/ml_prediction/models/random_forest_predictor.py`

```python
class RandomForestFrequencyPredictor:
    """Random Forest-based frequency predictor"""

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.model = RandomForestRegressor(**self.config['model_params'])
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_selector = None

    def _default_config(self) -> dict:
        return {
            'model_params': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'feature_selection': {
                'method': 'importance',
                'top_k': 20
            }
        }

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training"""
        # Separate categorical and numerical features
        categorical_features = ['gpu_type', 'app_type']
        numerical_features = [col for col in df.columns
                            if col not in categorical_features + ['optimal_frequency']]

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features)

        # Prepare feature matrix
        feature_columns = [col for col in df_encoded.columns
                          if col != 'optimal_frequency']
        X = df_encoded[feature_columns]
        y = df_encoded['optimal_frequency']

        return X, y, feature_columns

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train the model with cross-validation"""

        # Feature scaling
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Feature selection
        if self.config['feature_selection']['method'] == 'importance':
            self.feature_selector = SelectKBest(
                f_regression,
                k=self.config['feature_selection']['top_k']
            )
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_scaled)
        else:
            X_train_selected = X_train_scaled

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_selected, y_train_scaled,
            cv=5, scoring='neg_mean_absolute_error'
        )

        # Final training
        self.model.fit(X_train_selected, y_train_scaled)

        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self._analyze_feature_importance(X_train.columns)
        else:
            feature_importance = {}

        return {
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_test_scaled = self.feature_scaler.transform(X_test)

        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_test_selected = X_test_scaled

        y_pred_scaled = self.model.predict(X_test_selected)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Round to valid frequencies (30 MHz increments)
        return np.round(y_pred / 30) * 30

    def predict_single(self, profile_data: pd.DataFrame,
                      metadata: dict) -> dict:
        """Predict optimal frequency for a single profiling run"""

        # Extract features
        features = self.feature_extractor.extract_features(profile_data, metadata)
        features_df = pd.DataFrame([features])

        # Make prediction
        predicted_freq = self.predict(features_df)[0]

        # Calculate confidence score
        confidence = self._calculate_confidence(features_df)

        return {
            'predicted_frequency': int(predicted_freq),
            'confidence_score': confidence,
            'features_used': list(features.keys())
        }
```

**Tasks**:
- [ ] Implement Random Forest predictor
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Create prediction confidence estimation
- [ ] Add feature importance analysis

### 2.2 XGBoost Implementation

**File**: `tools/ml_prediction/models/xgboost_predictor.py`

```python
class XGBoostFrequencyPredictor:
    """XGBoost-based frequency predictor with advanced features"""

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.model = xgb.XGBRegressor(**self.config['model_params'])
        self.feature_scaler = RobustScaler()  # More robust to outliers

    def _default_config(self) -> dict:
        return {
            'model_params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 20
            }
        }

    def train_with_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Train with validation set for early stopping"""

        # Prepare data
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        # Training with early stopping
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            eval_metric='mae',
            verbose=False
        )

        # Get best iteration
        best_iteration = self.model.best_iteration

        # Validation metrics
        y_pred = self.model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return {
            'best_iteration': best_iteration,
            'validation_mae': val_mae,
            'validation_rmse': val_rmse,
            'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_))
        }
```

**Tasks**:
- [ ] Implement XGBoost predictor
- [ ] Add early stopping and validation
- [ ] Implement SHAP for explainability
- [ ] Add hyperparameter optimization with Optuna

### 2.3 Neural Network Implementation

**File**: `tools/ml_prediction/models/neural_network_predictor.py`

```python
class NeuralNetworkFrequencyPredictor:
    """Deep learning approach with temporal features"""

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def _default_config(self) -> dict:
        return {
            'architecture': {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'activation': 'relu'
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 15
            }
        }

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """Build neural network architecture"""

        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs

        # Hidden layers
        for units in self.config['architecture']['hidden_layers']:
            x = tf.keras.layers.Dense(
                units,
                activation=self.config['architecture']['activation']
            )(x)
            x = tf.keras.layers.Dropout(
                self.config['architecture']['dropout_rate']
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)

        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Train neural network with callbacks"""

        # Prepare data
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()

        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['training']['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-6
            )
        ]

        # Training
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=0
        )

        return {
            'history': history.history,
            'best_val_loss': min(history.history['val_loss']),
            'best_val_mae': min(history.history['val_mae'])
        }
```

**Tasks**:
- [ ] Implement neural network predictor
- [ ] Add temporal sequence modeling (LSTM/GRU)
- [ ] Implement attention mechanisms
- [ ] Add uncertainty quantification with MC Dropout

---

## 🧪 Phase 3: Advanced Model Development (Week 5-6)

### 3.1 Ensemble Methods

**File**: `tools/ml_prediction/models/ensemble_predictor.py`

```python
class EnsembleFrequencyPredictor:
    """Ensemble of multiple models for robust predictions"""

    def __init__(self):
        self.models = {
            'random_forest': RandomForestFrequencyPredictor(),
            'xgboost': XGBoostFrequencyPredictor(),
            'neural_network': NeuralNetworkFrequencyPredictor()
        }
        self.weights = None
        self.meta_model = LinearRegression()

    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Train ensemble with stacking"""

        # Stage 1: Train base models
        base_predictions = {}
        training_results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")

            if name == 'neural_network':
                result = model.train(X_train, y_train, X_val, y_val)
            else:
                result = model.train(X_train, y_train)

            training_results[name] = result

            # Get out-of-fold predictions for stacking
            predictions = model.predict(X_val)
            base_predictions[name] = predictions

        # Stage 2: Train meta-model (stacking)
        stacking_features = np.column_stack(list(base_predictions.values()))
        self.meta_model.fit(stacking_features, y_val)

        # Calculate ensemble weights based on validation performance
        val_errors = {}
        for name, predictions in base_predictions.items():
            val_errors[name] = mean_absolute_error(y_val, predictions)

        # Inverse error weighting
        total_inv_error = sum(1/error for error in val_errors.values())
        self.weights = {name: (1/error)/total_inv_error
                       for name, error in val_errors.items()}

        return {
            'individual_results': training_results,
            'ensemble_weights': self.weights,
            'validation_errors': val_errors
        }

    def predict_ensemble(self, X_test: pd.DataFrame) -> dict:
        """Make ensemble predictions with confidence"""

        # Get predictions from all models
        individual_predictions = {}
        for name, model in self.models.items():
            individual_predictions[name] = model.predict(X_test)

        # Weighted average
        weighted_pred = np.zeros(len(X_test))
        for name, predictions in individual_predictions.items():
            weighted_pred += self.weights[name] * predictions

        # Stacking prediction
        stacking_features = np.column_stack(list(individual_predictions.values()))
        stacked_pred = self.meta_model.predict(stacking_features)

        # Confidence based on prediction agreement
        pred_std = np.std(list(individual_predictions.values()), axis=0)
        confidence = 1 / (1 + pred_std / np.mean(weighted_pred))

        return {
            'weighted_prediction': weighted_pred,
            'stacked_prediction': stacked_pred,
            'individual_predictions': individual_predictions,
            'confidence_scores': confidence
        }
```

**Tasks**:
- [ ] Implement ensemble methods
- [ ] Add model selection strategies
- [ ] Create confidence estimation
- [ ] Add adaptive weight adjustment

### 3.2 Transfer Learning Framework

**File**: `tools/ml_prediction/transfer_learning.py`

```python
class TransferLearningPredictor:
    """Transfer learning for new GPU architectures and workloads"""

    def __init__(self, base_model):
        self.base_model = base_model
        self.domain_adapters = {}

    def create_domain_adapter(self, source_domain: str,
                            target_domain: str) -> dict:
        """Create domain adaptation layer"""

        # Domain-specific feature transformations
        adapter = {
            'feature_mapper': self._create_feature_mapper(source_domain, target_domain),
            'bias_corrector': self._create_bias_corrector(),
            'uncertainty_estimator': self._create_uncertainty_estimator()
        }

        return adapter

    def fine_tune_for_new_workload(self, new_workload_data: pd.DataFrame,
                                  source_workloads: list) -> dict:
        """Fine-tune model for new workload type"""

        # Find most similar source workload
        similarity_scores = {}
        for source_workload in source_workloads:
            similarity = self._calculate_workload_similarity(
                new_workload_data, source_workload
            )
            similarity_scores[source_workload] = similarity

        best_source = max(similarity_scores, key=similarity_scores.get)

        # Transfer learning with limited data
        adapted_model = self._adapt_model(
            base_model=self.base_model,
            source_domain=best_source,
            target_data=new_workload_data
        )

        return {
            'adapted_model': adapted_model,
            'source_similarity': similarity_scores,
            'confidence_adjustment': self._calculate_confidence_adjustment(
                similarity_scores[best_source]
            )
        }
```

**Tasks**:
- [ ] Implement transfer learning framework
- [ ] Add domain adaptation techniques
- [ ] Create workload similarity metrics
- [ ] Add few-shot learning capabilities

---

## 🎯 Phase 4: Integration & Validation (Week 7-8)

### 4.1 Real-time Prediction Pipeline

**File**: `tools/ml_prediction/real_time_predictor.py`

```python
class RealTimeFrequencyPredictor:
    """Production-ready prediction pipeline"""

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.feature_extractor = ProfileFeatureExtractor()
        self.validator = PredictionValidator()

    def predict_from_live_profiling(self, profile_duration: int = 120) -> dict:
        """Make prediction from live profiling data"""

        # 1. Collect profiling data
        print(f"Collecting {profile_duration}s of profiling data...")
        profile_data = self._collect_live_profile(profile_duration)

        # 2. Extract features
        features = self.feature_extractor.extract_features(
            profile_data['metrics'],
            profile_data['metadata']
        )

        # 3. Make prediction
        prediction_result = self.model.predict_single(
            profile_data['metrics'],
            profile_data['metadata']
        )

        # 4. Validate prediction
        validation_result = self.validator.validate_prediction(
            prediction_result, profile_data
        )

        # 5. Generate recommendation
        recommendation = self._generate_recommendation(
            prediction_result, validation_result
        )

        return {
            'predicted_frequency': prediction_result['predicted_frequency'],
            'confidence_score': prediction_result['confidence_score'],
            'validation_passed': validation_result['passed'],
            'recommendation': recommendation,
            'profiling_summary': self._summarize_profiling_data(profile_data)
        }

    def _collect_live_profile(self, duration: int) -> dict:
        """Collect live profiling data using existing tools"""

        # Use existing profiling infrastructure
        from sample_collection_scripts.profile import profile_smi

        # Start profiling
        profile_data = profile_smi.collect_baseline_profile(
            duration=duration,
            frequency='max'
        )

        return profile_data

    def _generate_recommendation(self, prediction: dict,
                               validation: dict) -> dict:
        """Generate actionable frequency recommendation"""

        if not validation['passed']:
            return {
                'action': 'collect_more_data',
                'reason': validation['failure_reason'],
                'suggested_profiling_duration': validation['suggested_duration']
            }

        confidence = prediction['confidence_score']

        if confidence > 0.8:
            return {
                'action': 'apply_frequency',
                'frequency': prediction['predicted_frequency'],
                'expected_savings': self._estimate_energy_savings(prediction),
                'risk_level': 'low'
            }
        elif confidence > 0.6:
            return {
                'action': 'apply_with_monitoring',
                'frequency': prediction['predicted_frequency'],
                'monitoring_duration': 300,  # 5 minutes
                'fallback_frequency': 'max',
                'risk_level': 'medium'
            }
        else:
            return {
                'action': 'run_limited_sweep',
                'frequency_range': self._suggest_frequency_range(prediction),
                'num_frequencies': 5,
                'risk_level': 'high'
            }
```

**Tasks**:
- [ ] Implement real-time prediction pipeline
- [ ] Add live profiling integration
- [ ] Create prediction validation
- [ ] Add recommendation engine

### 4.2 Validation Framework

**File**: `tools/ml_prediction/validation/comprehensive_validator.py`

```python
class ComprehensiveValidator:
    """Comprehensive validation of ML predictions"""

    def __init__(self):
        self.validation_tests = [
            self._test_cross_workload_generalization,
            self._test_cross_gpu_generalization,
            self._test_prediction_accuracy,
            self._test_energy_impact,
            self._test_runtime_performance
        ]

    def run_comprehensive_validation(self, model, test_data: dict) -> dict:
        """Run all validation tests"""

        results = {}
        overall_score = 0.0

        for test_func in self.validation_tests:
            test_name = test_func.__name__.replace('_test_', '')
            print(f"Running {test_name}...")

            test_result = test_func(model, test_data)
            results[test_name] = test_result
            overall_score += test_result['score'] * test_result['weight']

        results['overall_score'] = overall_score
        results['recommendation'] = self._generate_validation_recommendation(overall_score)

        return results

    def _test_cross_workload_generalization(self, model, test_data: dict) -> dict:
        """Test generalization across different workloads"""

        workloads = test_data['workloads']
        results = {}

        for test_workload in workloads:
            # Train on all except test workload
            train_workloads = [w for w in workloads if w != test_workload]
            train_data = self._combine_workload_data(train_workloads, test_data)
            test_workload_data = test_data[test_workload]

            # Train model
            model.train(train_data['X'], train_data['y'])

            # Test on held-out workload
            predictions = model.predict(test_workload_data['X'])
            actual = test_workload_data['y']

            # Calculate metrics
            mae = mean_absolute_error(actual, predictions)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            energy_impact = self._calculate_energy_impact(predictions, actual)

            results[test_workload] = {
                'mae': mae,
                'mape': mape,
                'energy_impact': energy_impact
            }

        # Overall generalization score
        avg_mape = np.mean([r['mape'] for r in results.values()])
        score = max(0, 1 - avg_mape / 100)  # Convert to 0-1 score

        return {
            'score': score,
            'weight': 0.3,
            'detailed_results': results,
            'summary': f"Average MAPE: {avg_mape:.2f}%"
        }

    def _test_energy_impact(self, model, test_data: dict) -> dict:
        """Test actual energy impact of predictions"""

        total_energy_savings = 0
        total_energy_potential = 0

        for workload, data in test_data['workloads'].items():
            predictions = model.predict(data['X'])
            optimal_frequencies = data['y']

            for pred_freq, optimal_freq in zip(predictions, optimal_frequencies):
                # Calculate energy savings
                pred_savings = self._calculate_energy_savings(pred_freq, workload)
                optimal_savings = self._calculate_energy_savings(optimal_freq, workload)

                total_energy_savings += pred_savings
                total_energy_potential += optimal_savings

        efficiency_ratio = total_energy_savings / total_energy_potential
        score = min(1.0, efficiency_ratio)

        return {
            'score': score,
            'weight': 0.4,
            'energy_efficiency_ratio': efficiency_ratio,
            'total_energy_savings': total_energy_savings,
            'summary': f"Energy efficiency: {efficiency_ratio:.2%}"
        }
```

**Tasks**:
- [ ] Implement comprehensive validation framework
- [ ] Add cross-validation strategies
- [ ] Create energy impact assessment
- [ ] Add statistical significance testing

---

## 📊 Phase 5: Production Deployment (Week 9-10)

### 5.1 Production API

**File**: `tools/ml_prediction/api/prediction_api.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="GPU Frequency Prediction API")

class PredictionRequest(BaseModel):
    gpu_type: str
    application: str
    profiling_duration: int = 120
    confidence_threshold: float = 0.7

class PredictionResponse(BaseModel):
    predicted_frequency: int
    confidence_score: float
    energy_savings_estimate: float
    recommendation: dict
    profiling_summary: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_optimal_frequency(request: PredictionRequest):
    """Predict optimal frequency for given configuration"""

    try:
        # Initialize predictor
        predictor = RealTimeFrequencyPredictor("/models/production_model.pkl")

        # Make prediction
        result = predictor.predict_from_live_profiling(
            profile_duration=request.profiling_duration
        )

        # Validate confidence threshold
        if result['confidence_score'] < request.confidence_threshold:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction confidence {result['confidence_score']:.2f} "
                       f"below threshold {request.confidence_threshold}"
            )

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Tasks**:
- [ ] Implement production API
- [ ] Add authentication and rate limiting
- [ ] Create monitoring and logging
- [ ] Add model versioning and rollback

### 5.2 Command Line Interface

**File**: `tools/ml_prediction/cli/predict_cli.py`

```python
import click
from pathlib import Path

@click.group()
def cli():
    """GPU Frequency Prediction CLI"""
    pass

@cli.command()
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--gpu-type', required=True, help='GPU type (V100/A100/H100)')
@click.option('--application', required=True, help='Application name')
@click.option('--duration', default=120, help='Profiling duration in seconds')
@click.option('--output', help='Output file for results')
def predict(model_path: str, gpu_type: str, application: str,
           duration: int, output: str):
    """Make frequency prediction for current workload"""

    predictor = RealTimeFrequencyPredictor(model_path)

    click.echo(f"Starting prediction for {application} on {gpu_type}")
    click.echo(f"Profiling duration: {duration} seconds")

    # Make prediction
    with click.progressbar(length=duration, label='Collecting profile') as bar:
        result = predictor.predict_from_live_profiling(duration)
        bar.update(duration)

    # Display results
    click.echo("\n" + "="*50)
    click.echo("PREDICTION RESULTS")
    click.echo("="*50)
    click.echo(f"Predicted Optimal Frequency: {result['predicted_frequency']} MHz")
    click.echo(f"Confidence Score: {result['confidence_score']:.2%}")
    click.echo(f"Recommendation: {result['recommendation']['action']}")

    if result['recommendation']['action'] == 'apply_frequency':
        click.echo(f"Expected Energy Savings: {result['recommendation']['expected_savings']:.1%}")

    # Save results if requested
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"\nResults saved to: {output}")

@cli.command()
@click.option('--results-dir', required=True, help='Directory with profiling results')
@click.option('--output-dir', default='./models', help='Output directory for trained model')
@click.option('--model-type', default='ensemble', help='Model type to train')
def train(results_dir: str, output_dir: str, model_type: str):
    """Train frequency prediction model"""

    click.echo(f"Training {model_type} model...")
    click.echo(f"Results directory: {results_dir}")

    # Build dataset
    dataset_builder = FrequencyPredictionDataset(Path(results_dir))
    dataset = dataset_builder.create_training_dataset()

    click.echo(f"Created dataset with {len(dataset)} samples")

    # Train model
    if model_type == 'ensemble':
        model = EnsembleFrequencyPredictor()
    elif model_type == 'xgboost':
        model = XGBoostFrequencyPredictor()
    else:
        model = RandomForestFrequencyPredictor()

    # Training process with progress
    X, y, _ = model.prepare_features(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with click.progressbar(label='Training model') as bar:
        training_result = model.train(X_train, y_train)
        bar.update(1)

    # Validation
    test_predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)

    click.echo(f"\nTraining completed!")
    click.echo(f"Test MAE: {test_mae:.1f} MHz")

    # Save model
    output_path = Path(output_dir) / f"{model_type}_frequency_predictor.pkl"
    model.save(output_path)
    click.echo(f"Model saved to: {output_path}")

if __name__ == '__main__':
    cli()
```

**Tasks**:
- [ ] Implement CLI interface
- [ ] Add progress tracking
- [ ] Create configuration management
- [ ] Add batch prediction capabilities

---

## 📈 Phase 6: Continuous Improvement (Ongoing)

### 6.1 Online Learning System

**File**: `tools/ml_prediction/online_learning.py`

```python
class OnlineLearningSystem:
    """Continuously improve model with new data"""

    def __init__(self, base_model):
        self.base_model = base_model
        self.feedback_buffer = []
        self.performance_tracker = PerformanceTracker()

    def collect_feedback(self, prediction: dict, actual_result: dict):
        """Collect feedback from actual frequency optimization"""

        feedback = {
            'timestamp': datetime.now(),
            'predicted_frequency': prediction['predicted_frequency'],
            'actual_optimal_frequency': actual_result['optimal_frequency'],
            'actual_energy_savings': actual_result['energy_savings'],
            'workload_features': prediction['features_used'],
            'prediction_confidence': prediction['confidence_score']
        }

        self.feedback_buffer.append(feedback)

        # Trigger retraining if buffer is full
        if len(self.feedback_buffer) >= 100:
            self.retrain_model()

    def retrain_model(self):
        """Retrain model with accumulated feedback"""

        # Prepare incremental training data
        new_features = []
        new_targets = []

        for feedback in self.feedback_buffer:
            new_features.append(feedback['workload_features'])
            new_targets.append(feedback['actual_optimal_frequency'])

        # Incremental learning
        self.base_model.partial_fit(new_features, new_targets)

        # Clear buffer
        self.feedback_buffer = []

        # Update performance metrics
        self.performance_tracker.update_metrics()
```

**Tasks**:
- [ ] Implement online learning system
- [ ] Add feedback collection mechanisms
- [ ] Create performance monitoring
- [ ] Add automatic model updates

### 6.2 A/B Testing Framework

**File**: `tools/ml_prediction/ab_testing.py`

```python
class ABTestingFramework:
    """A/B testing for model improvements"""

    def __init__(self):
        self.experiments = {}
        self.traffic_splitter = TrafficSplitter()

    def create_experiment(self, name: str, models: dict,
                         traffic_split: dict):
        """Create new A/B test experiment"""

        experiment = {
            'name': name,
            'models': models,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'metrics': {},
            'status': 'active'
        }

        self.experiments[name] = experiment

    def route_prediction(self, experiment_name: str,
                        request: dict) -> dict:
        """Route prediction to appropriate model variant"""

        experiment = self.experiments[experiment_name]

        # Determine model variant
        variant = self.traffic_splitter.assign_variant(
            request, experiment['traffic_split']
        )

        # Make prediction
        model = experiment['models'][variant]
        prediction = model.predict_single(request)

        # Track for analysis
        self._track_prediction(experiment_name, variant, prediction)

        return prediction
```

**Tasks**:
- [ ] Implement A/B testing framework
- [ ] Add statistical significance testing
- [ ] Create experiment management
- [ ] Add automated winner selection

---

## 🗓️ Implementation Timeline

### **Week 1-2: Foundation**
- [ ] Data pipeline development
- [ ] Feature engineering
- [ ] Dataset creation and validation

### **Week 3-4: Baseline Models**
- [ ] Random Forest implementation
- [ ] XGBoost implementation
- [ ] Initial neural network

### **Week 5-6: Advanced Models**
- [ ] Ensemble methods
- [ ] Transfer learning
- [ ] Model optimization

### **Week 7-8: Integration**
- [ ] Real-time prediction pipeline
- [ ] Comprehensive validation
- [ ] Performance testing

### **Week 9-10: Deployment**
- [ ] Production API
- [ ] CLI interface
- [ ] Documentation and tutorials

### **Ongoing: Improvement**
- [ ] Online learning
- [ ] A/B testing
- [ ] Model monitoring

---

## 📋 Success Metrics

### **Model Performance**
- **Prediction Accuracy**: MAE < 50 MHz (vs. optimal)
- **Energy Efficiency**: Capture >80% of potential energy savings
- **Confidence Calibration**: High-confidence predictions accurate >90%

### **Operational Metrics**
- **Speedup**: 100x faster than full DVFS sweep (2 hours → 2 minutes)
- **Resource Usage**: <1% GPU utilization for prediction
- **Generalization**: Work on unseen workloads with >70% accuracy

### **Business Impact**
- **Energy Savings**: 15-30% reduction in inference energy
- **Deployment Time**: <5 minutes to get frequency recommendation
- **Adoption**: Works across all GPU types and applications

This comprehensive plan provides a roadmap for developing a production-ready ML system for GPU frequency prediction that can significantly reduce energy profiling time while maintaining high accuracy.
