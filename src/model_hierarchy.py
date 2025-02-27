"""Model hierarchy module for Stock Potential Identifier.

This module handles model training, evaluation, and prediction for different
time horizons using various algorithms.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress Prophet warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utils import load_config, save_model, load_model

logger = logging.getLogger(__name__)

class ModelHierarchy:
    """Handles all model training, evaluation, and prediction."""

    def __init__(self, horizon: str, config_path: str = None):
        """Initialize model hierarchy for the given time horizon.
        
        Args:
            horizon: Time horizon ('short_term', 'medium_term', 'long_term')
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.models_config = self.config['models']
        self.horizon = horizon
        self.horizon_config = self.models_config['time_horizons'][horizon]
        
        # Initialize models dict
        self.models = {}
        self.model_metadata = {}
        self.fitted = False
        
        # Initialize history of training metrics
        self.training_history = []
        
        # Determine where to save models
        self.models_dir = os.environ.get('MODELS_DIR', 'models')
        self.models_subdir = os.path.join(self.models_dir, horizon)
        os.makedirs(self.models_subdir, exist_ok=True)
    
    def build_models(self) -> None:
        """Build model instances for the given horizon."""
        logger.info(f"Building models for {self.horizon} horizon")
        
        # Get enabled model types
        ml_enabled = self.models_config['configurations']['machine_learning']['enabled']
        ts_enabled = self.models_config['configurations']['time_series']['enabled']
        dl_enabled = self.models_config['configurations']['deep_learning']['enabled']
        
        # Build machine learning models
        if ml_enabled:
            ml_types = self.models_config['configurations']['machine_learning']['types']
            
            for model_type in ml_types:
                if model_type == 'xgboost':
                    self._build_xgboost_model()
                elif model_type == 'lightgbm':
                    self._build_lightgbm_model()
                elif model_type == 'random_forest':
                    self._build_random_forest_model()
        
        # Build time series models
        if ts_enabled:
            ts_types = self.models_config['configurations']['time_series']['types']
            
            for model_type in ts_types:
                if model_type == 'prophet':
                    self._build_prophet_model()
                elif model_type == 'arima':
                    # ARIMA would be implemented through statsmodels
                    # Not implemented in this version
                    pass
        
        # Build deep learning models
        if dl_enabled:
            dl_types = self.models_config['configurations']['deep_learning']['types']
            
            for model_type in dl_types:
                if model_type == 'lstm':
                    self._build_lstm_model()
    
    def _build_xgboost_model(self) -> None:
        """Build XGBoost model with appropriate hyperparameters."""
        # Get hyperparameters for this horizon
        params = self.models_config['hyperparameters']['xgboost'].get(
            self.horizon, self.models_config['hyperparameters']['xgboost']['short_term']
        )
        
        # Create model
        model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 200),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            objective='binary:logistic',
            n_jobs=-1,
            random_state=42
        )
        
        # Add to models dict
        self.models['xgboost'] = model
        
        # Add metadata
        self.model_metadata['xgboost'] = {
            'type': 'classification',
            'description': 'XGBoost classifier for predicting price direction',
            'params': params,
            'features': None,  # Will be set during training
            'training_date': None,  # Will be set during training
            'last_accuracy': None  # Will be set during evaluation
        }
        
        logger.info(f"Built XGBoost model for {self.horizon}")
    
    def _build_lightgbm_model(self) -> None:
        """Build LightGBM model with appropriate hyperparameters."""
        # Get hyperparameters for this horizon
        params = self.models_config['hyperparameters']['lightgbm'].get(
            self.horizon, self.models_config['hyperparameters']['lightgbm']['short_term']
        )
        
        # Create model
        model = lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 200),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            objective='binary',
            n_jobs=-1,
            random_state=42
        )
        
        # Add to models dict
        self.models['lightgbm'] = model
        
        # Add metadata
        self.model_metadata['lightgbm'] = {
            'type': 'classification',
            'description': 'LightGBM classifier for predicting price direction',
            'params': params,
            'features': None,  # Will be set during training
            'training_date': None,  # Will be set during training
            'last_accuracy': None  # Will be set during evaluation
        }
        
        logger.info(f"Built LightGBM model for {self.horizon}")
    
    def _build_random_forest_model(self) -> None:
        """Build Random Forest model."""
        # Create model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        
        # Add to models dict
        self.models['random_forest'] = model
        
        # Add metadata
        self.model_metadata['random_forest'] = {
            'type': 'classification',
            'description': 'Random Forest classifier for predicting price direction',
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            },
            'features': None,  # Will be set during training
            'training_date': None,  # Will be set during training
            'last_accuracy': None  # Will be set during evaluation
        }
        
        logger.info(f"Built Random Forest model for {self.horizon}")
    
    def _build_prophet_model(self) -> None:
        """Build Prophet model for time series forecasting."""
        # Prophet is instantiated during training since it doesn't follow
        # the same fit/predict pattern as scikit-learn models
        
        # Add metadata entry
        self.model_metadata['prophet'] = {
            'type': 'regression',
            'description': 'Facebook Prophet for time series forecasting',
            'params': {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'changepoint_prior_scale': 0.05
            },
            'features': None,  # Prophet uses different features
            'training_date': None,  # Will be set during training
            'last_accuracy': None  # Will be set during evaluation
        }
        
        logger.info(f"Added Prophet model metadata for {self.horizon}")
    
    def _build_lstm_model(self) -> None:
        """Build LSTM deep learning model."""
        # LSTM model will be built during training when input shape is known
        
        # Get hyperparameters for this horizon
        params = self.models_config['hyperparameters']['lstm'].get(
            self.horizon, self.models_config['hyperparameters']['lstm']['short_term']
        )
        
        # Add metadata entry
        self.model_metadata['lstm'] = {
            'type': 'classification',
            'description': 'LSTM deep learning model for time series prediction',
            'params': params,
            'features': None,  # Will be set during training
            'training_date': None,  # Will be set during training
            'last_accuracy': None  # Will be set during evaluation
        }
        
        logger.info(f"Added LSTM model metadata for {self.horizon}")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict:
        """Train all models on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            Dictionary of training metrics
        """
        if not self.models:
            logger.info("No models have been built yet, building now")
            self.build_models()
        
        logger.info(f"Training models for {self.horizon} horizon")
        
        # Store feature names
        feature_names = X_train.columns.tolist()
        
        # Training results
        train_results = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            
            try:
                # Time series CV for ML models
                if name in ['xgboost', 'lightgbm', 'random_forest']:
                    # Train with time series cross-validation
                    metrics = self._train_ml_model(name, model, X_train, y_train, validation_data)
                    
                    # Update metadata
                    self.model_metadata[name]['features'] = feature_names
                    self.model_metadata[name]['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.model_metadata[name]['last_accuracy'] = metrics.get('accuracy')
                    
                    # Save model
                    model_filename = f"{name}_{self.horizon}.joblib"
                    save_model(model, model_filename, self.models_subdir)
                    
                    # Add to results
                    train_results[name] = metrics
                
                # LSTM model training
                elif name == 'lstm':
                    # Build and train LSTM model
                    metrics = self._train_lstm_model(X_train, y_train, validation_data)
                    
                    # Update metadata
                    self.model_metadata[name]['features'] = feature_names
                    self.model_metadata[name]['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.model_metadata[name]['last_accuracy'] = metrics.get('accuracy')
                    
                    # Add to results
                    train_results[name] = metrics
                
                # Prophet model training
                elif name == 'prophet':
                    # Train Prophet model
                    metrics = self._train_prophet_model(X_train, y_train)
                    
                    # Update metadata
                    self.model_metadata[name]['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.model_metadata[name]['last_accuracy'] = metrics.get('rmse')
                    
                    # Add to results
                    train_results[name] = metrics
                    
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
                train_results[name] = {'error': str(e)}
        
        # Set fitted flag
        self.fitted = True
        
        # Add to training history
        self.training_history.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'horizon': self.horizon,
            'results': train_results
        })
        
        return train_results
    
    def _train_ml_model(self, name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                      validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict:
        """Train a machine learning model with proper evaluation.
        
        Args:
            name: Model name
            model: Model instance
            X_train: Training features
            y_train: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict on training data
        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate training metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, zero_division=0),
            'recall': recall_score(y_train, y_pred, zero_division=0),
            'f1': f1_score(y_train, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_train, y_proba)
        
        # Calculate validation metrics if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            
            # Predict on validation data
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate validation metrics
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred, zero_division=0),
                'val_recall': recall_score(y_val, val_pred, zero_division=0),
                'val_f1': f1_score(y_val, val_pred, zero_division=0)
            }
            
            if val_proba is not None:
                val_metrics['val_roc_auc'] = roc_auc_score(y_val, val_proba)
            
            # Add validation metrics to results
            metrics.update(val_metrics)
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store top 10 features
            metrics['top_features'] = feature_importance.head(10).to_dict('records')
        
        logger.info(f"Trained {name} model, accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def _train_lstm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict:
        """Train LSTM model for time series prediction.
        
        Args:
            X_train: Training features
            y_train: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get hyperparameters for this horizon
        params = self.model_metadata['lstm']['params']
        
        # Reshape data for LSTM [samples, timesteps, features]
        # For now, we're using a simple approach without time steps
        X_train_arr = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_arr = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
            val_data = (X_val_arr, y_val.values)
        
        # Build LSTM model
        input_shape = (1, X_train.shape[1])
        model = Sequential()
        
        # Add LSTM layers
        model.add(LSTM(params.get('units', 50), return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        for _ in range(params.get('layers', 2) - 1):
            model.add(LSTM(params.get('units', 50), return_sequences=False))
            model.add(Dropout(0.2))
        
        # Add output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        
        history = model.fit(
            X_train_arr, y_train.values,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            verbose=0
        )
        
        # Store model
        self.models['lstm'] = model
        
        # Evaluate model
        train_loss, train_acc = model.evaluate(X_train_arr, y_train.values, verbose=0)
        
        metrics = {
            'accuracy': train_acc,
            'loss': train_loss
        }
        
        # Add validation metrics if available
        if validation_data is not None:
            val_loss, val_acc = model.evaluate(X_val_arr, y_val.values, verbose=0)
            metrics.update({
                'val_accuracy': val_acc,
                'val_loss': val_loss
            })
        
        # Save model
        model_path = os.path.join(self.models_subdir, f"lstm_{self.horizon}")
        model.save(model_path)
        logger.info(f"Saved LSTM model to {model_path}")
        
        return metrics
    
    def _train_prophet_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train Prophet model for time series forecasting.
        
        Args:
            X_train: Training features (not used directly by Prophet)
            y_train: Training target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prophet requires a specific data format
        # For stock prediction, we need to convert our data
        
        # Check if we have a datetime index
        if not isinstance(X_train.index, pd.DatetimeIndex):
            logger.error("Prophet requires DatetimeIndex")
            return {'error': 'DatetimeIndex required for Prophet model'}
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': X_train.index,
            'y': y_train.values if isinstance(y_train, pd.Series) else y_train
        })
        
        # Create and train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors from features if needed
        # (simplified for binary classification)
        
        # Fit model
        model.fit(df)
        
        # Store model
        self.models['prophet'] = model
        
        # Make predictions on training data
        forecast = model.predict(df)
        
        # Calculate RMSE (continuous prediction)
        y_pred_prophet = forecast['yhat'].values
        rmse = np.sqrt(np.mean((y_train.values - y_pred_prophet) ** 2))
        
        # For binary classification, convert to probabilities
        probs = 1 / (1 + np.exp(-y_pred_prophet))
        y_pred_binary = (probs > 0.5).astype(int)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_train, y_pred_binary)
        
        metrics = {
            'rmse': rmse,
            'accuracy': accuracy
        }
        
        # Save model
        model_path = os.path.join(self.models_subdir, f"prophet_{self.horizon}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved Prophet model to {model_path}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from all trained models.
        
        Args:
            X: Feature DataFrame to predict on
            
        Returns:
            Dictionary of model name -> predictions
        """
        if not self.fitted:
            logger.warning("Models have not been trained, loading from disk if available")
            self.load_models()
        
        logger.info(f"Generating predictions for {self.horizon} horizon")
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Skip Prophet for simplicity (needs different prediction format)
                if name == 'prophet':
                    continue
                
                # LSTM needs special handling
                if name == 'lstm':
                    # Reshape for LSTM
                    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
                    y_pred_proba = model.predict(X_lstm).flatten()
                    y_pred_class = (y_pred_proba > 0.5).astype(int)
                    
                    predictions[name] = {
                        'class': y_pred_class,
                        'probability': y_pred_proba
                    }
                
                # Standard ML models
                elif hasattr(model, 'predict_proba'):
                    y_pred_class = model.predict(X)
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    
                    predictions[name] = {
                        'class': y_pred_class,
                        'probability': y_pred_proba
                    }
                
                # Models without probability output
                else:
                    y_pred_class = model.predict(X)
                    
                    predictions[name] = {
                        'class': y_pred_class,
                        'probability': None
                    }
                    
            except Exception as e:
                logger.error(f"Error generating predictions for {name} model: {e}")
        
        return predictions
    
    def load_models(self) -> None:
        """Load all saved models from disk."""
        logger.info(f"Loading models for {self.horizon} horizon")
        
        # Find model files in the models directory
        model_files = os.listdir(self.models_subdir) if os.path.exists(self.models_subdir) else []
        
        # Pattern matching for model files
        for file in model_files:
            if file.endswith('.joblib'):
                # Extract model name from filename
                model_name = file.split('_')[0]
                
                try:
                    # Load model
                    model_path = os.path.join(self.models_subdir, file)
                    model = joblib.load(model_path)
                    
                    # Add to models dict
                    self.models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading model from {file}: {e}")
            
            # Load LSTM model
            elif 'lstm' in file and not file.endswith('.joblib'):
                try:
                    model_path = os.path.join(self.models_subdir, file)
                    model = tf.keras.models.load_model(model_path)
                    
                    # Add to models dict
                    self.models['lstm'] = model
                    logger.info(f"Loaded LSTM model from {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading LSTM model from {file}: {e}")
        
        # Set fitted flag if models were loaded
        self.fitted = len(self.models) > 0
    
    def get_training_metrics(self) -> Dict:
        """Get the latest training metrics.
        
        Returns:
            Dictionary of metrics by model
        """
        if not self.training_history:
            return {}
        
        return self.training_history[-1]['results']