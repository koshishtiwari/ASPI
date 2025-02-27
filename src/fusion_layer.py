"""Fusion layer module for Stock Potential Identifier.

This module handles model weight determination, market regime detection,
and consensus prediction generation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from sklearn.metrics import log_loss, accuracy_score

from .utils import load_config

logger = logging.getLogger(__name__)

class FusionLayer:
    """Handles model fusion and consensus generation."""

    def __init__(self, horizon: str, config_path: str = None):
        """Initialize the fusion layer.
        
        Args:
            horizon: Time horizon ('short_term', 'medium_term', 'long_term')
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.fusion_config = self.config['fusion']
        self.horizon = horizon
        
        # Initialize model weights
        self.model_weights = {}
        
        # Initialize historical performance tracking
        self.lookback_window = self.fusion_config['weighting']['lookback_window']
        self.historical_performance = {}
        
        # Initialize market regime tracking
        self.current_regime = 'unknown'
        
        # Weights for different regimes
        self.regime_weights = {
            'trending': {
                'lightgbm': 1.2,  # Boost trend-following models
                'xgboost': 1.2,
                'random_forest': 1.0,
                'lstm': 1.1,
                'prophet': 0.8  # Reduce mean-reverting models
            },
            'mean_reverting': {
                'lightgbm': 0.8,  # Reduce trend-following models
                'xgboost': 0.8,
                'random_forest': 1.0,
                'lstm': 0.9,
                'prophet': 1.2  # Boost mean-reverting models
            },
            'volatile': {
                'lightgbm': 1.1,  # Volatility-aware models
                'xgboost': 1.1,
                'random_forest': 0.9,
                'lstm': 1.2,  # Better with irregular patterns
                'prophet': 0.7  # Not good with high volatility
            },
            'unknown': {
                'lightgbm': 1.0,
                'xgboost': 1.0,
                'random_forest': 1.0,
                'lstm': 1.0,
                'prophet': 1.0
            }
        }
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect the current market regime.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Market regime ('trending', 'mean_reverting', 'volatile', 'unknown')
        """
        # Check if regime detection is enabled
        if not self.fusion_config['regime_detection']['enabled']:
            return 'unknown'
        
        try:
            # Calculate key metrics for regime detection
            
            # 1. Calculate returns
            returns = market_data['Close'].pct_change().dropna()
            
            # 2. Calculate volatility (21-day rolling std)
            volatility = returns.rolling(21).std().iloc[-1]
            
            # 3. Calculate trend strength using ADX-like approach
            prices = market_data['Close']
            if len(prices) < 50:
                return 'unknown'
            
            # Calculate 20-day moving average
            sma20 = prices.rolling(20).mean()
            
            # Calculate distance of price from MA (normalized)
            dist_from_ma = (prices - sma20) / sma20
            
            # Calculate trend directional movement
            up_move = prices.diff()
            down_move = -prices.diff()
            
            pos_dm = up_move.copy()
            pos_dm[pos_dm < 0] = 0
            
            neg_dm = down_move.copy()
            neg_dm[neg_dm < 0] = 0
            
            # Sum directional movement over 14 days
            pos_dm_14 = pos_dm.rolling(14).sum()
            neg_dm_14 = neg_dm.rolling(14).sum()
            
            # Calculate directional index
            di_diff = abs(pos_dm_14 - neg_dm_14)
            di_sum = pos_dm_14 + neg_dm_14
            
            # Trend strength (simplified ADX calculation)
            trend_strength = (di_diff / di_sum).iloc[-1] if di_sum.iloc[-1] > 0 else 0
            
            # Determine regime based on volatility and trend strength
            volatility_threshold = self.fusion_config['regime_detection']['volatility_threshold']
            trend_threshold = self.fusion_config['regime_detection']['trend_threshold']
            
            if volatility > volatility_threshold:
                regime = 'volatile'
            elif trend_strength > trend_threshold:
                regime = 'trending'
            else:
                regime = 'mean_reverting'
            
            logger.info(f"Detected market regime: {regime} (volatility: {volatility:.4f}, trend: {trend_strength:.4f})")
            
            # Store current regime
            self.current_regime = regime
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'unknown'
    
    def update_model_weights(self, predictions: Dict[str, Dict[str, np.ndarray]], 
                            actual_values: np.ndarray) -> None:
        """Update model weights based on recent performance.
        
        Args:
            predictions: Dictionary of model predictions
            actual_values: Actual target values
        """
        # Calculate performance metrics for each model
        for model_name, pred in predictions.items():
            # Skip models without probability predictions
            if pred['probability'] is None:
                continue
            
            try:
                # Calculate loss and accuracy
                probs = pred['probability']
                pred_class = pred['class']
                
                loss = log_loss(actual_values, probs)
                acc = accuracy_score(actual_values, pred_class)
                
                # Initialize historical performance tracking if needed
                if model_name not in self.historical_performance:
                    self.historical_performance[model_name] = []
                
                # Add performance metrics
                self.historical_performance[model_name].append({
                    'timestamp': datetime.now(),
                    'loss': loss,
                    'accuracy': acc,
                    'horizon': self.horizon
                })
                
                logger.debug(f"Updated performance metrics for {model_name}: loss={loss:.4f}, accuracy={acc:.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating performance metrics for {model_name}: {e}")
        
        # Calculate weights based on recent performance
        self._calculate_model_weights()
    
    def _calculate_model_weights(self) -> None:
        """Calculate model weights based on historical performance."""
        # Get recent performance for each model
        recent_performance = {}
        
        # Current date for filtering
        now = datetime.now()
        cutoff_date = now - timedelta(days=self.lookback_window)
        
        for model_name, history in self.historical_performance.items():
            # Filter by recency and horizon
            relevant_history = [
                h for h in history 
                if h['horizon'] == self.horizon and h['timestamp'] > cutoff_date
            ]
            
            if relevant_history:
                # Average loss over recent history
                avg_loss = np.mean([h['loss'] for h in relevant_history])
                recent_performance[model_name] = avg_loss
        
        # Calculate weights based on inverse loss
        if recent_performance:
            # Inverse loss weighting (lower loss = higher weight)
            inverse_loss = {model: 1/max(loss, 0.0001) for model, loss in recent_performance.items()}
            total = sum(inverse_loss.values())
            
            # Normalize weights
            self.model_weights = {model: weight/total for model, weight in inverse_loss.items()}
            
            logger.info(f"Updated model weights for {self.horizon}: {self.model_weights}")
        else:
            # Equal weights if no history
            models = list(self.historical_performance.keys())
            if models:
                weight = 1.0 / len(models)
                self.model_weights = {model: weight for model in models}
                logger.info(f"Set equal weights for {self.horizon}: {self.model_weights}")
    
    def generate_consensus(self, predictions: Dict[str, Dict[str, np.ndarray]], 
                          market_regime: Optional[str] = None) -> Dict:
        """Generate consensus prediction with dynamic weighting.
        
        Args:
            predictions: Dictionary of model predictions
            market_regime: Optional market regime override
            
        Returns:
            Dictionary with consensus prediction
        """
        # Use provided regime or current regime
        regime = market_regime if market_regime is not None else self.current_regime
        
        # Check if we have weights
        if not self.model_weights:
            # Equal weights if not established yet
            models = list(predictions.keys())
            weight = 1.0 / len(models)
            self.model_weights = {model: weight for model in models}
            logger.info(f"Set initial equal weights for {self.horizon}: {self.model_weights}")
        
        # Adjust weights based on market regime
        adjusted_weights = {}
        for model_name, base_weight in self.model_weights.items():
            # Apply regime-specific multiplier if available
            if model_name in self.regime_weights.get(regime, {}):
                adjusted_weights[model_name] = base_weight * self.regime_weights[regime][model_name]
            else:
                adjusted_weights[model_name] = base_weight
        
        # Normalize adjusted weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {model: weight/total_weight for model, weight in adjusted_weights.items()}
        
        # Initialize consensus arrays
        sample_pred = next(iter(predictions.values()), None)
        if sample_pred is None or 'class' not in sample_pred:
            logger.error("No valid predictions provided")
            return {'error': 'No valid predictions'}
        
        # Get dimensions from first prediction
        n_samples = len(sample_pred['class'])
        
        # Initialize consensus arrays
        consensus_probability = np.zeros(n_samples)
        weight_sum = 0
        
        # Apply weights and generate consensus
        for model_name, model_preds in predictions.items():
            # Skip models not in weights
            if model_name not in adjusted_weights:
                continue
            
            # Skip models without probability predictions
            if 'probability' not in model_preds or model_preds['probability'] is None:
                continue
            
            # Get weight
            weight = adjusted_weights[model_name]
            
            # Add weighted probability
            consensus_probability += weight * model_preds['probability']
            weight_sum += weight
        
        # Normalize if not all models had probabilities
        if weight_sum > 0 and weight_sum < 1.0:
            consensus_probability /= weight_sum
        
        # Threshold probabilities to generate class predictions
        consensus_class = (consensus_probability > 0.5).astype(int)
        
        # Create confidence levels based on probability distance from 0.5
        confidence = np.abs(consensus_probability - 0.5) * 2  # Scale to 0-1
        
        logger.info(f"Generated consensus prediction for {self.horizon} with regime {regime}")
        
        return {
            'probability': consensus_probability,
            'class': consensus_class,
            'confidence': confidence,
            'model_weights': adjusted_weights,
            'market_regime': regime
        }
    
    def get_model_performance(self) -> pd.DataFrame:
        """Get historical performance for all models.
        
        Returns:
            DataFrame with performance metrics
        """
        # Flatten historical performance into DataFrame
        data = []
        
        for model_name, history in self.historical_performance.items():
            for record in history:
                entry = {
                    'model': model_name,
                    'timestamp': record['timestamp'],
                    'loss': record['loss'],
                    'accuracy': record['accuracy'],
                    'horizon': record['horizon']
                }
                data.append(entry)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df