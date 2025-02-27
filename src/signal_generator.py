"""Signal generator module for Stock Potential Identifier.

This module handles converting model predictions into actionable trading signals
with risk assessment and position sizing recommendations.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from .utils import load_config

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Handles signal generation from model predictions."""

    def __init__(self, config_path: str = None):
        """Initialize signal generator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.signals_config = self.config['signals']
        
        # Get risk parameters
        self.risk_free_rate = self.signals_config['risk']['risk_free_rate']
        self.stop_loss_volatility_multiplier = self.signals_config['risk']['stop_loss_volatility_multiplier']
        
        # Get signal quality thresholds
        self.min_probability = self.signals_config['quality_thresholds']['min_probability']
        self.min_risk_reward = self.signals_config['quality_thresholds']['min_risk_reward']
        self.min_sharpe = self.signals_config['quality_thresholds']['min_sharpe']
    
    def generate_signals(self, market_data: pd.DataFrame, consensus_prediction: Dict, 
                        horizon: str, symbol: str) -> List[Dict]:
        """Generate trading signals with risk metrics.
        
        Args:
            market_data: DataFrame with market data
            consensus_prediction: Dictionary with consensus predictions
            horizon: Time horizon ('short_term', 'medium_term', 'long_term')
            symbol: Stock symbol
            
        Returns:
            List of signal dictionaries
        """
        logger.info(f"Generating signals for {symbol} ({horizon})")
        
        # Initialize signals list
        signals = []
        
        try:
            # Get the latest data point
            latest = market_data.iloc[-1]
            current_price = latest['Close']
            
            # Recent volatility for risk sizing
            returns = market_data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=21).std().iloc[-1]
            
            # Get predictions
            prediction_class = consensus_prediction['class']
            prediction_proba = consensus_prediction['probability']
            confidence = consensus_prediction['confidence']
            market_regime = consensus_prediction['market_regime']
            
            # For each prediction (usually just one for the latest date)
            for i in range(len(prediction_class)):
                # Get prediction for this sample
                pred_class = prediction_class[i]
                prob = prediction_proba[i]
                conf = confidence[i]
                
                # Generate signal based on prediction
                if pred_class == 1 and prob >= self.min_probability:
                    # Bullish signal
                    signal = self._generate_bullish_signal(
                        symbol, current_price, volatility, prob, conf, horizon, market_regime
                    )
                    
                    # Add signal if it meets quality criteria
                    if signal is not None and self._check_signal_quality(signal):
                        signals.append(signal)
                
                elif pred_class == 0 and (1 - prob) >= self.min_probability:
                    # Bearish signal
                    signal = self._generate_bearish_signal(
                        symbol, current_price, volatility, 1 - prob, conf, horizon, market_regime
                    )
                    
                    # Add signal if it meets quality criteria
                    if signal is not None and self._check_signal_quality(signal):
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _generate_bullish_signal(self, symbol: str, current_price: float, volatility: float,
                               probability: float, confidence: float, horizon: str,
                               market_regime: str) -> Optional[Dict]:
        """Generate bullish signal with risk assessment.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            volatility: Historical volatility
            probability: Win probability
            confidence: Model confidence
            horizon: Time horizon
            market_regime: Market regime
            
        Returns:
            Signal dictionary or None if invalid
        """
        # Calculate expected move based on volatility and horizon
        expected_move = self._calculate_expected_move(current_price, volatility, horizon, 'up')
        
        # Calculate reasonable stop loss
        stop_loss = self._calculate_stop_loss(current_price, volatility)
        
        # If stop loss is too close, skip signal
        if current_price - stop_loss < 0.01 * current_price:
            logger.debug(f"Stop loss too close for {symbol}, skipping")
            return None
        
        # Calculate risk-reward ratio
        risk = current_price - stop_loss
        reward = expected_move
        risk_reward = reward / risk if risk > 0 else 0
        
        # Adjust risk-reward based on market regime
        if market_regime == 'trending':
            risk_reward *= 1.1  # Boost in trending markets
        elif market_regime == 'volatile':
            risk_reward *= 0.9  # Reduce in volatile markets
        
        # Calculate expected value
        win_probability = probability
        expected_value = (win_probability * reward) - ((1 - win_probability) * risk)
        
        # Calculate position size recommendation based on risk
        position_size = self._calculate_position_size(current_price, stop_loss, volatility)
        
        # Calculate Sharpe ratio estimate
        expected_return = expected_value / current_price
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Create signal
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'LONG',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'target': current_price + expected_move,
            'probability': probability,
            'confidence': confidence,
            'risk_reward': risk_reward,
            'expected_value': expected_value,
            'sharpe': sharpe,
            'horizon': horizon,
            'volatility': volatility,
            'position_size': position_size,
            'market_regime': market_regime
        }
        
        return signal
    
    def _generate_bearish_signal(self, symbol: str, current_price: float, volatility: float,
                               probability: float, confidence: float, horizon: str,
                               market_regime: str) -> Optional[Dict]:
        """Generate bearish signal with risk assessment.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            volatility: Historical volatility
            probability: Win probability
            confidence: Model confidence
            horizon: Time horizon
            market_regime: Market regime
            
        Returns:
            Signal dictionary or None if invalid
        """
        # Calculate expected move based on volatility and horizon
        expected_move = self._calculate_expected_move(current_price, volatility, horizon, 'down')
        
        # Calculate reasonable stop loss (above current price for shorts)
        stop_loss = self._calculate_stop_loss(current_price, volatility, direction='up')
        
        # If stop loss is too close, skip signal
        if stop_loss - current_price < 0.01 * current_price:
            logger.debug(f"Stop loss too close for {symbol}, skipping")
            return None
        
        # Calculate risk-reward ratio
        risk = stop_loss - current_price
        reward = expected_move
        risk_reward = reward / risk if risk > 0 else 0
        
        # Adjust risk-reward based on market regime
        if market_regime == 'trending':
            risk_reward *= 1.1  # Boost in trending markets
        elif market_regime == 'volatile':
            risk_reward *= 0.9  # Reduce in volatile markets
        
        # Calculate expected value
        win_probability = probability
        expected_value = (win_probability * reward) - ((1 - win_probability) * risk)
        
        # Calculate position size recommendation based on risk
        position_size = self._calculate_position_size(current_price, stop_loss, volatility)
        
        # Calculate Sharpe ratio estimate
        expected_return = expected_value / current_price
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Create signal
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'SHORT',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'target': current_price - expected_move,
            'probability': probability,
            'confidence': confidence,
            'risk_reward': risk_reward,
            'expected_value': expected_value,
            'sharpe': sharpe,
            'horizon': horizon,
            'volatility': volatility,
            'position_size': position_size,
            'market_regime': market_regime
        }
        
        return signal
    
    def _calculate_expected_move(self, price: float, volatility: float, 
                               horizon: str, direction: str = 'up') -> float:
        """Calculate expected price move based on volatility and horizon.
        
        Args:
            price: Current price
            volatility: Historical volatility
            horizon: Time horizon
            direction: Direction of expected move ('up' or 'down')
            
        Returns:
            Expected price move
        """
        # Convert horizon to days
        if horizon == 'short_term':
            days = 5
        elif horizon == 'medium_term':
            days = 21
        else:
            days = 63
        
        # Calculate expected move based on volatility
        # Standard formula: price * volatility * sqrt(days)
        expected_move_pct = volatility * np.sqrt(days)
        
        # Add multiplier for positive skew in up moves
        if direction == 'up':
            expected_move_pct *= 1.5  # Upside potential typically higher
            return price * expected_move_pct
        else:
            expected_move_pct *= 1.2  # Downside moves typically faster but smaller
            return price * expected_move_pct
    
    def _calculate_stop_loss(self, price: float, volatility: float, 
                           direction: str = 'down') -> float:
        """Calculate optimal stop loss based on volatility.
        
        Args:
            price: Current price
            volatility: Historical volatility
            direction: Direction of stop ('down' for longs, 'up' for shorts)
            
        Returns:
            Stop loss price
        """
        # ATR-like stop calculation
        stop_distance = price * volatility * self.stop_loss_volatility_multiplier
        
        if direction == 'down':
            return max(price - stop_distance, price * 0.9)  # Prevent extreme stops
        else:
            return min(price + stop_distance, price * 1.1)  # Prevent extreme stops
    
    def _calculate_position_size(self, price: float, stop_loss: float, 
                               volatility: float) -> float:
        """Calculate recommended position size based on risk.
        
        Args:
            price: Current price
            stop_loss: Stop loss price
            volatility: Historical volatility
            
        Returns:
            Recommended position size (0-1 scale)
        """
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)
        
        # Base position size on volatility
        if volatility < 0.01:  # Very low volatility
            base_size = 1.0
        elif volatility < 0.02:  # Low volatility
            base_size = 0.8
        elif volatility < 0.03:  # Medium volatility
            base_size = 0.6
        elif volatility < 0.04:  # High volatility
            base_size = 0.4
        else:  # Very high volatility
            base_size = 0.2
        
        # Adjust based on risk per share
        risk_adjustment = 0.05 / (risk_per_share / price)  # Normalize to percentage
        risk_adjustment = max(0.5, min(1.5, risk_adjustment))  # Limit adjustment
        
        position_size = base_size * risk_adjustment
        
        # Ensure position size is within bounds
        position_size = max(0.1, min(1.0, position_size))
        
        return round(position_size, 2)
    
    def _check_signal_quality(self, signal: Dict) -> bool:
        """Check if signal meets quality thresholds.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if signal meets quality criteria, False otherwise
        """
        # Check probability threshold
        if signal['probability'] < self.min_probability:
            return False
        
        # Check risk-reward threshold
        if signal['risk_reward'] < self.min_risk_reward:
            return False
        
        # Check Sharpe ratio threshold
        if signal['sharpe'] < self.min_sharpe:
            return False
        
        return True
    
    def generate_multi_timeframe_analysis(self, signals: Dict[str, List[Dict]]) -> Dict:
        """Generate a multi-timeframe analysis for conflicting signals.
        
        Args:
            signals: Dictionary of signals by horizon
            
        Returns:
            Dictionary with multi-timeframe analysis
        """
        # This is a more advanced feature that analyzes signals across
        # different time horizons to identify opportunities with
        # aligned signals or to resolve conflicts
        
        # Group signals by symbol
        symbols = set()
        for horizon, horizon_signals in signals.items():
            for signal in horizon_signals:
                symbols.add(signal['symbol'])
        
        results = {}
        
        for symbol in symbols:
            # Get signals for this symbol across horizons
            symbol_signals = {}
            
            for horizon, horizon_signals in signals.items():
                symbol_horizon_signals = [s for s in horizon_signals if s['symbol'] == symbol]
                if symbol_horizon_signals:
                    symbol_signals[horizon] = symbol_horizon_signals[0]
            
            # Skip if not enough horizons
            if len(symbol_signals) < 2:
                continue
            
            # Check for alignment or conflict
            signal_types = [s['type'] for s in symbol_signals.values()]
            
            if all(t == 'LONG' for t in signal_types):
                alignment = 'bullish'
            elif all(t == 'SHORT' for t in signal_types):
                alignment = 'bearish'
            else:
                alignment = 'mixed'
            
            # Calculate aggregate metrics
            avg_probability = np.mean([s['probability'] for s in symbol_signals.values()])
            avg_risk_reward = np.mean([s['risk_reward'] for s in symbol_signals.values()])
            
            # Store results
            results[symbol] = {
                'alignment': alignment,
                'horizons': list(symbol_signals.keys()),
                'avg_probability': avg_probability,
                'avg_risk_reward': avg_risk_reward,
                'signals': symbol_signals
            }
        
        return results