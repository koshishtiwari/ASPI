"""Feature engineering module for Stock Potential Identifier.

This module handles generating technical, fundamental, and alternative
features from raw market data.
"""

import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler

from .utils import load_config
from .utils.pattern_utils import detect_doji, detect_hammer, detect_engulfing, detect_morningstar, apply_pattern_detection
from .utils.date_utils.py import safe_convert_to_float, is_date, clean_dataframe, parse_date_column


logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Handles all feature engineering."""

    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = load_config(config_path)
        self.feature_config = self.config['feature_engineering']
        
        # Initialize scalers
        self.scalers = {
            'technical': StandardScaler(),
            'fundamental': StandardScaler(),
            'alternative': StandardScaler()
        }
        
        # Fitted flag for scalers
        self.scalers_fitted = {
            'technical': False,
            'fundamental': False,
            'alternative': False
        }
    
    def generate_technical_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators from market data.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        if market_data.empty:
            logger.warning("Empty market data provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = market_data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Get enabled indicator categories
        tech_config = self.feature_config['technical']
        
        # Generate momentum indicators
        if 'momentum_indicators' in tech_config:
            df = self._generate_momentum_indicators(df, tech_config['momentum_indicators'])
        
        # Generate volatility indicators
        if 'volatility_indicators' in tech_config:
            df = self._generate_volatility_indicators(df, tech_config['volatility_indicators'])
        
        # Generate volume indicators
        if 'volume_indicators' in tech_config:
            df = self._generate_volume_indicators(df, tech_config['volume_indicators'])
        
        # Generate trend indicators
        if 'trend_indicators' in tech_config:
            df = self._generate_trend_indicators(df, tech_config['trend_indicators'])
        
        # Generate pattern recognition features
        if tech_config.get('pattern_recognition', {}).get('enabled', False):
            df = self._generate_pattern_features(df, tech_config['pattern_recognition'].get('patterns', []))
        
        # Handle NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values created by indicators with lookback periods
        df = df.fillna(method='ffill')
        
        # Any remaining NaNs fill with zeros (should be minimal at this point)
        df = df.fillna(0)
        
        return df
    
    def _generate_momentum_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Generate momentum indicators."""
        # Get price and volume data
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        volume = df['Volume']
        
        # Generate indicators based on configuration
        for indicator in indicators:
            if indicator == 'rsi':
                # Relative Strength Index
                df['rsi_14'] = ta.rsi(close, length=14)
                
                # Additional RSI periods
                df['rsi_7'] = ta.rsi(close, length=7)
                df['rsi_21'] = ta.rsi(close, length=21)
                
                # RSI divergence (simple)
                if len(df) > 14:
                    df['rsi_trend'] = df['rsi_14'].diff(1)
                    df['price_trend'] = df['Close'].diff(1)
                    df['rsi_divergence'] = ((df['rsi_trend'] > 0) & (df['price_trend'] < 0)) | ((df['rsi_trend'] < 0) & (df['price_trend'] > 0))
                    df['rsi_divergence'] = df['rsi_divergence'].astype(int)
            
            elif indicator == 'macd':
                # MACD (Moving Average Convergence Divergence)
                macd = ta.macd(close)
                df = df.join(macd)
                
                # MACD signal crossovers
                if 'MACDh_12_26_9' in df.columns:
                    df['macd_crossover'] = np.where(df['MACDh_12_26_9'] > 0, 1, -1)
                    df['macd_crossover_change'] = df['macd_crossover'].diff().fillna(0)
            
            elif indicator == 'roc':
                # Rate of Change
                df['roc_10'] = ta.roc(close, length=10)
                df['roc_50'] = ta.roc(close, length=50)
                df['roc_200'] = ta.roc(close, length=200)
            
            elif indicator == 'mom':
                # Momentum
                df['mom_10'] = ta.mom(close, length=10)
                df['mom_50'] = ta.mom(close, length=50)
            
            elif indicator == 'cci':
                # Commodity Channel Index
                df['cci_20'] = ta.cci(high, low, close, length=20)
                # Overbought/oversold signals
                df['cci_signal'] = np.where(df['cci_20'] > 100, 1, np.where(df['cci_20'] < -100, -1, 0))
            
            elif indicator == 'stoch':
                # Stochastic Oscillator
                stoch = ta.stoch(high, low, close)
                df = df.join(stoch)
                
                # Stochastic crossovers
                if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
                    df['stoch_crossover'] = np.where(df['STOCHk_14_3_3'] > df['STOCHd_14_3_3'], 1, -1)
                    df['stoch_crossover_change'] = df['stoch_crossover'].diff().fillna(0)
            
            elif indicator == 'williams':
                # Williams %R
                df['willr_14'] = ta.willr(high, low, close)
                
                # Williams %R signals
                df['willr_signal'] = np.where(df['willr_14'] > -20, 1, np.where(df['willr_14'] < -80, -1, 0))
        
        return df
    
    def _generate_volatility_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Generate volatility indicators."""
        # Get price data
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Generate indicators based on configuration
        for indicator in indicators:
            if indicator == 'bbands':
                # Bollinger Bands
                bbands = ta.bbands(close, length=20, std=2.0)
                df = df.join(bbands)
                
                # Calculate BB width and %B
                if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
                    # BB Width - normalized by middle band
                    df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
                    
                    # BB Squeeze - when bands are narrowing (lower width)
                    if len(df) > 20:
                        df['bb_width_sma5'] = df['bb_width'].rolling(5).mean()
                        df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width_sma5'], 1, 0)
                    
                    # %B - position within bands (0 to 1 scale)
                    df['bb_percent_b'] = (close - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
                    
                    # BB Trend signals
                    df['bb_upper_cross'] = np.where(close > df['BBU_20_2.0'], 1, 0)
                    df['bb_lower_cross'] = np.where(close < df['BBL_20_2.0'], 1, 0)
            
            elif indicator == 'atr':
                # Average True Range
                df['atr_14'] = ta.atr(high, low, close, length=14)
                
                # ATR percentage (ATR relative to price)
                df['atr_percent'] = df['atr_14'] / close * 100
                
                # ATR-based volatility state
                if len(df) > 20:
                    df['atr_sma20'] = df['atr_14'].rolling(20).mean()
                    df['volatility_state'] = np.where(df['atr_14'] > df['atr_sma20'] * 1.2, 2,  # High volatility
                                                np.where(df['atr_14'] < df['atr_sma20'] * 0.8, 0, 1))  # Low or normal
            
            elif indicator == 'historical_volatility':
                # Historical Volatility (annualized)
                returns = df['Close'].pct_change().dropna()
                
                # Calculate HV for different lookback periods
                for period in [10, 20, 50]:
                    # Standard deviation of returns
                    rolling_std = returns.rolling(period).std()
                    
                    # Annualized volatility (standard formula: std * sqrt(252))
                    df[f'hv_{period}'] = rolling_std * np.sqrt(252)
                
                # Volatility regime based on 20-day HV
                if 'hv_20' in df.columns and len(df) > 60:
                    df['hv_20_sma60'] = df['hv_20'].rolling(60).mean()
                    df['vol_regime'] = np.where(df['hv_20'] > df['hv_20_sma60'] * 1.2, 2,  # High vol regime
                                           np.where(df['hv_20'] < df['hv_20_sma60'] * 0.8, 0, 1))  # Low or normal
            
            elif indicator == 'keltner':
                # Keltner Channels
                keltner = ta.kc(high, low, close)
                df = df.join(keltner)
                
                # Keltner width
                if 'KCU_20_2.0' in df.columns and 'KCL_20_2.0' in df.columns:
                    df['kc_width'] = (df['KCU_20_2.0'] - df['KCL_20_2.0']) / df['KCM_20_2.0']
            
            elif indicator == 'donchian':
                # Donchian Channels
                donchian = ta.donchian(high, low, close, length=20)
                df = df.join(donchian)
                
                # Donchian width
                if 'DCU_20_20' in df.columns and 'DCL_20_20' in df.columns:
                    df['dc_width'] = (df['DCU_20_20'] - df['DCL_20_20']) / close * 100
        
        return df
    
    def _generate_volume_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Generate volume indicators."""
        # Get price and volume data
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Generate indicators based on configuration
        for indicator in indicators:
            if indicator == 'obv':
                # On-Balance Volume
                df['obv'] = ta.obv(close, volume)
                
                # OBV simple trend
                if len(df) > 20:
                    df['obv_sma20'] = df['obv'].rolling(20).mean()
                    df['obv_trend'] = np.where(df['obv'] > df['obv_sma20'], 1, -1)
            
            elif indicator == 'cmf':
                # Chaikin Money Flow
                df['cmf_20'] = ta.cmf(high, low, close, volume, length=20)
                
                # CMF signals
                df['cmf_signal'] = np.where(df['cmf_20'] > 0.05, 1, np.where(df['cmf_20'] < -0.05, -1, 0))
            
            elif indicator == 'vwap':
                # Volume Weighted Average Price
                df['vwap'] = ta.vwap(high, low, close, volume)
                
                # VWAP signals
                df['vwap_signal'] = np.where(close > df['vwap'], 1, -1)
            
            elif indicator == 'ad':
                # Accumulation/Distribution Line
                df['ad'] = ta.ad(high, low, close, volume)
                
                # A/D trend
                if len(df) > 20:
                    df['ad_sma20'] = df['ad'].rolling(20).mean()
                    df['ad_trend'] = np.where(df['ad'] > df['ad_sma20'], 1, -1)
            
            elif indicator == 'volume_sma':
                # Volume SMAs for different periods
                for period in [10, 20, 50]:
                    df[f'volume_sma_{period}'] = volume.rolling(period).mean()
                
                # Volume relative to moving averages
                for period in [10, 20, 50]:
                    df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']
                
                # Volume spike detection
                df['volume_spike'] = np.where(df['volume_ratio_20'] > 2, 1, 0)
            
            elif indicator == 'pvi':
                # Positive Volume Index
                df['pvi'] = ta.pvi(close, volume)
            
            elif indicator == 'nvi':
                # Negative Volume Index
                df['nvi'] = ta.nvi(close, volume)
        
        return df
    
    def _generate_trend_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Generate trend indicators."""
        # Get price data
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Generate indicators based on configuration
        for indicator in indicators:
            if indicator == 'sma':
                # Simple Moving Averages
                for period in [10, 20, 50, 100, 200]:
                    df[f'sma_{period}'] = ta.sma(close, length=period)
                    
                    # Calculate distance from SMA (percentage)
                    df[f'dist_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
                
                # SMA crossovers
                if 'sma_20' in df.columns and 'sma_50' in df.columns:
                    df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
                    df['sma_20_50_cross_change'] = df['sma_20_50_cross'].diff().fillna(0)
                
                if 'sma_50' in df.columns and 'sma_200' in df.columns:
                    df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
                    df['sma_50_200_cross_change'] = df['sma_50_200_cross'].diff().fillna(0)
            
            elif indicator == 'ema':
                # Exponential Moving Averages
                for period in [10, 20, 50, 100, 200]:
                    df[f'ema_{period}'] = ta.ema(close, length=period)
                    
                    # Calculate distance from EMA (percentage)
                    df[f'dist_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
                
                # EMA crossovers
                if 'ema_20' in df.columns and 'ema_50' in df.columns:
                    df['ema_20_50_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
                    df['ema_20_50_cross_change'] = df['ema_20_50_cross'].diff().fillna(0)
                
                if 'ema_50' in df.columns and 'ema_200' in df.columns:
                    df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
                    df['ema_50_200_cross_change'] = df['ema_50_200_cross'].diff().fillna(0)
            
            elif indicator == 'adx':
                # Average Directional Index
                adx = ta.adx(high, low, close, length=14)
                df = df.join(adx)
                
                # ADX trend strength
                if 'ADX_14' in df.columns:
                    # Trend strength categories (0: no trend, 1: weak, 2: moderate, 3: strong)
                    df['adx_trend_strength'] = np.where(df['ADX_14'] > 40, 3,
                                                   np.where(df['ADX_14'] > 25, 2,
                                                       np.where(df['ADX_14'] > 15, 1, 0)))
                
                # Trend direction from DI lines
                if 'DMP_14' in df.columns and 'DMN_14' in df.columns:
                    df['di_trend'] = np.where(df['DMP_14'] > df['DMN_14'], 1, -1)
                    df['di_cross_change'] = df['di_trend'].diff().fillna(0)
            
            elif indicator == 'vwma':
                # Volume Weighted Moving Average
                if 'Volume' in df.columns:
                    df['vwma_20'] = ta.vwma(close, df['Volume'], length=20)
                    df['vwma_50'] = ta.vwma(close, df['Volume'], length=50)
                    
                    # Price relative to VWMA
                    df['dist_vwma_20'] = (close - df['vwma_20']) / df['vwma_20'] * 100
            
            elif indicator == 'supertrend':
                # SuperTrend indicator
                supertrend = ta.supertrend(high, low, close, length=10, multiplier=3.0)
                df = df.join(supertrend)
                
                # SuperTrend signals
                if 'SUPERT_10_3.0' in df.columns:
                    df['supertrend_signal'] = np.where(close > df['SUPERT_10_3.0'], 1, -1)
                    df['supertrend_change'] = df['supertrend_signal'].diff().fillna(0)
            
            elif indicator == 'ichimoku':
                # Ichimoku Cloud
                ichimoku = ta.ichimoku(high, low, close)
                df = df.join(ichimoku)
                
                # Ichimoku signals (simplified)
                if all(col in df.columns for col in ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26']):
                    # TK Cross
                    df['tk_cross'] = np.where(df['ITS_9'] > df['IKS_26'], 1, -1)
                    df['tk_cross_change'] = df['tk_cross'].diff().fillna(0)
                    
                    # Price relative to cloud
                    df['above_cloud'] = np.where(
                        (close > df['ISA_9']) & (close > df['ISB_26']), 1,
                        np.where((close < df['ISA_9']) & (close < df['ISB_26']), -1, 0)
                    )
        
        return df
    
    def _generate_pattern_features(self, df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
        """Generate pattern recognition features.
        
        Args:
            df: DataFrame with OHLC data
            patterns: List of patterns to detect
            
        Returns:
            DataFrame with pattern features added
        """
        # Import the custom pattern detection module
        from .pattern_utils import detect_doji, detect_hammer, detect_engulfing, detect_morningstar
        
        # Get OHLC data
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Generate pattern features based on configuration
        for pattern in patterns:
            if pattern == 'hammer':
                df['hammer'] = detect_hammer(df)
            
            elif pattern == 'engulfing':
                engulfing = detect_engulfing(df)
                df['bullish_engulfing'] = engulfing['bullish_engulfing']
                df['bearish_engulfing'] = engulfing['bearish_engulfing']
                
                # Combined engulfing signal (1 for bullish, -1 for bearish, 0 for none)
                df['engulfing_signal'] = df['bullish_engulfing'] - df['bearish_engulfing']
            
            elif pattern == 'doji':
                df['doji'] = detect_doji(df)
            
            elif pattern == 'morningstar':
                df['morning_star'] = detect_morningstar(df)
        
        return df
    
    def generate_fundamental_features(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Generate fundamental features and ratios.
        
        Args:
            fundamental_data: DataFrame with fundamental data
            
        Returns:
            DataFrame with fundamental features
        """
        if fundamental_data.empty:
            logger.warning("Empty fundamental data provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Check if fundamental features are enabled
        if not self.feature_config['fundamental']['enabled']:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = fundamental_data.copy()
        
        # Calculate additional ratios if they don't already exist
        
        # P/E ratio
        if 'pe_ratio' not in df.columns and 'market_cap' in df.columns and 'net_income' in df.columns:
            df['pe_ratio'] = df['market_cap'] / df['net_income']
        
        # P/S ratio
        if 'ps_ratio' not in df.columns and 'market_cap' in df.columns and 'revenue' in df.columns:
            df['ps_ratio'] = df['market_cap'] / df['revenue']
        
        # P/B ratio
        if 'price_to_book' not in df.columns and 'market_cap' in df.columns and 'total_equity' in df.columns:
            df['price_to_book'] = df['market_cap'] / df['total_equity']
        
        # EV/EBITDA
        if 'ev_to_ebitda' not in df.columns and 'market_cap' in df.columns and 'debt' in df.columns and 'cash_and_equivalents' in df.columns and 'ebitda' in df.columns:
            enterprise_value = df['market_cap'] + df['debt'] - df['cash_and_equivalents']
            df['ev_to_ebitda'] = enterprise_value / df['ebitda']
        
        # Debt to Equity
        if 'debt_to_equity' not in df.columns and 'debt' in df.columns and 'total_equity' in df.columns:
            df['debt_to_equity'] = df['debt'] / df['total_equity']
        
        # Current ratio (if balance sheet data available)
        if 'current_ratio' not in df.columns and 'total_current_assets' in df.columns and 'total_current_liabilities' in df.columns:
            df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities']
        
        # Return on Equity (ROE)
        if 'roe' not in df.columns and 'net_income' in df.columns and 'total_equity' in df.columns:
            df['roe'] = df['net_income'] / df['total_equity']
        
        # Return on Assets (ROA)
        if 'roa' not in df.columns and 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = df['net_income'] / df['total_assets']
        
        # Operating Margin
        if 'operating_margin' not in df.columns and 'operating_income' in df.columns and 'revenue' in df.columns:
            df['operating_margin'] = df['operating_income'] / df['revenue']
        
        # Net Margin
        if 'net_margin' not in df.columns and 'net_income' in df.columns and 'revenue' in df.columns:
            df['net_margin'] = df['net_income'] / df['revenue']
        
        # Free Cash Flow Yield
        if 'fcf_yield' not in df.columns and 'free_cash_flow' in df.columns and 'market_cap' in df.columns:
            df['fcf_yield'] = df['free_cash_flow'] / df['market_cap']
        
        # Replace infinities and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Select the metrics specified in config
        metrics = self.feature_config['fundamental'].get('metrics', [])
        if metrics:
            # Keep only specified metrics
            available_metrics = [col for col in metrics if col in df.columns]
            df = df[available_metrics]
        
        return df
    
    def generate_alternative_features(self, alternative_data: pd.DataFrame) -> pd.DataFrame:
        """Generate alternative features from alternative data.
        
        Args:
            alternative_data: DataFrame with alternative data
            
        Returns:
            DataFrame with alternative features
        """
        if alternative_data.empty:
            logger.warning("Empty alternative data provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Check if alternative features are enabled
        alt_config = self.feature_config['alternative']
        
        # Make a copy to avoid modifying the original
        df = alternative_data.copy()
        
        # Process insider trading data if available and enabled
        if alt_config['insider_activity']['enabled'] and 'transaction_type' in df.columns:
            # Calculate net insider transactions (Buy = +1, Sell = -1)
            df['insider_buy'] = (df['transaction_type'] == 'Buy').astype(int)
            df['insider_sell'] = (df['transaction_type'] == 'Sell').astype(int)
            
            # Aggregate by date if multiple transactions per date
            if df.index.name == 'date':
                daily_net = df.groupby(level=0)[['insider_buy', 'insider_sell']].sum()
                daily_net['net_insider_transactions'] = daily_net['insider_buy'] - daily_net['insider_sell']
                
                # Calculate rolling metrics (last 30 days)
                rolling_net = daily_net['net_insider_transactions'].rolling(30).sum()
                
                # Convert to strength signal
                insider_signal = pd.DataFrame(index=daily_net.index)
                insider_signal['insider_signal'] = np.where(rolling_net > 2, 1,  # Strong buying
                                                      np.where(rolling_net < -2, -1, 0))  # Strong selling or neutral
                
                return insider_signal
        
        # Process news sentiment if enabled
        if alt_config.get('news_sentiment', {}).get('enabled', False):
            # This would process news sentiment data
            # For now return empty DataFrame as it requires API access
            pass
        
        # Process social sentiment if enabled
        if alt_config.get('social_sentiment', {}).get('enabled', False):
            # This would process social media sentiment data
            # For now return empty DataFrame as it requires API access
            pass
        
        # Return processed data or empty DataFrame if no processing was done
        return pd.DataFrame(index=df.index)
    
    def combine_feature_sets(self, technical_features: pd.DataFrame, 
                            fundamental_features: pd.DataFrame = None,
                            alternative_features: pd.DataFrame = None,
                            normalize: bool = True) -> pd.DataFrame:
        """Combine all feature sets with proper alignment.
        
        Args:
            technical_features: DataFrame with technical features
            fundamental_features: DataFrame with fundamental features
            alternative_features: DataFrame with alternative features
            normalize: Whether to normalize features
            
        Returns:
            Combined DataFrame with all features
        """
        if technical_features.empty:
            logger.warning("Empty technical features provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Start with technical features
        combined = technical_features.copy()
        
        # Add fundamental features if available (broadcast to all dates)
        if fundamental_features is not None and not fundamental_features.empty:
            # For each fundamental feature, broadcast to all dates
            for col in fundamental_features.columns:
                # Use the latest value for all dates
                combined[f'fundamental_{col}'] = fundamental_features[col].iloc[0]
        
        # Add alternative features if available (with date alignment)
        if alternative_features is not None and not alternative_features.empty:
            # Reindex to match technical features dates
            aligned_alt = alternative_features.reindex(combined.index, method='ffill')
            
            # Add alternative features with prefix
            for col in aligned_alt.columns:
                combined[f'alt_{col}'] = aligned_alt[col]
        
        # Normalize features if requested
        if normalize:
            combined = self._normalize_features(combined)
        
        return combined
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to have zero mean and unit variance."""
        # Copy to avoid modifying the original
        df = features.copy()
        
        # Separate feature types (by prefix)
        technical_cols = [col for col in df.columns if not col.startswith(('fundamental_', 'alt_'))]
        fundamental_cols = [col for col in df.columns if col.startswith('fundamental_')]
        alternative_cols = [col for col in df.columns if col.startswith('alt_')]
        
        # Skip normalization for categorical columns
        categorical_cols = []
        
        # Normalize technical features
        if technical_cols:
            # Keep original columns that should not be normalized
            tech_categorical = list(set(technical_cols) & set(categorical_cols))
            tech_numeric = list(set(technical_cols) - set(tech_categorical))
            
            if tech_numeric:
                tech_data = df[tech_numeric].values
                
                if not self.scalers_fitted['technical']:
                    # Fit scaler
                    self.scalers['technical'].fit(tech_data)
                    self.scalers_fitted['technical'] = True
                
                # Transform
                tech_normalized = self.scalers['technical'].transform(tech_data)
                df[tech_numeric] = tech_normalized
        
        # Normalize fundamental features
        if fundamental_cols:
            # Keep original columns that should not be normalized
            fund_categorical = list(set(fundamental_cols) & set(categorical_cols))
            fund_numeric = list(set(fundamental_cols) - set(fund_categorical))
            
            if fund_numeric:
                fund_data = df[fund_numeric].values
                
                if not self.scalers_fitted['fundamental']:
                    # Fit scaler
                    self.scalers['fundamental'].fit(fund_data)
                    self.scalers_fitted['fundamental'] = True
                
                # Transform
                fund_normalized = self.scalers['fundamental'].transform(fund_data)
                df[fund_numeric] = fund_normalized
        
        # Normalize alternative features
        if alternative_cols:
            # Keep original columns that should not be normalized
            alt_categorical = list(set(alternative_cols) & set(categorical_cols))
            alt_numeric = list(set(alternative_cols) - set(alt_categorical))
            
            if alt_numeric:
                alt_data = df[alt_numeric].values
                
                if not self.scalers_fitted['alternative']:
                    # Fit scaler
                    self.scalers['alternative'].fit(alt_data)
                    self.scalers_fitted['alternative'] = True
                
                # Transform
                alt_normalized = self.scalers['alternative'].transform(alt_data)
                df[alt_numeric] = alt_normalized
        
        return df
    
    def generate_target_labels(self, market_data: pd.DataFrame, horizon: str) -> pd.DataFrame:
        """Generate target labels for ML models.
        
        Args:
            market_data: DataFrame with OHLC data
            horizon: Time horizon ('short_term', 'medium_term', 'long_term')
            
        Returns:
            DataFrame with binary target labels
        """
        from .date_utils import safe_convert_to_float, clean_dataframe
        
        if market_data.empty:
            logger.warning("Empty market data provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Get horizon parameters from config
        horizons_config = self.config['models']['time_horizons']
        horizon_days = horizons_config[horizon]['days']
        
        # Make a copy of the market data and ensure numeric types
        df = clean_dataframe(market_data.copy())
        
        # Calculate future returns
        df['future_return'] = df['Close'].shift(-horizon_days) / df['Close'] - 1
        
        # Create binary target label (1 for positive return, 0 for negative)
        df['target'] = np.where(df['future_return'] > 0, 1, 0)
        
        # Optionally create multi-class targets
        # 0: Strong Negative (< -5%), 1: Negative, 2: Neutral, 3: Positive, 4: Strong Positive (> 5%)
        df['target_multiclass'] = np.where(df['future_return'] < -0.05, 0,
                                    np.where(df['future_return'] < 0, 1,
                                        np.where(df['future_return'] < 0.03, 2,
                                            np.where(df['future_return'] < 0.05, 3, 4))))
        
        # Remove NaN targets (will be at the end due to future returns)
        df = df.dropna(subset=['target'])
        
        return df[['target', 'target_multiclass', 'future_return']]
    
    def select_features(self, features: pd.DataFrame, horizon: str) -> pd.DataFrame:
        """Select relevant features for a specific time horizon.
        
        Args:
            features: DataFrame with all features
            horizon: Time horizon ('short_term', 'medium_term', 'long_term')
            
        Returns:
            DataFrame with selected features
        """
        if features.empty:
            logger.warning("Empty features provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # For now, use all features for all horizons
        # In a more sophisticated implementation, you could select different
        # feature sets for different horizons based on domain knowledge
        
        # For example:
        # - Short-term: Technical indicators (momentum, volatility)
        # - Medium-term: Technical + alternative data
        # - Long-term: Fundamental + trend technical indicators
        
        return features