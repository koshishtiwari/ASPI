"""Custom pattern recognition functions without TA-Lib dependency."""

import pandas as pd
import numpy as np
from typing import Optional

def detect_doji(df: pd.DataFrame, tolerance: float = 0.05) -> pd.Series:
    """Detect doji candlestick patterns.
    
    A doji occurs when the open and close prices are very close to each other.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Maximum percentage difference between open and close to be considered a doji
        
    Returns:
        Series with 1 for doji detection, 0 otherwise
    """
    # Get OHLC data
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate body size (absolute difference between open and close)
    body_size = abs(close - open_price)
    
    # Calculate shadow size (high - low)
    shadow_size = high - low
    
    # Calculate body to shadow ratio
    body_shadow_ratio = body_size / shadow_size
    
    # Doji has a very small body compared to its shadow
    doji = (body_shadow_ratio < tolerance) & (shadow_size > 0)
    
    # Return as integer (1 for doji, 0 otherwise)
    return doji.astype(int)

def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect hammer candlestick patterns.
    
    A hammer has a small body at the top and a long lower shadow.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Series with 1 for hammer detection, 0 otherwise
    """
    # Get OHLC data
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate body size
    body_size = abs(close - open_price)
    
    # Calculate upper shadow
    upper_shadow = high - np.maximum(open_price, close)
    
    # Calculate lower shadow
    lower_shadow = np.minimum(open_price, close) - low
    
    # Hammer criteria:
    # 1. Lower shadow is at least 2 times the body size
    # 2. Upper shadow is small (less than half the body size)
    # 3. Body is in the upper part of the candle
    hammer = (
        (lower_shadow >= 2 * body_size) &  # Long lower shadow
        (upper_shadow <= 0.5 * body_size) &  # Small upper shadow
        (body_size > 0)  # Ensure we have a body
    )
    
    # Return as integer (1 for hammer, 0 otherwise)
    return hammer.astype(int)

def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish engulfing patterns.
    
    An engulfing pattern is a two-candle pattern where the second candle completely
    engulfs the body of the first candle.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with 'bullish_engulfing' and 'bearish_engulfing' columns
    """
    # Get OHLC data
    open_price = df['Open']
    close = df['Close']
    
    # Calculate previous candle open and close
    prev_open = open_price.shift(1)
    prev_close = close.shift(1)
    
    # Calculate current and previous candle direction
    current_bullish = close > open_price
    prev_bullish = prev_close > prev_open
    
    # Bullish engulfing:
    # 1. Previous candle is bearish (close < open)
    # 2. Current candle is bullish (close > open)
    # 3. Current open is <= previous close
    # 4. Current close is >= previous open
    bullish_engulfing = (
        (~prev_bullish) &  # Previous candle is bearish
        current_bullish &  # Current candle is bullish
        (open_price <= prev_close) &  # Current open <= previous close
        (close >= prev_open)  # Current close >= previous open
    )
    
    # Bearish engulfing:
    # 1. Previous candle is bullish (close > open)
    # 2. Current candle is bearish (close < open)
    # 3. Current open is >= previous close
    # 4. Current close is <= previous open
    bearish_engulfing = (
        prev_bullish &  # Previous candle is bullish
        (~current_bullish) &  # Current candle is bearish
        (open_price >= prev_close) &  # Current open >= previous close
        (close <= prev_open)  # Current close <= previous open
    )
    
    # Create result DataFrame
    result = pd.DataFrame({
        'bullish_engulfing': bullish_engulfing.astype(int),
        'bearish_engulfing': bearish_engulfing.astype(int)
    }, index=df.index)
    
    return result

def detect_morningstar(df: pd.DataFrame) -> pd.Series:
    """Detect morning star patterns.
    
    A morning star is a three-candle bullish reversal pattern:
    1. First candle is a large bearish candle
    2. Second candle is a small-bodied candle (like a doji) that gaps down
    3. Third candle is a large bullish candle that gaps up and closes above the midpoint of the first candle
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Series with 1 for morning star detection, 0 otherwise
    """
    # Get OHLC data
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate body size
    body_size = abs(close - open_price)
    
    # Calculate previous and second previous candle data
    prev_open = open_price.shift(1)
    prev_close = close.shift(1)
    prev_body_size = abs(prev_close - prev_open)
    
    prev2_open = open_price.shift(2)
    prev2_close = close.shift(2)
    prev2_body_size = abs(prev2_close - prev2_open)
    
    # Identify candle directions
    current_bullish = close > open_price
    prev_bullish = prev_close > prev_open
    prev2_bullish = prev2_close > prev2_open
    
    # Morning star criteria:
    # 1. First candle (prev2) is bearish and has a large body
    # 2. Second candle (prev) has a small body
    # 3. Third candle (current) is bullish and has a large body
    # 4. There is a gap down between first and second candles
    # 5. There is a gap up between second and third candles
    # 6. Third candle closes above midpoint of first candle
    
    morning_star = (
        (~prev2_bullish) &  # First candle is bearish
        (prev2_body_size > np.mean(body_size.rolling(10).mean())) &  # First candle has large body
        (prev_body_size < 0.5 * np.mean(body_size.rolling(10).mean())) &  # Second candle has small body
        current_bullish &  # Current candle is bullish
        (body_size > np.mean(body_size.rolling(10).mean())) &  # Current candle has large body
        (close > (prev2_open + (prev2_close - prev2_open) / 2))  # Current close above midpoint of first candle
    )
    
    # Return as integer (1 for morning star, 0 otherwise)
    return morning_star.astype(int)

def apply_pattern_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all pattern detection functions to a DataFrame with OHLC data.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Original DataFrame with pattern detection columns added
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Detect patterns
    result['doji'] = detect_doji(df)
    result['hammer'] = detect_hammer(df)
    
    # Add engulfing patterns
    engulfing = detect_engulfing(df)
    result['bullish_engulfing'] = engulfing['bullish_engulfing']
    result['bearish_engulfing'] = engulfing['bearish_engulfing']
    
    # Add morning star pattern
    result['morning_star'] = detect_morningstar(df)
    
    return result