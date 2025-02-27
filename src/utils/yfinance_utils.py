"""Improved functions for more robust data acquisition with retries."""

import time
import random
import logging
import pandas as pd
import yfinance as yf
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def download_with_retry(symbol: str, period: str = "2y", interval: str = "1d", 
                      max_retries: int = 5, initial_delay: float = 2.0) -> Optional[pd.DataFrame]:
    """Download stock data with exponential backoff retry logic.
    
    Args:
        symbol: Stock symbol
        period: Time period to fetch
        interval: Data interval
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying
        
    Returns:
        DataFrame with market data or None if all retries fail
    """
    retry_count = 0
    delay = initial_delay
    
    while retry_count < max_retries:
        try:
            # Add random jitter to avoid API rate limits
            jitter = random.uniform(0.1, 0.5)
            
            if retry_count > 0:
                logger.info(f"Retry {retry_count}/{max_retries} for {symbol} after {delay:.2f}s delay")
                time.sleep(delay + jitter)
            
            # Download data
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Check if data is empty or only has NaN values
            if data.empty or data.isnull().all().all():
                logger.warning(f"No data returned for {symbol}")
                retry_count += 1
                delay *= 2  # Exponential backoff
                continue
                
            return data
            
        except Exception as e:
            logger.warning(f"Error downloading {symbol}: {str(e)}")
            retry_count += 1
            delay *= 2  # Exponential backoff
    
    logger.error(f"Failed to download {symbol} after {max_retries} retries")
    return None

def batch_download_with_retry(symbols: list, period: str = "2y", interval: str = "1d", 
                            max_workers: int = 5) -> dict:
    """Download data for multiple symbols with retry logic and parallel execution.
    
    Args:
        symbols: List of stock symbols
        period: Time period to fetch
        interval: Data interval
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary of symbol -> market data DataFrame
    """
    results = {}
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = {
            executor.submit(download_with_retry, symbol, period, interval): symbol
            for symbol in symbols
        }
        
        # Process results as they complete
        for future in futures:
            symbol = futures[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    results[symbol] = data
                    logger.info(f"Successfully downloaded data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
    
    logger.info(f"Successfully downloaded data for {len(results)}/{len(symbols)} symbols")
    return results