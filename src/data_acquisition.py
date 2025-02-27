"""Data acquisition module for Stock Potential Identifier.

This module handles fetching market data from various free sources with
fallback mechanisms, caching, and rate limiting.
"""

import os
import logging
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

from .utils import create_directory, load_config

logger = logging.getLogger(__name__)

class DataAcquisition:
    """Handles all data acquisition with caching and rate limiting."""

    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = load_config(config_path)
        self.cache_dir = self.config['data_acquisition']['cache_dir']
        create_directory(self.cache_dir)
        
        # Initialize API rate limiters
        self.api_limiters = {}
        self._setup_rate_limiters()
        
        # Request sessions
        self.session = None  # Will be initialized in start method
    
    async def start(self):
        """Start the data acquisition service."""
        self.session = aiohttp.ClientSession()
        logger.info("Data acquisition service started")
    
    async def stop(self):
        """Stop the data acquisition service."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Data acquisition service stopped")
    
    def _setup_rate_limiters(self):
        """Set up API rate limiters based on config."""
        apis = self.config['data_acquisition']['apis']
        
        for api_name, api_config in apis.items():
            if api_config.get('enabled', False):
                if 'calls_per_minute' in api_config:
                    self.api_limiters[api_name] = {
                        'calls_per_minute': api_config['calls_per_minute'],
                        'calls_per_day': api_config.get('calls_per_day', float('inf')),
                        'last_call': 0,
                        'calls_today': 0,
                        'day_start': datetime.now().date()
                    }
    
    async def _wait_for_rate_limit(self, api_name: str):
        """Wait if necessary to respect API rate limits."""
        if api_name not in self.api_limiters:
            return
        
        limiter = self.api_limiters[api_name]
        current_time = time.time()
        current_date = datetime.now().date()
        
        # Reset daily counter if it's a new day
        if current_date > limiter['day_start']:
            limiter['day_start'] = current_date
            limiter['calls_today'] = 0
        
        # Check daily limit
        if limiter['calls_today'] >= limiter['calls_per_day']:
            logger.warning(f"Daily limit reached for {api_name}. Waiting until next day.")
            # Wait until midnight
            tomorrow = datetime.combine(current_date + timedelta(days=1), datetime.min.time())
            seconds_until_midnight = (tomorrow - datetime.now()).total_seconds()
            await asyncio.sleep(seconds_until_midnight + 1)
            limiter['day_start'] = datetime.now().date()
            limiter['calls_today'] = 0
        
        # Check rate limit (per minute)
        seconds_since_last_call = current_time - limiter['last_call']
        seconds_per_call = 60 / limiter['calls_per_minute']
        
        if seconds_since_last_call < seconds_per_call:
            wait_time = seconds_per_call - seconds_since_last_call
            logger.debug(f"Rate limiting {api_name}, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Update call counters
        limiter['last_call'] = time.time()
        limiter['calls_today'] += 1
    
    def _get_cache_file_path(self, symbol: str, data_type: str, interval: str = '1d', period: str = '2y') -> str:
        """Get the path to the cache file for the specified data."""
        # Use CSV instead of parquet to avoid dependency issues
        filename = f"{symbol}_{data_type}_{interval}_{period}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if the cache file is still valid based on config."""
        if not os.path.exists(cache_file):
            return False
        
        cache_expiry_days = self.config['data_acquisition']['cache_expiry_days']
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_age_days = (datetime.now() - file_modified_time).days
        
        return cache_age_days < cache_expiry_days
    
    async def get_market_data(self, symbol: str, interval: str = '1d', period: str = '2y') -> pd.DataFrame:
        """Get market data with caching to avoid API limits.
        
        Args:
            symbol: The stock symbol
            interval: Data interval ('1d', '1h', etc.)
            period: Time period to fetch ('2y', '5y', etc.)
            
        Returns:
            DataFrame with market data
        """
        cache_file = self._get_cache_file_path(symbol, 'market', interval, period)
        
        # Check if we have valid cached data
        if self._is_cache_valid(cache_file):
            try:
                # Load from CSV cache
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.debug(f"Using cached market data for {symbol}")
                return data
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}")
        
        # If no valid cache, fetch fresh data
        logger.info(f"Fetching fresh market data for {symbol}")
        
        # Try different data sources with fallback
        data = None
        
        # Try yfinance first if enabled
        if self.config['data_acquisition']['apis']['yfinance']['enabled']:
            try:
                # Use ThreadPoolExecutor for yfinance which doesn't support asyncio natively
                with ThreadPoolExecutor() as executor:
                    data = await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        lambda: yf.download(symbol, period=period, interval=interval, progress=False)
                    )
                
                if not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from yfinance: {e}")
        
        # Fallback to Alpha Vantage if enabled
        if self.config['data_acquisition']['apis']['alpha_vantage']['enabled']:
            try:
                await self._wait_for_rate_limit('alpha_vantage')
                data = await self._fetch_from_alpha_vantage(symbol, interval)
                
                if data is not None and not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from Alpha Vantage: {e}")
        
        # Fallback to FMP if enabled
        if self.config['data_acquisition']['apis']['financial_modeling_prep']['enabled']:
            try:
                await self._wait_for_rate_limit('financial_modeling_prep')
                data = await self._fetch_from_fmp(symbol, interval)
                
                if data is not None and not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from FMP: {e}")
        
        if data is None or data.empty:
            logger.error(f"Failed to fetch market data for {symbol} from all sources")
            return pd.DataFrame()
        
        return data
    
    async def _fetch_from_alpha_vantage(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        api_key = os.environ.get(self.config['data_acquisition']['apis']['alpha_vantage']['api_key_env'])
        if not api_key:
            logger.warning("Alpha Vantage API key not found in environment variables")
            return None
        
        base_url = self.config['data_acquisition']['apis']['alpha_vantage']['base_url']
        
        # Map interval to Alpha Vantage interval
        av_interval = '1day'
        if interval == '1h':
            av_interval = '60min'
        elif interval == '15m':
            av_interval = '15min'
        
        # Determine function based on interval
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        if av_interval in ['60min', '15min']:
            function = 'TIME_SERIES_INTRADAY'
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        if function == 'TIME_SERIES_INTRADAY':
            params['interval'] = av_interval
        
        try:
            async with self.session.get(base_url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                # Extract and process the time series data
                time_series_key = [k for k in data.keys() if 'Time Series' in k]
                if not time_series_key:
                    return None
                
                time_series = data[time_series_key[0]]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                
                # Rename columns
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                
                # Convert index to datetime
                df.index = pd.to_datetime(df.index)
                
                # Convert columns to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Sort by date
                df = df.sort_index()
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")
            return None
    
    async def _fetch_from_fmp(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch data from Financial Modeling Prep API."""
        api_key = os.environ.get(self.config['data_acquisition']['apis']['financial_modeling_prep']['api_key_env'])
        if not api_key:
            logger.warning("FMP API key not found in environment variables")
            return None
        
        base_url = self.config['data_acquisition']['apis']['financial_modeling_prep']['base_url']
        
        # Determine endpoint based on interval
        endpoint = 'v3/historical-price-full'
        
        url = f"{base_url}/{endpoint}/{symbol}"
        params = {'apikey': api_key}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if 'historical' not in data:
                    return None
                
                # Extract historical data
                historical = data['historical']
                df = pd.DataFrame(historical)
                
                # Rename columns
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Set date as index
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                
                # Sort by date
                df = df.sort_index()
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching from FMP: {e}")
            return None
        
    async def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Get fundamental data with multiple source fallbacks.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            DataFrame with fundamental data
        """
        cache_file = self._get_cache_file_path(symbol, 'fundamental')
        
        # Check if we have valid cached data
        if self._is_cache_valid(cache_file):
            try:
                # Load from CSV cache
                data = pd.read_csv(cache_file, index_col=0)
                logger.debug(f"Using cached fundamental data for {symbol}")
                return data
            except Exception as e:
                logger.warning(f"Failed to read fundamental cache for {symbol}: {e}")
        
        # If no valid cache, fetch fresh data
        logger.info(f"Fetching fresh fundamental data for {symbol}")
        
        # Try different data sources with fallback
        data = None
        
        # Try FMP first if enabled
        if self.config['data_acquisition']['apis']['financial_modeling_prep']['enabled']:
            try:
                await self._wait_for_rate_limit('financial_modeling_prep')
                data = await self._fetch_fundamentals_from_fmp(symbol)
                
                if data is not None and not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol} from FMP: {e}")
        
        # Fallback to Alpha Vantage if enabled
        if self.config['data_acquisition']['apis']['alpha_vantage']['enabled']:
            try:
                await self._wait_for_rate_limit('alpha_vantage')
                data = await self._fetch_fundamentals_from_alpha_vantage(symbol)
                
                if data is not None and not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol} from Alpha Vantage: {e}")
        
        # Last resort: scrape from SEC EDGAR if enabled
        if self.config['data_acquisition']['apis']['sec_edgar']['enabled']:
            try:
                data = await self._fetch_fundamentals_from_sec(symbol)
                
                if data is not None and not data.empty:
                    # Save to CSV cache
                    data.to_csv(cache_file)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol} from SEC EDGAR: {e}")
        
        if data is None or data.empty:
            logger.error(f"Failed to fetch fundamental data for {symbol} from all sources")
            return pd.DataFrame()
        
        return data
    
    async def _fetch_fundamentals_from_fmp(self, symbol: str) -> pd.DataFrame:
        """Fetch fundamental data from Financial Modeling Prep API."""
        api_key = os.environ.get(self.config['data_acquisition']['apis']['financial_modeling_prep']['api_key_env'])
        if not api_key:
            logger.warning("FMP API key not found in environment variables")
            return None
        
        base_url = self.config['data_acquisition']['apis']['financial_modeling_prep']['base_url']
        
        # Get income statement
        income_statement_url = f"{base_url}/v3/income-statement/{symbol}"
        income_params = {'apikey': api_key, 'limit': 10}
        
        # Get balance sheet
        balance_sheet_url = f"{base_url}/v3/balance-sheet-statement/{symbol}"
        balance_params = {'apikey': api_key, 'limit': 10}
        
        # Get cash flow statement
        cash_flow_url = f"{base_url}/v3/cash-flow-statement/{symbol}"
        cash_flow_params = {'apikey': api_key, 'limit': 10}
        
        # Get ratios
        ratios_url = f"{base_url}/v3/ratios/{symbol}"
        ratios_params = {'apikey': api_key, 'limit': 10}
        
        # Get key metrics
        metrics_url = f"{base_url}/v3/key-metrics/{symbol}"
        metrics_params = {'apikey': api_key, 'limit': 10}
        
        try:
            # Fetch all data concurrently
            income_task = self.session.get(income_statement_url, params=income_params)
            balance_task = self.session.get(balance_sheet_url, params=balance_params)
            cash_flow_task = self.session.get(cash_flow_url, params=cash_flow_params)
            ratios_task = self.session.get(ratios_url, params=ratios_params)
            metrics_task = self.session.get(metrics_url, params=metrics_params)
            
            responses = await asyncio.gather(
                income_task, balance_task, cash_flow_task, ratios_task, metrics_task,
                return_exceptions=True
            )
            
            income_data, balance_data, cash_flow_data, ratios_data, metrics_data = [], [], [], [], []
            
            # Process income statement
            if not isinstance(responses[0], Exception) and responses[0].status == 200:
                income_data = await responses[0].json()
            
            # Process balance sheet
            if not isinstance(responses[1], Exception) and responses[1].status == 200:
                balance_data = await responses[1].json()
            
            # Process cash flow
            if not isinstance(responses[2], Exception) and responses[2].status == 200:
                cash_flow_data = await responses[2].json()
            
            # Process ratios
            if not isinstance(responses[3], Exception) and responses[3].status == 200:
                ratios_data = await responses[3].json()
            
            # Process metrics
            if not isinstance(responses[4], Exception) and responses[4].status == 200:
                metrics_data = await responses[4].json()
            
            # Process and combine all data
            result = {}
            
            # Process data into a usable format
            # This is simplified, in a real implementation you'd want to carefully
            # merge these datasets with proper date alignment
            
            # Extract the most recent data point from each dataset
            if income_data and isinstance(income_data, list) and len(income_data) > 0:
                latest_income = income_data[0]
                result.update({
                    'revenue': latest_income.get('revenue'),
                    'gross_profit': latest_income.get('grossProfit'),
                    'net_income': latest_income.get('netIncome'),
                    'ebitda': latest_income.get('ebitda'),
                    'income_date': latest_income.get('date')
                })
            
            if balance_data and isinstance(balance_data, list) and len(balance_data) > 0:
                latest_balance = balance_data[0]
                result.update({
                    'total_assets': latest_balance.get('totalAssets'),
                    'total_liabilities': latest_balance.get('totalLiabilities'),
                    'total_equity': latest_balance.get('totalEquity'),
                    'cash_and_equivalents': latest_balance.get('cashAndCashEquivalents'),
                    'debt': latest_balance.get('totalDebt'),
                    'balance_date': latest_balance.get('date')
                })
            
            if cash_flow_data and isinstance(cash_flow_data, list) and len(cash_flow_data) > 0:
                latest_cash_flow = cash_flow_data[0]
                result.update({
                    'operating_cash_flow': latest_cash_flow.get('operatingCashFlow'),
                    'capital_expenditure': latest_cash_flow.get('capitalExpenditure'),
                    'free_cash_flow': latest_cash_flow.get('freeCashFlow'),
                    'cash_flow_date': latest_cash_flow.get('date')
                })
            
            if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0:
                latest_ratios = ratios_data[0]
                result.update({
                    'pe_ratio': latest_ratios.get('priceEarningsRatio'),
                    'price_to_book': latest_ratios.get('priceToBookRatio'),
                    'debt_to_equity': latest_ratios.get('debtToEquity'),
                    'roe': latest_ratios.get('returnOnEquity'),
                    'roa': latest_ratios.get('returnOnAssets'),
                    'ratios_date': latest_ratios.get('date')
                })
            
            if metrics_data and isinstance(metrics_data, list) and len(metrics_data) > 0:
                latest_metrics = metrics_data[0]
                result.update({
                    'market_cap': latest_metrics.get('marketCap'),
                    'ev_to_ebitda': latest_metrics.get('enterpriseValueOverEBITDA'),
                    'peg_ratio': latest_metrics.get('pegRatio'),
                    'dividend_yield': latest_metrics.get('dividendYield'),
                    'metrics_date': latest_metrics.get('date')
                })
            
            # Convert to DataFrame
            df = pd.DataFrame([result])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals from FMP: {e}")
            return None
    
    async def _fetch_fundamentals_from_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Fetch fundamental data from Alpha Vantage API."""
        api_key = os.environ.get(self.config['data_acquisition']['apis']['alpha_vantage']['api_key_env'])
        if not api_key:
            logger.warning("Alpha Vantage API key not found in environment variables")
            return None
        
        base_url = self.config['data_acquisition']['apis']['alpha_vantage']['base_url']
        
        # Get overview
        overview_params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': api_key
        }
        
        # Get income statement
        income_params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol,
            'apikey': api_key
        }
        
        # Get balance sheet
        balance_params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol,
            'apikey': api_key
        }
        
        # Get cash flow
        cash_flow_params = {
            'function': 'CASH_FLOW',
            'symbol': symbol,
            'apikey': api_key
        }
        
        try:
            # Fetch overview
            await self._wait_for_rate_limit('alpha_vantage')
            async with self.session.get(base_url, params=overview_params) as overview_response:
                if overview_response.status != 200:
                    return None
                
                overview_data = await overview_response.json()
            
            # Fetch income statement
            await self._wait_for_rate_limit('alpha_vantage')
            async with self.session.get(base_url, params=income_params) as income_response:
                if income_response.status != 200:
                    return None
                
                income_data = await income_response.json()
            
            # Fetch balance sheet
            await self._wait_for_rate_limit('alpha_vantage')
            async with self.session.get(base_url, params=balance_params) as balance_response:
                if balance_response.status != 200:
                    return None
                
                balance_data = await balance_response.json()
            
            # Fetch cash flow
            await self._wait_for_rate_limit('alpha_vantage')
            async with self.session.get(base_url, params=cash_flow_params) as cash_flow_response:
                if cash_flow_response.status != 200:
                    return None
                
                cash_flow_data = await cash_flow_response.json()
            
            # Process and combine data
            result = {}
            
            # Process overview data
            result.update({
                'market_cap': float(overview_data.get('MarketCapitalization', 0)),
                'pe_ratio': float(overview_data.get('PERatio', 0)),
                'peg_ratio': float(overview_data.get('PEGRatio', 0)),
                'dividend_yield': float(overview_data.get('DividendYield', 0)),
                'eps': float(overview_data.get('EPS', 0)),
                'beta': float(overview_data.get('Beta', 0)),
                'price_to_book': float(overview_data.get('PriceToBookRatio', 0)),
                'profit_margin': float(overview_data.get('ProfitMargin', 0)),
                'roe': float(overview_data.get('ReturnOnEquityTTM', 0)),
                'roa': float(overview_data.get('ReturnOnAssetsTTM', 0)),
                'overview_date': datetime.now().strftime('%Y-%m-%d')  # No date in overview
            })
            
            # Process income statement (latest annual)
            if 'annualReports' in income_data and len(income_data['annualReports']) > 0:
                latest_income = income_data['annualReports'][0]
                result.update({
                    'revenue': float(latest_income.get('totalRevenue', 0)),
                    'gross_profit': float(latest_income.get('grossProfit', 0)),
                    'net_income': float(latest_income.get('netIncome', 0)),
                    'ebitda': float(latest_income.get('ebitda', 0)),
                    'income_date': latest_income.get('fiscalDateEnding')
                })
            
            # Process balance sheet (latest annual)
            if 'annualReports' in balance_data and len(balance_data['annualReports']) > 0:
                latest_balance = balance_data['annualReports'][0]
                result.update({
                    'total_assets': float(latest_balance.get('totalAssets', 0)),
                    'total_liabilities': float(latest_balance.get('totalLiabilities', 0)),
                    'total_equity': float(latest_balance.get('totalShareholderEquity', 0)),
                    'cash_and_equivalents': float(latest_balance.get('cashAndCashEquivalentsAtCarryingValue', 0)),
                    'debt': float(latest_balance.get('shortLongTermDebtTotal', 0)),
                    'balance_date': latest_balance.get('fiscalDateEnding')
                })
            
            # Process cash flow (latest annual)
            if 'annualReports' in cash_flow_data and len(cash_flow_data['annualReports']) > 0:
                latest_cash_flow = cash_flow_data['annualReports'][0]
                result.update({
                    'operating_cash_flow': float(latest_cash_flow.get('operatingCashflow', 0)),
                    'capital_expenditure': float(latest_cash_flow.get('capitalExpenditures', 0)),
                    'free_cash_flow': float(latest_cash_flow.get('operatingCashflow', 0)) - 
                                      float(latest_cash_flow.get('capitalExpenditures', 0)),
                    'cash_flow_date': latest_cash_flow.get('fiscalDateEnding')
                })
            
            # Calculate additional ratios
            if result.get('total_equity', 0) > 0 and result.get('debt', 0) > 0:
                result['debt_to_equity'] = result['debt'] / result['total_equity']
            
            # Convert to DataFrame
            df = pd.DataFrame([result])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals from Alpha Vantage: {e}")
            return None
    
    async def _fetch_fundamentals_from_sec(self, symbol: str) -> pd.DataFrame:
        """Fetch fundamental data from SEC EDGAR (simplified implementation)."""
        # This is a placeholder for a real SEC EDGAR scraper
        # In a production system, you'd implement a proper SEC filing parser
        # which is complex and beyond the scope of this example
        
        logger.warning("SEC EDGAR scraping not fully implemented")
        return None
    
    async def get_alternative_data(self, symbol: str) -> pd.DataFrame:
        """Get alternative data like insider trading, news sentiment.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            DataFrame with alternative data
        """
        # This would implement fetching alternative data from free sources
        # For now we'll return a placeholder DataFrame with basic insider data
        
        # Check if insider activity is enabled
        if not self.config['feature_engineering']['alternative']['insider_activity']['enabled']:
            return pd.DataFrame()
        
        cache_file = self._get_cache_file_path(symbol, 'alternative')
        
        # Check if we have valid cached data
        if self._is_cache_valid(cache_file):
            try:
                # Load from CSV cache
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.debug(f"Using cached alternative data for {symbol}")
                return data
            except Exception as e:
                logger.warning(f"Failed to read alternative data cache for {symbol}: {e}")
        
        # If no valid cache, fetch fresh data
        logger.info(f"Fetching fresh alternative data for {symbol}")
        
        # For now, use SEC EDGAR for insider trading (if enabled)
        if self.config['data_acquisition']['apis']['sec_edgar']['enabled']:
            try:
                insider_data = await self._fetch_insider_trading(symbol)
                
                if insider_data is not None and not insider_data.empty:
                    # Save to CSV cache
                    insider_data.to_csv(cache_file)
                    return insider_data
            except Exception as e:
                logger.warning(f"Failed to fetch insider trading for {symbol}: {e}")
        
        return pd.DataFrame()
    
    async def _fetch_insider_trading(self, symbol: str) -> pd.DataFrame:
        """Fetch insider trading data from SEC (simplified implementation)."""
        # This is a placeholder for a real SEC Form 4 parser
        # In a production system, you'd implement a proper SEC Form 4 parser
        
        # For now, create a mock dataset with random insider transactions
        # This would be replaced with real data in a production system
        np.random.seed(42)  # For reproducibility
        
        dates = pd.date_range(end=datetime.now(), periods=10, freq='M')
        
        data = {
            'date': dates,
            'insider_name': [f"Insider {i}" for i in range(1, 11)],
            'relationship': np.random.choice(['CEO', 'CFO', 'Director', 'VP'], size=10),
            'transaction_type': np.random.choice(['Buy', 'Sell'], size=10),
            'shares': np.random.randint(1000, 10000, size=10),
            'price': np.random.uniform(50, 200, size=10),
            'total_value': np.random.uniform(50000, 2000000, size=10)
        }
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df
    
    async def get_stock_universe(self) -> List[str]:
        """Get the configured stock universe.
        
        Returns:
            List of stock symbols
        """
        # Get universe configuration
        universe_config = self.config['stock_universe']
        max_stocks = universe_config['max_stocks']
        
        # Check if a custom universe file is specified
        custom_path = universe_config.get('custom_universe_path')
        if custom_path and os.path.exists(custom_path):
            try:
                custom_df = pd.read_csv(custom_path)
                if 'symbol' in custom_df.columns:
                    symbols = custom_df['symbol'].tolist()
                    return symbols[:max_stocks]
            except Exception as e:
                logger.error(f"Failed to load custom universe from {custom_path}: {e}")
        
        # Fall back to default stocks
        default_stocks = universe_config['default_stocks']
        return default_stocks[:max_stocks]