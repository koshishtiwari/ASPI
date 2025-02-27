"""Unit tests for data acquisition module."""

import os
import sys
import unittest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition import DataAcquisition

class TestDataAcquisition(unittest.TestCase):
    """Test data acquisition functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a minimal test config
        self.test_config = {
            'data_acquisition': {
                'cache_dir': 'tests/test_data',
                'cache_expiry_days': 1,
                'apis': {
                    'yfinance': {'enabled': True},
                    'alpha_vantage': {'enabled': False},
                    'financial_modeling_prep': {'enabled': False},
                    'sec_edgar': {'enabled': False}
                }
            },
            'stock_universe': {
                'default_stocks': ['AAPL', 'MSFT'],
                'max_stocks': 5
            }
        }
        
        # Create test directory
        os.makedirs('tests/test_data', exist_ok=True)
        
        # Set up event loop
        self.loop = asyncio.get_event_loop()
        
        # Create data acquisition with mock config
        with patch('src.data_acquisition.load_config', return_value=self.test_config):
            self.data_engine = DataAcquisition()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test files
        import shutil
        if os.path.exists('tests/test_data'):
            shutil.rmtree('tests/test_data')
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.data_engine.cache_dir, 'tests/test_data')
        self.assertEqual(self.data_engine.config['data_acquisition']['cache_expiry_days'], 1)
    
    def test_get_cache_file_path(self):
        """Test cache file path generation."""
        cache_path = self.data_engine._get_cache_file_path('AAPL', 'market', '1d', '2y')
        expected_path = os.path.join('tests/test_data', 'AAPL_market_1d_2y.parquet')
        self.assertEqual(cache_path, expected_path)
    
    def test_is_cache_valid(self):
        """Test cache validity check."""
        # Create a mock cache file
        test_file = os.path.join('tests/test_data', 'test_cache.parquet')
        pd.DataFrame().to_parquet(test_file)
        
        # Should be valid (created just now)
        self.assertTrue(self.data_engine._is_cache_valid(test_file))
        
        # Modify the file time to be older than cache_expiry_days
        old_time = datetime.now() - timedelta(days=2)
        os.utime(test_file, (old_time.timestamp(), old_time.timestamp()))
        
        # Should be invalid now
        self.assertFalse(self.data_engine._is_cache_valid(test_file))
    
    @patch('yfinance.download')
    def test_get_market_data(self, mock_download):
        """Test market data acquisition."""
        # Mock yfinance download
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        mock_download.return_value = mock_data
        
        # Run the async function using the event loop
        async def run_get_market_data():
            await self.data_engine.start()
            data = await self.data_engine.get_market_data('AAPL')
            await self.data_engine.stop()
            return data
        
        result = self.loop.run_until_complete(run_get_market_data())
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue('Close' in result.columns)
        
        # Verify yfinance was called correctly
        mock_download.assert_called_once_with('AAPL', period='2y', interval='1d', progress=False)
    
    def test_get_stock_universe(self):
        """Test stock universe retrieval."""
        # Run the async function
        async def run_get_stock_universe():
            return await self.data_engine.get_stock_universe()
        
        result = self.loop.run_until_complete(run_get_stock_universe())
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'AAPL')
        self.assertEqual(result[1], 'MSFT')

if __name__ == '__main__':
    unittest.main()