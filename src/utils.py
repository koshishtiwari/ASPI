"""Utility functions for Stock Potential Identifier."""

import os
import logging
import yaml
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file, uses CONFIG_PATH env var if None
        
    Returns:
        Config dictionary
    """
    # Get config path from environment if not provided
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        # Return a default minimal config
        return {
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'reports_dir': 'reports',
            },
            'data_acquisition': {
                'cache_dir': 'data/market_data',
                'cache_expiry_days': 1,
                'apis': {
                    'yfinance': {'enabled': True}
                }
            },
            'stock_universe': {
                'default_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'max_stocks': 10
            },
            'feature_engineering': {
                'technical': {
                    'momentum_indicators': ['rsi', 'macd'],
                    'volatility_indicators': ['bbands', 'atr'],
                    'volume_indicators': ['obv'],
                    'trend_indicators': ['sma', 'ema']
                },
                'fundamental': {'enabled': False},
                'alternative': {'insider_activity': {'enabled': False}}
            },
            'models': {
                'time_horizons': {
                    'short_term': {'days': 5, 'training_frequency_days': 7},
                    'medium_term': {'days': 21, 'training_frequency_days': 14},
                    'long_term': {'days': 63, 'training_frequency_days': 30}
                }
            },
            'execution': {
                'check_frequency_minutes': 60,
                'report_output_formats': ['json']
            }
        }

def create_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to create
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Created directory: {directory_path}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")

def save_model(model: Any, filename: str, directory: Optional[str] = None) -> str:
    """Save model to disk.
    
    Args:
        model: Model object to save
        filename: Filename to save as
        directory: Directory to save in, uses MODELS_DIR env var if None
        
    Returns:
        Full path to saved model
    """
    if directory is None:
        directory = os.environ.get('MODELS_DIR', 'models')
    
    create_directory(directory)
    
    if not filename.endswith('.joblib'):
        filename += '.joblib'
    
    full_path = os.path.join(directory, filename)
    
    try:
        joblib.dump(model, full_path)
        logger.info(f"Saved model to {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Failed to save model to {full_path}: {e}")
        return ""

def load_model(filename: str, directory: Optional[str] = None) -> Any:
    """Load model from disk.
    
    Args:
        filename: Filename to load
        directory: Directory to load from, uses MODELS_DIR env var if None
        
    Returns:
        Loaded model object
    """
    if directory is None:
        directory = os.environ.get('MODELS_DIR', 'models')
    
    if not filename.endswith('.joblib'):
        filename += '.joblib'
    
    full_path = os.path.join(directory, filename)
    
    try:
        model = joblib.load(full_path)
        logger.info(f"Loaded model from {full_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {full_path}: {e}")
        return None

def save_report(data: Any, filename: str, directory: Optional[str] = None, format: str = 'json') -> str:
    """Save report to disk in specified format.
    
    Args:
        data: Data to save (DataFrame, dict, etc.)
        filename: Filename to save as
        directory: Directory to save in, uses REPORTS_DIR env var if None
        format: Format to save as ('json', 'csv', 'html', 'pickle')
        
    Returns:
        Full path to saved report
    """
    if directory is None:
        directory = os.environ.get('REPORTS_DIR', 'reports')
    
    create_directory(directory)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(filename)[0]
    filename = f"{base_name}_{timestamp}"
    
    # Add extension based on format
    if format == 'json':
        if not filename.endswith('.json'):
            filename += '.json'
    elif format == 'csv':
        if not filename.endswith('.csv'):
            filename += '.csv'
    elif format == 'html':
        if not filename.endswith('.html'):
            filename += '.html'
    elif format == 'pickle':
        if not filename.endswith('.pkl'):
            filename += '.pkl'
    
    full_path = os.path.join(directory, filename)
    
    try:
        if format == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(full_path, orient='records', date_format='iso')
            else:
                import json
                with open(full_path, 'w') as f:
                    json.dump(data, f, default=str, indent=2)
        
        elif format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(full_path, index=True)
            else:
                logger.warning(f"Data is not a DataFrame, converting to DataFrame first")
                pd.DataFrame(data).to_csv(full_path, index=True)
        
        elif format == 'html':
            if isinstance(data, pd.DataFrame):
                data.to_html(full_path)
            else:
                logger.warning(f"Data is not a DataFrame, converting to DataFrame first")
                pd.DataFrame(data).to_html(full_path)
        
        elif format == 'pickle':
            if isinstance(data, pd.DataFrame):
                data.to_pickle(full_path)
            else:
                import pickle
                with open(full_path, 'wb') as f:
                    pickle.dump(data, f)
        
        logger.info(f"Saved report to {full_path}")
        return full_path
    
    except Exception as e:
        logger.error(f"Failed to save report to {full_path}: {e}")
        return ""

def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20, 
                           title: str = "Feature Importance", 
                           filename: Optional[str] = None) -> plt.Figure:
    """Plot feature importance.
    
    Args:
        feature_importance: DataFrame with columns 'feature' and 'importance'
        top_n: Number of top features to show
        title: Plot title
        filename: If provided, save plot to this filename
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by importance and get top N
    df = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Create bar plot
    ax = sns.barplot(x='importance', y='feature', data=df)
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    
    # Add value labels
    for i, v in enumerate(df['importance']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        directory = os.environ.get('REPORTS_DIR', 'reports')
        create_directory(directory)
        
        if not filename.endswith(('.png', '.jpg', '.pdf')):
            filename += '.png'
        
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {full_path}")
    
    return plt.gcf()

def plot_model_performance(predictions: pd.DataFrame, actual: pd.Series, model_name: str,
                          title: str = "Model Performance", filename: Optional[str] = None) -> plt.Figure:
    """Plot model performance.
    
    Args:
        predictions: DataFrame with prediction columns
        actual: Series with actual values
        model_name: Name of the model to highlight
        title: Plot title
        filename: If provided, save plot to this filename
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    
    # Plot actual values
    ax1.plot(actual.index, actual, label='Actual', linewidth=2)
    
    # Plot predictions if available
    if model_name in predictions.columns:
        ax1.plot(predictions.index, predictions[model_name], label=f'Predicted ({model_name})', 
                linewidth=2, alpha=0.7)
    
    # Add other models if available
    for col in predictions.columns:
        if col != model_name:
            ax1.plot(predictions.index, predictions[col], label=f'Predicted ({col})', 
                    linewidth=1, alpha=0.4)
    
    ax1.set_title(f"{title} - Actual vs Predicted", fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    ax2 = axes[1]
    
    if model_name in predictions.columns:
        # Calculate errors
        error = actual - predictions[model_name]
        
        # Plot error distribution
        sns.histplot(error, kde=True, ax=ax2)
        
        # Add mean and std lines
        mean_error = error.mean()
        std_error = error.std()
        
        ax2.axvline(mean_error, color='r', linestyle='--', 
                   label=f'Mean Error: {mean_error:.4f}')
        ax2.axvline(mean_error + std_error, color='g', linestyle='--', 
                   label=f'Std Dev: {std_error:.4f}')
        ax2.axvline(mean_error - std_error, color='g', linestyle='--')
        
        ax2.set_title(f"{title} - Error Distribution", fontsize=14)
        ax2.set_xlabel('Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        directory = os.environ.get('REPORTS_DIR', 'reports')
        create_directory(directory)
        
        if not filename.endswith(('.png', '.jpg', '.pdf')):
            filename += '.png'
        
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model performance plot to {full_path}")
    
    return fig

def calculate_returns(signals: pd.DataFrame, market_data: pd.DataFrame, 
                     initial_capital: float = 10000.0) -> pd.DataFrame:
    """Calculate returns based on trading signals.
    
    Args:
        signals: DataFrame with signal columns (1 for buy, -1 for sell, 0 for hold)
        market_data: DataFrame with market data
        initial_capital: Initial capital
        
    Returns:
        DataFrame with portfolio value and returns
    """
    # Ensure we have Close prices
    if 'Close' not in market_data.columns:
        logger.error("Market data missing 'Close' column")
        return pd.DataFrame()
    
    # Ensure dates align
    signals = signals.reindex(market_data.index)
    
    # Calculate position and holdings
    positions = pd.DataFrame(index=signals.index)
    
    # For each signal column
    for col in signals.columns:
        # Calculate position for this signal
        positions[f'position_{col}'] = signals[col].cumsum()
        
        # Calculate holdings (position * price)
        positions[f'holdings_{col}'] = positions[f'position_{col}'] * market_data['Close']
        
        # Add cash (initial capital - cumulative investment)
        positions[f'cash_{col}'] = initial_capital - (signals[col] * market_data['Close']).cumsum()
        
        # Calculate total value (holdings + cash)
        positions[f'total_{col}'] = positions[f'holdings_{col}'] + positions[f'cash_{col}']
        
        # Calculate returns
        positions[f'returns_{col}'] = positions[f'total_{col}'].pct_change()
    
    # Calculate benchmark (buy and hold)
    positions['position_benchmark'] = 1  # Always fully invested
    positions['holdings_benchmark'] = positions['position_benchmark'] * market_data['Close']
    positions['cash_benchmark'] = 0  # All cash is invested
    positions['total_benchmark'] = positions['holdings_benchmark']
    positions['returns_benchmark'] = positions['total_benchmark'].pct_change()
    
    return positions

def calculate_performance_metrics(returns: pd.DataFrame, risk_free_rate: float = 0.03) -> pd.DataFrame:
    """Calculate performance metrics for each strategy.
    
    Args:
        returns: DataFrame with returns columns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with performance metrics
    """
    # Get returns columns
    returns_cols = [col for col in returns.columns if col.startswith('returns_')]
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Initialize results
    results = {}
    
    for col in returns_cols:
        strategy = col.replace('returns_', '')
        
        # Get returns data
        ret = returns[col].dropna()
        
        if len(ret) < 5:
            logger.warning(f"Not enough data for {strategy}, skipping")
            continue
        
        # Calculate metrics
        total_return = (ret + 1).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(ret)) - 1
        daily_std = ret.std()
        annual_std = daily_std * np.sqrt(252)
        
        # Sharpe and Sortino
        excess_ret = ret - daily_rf
        sharpe = (excess_ret.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        
        # Sortino (only considers downside deviation)
        downside_ret = ret[ret < 0]
        downside_std = downside_ret.std() if len(downside_ret) > 0 else 0
        sortino = (excess_ret.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + ret).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate and other metrics
        win_rate = len(ret[ret > 0]) / len(ret)
        loss_rate = len(ret[ret < 0]) / len(ret)
        avg_win = ret[ret > 0].mean() if len(ret[ret > 0]) > 0 else 0
        avg_loss = ret[ret < 0].mean() if len(ret[ret < 0]) > 0 else 0
        profit_factor = (ret[ret > 0].sum() / -ret[ret < 0].sum()) if ret[ret < 0].sum() < 0 else float('inf')
        
        results[strategy] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_std,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    return pd.DataFrame(results).T

def plot_equity_curve(positions: pd.DataFrame, title: str = "Strategy Performance",
                     filename: Optional[str] = None) -> plt.Figure:
    """Plot equity curves for each strategy.
    
    Args:
        positions: DataFrame with position and total value columns
        title: Plot title
        filename: If provided, save plot to this filename
        
    Returns:
        Matplotlib figure
    """
    # Get total value columns
    total_cols = [col for col in positions.columns if col.startswith('total_')]
    
    if not total_cols:
        logger.error("No total value columns found")
        return None
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each equity curve
    for col in total_cols:
        strategy = col.replace('total_', '')
        plt.plot(positions.index, positions[col], label=strategy)
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        directory = os.environ.get('REPORTS_DIR', 'reports')
        create_directory(directory)
        
        if not filename.endswith(('.png', '.jpg', '.pdf')):
            filename += '.png'
        
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve plot to {full_path}")
    
    return plt.gcf()

def load_stock_data(symbol: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load cached stock data from disk.
    
    Args:
        symbol: Stock symbol
        data_dir: Directory to load from, uses DATA_DIR env var if None
        
    Returns:
        DataFrame with market data
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data')
    
    # Check for parquet file
    parquet_pattern = f"{symbol}_market_1d_*.parquet"
    
    # Use pathlib for easier file matching
    path = Path(data_dir)
    files = list(path.glob(parquet_pattern))
    
    if not files:
        logger.warning(f"No cached data found for {symbol}")
        return pd.DataFrame()
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Load newest file
    try:
        df = pd.read_parquet(files[0])
        logger.info(f"Loaded cached data for {symbol} from {files[0]}")
        return df
    except Exception as e:
        logger.error(f"Failed to load cached data for {symbol}: {e}")
        return pd.DataFrame()