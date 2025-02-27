"""Example of analyzing a single stock."""

import os
import sys
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition import DataAcquisition
from src.feature_engineering import FeatureEngineering
from src.model_hierarchy import ModelHierarchy
from src.fusion_layer import FusionLayer
from src.signal_generator import SignalGenerator

async def analyze_stock(symbol='AAPL', horizon='medium_term'):
    """Analyze a single stock and generate signals."""
    print(f"Analyzing {symbol} for {horizon} horizon...")
    
    # Initialize components
    data_engine = DataAcquisition()
    feature_engine = FeatureEngineering()
    model_engine = ModelHierarchy(horizon)
    fusion_engine = FusionLayer(horizon)
    signal_engine = SignalGenerator()
    
    # Start data acquisition service
    await data_engine.start()
    
    try:
        # Get market data
        print(f"Fetching market data for {symbol}...")
        market_data = await data_engine.get_market_data(symbol, period="2y", interval="1d")
        
        if market_data.empty:
            print(f"No market data available for {symbol}")
            return None
        
        # Generate features
        print("Generating features...")
        tech_features = feature_engine.generate_technical_features(market_data)
        
        # Get fundamental data
        fundamental_data = await data_engine.get_fundamentals(symbol)
        fund_features = None
        
        if not fundamental_data.empty:
            fund_features = feature_engine.generate_fundamental_features(fundamental_data)
            print("Fundamental features generated")
        else:
            print("No fundamental data available")
        
        # Get alternative data
        alt_data = await data_engine.get_alternative_data(symbol)
        alt_features = None
        
        if not alt_data.empty:
            alt_features = feature_engine.generate_alternative_features(alt_data)
            print("Alternative features generated")
        else:
            print("No alternative data available")
        
        # Combine features
        combined_features = feature_engine.combine_feature_sets(
            tech_features, fund_features, alt_features
        )
        
        # Load models
        print("Loading models...")
        model_engine.load_models()
        
        # If models not loaded, build them (they won't be trained)
        if not model_engine.fitted:
            print("Models not found, building new models (not trained)")
            model_engine.build_models()
        
        # Get latest data point for prediction
        latest_features = combined_features.iloc[[-1]]
        
        # Generate predictions
        print("Generating predictions...")
        predictions = model_engine.predict(latest_features)
        
        if not predictions:
            print(f"No predictions generated for {symbol}")
            return None
        
        # Generate consensus
        market_regime = fusion_engine.detect_market_regime(market_data)
        print(f"Detected market regime: {market_regime}")
        
        consensus = fusion_engine.generate_consensus(predictions, market_regime)
        
        # Generate signals
        signals = signal_engine.generate_signals(
            market_data, consensus, horizon, symbol
        )
        
        # Display results
        if signals:
            print("\nSignal generated:")
            signal = signals[0]  # Take the first signal
            print(f"  Type: {signal['type']}")
            print(f"  Entry Price: ${signal['entry_price']:.2f}")
            print(f"  Stop Loss: ${signal['stop_loss']:.2f}")
            print(f"  Target: ${signal['target']:.2f}")
            print(f"  Risk-Reward: {signal['risk_reward']:.2f}")
            print(f"  Probability: {signal['probability']:.2f}")
            print(f"  Sharpe: {signal['sharpe']:.2f}")
            
            # Visualize the signal
            plt.figure(figsize=(12, 6))
            
            # Plot price history
            plt.plot(market_data.index, market_data['Close'], label='Close Price')
            
            # Get the latest date
            latest_date = market_data.index[-1]
            
            # Add entry point
            plt.scatter(latest_date, signal['entry_price'], color='blue', s=100, label='Entry')
            
            # Add stop loss and target
            if signal['type'] == 'LONG':
                stop_color = 'red'
                target_color = 'green'
            else:  # SHORT
                stop_color = 'green'
                target_color = 'red'
            
            plt.scatter(latest_date, signal['stop_loss'], color=stop_color, s=100, label='Stop Loss')
            plt.scatter(latest_date, signal['target'], color=target_color, s=100, label='Target')
            
            # Add horizontal lines
            plt.axhline(y=signal['entry_price'], color='blue', linestyle='--', alpha=0.3)
            plt.axhline(y=signal['stop_loss'], color=stop_color, linestyle='--', alpha=0.3)
            plt.axhline(y=signal['target'], color=target_color, linestyle='--', alpha=0.3)
            
            plt.title(f"{symbol} - {signal['type']} Signal")
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('examples/output', exist_ok=True)
            plt.savefig(f'examples/output/{symbol}_{horizon}_signal.png')
            plt.show()
            
            return signals
        else:
            print("No signals generated")
            return None
    
    finally:
        # Stop data acquisition service
        await data_engine.stop()

async def main():
    """Main function."""
    # Set the symbol to analyze
    symbol = input("Enter stock symbol to analyze (default: AAPL): ") or "AAPL"
    
    # Set the horizon
    horizon_options = {
        '1': 'short_term',
        '2': 'medium_term',
        '3': 'long_term'
    }
    
    print("\nSelect time horizon:")
    print("1. Short-term (1-5 days)")
    print("2. Medium-term (1-4 weeks)")
    print("3. Long-term (1-3 months)")
    
    horizon_choice = input("Enter choice (default: 2): ") or "2"
    horizon = horizon_options.get(horizon_choice, 'medium_term')
    
    # Run the analysis
    await analyze_stock(symbol, horizon)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())