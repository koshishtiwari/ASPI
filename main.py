"""Main entry point for Stock Potential Identifier System."""

import os
import sys
import asyncio
import logging
import argparse
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_acquisition import DataAcquisition
from src.feature_engineering import FeatureEngineering
from src.model_hierarchy import ModelHierarchy
from src.fusion_layer import FusionLayer
from src.signal_generator import SignalGenerator
from src.decision_engine import DecisionEngine
from src.utils import load_config, create_directory, save_report

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/stock_identifier.log')
    ]
)

logger = logging.getLogger('StockPotentialIdentifier')

class StockPotentialIdentifier:
    """Main system controller for Stock Potential Identifier."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize system components
        self.data_engine = DataAcquisition(config_path)
        self.feature_engine = FeatureEngineering(config_path)
        
        # Initialize models for each horizon
        self.model_engines = {
            'short_term': ModelHierarchy('short_term', config_path),
            'medium_term': ModelHierarchy('medium_term', config_path),
            'long_term': ModelHierarchy('long_term', config_path)
        }
        
        # Initialize fusion layers for each horizon
        self.fusion_engines = {
            'short_term': FusionLayer('short_term', config_path),
            'medium_term': FusionLayer('medium_term', config_path),
            'long_term': FusionLayer('long_term', config_path)
        }
        
        # Initialize signal generator and decision engine
        self.signal_engine = SignalGenerator(config_path)
        self.decision_engine = DecisionEngine(config_path)
        
        # System state
        self.last_training = {
            'short_term': None,
            'medium_term': None,
            'long_term': None
        }
        
        self.is_running = False
    
    def _create_directories(self) -> None:
        """Create necessary directories for the system."""
        # Get paths from config
        paths = self.config['paths']
        
        create_directory(paths['data_dir'])
        create_directory(paths['models_dir'])
        create_directory(paths['reports_dir'])
        
        # Create logs directory (to fix the Docker issue)
        create_directory('logs')
        
        # Additional directories
        create_directory(os.path.join(paths['data_dir'], 'market_data'))
        create_directory(os.path.join(paths['models_dir'], 'short_term'))
        create_directory(os.path.join(paths['models_dir'], 'medium_term'))
        create_directory(os.path.join(paths['models_dir'], 'long_term'))
    
    async def start(self) -> None:
        """Start the system main loop."""
        logger.info("Starting Stock Potential Identifier System")
        self.is_running = True
        
        # Start data acquisition service
        await self.data_engine.start()
        
        try:
            while self.is_running:
                try:
                    # 1. Check if we need to retrain any models
                    for horizon in ['short_term', 'medium_term', 'long_term']:
                        if self._should_train(horizon):
                            await self._train_models(horizon)
                    
                    # 2. Get stock universe
                    stock_universe = await self.data_engine.get_stock_universe()
                    logger.info(f"Analyzing {len(stock_universe)} stocks")
                    
                    # 3. Get market data for all stocks
                    market_data = {}
                    for symbol in stock_universe:
                        data = await self.data_engine.get_market_data(symbol, period="2y", interval="1d")
                        if not data.empty:
                            market_data[symbol] = data
                    
                    # 4. Generate signals for each stock and horizon
                    all_signals = {}
                    
                    for horizon in ['short_term', 'medium_term', 'long_term']:
                        horizon_signals = []
                        
                        for symbol, data in market_data.items():
                            # Generate predictions and signals for this stock
                            signals = await self._analyze_stock(symbol, data, horizon)
                            if signals:
                                horizon_signals.extend(signals)
                        
                        all_signals[horizon] = horizon_signals
                    
                    # 5. Calculate correlation matrix for portfolio constraints
                    correlation_matrix = self.decision_engine.calculate_correlation_matrix(market_data)
                    self.decision_engine.set_correlation_matrix(correlation_matrix)
                    
                    # 6. Combine signals from all horizons and rank opportunities
                    all_opportunities = []
                    for horizon, signals in all_signals.items():
                        all_opportunities.extend(signals)
                    
                    ranked_opportunities = self.decision_engine.rank_opportunities(all_opportunities)
                    
                    # 7. Generate market context
                    # Use the S&P 500 as a proxy for market context
                    market_index = await self.data_engine.get_market_data("SPY", period="2y", interval="1d")
                    market_regime = "unknown"
                    
                    if not market_index.empty:
                        market_regime = self.fusion_engines['medium_term'].detect_market_regime(market_index)
                    
                    market_context = {
                        'regime': market_regime,
                        'volatility': 'normal'  # Could be determined from VIX or similar
                    }
                    
                    # 8. Generate final decision report
                    decision_report = self.decision_engine.generate_decision_report(
                        ranked_opportunities, market_context
                    )
                    
                    # 9. Save decision report
                    self.decision_engine.save_decision_report(decision_report)
                    
                    # 10. Output summary
                    logger.info(f"Analysis complete. {len(ranked_opportunities)} opportunities identified.")
                    logger.info(f"Summary: {decision_report['summary']}")
                    
                    # Wait before next cycle
                    check_frequency = self.config['execution']['check_frequency_minutes']
                    await asyncio.sleep(check_frequency * 60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(60)  # Wait a minute before retry
        
        finally:
            # Clean shutdown
            await self.data_engine.stop()
            logger.info("Stock Potential Identifier System stopped")
    
    def stop(self) -> None:
        """Stop the system."""
        logger.info("Stopping Stock Potential Identifier System")
        self.is_running = False
    
    def _should_train(self, horizon: str) -> bool:
        """Check if we should train models for the given horizon.
        
        Args:
            horizon: Time horizon
            
        Returns:
            True if training is needed, False otherwise
        """
        # Get training frequency from config
        frequency_days = self.config['models']['time_horizons'][horizon]['training_frequency_days']
        
        # Check if we've trained before
        if self.last_training[horizon] is None:
            return True
        
        # Check if it's time to retrain
        days_since_training = (datetime.now() - self.last_training[horizon]).days
        return days_since_training >= frequency_days
    
    async def _train_models(self, horizon: str) -> None:
        """Train models for the given horizon.
        
        Args:
            horizon: Time horizon
        """
        logger.info(f"Training {horizon} models")
        
        try:
            # 1. Get stock universe for training
            stock_universe = await self.data_engine.get_stock_universe()
            
            # 2. Get historical data for training
            training_data = {}
            for symbol in stock_universe:
                # Get longer history for training
                data = await self.data_engine.get_market_data(symbol, period="5y", interval="1d")
                if not data.empty:
                    training_data[symbol] = data
            
            if not training_data:
                logger.warning("No training data available")
                return
            
            # 3. Prepare training datasets
            X_train_all, y_train_all = await self._prepare_training_data(training_data, horizon)
            
            if X_train_all.empty or y_train_all.empty:
                logger.warning("Failed to prepare training data")
                return
            
            # 4. Train models
            self.model_engines[horizon].train_models(X_train_all, y_train_all)
            
            # 5. Update last training timestamp
            self.last_training[horizon] = datetime.now()
            
            logger.info(f"Completed training {horizon} models")
            
        except Exception as e:
            logger.error(f"Error training {horizon} models: {str(e)}", exc_info=True)
    
    async def _prepare_training_data(self, training_data: Dict[str, pd.DataFrame], 
                                  horizon: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrices and target vectors for training.
        
        Args:
            training_data: Dictionary of DataFrames by symbol
            horizon: Time horizon
            
        Returns:
            Tuple of (X_train, y_train)
        """
        all_features = []
        all_targets = []
        
        for symbol, data in training_data.items():
            try:
                # 1. Generate technical features
                tech_features = self.feature_engine.generate_technical_features(data)
                
                # 2. Get fundamental data (if available)
                fundamental_data = await self.data_engine.get_fundamentals(symbol)
                fund_features = None
                
                if not fundamental_data.empty:
                    fund_features = self.feature_engine.generate_fundamental_features(fundamental_data)
                
                # 3. Get alternative data (if available)
                alt_data = await self.data_engine.get_alternative_data(symbol)
                alt_features = None
                
                if not alt_data.empty:
                    alt_features = self.feature_engine.generate_alternative_features(alt_data)
                
                # 4. Combine all features
                combined_features = self.feature_engine.combine_feature_sets(
                    tech_features, fund_features, alt_features
                )
                
                # 5. Generate target labels
                target_data = self.feature_engine.generate_target_labels(data, horizon)
                
                if not target_data.empty and not combined_features.empty:
                    # Align features and targets
                    aligned_features = combined_features.loc[target_data.index]
                    
                    # Add symbol column for later identification
                    aligned_features['symbol'] = symbol
                    
                    # Append to training data
                    all_features.append(aligned_features)
                    all_targets.append(target_data['target'])
                
            except Exception as e:
                logger.error(f"Error preparing training data for {symbol}: {str(e)}")
        
        if not all_features or not all_targets:
            return pd.DataFrame(), pd.Series()
        
        # Combine all training data
        X_train_all = pd.concat(all_features)
        y_train_all = pd.concat(all_targets)
        
        # Remove symbol column before training
        symbol_col = X_train_all['symbol']
        X_train_all = X_train_all.drop('symbol', axis=1)
        
        logger.info(f"Prepared training data: {X_train_all.shape[0]} samples, {X_train_all.shape[1]} features")
        
        return X_train_all, y_train_all
    
    async def _analyze_stock(self, symbol: str, market_data: pd.DataFrame, 
                          horizon: str) -> List[Dict]:
        """Analyze a stock and generate signals.
        
        Args:
            symbol: Stock symbol
            market_data: Market data DataFrame
            horizon: Time horizon
            
        Returns:
            List of signal dictionaries
        """
        try:
            # 1. Generate features
            tech_features = self.feature_engine.generate_technical_features(market_data)
            
            # 2. Get fundamental data (if available)
            fundamental_data = await self.data_engine.get_fundamentals(symbol)
            fund_features = None
            
            if not fundamental_data.empty:
                fund_features = self.feature_engine.generate_fundamental_features(fundamental_data)
            
            # 3. Get alternative data (if available)
            alt_data = await self.data_engine.get_alternative_data(symbol)
            alt_features = None
            
            if not alt_data.empty:
                alt_features = self.feature_engine.generate_alternative_features(alt_data)
            
            # 4. Combine all features
            combined_features = self.feature_engine.combine_feature_sets(
                tech_features, fund_features, alt_features
            )
            
            # 5. Select features for this horizon
            selected_features = self.feature_engine.select_features(combined_features, horizon)
            
            # Use only the latest data point for prediction
            latest_features = selected_features.iloc[[-1]]
            
            # 6. Generate predictions
            predictions = self.model_engines[horizon].predict(latest_features)
            
            # 7. Generate consensus prediction
            market_regime = self.fusion_engines[horizon].detect_market_regime(market_data)
            consensus = self.fusion_engines[horizon].generate_consensus(predictions, market_regime)
            
            # 8. Generate signals
            signals = self.signal_engine.generate_signals(
                market_data, consensus, horizon, symbol
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} for {horizon}: {str(e)}")
            return []

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stock Potential Identifier System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--train-only', action='store_true', help='Train models only, then exit')
    args = parser.parse_args()
    
    # Create system
    system = StockPotentialIdentifier(args.config)
    
    if args.train_only:
        # Train all models and exit
        for horizon in ['short_term', 'medium_term', 'long_term']:
            await system._train_models(horizon)
    else:
        # Start the main loop
        await system.start()

if __name__ == "__main__":
    asyncio.run(main())