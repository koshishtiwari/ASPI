"""Decision engine module for Stock Potential Identifier.

This module handles opportunity ranking, portfolio constraints, and
generates final decision reports.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from .utils import load_config, save_report

logger = logging.getLogger(__name__)

class DecisionEngine:
    """Handles ranking of opportunities and final decision reports."""

    def __init__(self, config_path: str = None):
        """Initialize decision engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.decisions_config = self.config['decisions']
        
        # Get portfolio constraints
        self.max_positions = self.decisions_config['portfolio']['max_positions']
        self.max_correlation = self.decisions_config['portfolio']['max_correlation']
        
        # Get ranking weights
        self.probability_weight = self.decisions_config['ranking']['probability_weight']
        self.risk_reward_weight = self.decisions_config['ranking']['risk_reward_weight']
        self.sharpe_weight = self.decisions_config['ranking']['sharpe_weight']
        
        # Current positions and correlation data
        self.current_positions = []
        self.correlation_matrix = None
    
    def set_current_positions(self, positions: List[Dict]) -> None:
        """Set current portfolio positions.
        
        Args:
            positions: List of position dictionaries
        """
        self.current_positions = positions
        logger.info(f"Set {len(positions)} current positions")
    
    def set_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """Set correlation matrix for portfolio diversification.
        
        Args:
            correlation_matrix: DataFrame with pairwise correlations
        """
        self.correlation_matrix = correlation_matrix
        logger.info(f"Set correlation matrix with {len(correlation_matrix)} symbols")
    
    def rank_opportunities(self, signals: List[Dict]) -> List[Dict]:
        """Rank trading opportunities by quality and portfolio fit.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            List of ranked opportunity dictionaries
        """
        if not signals:
            logger.info("No signals to rank")
            return []
        
        logger.info(f"Ranking {len(signals)} opportunities")
        
        try:
            # Convert to DataFrame for easier manipulation
            signals_df = pd.DataFrame(signals)
            
            # Calculate conviction score (composite quality metric)
            signals_df['conviction'] = (
                signals_df['probability'] * self.probability_weight + 
                signals_df['risk_reward'] * self.risk_reward_weight + 
                signals_df['sharpe'] * self.sharpe_weight
            )
            
            # Apply portfolio constraints
            filtered_signals = self._apply_portfolio_constraints(signals_df)
            
            # Sort by conviction
            ranked_signals = filtered_signals.sort_values('conviction', ascending=False)
            
            # Cap to available slots based on max positions
            available_slots = max(0, self.max_positions - len(self.current_positions))
            top_opportunities = ranked_signals.head(available_slots).to_dict('records')
            
            logger.info(f"Ranked opportunities, selected top {len(top_opportunities)}")
            return top_opportunities
            
        except Exception as e:
            logger.error(f"Error ranking opportunities: {e}")
            return []
    
    def _apply_portfolio_constraints(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Apply portfolio constraints to signal dataframe.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            Filtered and adjusted signals DataFrame
        """
        # Make a copy to avoid modifying the original
        filtered_signals = signals_df.copy()
        
        # If we have a correlation matrix and existing positions
        if self.correlation_matrix is not None and self.current_positions:
            for idx, row in filtered_signals.iterrows():
                symbol = row['symbol']
                
                # Check correlation with existing positions
                for pos in self.current_positions:
                    pos_symbol = pos['symbol']
                    
                    if symbol in self.correlation_matrix.index and pos_symbol in self.correlation_matrix.columns:
                        corr = self.correlation_matrix.loc[symbol, pos_symbol]
                        
                        # Penalize highly correlated opportunities
                        if abs(corr) > self.max_correlation:
                            # Reduce conviction by correlation excess
                            penalty = (abs(corr) - self.max_correlation) / (1 - self.max_correlation)
                            filtered_signals.loc[idx, 'conviction'] *= (1 - penalty)
                            
                            logger.debug(f"Penalized {symbol} due to {abs(corr):.2f} correlation with {pos_symbol}")
        
        # Check for conflicting signals (same symbol, different type/horizon)
        symbols = filtered_signals['symbol'].unique()
        
        for symbol in symbols:
            symbol_signals = filtered_signals[filtered_signals['symbol'] == symbol]
            
            if len(symbol_signals) > 1:
                # Check if signals conflict (both LONG and SHORT)
                signal_types = symbol_signals['type'].unique()
                
                if len(signal_types) > 1:
                    logger.debug(f"Conflicting signals for {symbol}, keeping highest conviction")
                    
                    # Keep only highest conviction signal
                    best_idx = symbol_signals['conviction'].idxmax()
                    drop_idx = symbol_signals.index[symbol_signals.index != best_idx]
                    filtered_signals = filtered_signals.drop(drop_idx)
        
        return filtered_signals
    
    def generate_decision_report(self, ranked_opportunities: List[Dict], 
                               market_context: Dict) -> Dict:
        """Generate comprehensive decision report with context.
        
        Args:
            ranked_opportunities: List of ranked opportunity dictionaries
            market_context: Dictionary with market context
            
        Returns:
            Report dictionary
        """
        timestamp = datetime.now()
        
        # If no opportunities, return minimal report
        if not ranked_opportunities:
            report = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'market_regime': market_context.get('regime', 'unknown'),
                'volatility_environment': market_context.get('volatility', 'normal'),
                'opportunities': [],
                'summary': "No actionable opportunities identified"
            }
            
            logger.info("Generated empty decision report")
            return report
        
        # Group opportunities by time horizon
        opportunities_by_horizon = {}
        for horizon in ['short_term', 'medium_term', 'long_term']:
            horizon_ops = [op for op in ranked_opportunities if op['horizon'] == horizon]
            opportunities_by_horizon[horizon] = horizon_ops
        
        # Also group by signal type
        long_ops = [op for op in ranked_opportunities if op['type'] == 'LONG']
        short_ops = [op for op in ranked_opportunities if op['type'] == 'SHORT']
        
        # Generate executive summary
        total_opportunities = len(ranked_opportunities)
        
        if total_opportunities > 0:
            best_opportunity = max(ranked_opportunities, key=lambda x: x['conviction'])
            
            summary = f"Identified {total_opportunities} actionable opportunities "
            summary += f"with highest conviction in {best_opportunity['symbol']} "
            summary += f"({best_opportunity['type']}, {best_opportunity['probability']:.1%} probability). "
            
            # Add market context
            summary += f"Current market regime appears to be {market_context.get('regime', 'mixed')}. "
            
            # Add signal type breakdown
            if long_ops and short_ops:
                summary += f"Signals are mixed with {len(long_ops)} bullish and {len(short_ops)} bearish opportunities."
            elif long_ops:
                summary += f"All signals are bullish, suggesting potential upside momentum."
            elif short_ops:
                summary += f"All signals are bearish, suggesting potential downside risk."
        else:
            summary = "No actionable opportunities identified at this time."
        
        # Compile report
        report = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'market_regime': market_context.get('regime', 'unknown'),
            'volatility_environment': market_context.get('volatility', 'normal'),
            'opportunities_by_horizon': opportunities_by_horizon,
            'long_opportunities': long_ops,
            'short_opportunities': short_ops,
            'total_opportunities': total_opportunities,
            'summary': summary
        }
        
        logger.info(f"Generated decision report with {total_opportunities} opportunities")
        
        return report
    
    def save_decision_report(self, report: Dict) -> Dict:
        """Save decision report to disk in specified formats.
        
        Args:
            report: Decision report dictionary
            
        Returns:
            Dictionary with saved file paths
        """
        # Get output formats from config
        formats = self.config['execution']['report_output_formats']
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for format_type in formats:
            try:
                filename = f"decision_report_{timestamp}"
                path = save_report(report, filename, format=format_type)
                
                if path:
                    saved_files[format_type] = path
                    
            except Exception as e:
                logger.error(f"Error saving report as {format_type}: {e}")
        
        return saved_files
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.DataFrame], 
                                   lookback_days: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix between symbols.
        
        Args:
            price_data: Dictionary of DataFrames by symbol
            lookback_days: Number of days to use for correlation
            
        Returns:
            Correlation matrix DataFrame
        """
        # Extract returns for each symbol
        returns_data = {}
        
        for symbol, data in price_data.items():
            # Calculate daily returns
            if 'Close' in data.columns:
                # Use only the last N days
                subset = data.iloc[-lookback_days:] if len(data) > lookback_days else data
                returns = subset['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        # Create DataFrame with all returns
        if not returns_data:
            logger.warning("No return data available for correlation calculation")
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        logger.info(f"Calculated correlation matrix for {len(correlation_matrix)} symbols")
        
        return correlation_matrix
    
    def portfolio_performance_analysis(self, positions: List[Dict], 
                                     historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze current portfolio performance.
        
        Args:
            positions: List of current positions
            historical_data: Dictionary of historical price data by symbol
            
        Returns:
            Dictionary with portfolio analysis
        """
        if not positions:
            return {'status': 'No positions to analyze'}
        
        # Extract position data
        symbols = [pos['symbol'] for pos in positions]
        entry_prices = {pos['symbol']: pos['entry_price'] for pos in positions}
        position_types = {pos['symbol']: pos['type'] for pos in positions}
        stop_losses = {pos['symbol']: pos['stop_loss'] for pos in positions}
        targets = {pos['symbol']: pos['target'] for pos in positions}
        
        # Calculate current performance
        performance = []
        
        for symbol in symbols:
            if symbol not in historical_data:
                logger.warning(f"No historical data for {symbol}, skipping")
                continue
            
            data = historical_data[symbol]
            
            if data.empty or 'Close' not in data.columns:
                continue
            
            current_price = data['Close'].iloc[-1]
            entry_price = entry_prices[symbol]
            position_type = position_types[symbol]
            stop_loss = stop_losses[symbol]
            target = targets[symbol]
            
            # Calculate return
            if position_type == 'LONG':
                return_pct = (current_price / entry_price) - 1
                risk = (entry_price - stop_loss) / entry_price
                reward = (target - entry_price) / entry_price
            else:  # SHORT
                return_pct = 1 - (current_price / entry_price)
                risk = (stop_loss - entry_price) / entry_price
                reward = (entry_price - target) / entry_price
            
            # Calculate distance to target/stop
            if position_type == 'LONG':
                pct_to_target = (target - current_price) / (target - entry_price) if target != entry_price else 0
                pct_to_stop = (current_price - stop_loss) / (entry_price - stop_loss) if entry_price != stop_loss else 0
            else:  # SHORT
                pct_to_target = (current_price - target) / (entry_price - target) if entry_price != target else 0
                pct_to_stop = (stop_loss - current_price) / (stop_loss - entry_price) if stop_loss != entry_price else 0
            
            performance.append({
                'symbol': symbol,
                'position_type': position_type,
                'entry_price': entry_price,
                'current_price': current_price,
                'return_pct': return_pct,
                'risk': risk,
                'reward': reward,
                'risk_reward': reward / risk if risk > 0 else 0,
                'pct_to_target': min(max(pct_to_target, 0), 1),  # Bound between 0-1
                'pct_to_stop': min(max(pct_to_stop, 0), 1)  # Bound between 0-1
            })
        
        # Calculate portfolio summary
        if performance:
            avg_return = np.mean([p['return_pct'] for p in performance])
            profitable_positions = sum(1 for p in performance if p['return_pct'] > 0)
            total_positions = len(performance)
            hit_rate = profitable_positions / total_positions if total_positions > 0 else 0
            
            summary = {
                'total_positions': total_positions,
                'profitable_positions': profitable_positions,
                'hit_rate': hit_rate,
                'avg_return': avg_return,
                'positions': performance
            }
        else:
            summary = {'status': 'No performance data available'}
        
        return summary