"""
Strategy Runner with Debug Logging - Comprehensive trade tracking
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from debug_trade_logger import TradeDebugLogger
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')


def run_strategy_with_debug(config_name='scalping', currency='AUDUSD', 
                           start_date='2023-07-30', end_date='2023-08-07',
                           generate_plots=True):
    """Run strategy with comprehensive debug logging"""
    
    print(f"\n{'='*60}")
    print(f"Running Strategy with Debug Logging")
    print(f"{'='*60}")
    print(f"Config: {config_name}")
    print(f"Currency: {currency}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Configuration setup
    if config_name == 'ultra_tight':
        config_name_str = "Config 1: Ultra-Tight Risk Management"
        config = OptimizedStrategyConfig(
            initial_capital=10000,
            risk_per_trade=0.003,
            relaxed_mode=False,
            exit_on_signal_flip=True,
            verbose=True,
            debug_decisions=True,
            # Ultra-tight specific settings
            sl_max_pips=5.0,
            tp_atr_multipliers=(0.6, 1.2, 2.0),
            confidence_thresholds=(50.0, 60.0, 70.0),
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.7,
            partial_profit_size_percent=0.7,
            tsl_activation_pips=10.0,
            tsl_min_profit_pips=6.0
        )
    else:  # scalping
        config_name_str = "Config 2: Scalping Strategy"
        config = OptimizedStrategyConfig(
            initial_capital=10000,
            risk_per_trade=0.003,
            relaxed_mode=False,
            exit_on_signal_flip=True,
            verbose=True,
            debug_decisions=True,
            # Scalping specific settings
            sl_max_pips=5.0,
            tp_atr_multipliers=(0.6, 1.2, 2.0),
            confidence_thresholds=(30.0, 50.0, 70.0),
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.7,
            partial_profit_size_percent=0.7,
            tsl_activation_pips=10.0,
            tsl_min_profit_pips=6.0
        )
    
    # Data loading
    data_path = f'/Users/williamsmith/Python_local_Mac/Trend_regime_trading/data/{currency}_MASTER_15M.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if df.empty:
        raise ValueError(f"No data found for the specified date range: {start_date} to {end_date}")
    
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Check if indicators already present
    required_indicators = ['NTI_Direction', 'MB_Bias', 'IC_Regime']
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        print(f"\nError: Missing required indicators: {missing_indicators}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Data file does not contain required technical indicators")
    
    # Create debug logger
    debug_logger = TradeDebugLogger(output_dir="debug")
    
    # Create enhanced strategy with debug logger
    strategy = DebugEnabledStrategy(config, debug_logger)
    
    # Run backtest
    print("\nRunning backtest with debug logging...")
    results = strategy.run_backtest(df, currency)
    
    # Save debug log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_logger.save_debug_log(timestamp)
    
    # Generate plots if requested
    if generate_plots and results.get('trades'):
        print("\nGenerating charts...")
        plot_production_results(
            df, 
            results,
            account_value_history=results.get('account_value_history', []),
            config=config,
            currency=currency,
            timestamp=timestamp,
            config_name=config_name_str
        )
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Return: {results.get('total_return', 0):.2f}%")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    
    # Print validation summary
    print("\n" + "="*60)
    print("DEBUG VALIDATION SUMMARY")
    print("="*60)
    
    valid_trades = sum(1 for t in debug_logger.trades.values() if len(t.validation_errors) == 0)
    invalid_trades = len(debug_logger.trades) - valid_trades
    
    print(f"\nTotal Trades Logged: {len(debug_logger.trades)}")
    print(f"Valid Trades: {valid_trades}")
    print(f"Invalid Trades: {invalid_trades}")
    
    if invalid_trades > 0:
        print("\nValidation Errors Found:")
        for trade in debug_logger.trades.values():
            if trade.validation_errors:
                print(f"\nTrade {trade.trade_id}:")
                for error in trade.validation_errors:
                    print(f"  - {error}")
    
    return results, debug_logger


class DebugEnabledStrategy(OptimizedProdStrategy):
    """Strategy with integrated debug logging"""
    
    def __init__(self, config: OptimizedStrategyConfig, debug_logger: TradeDebugLogger):
        super().__init__(config)
        self.debug_logger = debug_logger
        self.trade_id_map = {}  # Map internal trade objects to debug trade IDs
    
    def _execute_entry(self, df: pd.DataFrame, idx: int, current_row: pd.Series, 
                      direction, confidence: float, is_relaxed: bool = False) -> bool:
        """Override to add debug logging on entry"""
        # Call parent method
        result = super()._execute_entry(df, idx, current_row, direction, confidence, is_relaxed)
        
        if result and self.current_trade:
            # Create debug entry
            entry_logic = "Relaxed (NTI only)" if is_relaxed else "Standard (NTI+MB+IC)"
            
            debug_trade_id = self.debug_logger.create_trade(
                entry_time=df.index[idx],
                entry_price=self.current_trade.entry_price,
                direction=self.current_trade.direction.value,
                size_millions=self.current_trade.position_size / self.config.min_lot_size,
                confidence=confidence,
                is_relaxed=is_relaxed,
                entry_logic=entry_logic,
                sl_price=self.current_trade.stop_loss,
                tp1_price=self.current_trade.take_profits[0],
                tp2_price=self.current_trade.take_profits[1],
                tp3_price=self.current_trade.take_profits[2]
            )
            
            # Store mapping
            self.trade_id_map[id(self.current_trade)] = debug_trade_id
            
            if self.config.debug_decisions:
                print(f"  📝 Debug Trade ID: {debug_trade_id}")
        
        return result
    
    def _execute_partial_profit_take(self, row: pd.Series, current_time: pd.Timestamp, 
                                   trade: 'Trade', take_percentage: float, trigger_price: float):
        """Override to add PPT debug logging"""
        # Get debug trade ID
        debug_trade_id = self.trade_id_map.get(id(trade))
        
        # Call parent method
        result = super()._execute_partial_profit_take(row, current_time, trade, take_percentage, trigger_price)
        
        if result and debug_trade_id:
            # Log PPT to debug
            size_closed = trade.position_size * take_percentage
            pnl = trade.partial_exits[-1].pnl if trade.partial_exits else 0
            
            self.debug_logger.update_ppt(
                trade_id=debug_trade_id,
                trigger_time=current_time,
                trigger_price=trigger_price,
                size_closed=size_closed,
                pnl=pnl
            )
        
        return result
    
    def _execute_take_profit_hit(self, row: pd.Series, current_time: pd.Timestamp, 
                               trade: 'Trade', tp_level: int, tp_price: float, 
                               take_percentage: float = None):
        """Override to add TP debug logging"""
        # Get debug trade ID
        debug_trade_id = self.trade_id_map.get(id(trade))
        
        # Call parent method
        result = super()._execute_take_profit_hit(row, current_time, trade, tp_level, 
                                                tp_price, take_percentage)
        
        if result and debug_trade_id:
            # Find the latest partial exit for this TP level
            relevant_exits = [pe for pe in trade.partial_exits if pe.tp_level == tp_level]
            if relevant_exits:
                latest_exit = relevant_exits[-1]
                
                self.debug_logger.update_tp_hit(
                    trade_id=debug_trade_id,
                    tp_level=tp_level,
                    hit_time=current_time,
                    exit_price=latest_exit.price,
                    size_closed=latest_exit.size,
                    pnl=latest_exit.pnl
                )
        
        return result
    
    def _execute_full_exit(self, trade: 'Trade', current_time: pd.Timestamp, 
                         exit_price: float, exit_reason):
        """Override to add final exit debug logging"""
        # Get debug trade ID
        debug_trade_id = self.trade_id_map.get(id(trade))
        
        # Store remaining size before exit
        remaining_size = trade.current_position_size
        
        # Call parent method
        result = super()._execute_full_exit(trade, current_time, exit_price, exit_reason)
        
        if result and debug_trade_id:
            # Calculate final exit PnL for remaining position
            if trade.direction.value == 'long':
                final_pips = (exit_price - trade.entry_price) * 10000
            else:
                final_pips = (trade.entry_price - exit_price) * 10000
            
            final_pnl = (remaining_size / self.config.min_lot_size) * 100 * final_pips
            
            # Update max excursions if available
            if hasattr(trade, 'max_favorable_excursion'):
                self.debug_logger.update_excursions(
                    trade_id=debug_trade_id,
                    max_favorable_pips=trade.max_favorable_excursion,
                    max_adverse_pips=trade.max_adverse_excursion
                )
            
            # Log final exit
            self.debug_logger.update_final_exit(
                trade_id=debug_trade_id,
                exit_time=current_time,
                exit_price=exit_price,
                exit_reason=exit_reason.value,
                remaining_size=remaining_size,
                final_pnl=final_pnl,
                total_trade_pnl=trade.final_pnl
            )
        
        return result
    
    def _update_trade_excursions(self, trade: 'Trade', current_price: float):
        """Track maximum favorable and adverse excursions"""
        if trade.direction.value == 'long':
            favorable_pips = (current_price - trade.entry_price) * 10000
            adverse_pips = (trade.entry_price - current_price) * 10000
        else:
            favorable_pips = (trade.entry_price - current_price) * 10000
            adverse_pips = (current_price - trade.entry_price) * 10000
        
        # Initialize if not present
        if not hasattr(trade, 'max_favorable_excursion'):
            trade.max_favorable_excursion = 0
            trade.max_adverse_excursion = 0
        
        # Update maximums
        trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable_pips)
        trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse_pips)
    
    def _update_position(self, df: pd.DataFrame, idx: int):
        """Override to track excursions"""
        # Track excursions before updating position
        if self.current_trade:
            current_price = df.iloc[idx]['Close']
            self._update_trade_excursions(self.current_trade, current_price)
        
        # Call parent method
        super()._update_position(df, idx)


def main():
    parser = argparse.ArgumentParser(description='Run strategy with debug logging')
    parser.add_argument('--config', type=str, default='scalping', 
                       choices=['ultra_tight', 'scalping'],
                       help='Configuration to use')
    parser.add_argument('--currency', type=str, default='AUDUSD',
                       help='Currency pair to test')
    parser.add_argument('--start', type=str, default='2023-07-30',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-08-07',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        results, debug_logger = run_strategy_with_debug(
            config_name=args.config,
            currency=args.currency,
            start_date=args.start,
            end_date=args.end,
            generate_plots=not args.no_plots
        )
        
        print(f"\n✅ Strategy completed successfully!")
        print(f"📄 Debug log saved to: debug/trade_debug_log_*.json")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()