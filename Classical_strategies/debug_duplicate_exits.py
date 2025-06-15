"""
Debug script to investigate duplicate exit markers issue
Focuses on trade entered at 2025-03-30 21:15
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

def create_debug_strategy():
    """Create strategy with debug mode enabled"""
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        tsl_activation_pips=15,
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=1.2,
        tp_range_market_multiplier=0.5,
        tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        relaxed_position_multiplier=0.5,
        relaxed_mode=False,
        realistic_costs=True,
        verbose=False,
        debug_decisions=True,  # Enable debug mode
        use_daily_sharpe=True
    )
    return OptimizedProdStrategy(strategy_config)

def main():
    print("="*80)
    print("DEBUG: DUPLICATE EXIT MARKERS INVESTIGATION")
    print("Focusing on trade entered at 2025-03-30 21:15")
    print("="*80)
    
    # Load data
    if os.path.exists('data'):
        data_path = 'data'
    elif os.path.exists('../data'):
        data_path = '../data'
    else:
        raise FileNotFoundError("Cannot find data directory")
    
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take only the last 5000 rows (same as run_strategy_single.py)
    df = df_full.iloc[-5000:].copy()
    print(f"Using last 5,000 rows: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
    strategy = create_debug_strategy()
    
    # Override the _execute_partial_exit method to add detailed logging
    original_partial_exit = strategy._execute_partial_exit
    
    def debug_execute_partial_exit(trade, exit_time, exit_price, exit_percent, exit_reason=None):
        """Wrapper to log partial exit processing details"""
        if trade.entry_time.strftime('%Y-%m-%d %H:%M') == '2025-03-30 21:15':
            print(f"\n{'='*60}")
            print(f"DEBUG PARTIAL EXIT at {exit_time}")
            print(f"  Trade Entry: {trade.entry_time} at {trade.entry_price}")
            print(f"  Exit Price: {exit_price}")
            print(f"  Exit Percent: {exit_percent}")
            print(f"  Exit Reason: {exit_reason}")
            print(f"  Size to Exit: {trade.remaining_size * exit_percent}")
            print(f"  Current Remaining Size: {trade.remaining_size}")
            print(f"  Exit Count Before: {trade.exit_count}")
            print(f"  TP Hits Before: {trade.tp_hits}")
            print(f"  Partial Exits Before: {len(trade.partial_exits)}")
            
        # Call original method
        result = original_partial_exit(trade, exit_time, exit_price, exit_percent, exit_reason)
        
        if trade.entry_time.strftime('%Y-%m-%d %H:%M') == '2025-03-30 21:15':
            print(f"\n  After Partial Exit:")
            print(f"  Exit Count After: {trade.exit_count}")
            print(f"  TP Hits After: {trade.tp_hits}")
            print(f"  Partial Exits After: {len(trade.partial_exits)}")
            print(f"  Remaining Size After: {trade.remaining_size}")
            print(f"{'='*60}")
        
        return result
    
    strategy._execute_partial_exit = debug_execute_partial_exit
    
    # Also monitor the _execute_full_exit method
    original_full_exit = strategy._execute_full_exit
    
    def debug_execute_full_exit(trade, exit_time, exit_price, exit_reason):
        """Wrapper to log full exit processing"""
        if trade.entry_time.strftime('%Y-%m-%d %H:%M') == '2025-03-30 21:15':
            print(f"\n{'='*60}")
            print(f"DEBUG FULL EXIT at {exit_time}")
            print(f"  Trade Entry: {trade.entry_time} at {trade.entry_price}")
            print(f"  Exit Price: {exit_price}")
            print(f"  Exit Reason: {exit_reason}")
            print(f"  Remaining Size: {trade.remaining_size}")
            print(f"  Partial Exits Count: {len(trade.partial_exits)}")
            print(f"{'='*60}")
        
        return original_full_exit(trade, exit_time, exit_price, exit_reason)
    
    strategy._execute_full_exit = debug_execute_full_exit
    
    # Run backtest
    print("\nRunning backtest with debug logging...")
    print("Looking for trade entered at 2025-03-30 21:15...")
    results = strategy.run_backtest(df)
    
    # Find and analyze the specific trade
    print("\n" + "="*80)
    print("ANALYZING TARGET TRADE")
    print("="*80)
    
    target_trade = None
    for trade in results['trades']:
        if trade.entry_time.strftime('%Y-%m-%d %H:%M') == '2025-03-30 21:15':
            target_trade = trade
            break
    
    if target_trade:
        print(f"\nFound target trade:")
        print(f"  Entry: {target_trade.entry_time} at {target_trade.entry_price}")
        print(f"  Direction: {target_trade.direction}")
        print(f"  Initial Size: {target_trade.position_size:,.0f}")
        print(f"  Stop Loss: {target_trade.stop_loss}")
        print(f"  Take Profits: {target_trade.take_profits}")
        print(f"  Exit Time: {target_trade.exit_time}")
        print(f"  Exit Price: {target_trade.exit_price}")
        print(f"  Exit Reason: {target_trade.exit_reason}")
        print(f"  TP Hits: {target_trade.tp_hits}")
        print(f"  Final P&L: ${target_trade.pnl:,.2f}")
        
        print(f"\n  Partial Exits Detail:")
        for i, pe in enumerate(target_trade.partial_exits):
            print(f"    Exit {i+1}:")
            print(f"      Time: {pe.time}")
            print(f"      Price: {pe.price}")
            print(f"      Size: {pe.size:,.0f}")
            print(f"      TP Level: {pe.tp_level}")
            print(f"      P&L: ${pe.pnl:,.2f}")
            print(f"      Cumulative P&L: ${pe.cumulative_pnl:,.2f}")
    else:
        print("ERROR: Could not find trade entered at 2025-03-30 21:15")
    
    # Also check for trades around that time
    print("\n\nOther trades around that time:")
    for trade in results['trades']:
        entry_time = trade.entry_time
        if '2025-03-30 20:00' <= entry_time.strftime('%Y-%m-%d %H:%M') <= '2025-03-31 02:00':
            print(f"  {entry_time} - {trade.direction} - Exit: {trade.exit_time} - {trade.exit_reason}")

if __name__ == "__main__":
    main()