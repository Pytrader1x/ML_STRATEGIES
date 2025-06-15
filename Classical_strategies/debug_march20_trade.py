"""
Debug script for the specific trade entered at 2025-03-20 21:15
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
        risk_per_trade=0.002,
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
        debug_decisions=True,
        use_daily_sharpe=True
    )
    return OptimizedProdStrategy(strategy_config)

def main():
    print("="*80)
    print("DEBUG: 2025-03-20 21:15 TRADE ANALYSIS")
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
    
    # Take only the last 5000 rows
    df = df_full.iloc[-5000:].copy()
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
    strategy = create_debug_strategy()
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.run_backtest(df)
    
    # Find all trades around March 30, 2025 (NOT March 20)
    print("\n" + "="*80)
    print("SEARCHING FOR TRADES AROUND 2025-03-30")
    print("="*80)
    
    march30_trades = []
    for trade in results['trades']:
        entry_date = trade.entry_time.strftime('%Y-%m-%d')
        if '2025-03-29' <= entry_date <= '2025-03-31':
            march30_trades.append(trade)
            print(f"\nTrade found:")
            print(f"  Entry: {trade.entry_time} at {trade.entry_price}")
            print(f"  Direction: {trade.direction}")
            print(f"  Position Size: {trade.position_size:,.0f}")
    
    # Find the specific 21:15 trade
    target_trade = None
    for trade in march30_trades:
        if trade.entry_time.strftime('%Y-%m-%d %H:%M') == '2025-03-30 21:15':
            target_trade = trade
            break
    
    if target_trade:
        print("\n" + "="*80)
        print("DETAILED ANALYSIS OF 2025-03-30 21:15 TRADE")
        print("="*80)
        
        print(f"\nTRADE ENTRY:")
        print(f"  Time: {target_trade.entry_time}")
        print(f"  Price: {target_trade.entry_price}")
        print(f"  Direction: {target_trade.direction}")
        print(f"  Initial Position Size: {target_trade.position_size:,.0f} ({target_trade.position_size/1e6:.1f}M)")
        print(f"  Stop Loss: {target_trade.stop_loss}")
        print(f"  Take Profits: {target_trade.take_profits}")
        
        print(f"\nTRADE EXIT:")
        print(f"  Exit Time: {target_trade.exit_time}")
        print(f"  Exit Price: {target_trade.exit_price}")
        print(f"  Exit Reason: {target_trade.exit_reason}")
        print(f"  TP Hits: {target_trade.tp_hits}")
        print(f"  Total P&L: ${target_trade.pnl:,.2f}")
        
        print(f"\nPARTIAL EXITS BREAKDOWN:")
        print(f"  Number of Partial Exits: {len(target_trade.partial_exits)}")
        
        cumulative_size = 0
        for i, pe in enumerate(target_trade.partial_exits):
            cumulative_size += pe.size
            print(f"\n  Exit #{i+1}:")
            print(f"    Time: {pe.time}")
            print(f"    Price: {pe.price}")
            print(f"    Size: {pe.size:,.0f} ({pe.size/1e6:.2f}M)")
            print(f"    TP Level: {pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'}")
            print(f"    P&L: ${pe.pnl:,.2f}")
            print(f"    Cumulative Size Exited: {cumulative_size:,.0f} ({cumulative_size/1e6:.2f}M)")
            
            # Calculate pips
            if target_trade.direction.value == 'short':
                pips = (target_trade.entry_price - pe.price) / 0.0001
            else:
                pips = (pe.price - target_trade.entry_price) / 0.0001
            print(f"    Pips: {pips:.1f}")
        
        print(f"\nEXIT VERIFICATION:")
        print(f"  Total Size Exited: {cumulative_size:,.0f}")
        print(f"  Original Position: {target_trade.position_size:,.0f}")
        print(f"  Difference: {target_trade.position_size - cumulative_size:,.0f}")
        
        # Check for the "TP1 triggered twice" issue
        tp1_exits = [pe for pe in target_trade.partial_exits if hasattr(pe, 'tp_level') and pe.tp_level == 1]
        if len(tp1_exits) > 1:
            print(f"\n⚠️  WARNING: TP1 appears to have triggered {len(tp1_exits)} times!")
            for i, tp1 in enumerate(tp1_exits):
                print(f"    TP1 Exit #{i+1}: {tp1.time} - Size: {tp1.size/1e6:.2f}M")
    else:
        print("\nERROR: Could not find trade entered at 2025-03-30 21:15")
        print("\nAll trades on March 30:")
        for trade in march30_trades:
            print(f"  {trade.entry_time} - {trade.direction}")
    
    # Also look for trades near that time
    print("\n\nTrades within 1 hour of 2025-03-30 21:15:")
    target_time = pd.Timestamp('2025-03-30 21:15:00')
    for trade in results['trades']:
        time_diff = abs((trade.entry_time - target_time).total_seconds() / 60)  # minutes
        if time_diff <= 60:  # within 60 minutes
            print(f"  {trade.entry_time} - {trade.direction} - {trade.position_size/1e6:.1f}M - Exit: {trade.exit_reason}")

if __name__ == "__main__":
    main()