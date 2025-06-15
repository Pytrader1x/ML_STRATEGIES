"""
Debug script for the specific trade entered at 2025-04-02 03:30
Investigating TP exit sequence and P&L calculations
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
        partial_profit_before_sl=False,  # Disable PP to focus on TP exits
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        relaxed_position_multiplier=0.5,
        relaxed_mode=False,
        realistic_costs=True,
        verbose=False,
        debug_decisions=True,  # Enable detailed debug output
        use_daily_sharpe=True
    )
    return OptimizedProdStrategy(strategy_config)

def main():
    print("="*80)
    print("DEBUG: 2025-04-02 03:30 TRADE ANALYSIS")
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
    
    # Take last 5000 rows (same as single run script)
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
    
    # Find the specific trade around 2025-04-02 03:30
    print("\n" + "="*80)
    print("SEARCHING FOR TRADES AROUND 2025-04-02 03:30")
    print("="*80)
    
    target_date = '2025-04-02'
    target_time = '03:30'
    
    april02_trades = []
    for trade in results['trades']:
        entry_date = trade.entry_time.strftime('%Y-%m-%d')
        entry_time = trade.entry_time.strftime('%H:%M')
        
        # Look for trades on April 2nd or close to 03:30
        if entry_date == target_date or (
            entry_date in ['2025-04-01', '2025-04-02', '2025-04-03'] and 
            '02:00' <= entry_time <= '05:00'
        ):
            april02_trades.append(trade)
            print(f"\nTrade found:")
            print(f"  Entry: {trade.entry_time} at {trade.entry_price}")
            print(f"  Direction: {trade.direction}")
            print(f"  Position Size: {trade.position_size:,.0f} ({trade.position_size/1e6:.1f}M)")
    
    # Find the closest match to 03:30
    target_trade = None
    target_datetime = pd.Timestamp(f'{target_date} {target_time}:00')
    
    if april02_trades:
        # Find closest to target time
        closest_trade = min(april02_trades, 
                          key=lambda t: abs((t.entry_time - target_datetime).total_seconds()))
        
        time_diff_minutes = abs((closest_trade.entry_time - target_datetime).total_seconds() / 60)
        if time_diff_minutes <= 60:  # Within 1 hour
            target_trade = closest_trade
    
    if target_trade:
        print("\n" + "="*80)
        print(f"DETAILED ANALYSIS OF CLOSEST TRADE: {target_trade.entry_time}")
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
        
        print(f"\nPARTIAL EXITS ANALYSIS:")
        print(f"  Number of Partial Exits: {len(target_trade.partial_exits)}")
        
        if len(target_trade.partial_exits) == 0:
            print("  ⚠️  NO PARTIAL EXITS FOUND!")
            print("  This suggests the trade went straight to final exit without hitting any TP levels.")
        else:
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
        
        print(f"\nTP LEVEL ANALYSIS:")
        tp_levels = target_trade.take_profits
        entry_price = target_trade.entry_price
        direction = target_trade.direction.value
        
        for i, tp_price in enumerate(tp_levels, 1):
            if direction == 'short':
                pips_to_tp = (entry_price - tp_price) / 0.0001
            else:
                pips_to_tp = (tp_price - entry_price) / 0.0001
            
            print(f"  TP{i}: {tp_price:.5f} ({pips_to_tp:.1f} pips from entry)")
            
            # Check if this TP was hit
            tp_hit = any(
                hasattr(pe, 'tp_level') and pe.tp_level == i 
                for pe in target_trade.partial_exits
            )
            print(f"        Hit: {'✅ YES' if tp_hit else '❌ NO'}")
        
        print(f"\nEXIT VERIFICATION:")
        print(f"  Total Size Exited: {cumulative_size:,.0f}")
        print(f"  Original Position: {target_trade.position_size:,.0f}")
        print(f"  Difference: {target_trade.position_size - cumulative_size:,.0f}")
        
        # Check price movement during trade
        print(f"\nPRICE MOVEMENT ANALYSIS:")
        entry_idx = None
        exit_idx = None
        
        for i, timestamp in enumerate(df.index):
            if timestamp == target_trade.entry_time:
                entry_idx = i
            if timestamp == target_trade.exit_time:
                exit_idx = i
        
        if entry_idx is not None and exit_idx is not None:
            trade_data = df.iloc[entry_idx:exit_idx+1]
            print(f"  Trade Duration: {len(trade_data)} bars ({len(trade_data)*15} minutes)")
            print(f"  High during trade: {trade_data['High'].max():.5f}")
            print(f"  Low during trade: {trade_data['Low'].min():.5f}")
            
            # Check which TP levels were actually touched by price
            for i, tp_price in enumerate(tp_levels, 1):
                if direction == 'short':
                    # For shorts, TP is below entry, so check if Low touched TP
                    tp_touched = trade_data['Low'].min() <= tp_price
                else:
                    # For longs, TP is above entry, so check if High touched TP
                    tp_touched = trade_data['High'].max() >= tp_price
                
                print(f"  TP{i} ({tp_price:.5f}) touched by price: {'✅ YES' if tp_touched else '❌ NO'}")
        
    else:
        print("\nERROR: Could not find trade at specified time")
        print("\nAll trades on April 2nd:")
        april_2_trades = [t for t in results['trades'] 
                         if t.entry_time.strftime('%Y-%m-%d') == '2025-04-02']
        for trade in april_2_trades:
            print(f"  {trade.entry_time} - {trade.direction} - {trade.position_size/1e6:.1f}M")

if __name__ == "__main__":
    main()