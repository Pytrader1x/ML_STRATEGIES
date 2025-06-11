"""
Debug TSL behavior to understand why trades are exiting at specific levels
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_optimized_strategy, FOREX_PIP_SIZE
from technical_indicators_custom import TIC
from datetime import datetime

def debug_tsl_behavior():
    """Debug TSL behavior in detail"""
    
    print("Debugging TSL Behavior")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use a smaller sample for detailed analysis
    df_test = df.tail(5000).copy()
    
    # Calculate indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Run strategy with verbose logging
    strategy = create_optimized_strategy(
        initial_capital=100_000,
        tsl_activation_pips=15,
        tsl_min_profit_pips=5,
        tsl_initial_buffer_multiplier=2.0,
        verbose=False  # We'll do our own logging
    )
    
    # Track TSL behavior manually
    print("\nRunning backtest and tracking TSL behavior...")
    
    # We'll need to modify the strategy to log TSL updates
    # For now, let's run the backtest and analyze the results
    results = strategy.run_backtest(df_test)
    
    # Analyze all trades
    print(f"\nTotal trades: {len(results['trades'])}")
    
    # Group by exit reason
    exit_reasons = {}
    for trade in results['trades']:
        reason = trade.exit_reason.value if trade.exit_reason else 'unknown'
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(trade)
    
    print("\nExit reason breakdown:")
    for reason, trades in exit_reasons.items():
        print(f"  {reason}: {len(trades)} trades")
    
    # Focus on different exit types
    for exit_type in ['trailing_stop', 'take_profit_1', 'take_profit_2', 'stop_loss']:
        if exit_type in exit_reasons:
            trades = exit_reasons[exit_type]
            print(f"\n{exit_type.upper()} Analysis ({len(trades)} trades):")
            
            pip_gains = []
            for trade in trades[:5]:  # Show first 5
                if trade.direction.value == 'long':
                    pips = (trade.exit_price - trade.entry_price) * 10000
                else:
                    pips = (trade.entry_price - trade.exit_price) * 10000
                pip_gains.append(pips)
                
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                
                print(f"  {trade.direction.value.upper()}: Entry={trade.entry_price:.5f}, "
                      f"Exit={trade.exit_price:.5f}, Pips={pips:.1f}, "
                      f"Duration={duration:.1f}h")
            
            if pip_gains:
                print(f"  Average: {np.mean(pip_gains):.1f} pips")
                print(f"  Range: {np.min(pip_gains):.1f} - {np.max(pip_gains):.1f} pips")
    
    # Look for trades that might have been affected by the 15-pip issue
    print("\n" + "=" * 60)
    print("Looking for trades with suspicious exit patterns...")
    
    suspicious_trades = []
    for trade in results['trades']:
        if trade.direction.value == 'long':
            pips = (trade.exit_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - trade.exit_price) * 10000
        
        # Check if exit is suspiciously close to common levels
        if 14 <= pips <= 16:  # Near 15 pips
            suspicious_trades.append(('15_pip_area', trade, pips))
        elif 4 <= pips <= 6:  # Near 5 pips (minimum)
            suspicious_trades.append(('min_profit', trade, pips))
        elif -1 <= pips <= 1:  # Near break-even
            suspicious_trades.append(('breakeven', trade, pips))
    
    if suspicious_trades:
        print(f"\nFound {len(suspicious_trades)} trades with suspicious exits:")
        for pattern, trade, pips in suspicious_trades[:10]:
            print(f"  {pattern}: {trade.exit_reason.value if trade.exit_reason else 'unknown'} "
                  f"exit at {pips:.1f} pips")
    
    # Analyze ATR values at exit
    print("\n" + "=" * 60)
    print("Analyzing market conditions at TSL exits...")
    
    tsl_trades = [t for t in results['trades'] 
                  if t.exit_reason and t.exit_reason.value == 'trailing_stop']
    
    if tsl_trades:
        print(f"\nChecking ATR values for {len(tsl_trades)} TSL exits:")
        for i, trade in enumerate(tsl_trades[:3]):
            # Find the exit bar
            exit_idx = df_test.index.get_loc(trade.exit_time)
            if exit_idx > 0:
                exit_row = df_test.iloc[exit_idx]
                atr = exit_row['IC_ATR_Normalized']
                atr_pips = atr * 10000
                
                print(f"\nTrade {i+1}:")
                print(f"  ATR at exit: {atr:.5f} ({atr_pips:.1f} pips)")
                print(f"  ATR × 1.2: {atr * 1.2:.5f} ({atr_pips * 1.2:.1f} pips)")
                print(f"  ATR × 2.4 (buffered): {atr * 2.4:.5f} ({atr_pips * 2.4:.1f} pips)")
                
                # Calculate what the TSL would have been
                if trade.direction.value == 'long':
                    theoretical_tsl = trade.exit_price - (atr * 1.2)
                    theoretical_tsl_buffered = trade.exit_price - (atr * 2.4)
                    min_profit_tsl = trade.entry_price + (5 * FOREX_PIP_SIZE)
                    
                    print(f"  Theoretical TSL: {theoretical_tsl:.5f}")
                    print(f"  Theoretical TSL (buffered): {theoretical_tsl_buffered:.5f}")
                    print(f"  Min profit TSL: {min_profit_tsl:.5f}")
                    print(f"  Actual exit: {trade.exit_price:.5f}")

if __name__ == "__main__":
    debug_tsl_behavior()