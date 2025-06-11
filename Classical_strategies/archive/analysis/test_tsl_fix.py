"""
Test the TSL fix to verify it prevents immediate exits at 15 pips
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_optimized_strategy
from technical_indicators_custom import TIC
from datetime import datetime

def test_tsl_fix():
    """Test that the TSL fix prevents immediate exits"""
    
    print("Testing TSL Fix - Initial Buffer Implementation")
    print("=" * 60)
    
    # Load a small sample
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent data
    df_test = df.tail(10000).copy()
    
    # Calculate indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Test with different buffer multipliers
    buffer_multipliers = [1.0, 1.5, 2.0, 2.5]
    
    for buffer_mult in buffer_multipliers:
        print(f"\nTesting with TSL initial buffer multiplier: {buffer_mult}")
        print("-" * 40)
        
        strategy = create_optimized_strategy(
            initial_capital=100_000,
            tsl_activation_pips=15,
            tsl_min_profit_pips=5,
            tsl_initial_buffer_multiplier=buffer_mult,
            verbose=False
        )
        
        results = strategy.run_backtest(df_test)
        
        # Analyze TSL exits
        tsl_trades = [t for t in results['trades'] 
                      if t.exit_reason and t.exit_reason.value == 'trailing_stop']
        
        if tsl_trades:
            # Calculate pip statistics
            pip_gains = []
            for trade in tsl_trades:
                if trade.direction.value == 'long':
                    pips = (trade.exit_price - trade.entry_price) * 10000
                else:
                    pips = (trade.entry_price - trade.exit_price) * 10000
                pip_gains.append(pips)
            
            # Statistics
            avg_pips = np.mean(pip_gains)
            min_pips = np.min(pip_gains)
            max_pips = np.max(pip_gains)
            std_pips = np.std(pip_gains)
            
            # Count exits near 15 pips (within 2 pips)
            near_activation = sum(1 for p in pip_gains if 13 <= p <= 17)
            pct_near_activation = (near_activation / len(pip_gains)) * 100
            
            print(f"  TSL exits: {len(tsl_trades)}")
            print(f"  Average pips: {avg_pips:.1f}")
            print(f"  Min pips: {min_pips:.1f}")
            print(f"  Max pips: {max_pips:.1f}")
            print(f"  Std dev: {std_pips:.1f}")
            print(f"  Exits near 15 pips (13-17): {near_activation} ({pct_near_activation:.1f}%)")
            
            # Show distribution
            print(f"\n  Pip distribution:")
            bins = [5, 10, 15, 20, 25, 30, 40, 50, 100]
            for i in range(len(bins)-1):
                count = sum(1 for p in pip_gains if bins[i] <= p < bins[i+1])
                if count > 0:
                    print(f"    {bins[i]}-{bins[i+1]} pips: {count} trades")
            count_large = sum(1 for p in pip_gains if p >= 100)
            if count_large > 0:
                print(f"    100+ pips: {count_large} trades")
    
    print("\n" + "=" * 60)
    print("Analysis Summary:")
    print("- Buffer multiplier 1.0 = No buffer (original behavior)")
    print("- Buffer multiplier 2.0 = TSL placed 2x further on activation")
    print("- Higher buffer = More room for price to breathe after activation")
    print("- Goal: Reduce immediate exits at 15 pips while maintaining protection")

if __name__ == "__main__":
    test_tsl_fix()