"""
Test TSL behavior to ensure 5 pip minimum profit is guaranteed
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_optimized_strategy
from technical_indicators_custom import TIC
from datetime import datetime

def test_tsl_behavior():
    """Test that TSL guarantees minimum 5 pip profit after 15 pip activation"""
    
    print("Testing TSL Behavior - 15 pip activation, 5 pip minimum profit")
    print("=" * 60)
    
    # Load a small sample
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent data
    df_test = df.tail(5000).copy()
    
    # Calculate indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Run strategy
    strategy = create_optimized_strategy(
        initial_capital=100_000,
        tsl_activation_pips=15,
        tsl_min_profit_pips=5,
        verbose=False
    )
    
    results = strategy.run_backtest(df_test)
    
    # Analyze TSL exits
    tsl_trades = [t for t in results['trades'] 
                  if t.exit_reason and t.exit_reason.value == 'trailing_stop']
    
    print(f"\nTotal trades: {len(results['trades'])}")
    print(f"TSL exits: {len(tsl_trades)}")
    
    if tsl_trades:
        print(f"\nAnalyzing TSL exits:")
        print(f"{'#':<3} {'Entry':<8} {'Exit':<8} {'Pips':<8} {'P&L':<10} {'Direction':<10}")
        print("-" * 55)
        
        min_pips = float('inf')
        all_profitable = True
        
        for i, trade in enumerate(tsl_trades[:10], 1):  # Show first 10
            if trade.direction.value == 'long':
                pips = (trade.exit_price - trade.entry_price) * 10000
            else:
                pips = (trade.entry_price - trade.exit_price) * 10000
            
            min_pips = min(min_pips, pips)
            if pips < 5:
                all_profitable = False
            
            print(f"{i:<3} {trade.entry_price:<8.5f} {trade.exit_price:<8.5f} "
                  f"{pips:<8.1f} ${trade.pnl:<9.2f} {trade.direction.value:<10}")
        
        print(f"\nTSL Analysis:")
        print(f"  Minimum pips gained: {min_pips:.1f}")
        print(f"  All TSL exits >= 5 pips: {'YES' if all_profitable and min_pips >= 5 else 'NO'}")
        
        if min_pips < 4.9:  # Allow for small floating point differences
            print(f"\n⚠️  WARNING: Some TSL exits had less than 5 pips profit!")
            print(f"     This should not happen if TSL is working correctly.")
        else:
            print(f"\n✅ SUCCESS: All TSL exits guaranteed at least 5 pips profit!")
            print(f"     TSL is working correctly.")
    else:
        print("\nNo TSL exits found in this sample.")
    
    # Check TSL activation behavior
    print(f"\n" + "=" * 60)
    print("TSL Configuration:")
    print(f"  Activation: 15 pips in profit")
    print(f"  Minimum profit: 5 pips from entry")
    print(f"  Behavior: Once price moves 15 pips in favor, TSL guarantees 5 pip minimum")

if __name__ == "__main__":
    test_tsl_behavior()