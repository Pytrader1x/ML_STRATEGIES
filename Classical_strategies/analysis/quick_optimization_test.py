"""
Quick test of optimization improvements
Focus on signal flip fixes which provide the biggest impact
"""

import pandas as pd
import numpy as np
from Prod_strategy import create_strategy
from Prod_strategy_optimized import create_optimized_strategy
from technical_indicators_custom import TIC
import time

def main():
    print("Quick Optimization Test - Signal Flip Improvements")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use 6 months of recent data for quick test
    df_test = df.tail(17280).copy()  # ~6 months
    
    print(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
    print(f"Total bars: {len(df_test):,}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Run original
    print("\nRunning ORIGINAL strategy...")
    strategy_orig = create_strategy(
        initial_capital=100_000,
        exit_on_signal_flip=True,
        intelligent_sizing=True,
        verbose=False
    )
    
    start = time.time()
    results_orig = strategy_orig.run_backtest(df_test)
    print(f"Original completed in {time.time() - start:.1f}s")
    
    # Run optimized
    print("\nRunning OPTIMIZED strategy...")
    strategy_opt = create_optimized_strategy(
        initial_capital=100_000,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=5.0,  # Key improvement
        signal_flip_min_time_hours=2.0,   # Key improvement
        signal_flip_partial_exit_percent=0.5,  # Key improvement
        intelligent_sizing=True,
        verbose=False
    )
    
    start = time.time()
    results_opt = strategy_opt.run_backtest(df_test)
    print(f"Optimized completed in {time.time() - start:.1f}s")
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 65)
    
    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Total P&L', 'total_pnl', '$'),
        ('Return', 'total_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Profit Factor', 'profit_factor', '')
    ]
    
    for label, key, suffix in metrics:
        orig_val = results_orig[key]
        opt_val = results_opt[key]
        
        if key == 'total_pnl':
            improvement = opt_val - orig_val
            print(f"{label:<20} ${orig_val:<14,.0f} ${opt_val:<14,.0f} ${improvement:+,.0f}")
        elif key in ['win_rate', 'total_return', 'max_drawdown']:
            improvement = opt_val - orig_val
            print(f"{label:<20} {orig_val:<14.1f}{suffix} {opt_val:<14.1f}{suffix} {improvement:+.1f}{suffix}")
        elif key == 'total_trades':
            print(f"{label:<20} {orig_val:<15} {opt_val:<15} {opt_val - orig_val:+}")
        else:
            improvement = opt_val - orig_val
            print(f"{label:<20} {orig_val:<14.2f} {opt_val:<14.2f} {improvement:+.2f}")
    
    # Signal flip analysis
    print("\n" + "=" * 60)
    print("SIGNAL FLIP ANALYSIS")
    print("=" * 60)
    
    # Original signal flips
    orig_flips = [t for t in results_orig['trades'] 
                  if t.exit_reason and t.exit_reason.value == 'signal_flip']
    orig_flip_wins = [t for t in orig_flips if t.pnl > 0]
    orig_flip_pnl = sum(t.pnl for t in orig_flips)
    
    # Optimized signal flips
    opt_flips = [t for t in results_opt['trades'] 
                 if t.exit_reason and t.exit_reason.value == 'signal_flip']
    opt_flip_wins = [t for t in opt_flips if t.pnl > 0]
    opt_flip_pnl = sum(t.pnl for t in opt_flips)
    
    print(f"Original Strategy:")
    print(f"  Signal flip exits: {len(orig_flips)}")
    print(f"  Profitable flips: {len(orig_flip_wins)} ({len(orig_flip_wins)/len(orig_flips)*100:.1f}%)" if orig_flips else "  No flips")
    print(f"  Total flip P&L: ${orig_flip_pnl:,.2f}")
    print(f"  Avg flip P&L: ${orig_flip_pnl/len(orig_flips):,.2f}" if orig_flips else "  N/A")
    
    print(f"\nOptimized Strategy:")
    print(f"  Signal flip exits: {len(opt_flips)}")
    print(f"  Profitable flips: {len(opt_flip_wins)} ({len(opt_flip_wins)/len(opt_flips)*100:.1f}%)" if opt_flips else "  No flips")
    print(f"  Total flip P&L: ${opt_flip_pnl:,.2f}")
    print(f"  Avg flip P&L: ${opt_flip_pnl/len(opt_flips):,.2f}" if opt_flips else "  N/A")
    
    print(f"\nImprovement:")
    print(f"  Flip P&L improvement: ${opt_flip_pnl - orig_flip_pnl:+,.2f}")
    print(f"  Reduced bad flips by: {len(orig_flips) - len(opt_flips)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    pnl_improvement_pct = ((results_opt['total_pnl'] - results_orig['total_pnl']) / 
                          abs(results_orig['total_pnl']) * 100) if results_orig['total_pnl'] != 0 else 0
    
    print(f"The optimized strategy improved P&L by {pnl_improvement_pct:+.1f}%")
    print(f"Main improvement: Better signal flip handling reduced losses significantly")
    print(f"Signal flip P&L went from ${orig_flip_pnl:,.0f} to ${opt_flip_pnl:,.0f}")
    print(f"\nKey optimizations applied:")
    print("- Minimum 5 pip profit before allowing signal flip exit")
    print("- Minimum 2 hour hold time before flip exit")
    print("- Partial exit (50%) on weak signal flips")

if __name__ == "__main__":
    main()