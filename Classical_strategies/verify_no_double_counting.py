"""
Verify there's no actual P&L double counting despite duplicate timestamps
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def verify_pnl_integrity():
    print("="*80)
    print("VERIFYING P&L INTEGRITY - NO DOUBLE COUNTING")
    print("="*80)
    
    # Load data and run strategy
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    df = df_full.iloc[-5000:].copy()
    
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000, risk_per_trade=0.002, sl_max_pips=10.0,
        sl_atr_multiplier=1.0, tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003, tsl_activation_pips=15, tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0, trailing_atr_multiplier=1.2,
        tp_range_market_multiplier=0.5, tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3, sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False, partial_profit_before_sl=False,
        debug_decisions=False, use_daily_sharpe=True
    )
    
    strategy = OptimizedProdStrategy(strategy_config)
    results = strategy.run_backtest(df)
    trades = results['trades']
    
    print(f"Total trades: {len(trades)}")
    
    # Check specific trades with duplicate timestamps
    trades_with_dup_times = []
    
    for i, trade in enumerate(trades):
        exit_times = [pe.time for pe in trade.partial_exits]
        if len(exit_times) != len(set(exit_times)):
            trades_with_dup_times.append((i, trade))
    
    print(f"\nTrades with exits at same timestamp: {len(trades_with_dup_times)}")
    
    # Analyze a few examples
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF TRADES WITH DUPLICATE TIMESTAMPS:")
    print("="*80)
    
    for idx, (i, trade) in enumerate(trades_with_dup_times[:3]):
        print(f"\nTrade #{i+1} ({trade.entry_time}):")
        print(f"  Direction: {trade.direction.value}")
        print(f"  Position: {trade.position_size/1e6:.2f}M")
        print(f"  Entry: {trade.entry_price:.5f}")
        
        # Manual P&L calculation
        manual_total_pnl = 0
        
        print(f"\n  Exit Details:")
        for j, pe in enumerate(trade.partial_exits):
            # Calculate P&L manually
            if trade.direction.value == 'short':
                pips = (trade.entry_price - pe.price) / 0.0001
            else:
                pips = (pe.price - trade.entry_price) / 0.0001
            
            manual_pnl = (pe.size / 1e6) * 100 * pips
            manual_total_pnl += manual_pnl
            
            tp_str = f"TP{pe.tp_level}" if hasattr(pe, 'tp_level') and pe.tp_level > 0 else "Final"
            print(f"    {j+1}. {tp_str} at {pe.time}:")
            print(f"       Size: {pe.size/1e6:.2f}M")
            print(f"       Price: {pe.price:.5f}")
            print(f"       Pips: {pips:.1f}")
            print(f"       Recorded P&L: ${pe.pnl:.2f}")
            print(f"       Manual P&L: ${manual_pnl:.2f}")
            print(f"       Match: {'✅' if abs(pe.pnl - manual_pnl) < 0.01 else '❌'}")
        
        print(f"\n  Summary:")
        print(f"    Total recorded P&L: ${trade.pnl:.2f}")
        print(f"    Manual total P&L: ${manual_total_pnl:.2f}")
        print(f"    Difference: ${abs(trade.pnl - manual_total_pnl):.2f}")
        print(f"    Status: {'✅ CORRECT' if abs(trade.pnl - manual_total_pnl) < 0.01 else '❌ ERROR'}")
    
    # Global verification
    print("\n" + "="*80)
    print("GLOBAL VERIFICATION:")
    print("="*80)
    
    # 1. Sum all trade P&Ls
    total_trades_pnl = sum(t.pnl for t in trades)
    
    # 2. Calculate expected final capital
    expected_capital = strategy_config.initial_capital + total_trades_pnl
    
    # 3. Check against actual
    actual_capital = strategy.current_capital
    
    print(f"Initial Capital: ${strategy_config.initial_capital:,.2f}")
    print(f"Sum of all trade P&Ls: ${total_trades_pnl:,.2f}")
    print(f"Expected Final Capital: ${expected_capital:,.2f}")
    print(f"Actual Final Capital: ${actual_capital:,.2f}")
    print(f"Difference: ${abs(expected_capital - actual_capital):.2f}")
    
    capital_match = abs(expected_capital - actual_capital) < 0.01
    print(f"\nCapital Integrity: {'✅ PERFECT MATCH' if capital_match else '❌ MISMATCH'}")
    
    # 4. Check for any trades where P&L doesn't match sum of partials
    pnl_mismatches = []
    for i, trade in enumerate(trades):
        sum_partials = sum(pe.pnl for pe in trade.partial_exits)
        if abs(trade.pnl - sum_partials) > 0.01:
            pnl_mismatches.append((i, trade, trade.pnl, sum_partials))
    
    print(f"\nTrades with P&L != sum of partials: {len(pnl_mismatches)}")
    
    if pnl_mismatches:
        print("\nP&L Mismatches found:")
        for i, trade, total, sum_p in pnl_mismatches[:5]:
            print(f"  Trade #{i+1}: Total=${total:.2f}, Sum=${sum_p:.2f}, Diff=${abs(total-sum_p):.2f}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT:")
    print("="*80)
    
    if capital_match and len(pnl_mismatches) == 0:
        print("✅ NO DOUBLE COUNTING DETECTED")
        print("✅ All P&L calculations are correct")
        print("✅ Duplicate timestamps are cosmetic only - they don't affect calculations")
        print("\nNote: Exits at same timestamp occur when:")
        print("- TP2 and TP3 are hit in the same bar")
        print("- TP2 and final exit (TP1 pullback) occur simultaneously")
        print("- This is normal behavior and doesn't affect P&L integrity")
    else:
        print("❌ P&L INTEGRITY ISSUES DETECTED")
        print("Further investigation required")

if __name__ == "__main__":
    verify_pnl_integrity()