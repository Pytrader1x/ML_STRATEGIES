"""
Comprehensive P&L Integrity Validation
Validates that all trade P&L calculations are correct and consistent
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def validate_pnl_integrity():
    print("="*80)
    print("P&L INTEGRITY VALIDATION")
    print("="*80)
    
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take last 5000 rows for testing
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
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
    print(f"Total trades analyzed: {len(trades)}")
    
    # Validation counters
    position_errors = 0
    pnl_errors = 0
    tp_size_errors = 0
    cumulative_pnl_errors = 0
    
    # Track cumulative P&L
    cumulative_pnl = 0
    
    # Expected TP sizing (current implementation)
    expected_tp_percentages = {
        1: 0.5,      # TP1: 50% of initial
        2: 0.5,      # TP2: 50% of remaining
        3: 1.0       # TP3: 100% of remaining
    }
    
    print("\n" + "="*80)
    print("DETAILED VALIDATION")
    print("="*80)
    
    for i, trade in enumerate(trades):
        trade_issues = []
        
        # 1. POSITION SIZE VALIDATION
        total_exit_size = sum(pe.size for pe in trade.partial_exits)
        position_match = abs(trade.position_size - total_exit_size) < 1  # Allow 1 unit tolerance
        
        if not position_match:
            position_errors += 1
            trade_issues.append(f"Position mismatch: Initial={trade.position_size/1e6:.2f}M, Exited={total_exit_size/1e6:.2f}M")
        
        # 2. P&L CALCULATION VALIDATION
        calculated_pnl = 0
        remaining_position = trade.position_size
        
        for pe in trade.partial_exits:
            # Manual P&L calculation
            if trade.direction.value == 'short':
                price_change = trade.entry_price - pe.price
            else:
                price_change = pe.price - trade.entry_price
            
            pips = price_change / 0.0001
            manual_pnl = (pe.size / 1e6) * 100 * pips
            
            # Check individual exit P&L
            if abs(manual_pnl - pe.pnl) > 0.01:
                pnl_errors += 1
                trade_issues.append(f"P&L mismatch at {pe.time}: Calculated=${manual_pnl:.2f}, Reported=${pe.pnl:.2f}")
            
            calculated_pnl += pe.pnl
            
            # Validate TP sizing
            if hasattr(pe, 'tp_level') and pe.tp_level > 0:
                expected_percent = expected_tp_percentages.get(pe.tp_level, 1.0)
                
                if pe.tp_level == 1:
                    expected_size = trade.position_size * expected_percent
                else:
                    # Calculate remaining after previous TPs
                    prev_exits = sum(p.size for p in trade.partial_exits 
                                   if hasattr(p, 'tp_level') and p.tp_level < pe.tp_level)
                    remaining_before = trade.position_size - prev_exits
                    expected_size = remaining_before * expected_percent
                
                size_diff = abs(pe.size - expected_size)
                if size_diff > 1:  # Allow 1 unit tolerance
                    tp_size_errors += 1
                    trade_issues.append(f"TP{pe.tp_level} size mismatch: Expected={expected_size/1e6:.2f}M, Actual={pe.size/1e6:.2f}M")
        
        # Check total P&L
        if abs(calculated_pnl - trade.pnl) > 0.01:
            pnl_errors += 1
            trade_issues.append(f"Total P&L mismatch: Calculated=${calculated_pnl:.2f}, Reported=${trade.pnl:.2f}")
        
        # Update cumulative P&L
        cumulative_pnl += trade.pnl
        
        # Print issues for problematic trades
        if trade_issues:
            print(f"\nTrade #{i+1} ({trade.entry_time}):")
            print(f"  Direction: {trade.direction.value}")
            print(f"  Entry: {trade.entry_price:.5f}")
            print(f"  Exit Reason: {trade.exit_reason}")
            print(f"  Issues:")
            for issue in trade_issues:
                print(f"    ❌ {issue}")
    
    # 3. CUMULATIVE P&L VALIDATION
    strategy_final_pnl = results.get('total_pnl', 0)
    if abs(cumulative_pnl - strategy_final_pnl) > 0.01:
        cumulative_pnl_errors += 1
        print(f"\n❌ Cumulative P&L Error: Sum of trades=${cumulative_pnl:.2f}, Strategy total=${strategy_final_pnl:.2f}")
    
    # 4. CAPITAL TRACKING VALIDATION
    expected_final_capital = strategy_config.initial_capital + cumulative_pnl
    actual_final_capital = strategy.current_capital
    
    capital_match = abs(expected_final_capital - actual_final_capital) < 0.01
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n1. POSITION INTEGRITY:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Position Errors: {position_errors}")
    print(f"   Status: {'✅ PASS' if position_errors == 0 else '❌ FAIL'}")
    
    print(f"\n2. P&L CALCULATIONS:")
    print(f"   Individual P&L Errors: {pnl_errors}")
    print(f"   Total P&L Errors: {cumulative_pnl_errors}")
    print(f"   Status: {'✅ PASS' if pnl_errors == 0 else '❌ FAIL'}")
    
    print(f"\n3. TP SIZING:")
    print(f"   TP Size Errors: {tp_size_errors}")
    print(f"   Expected Sizing: TP1=50%, TP2=50% of remaining, TP3=100% of remaining")
    print(f"   Status: {'✅ PASS' if tp_size_errors == 0 else '❌ FAIL'}")
    
    print(f"\n4. CAPITAL TRACKING:")
    print(f"   Initial Capital: ${strategy_config.initial_capital:,.2f}")
    print(f"   Total P&L: ${cumulative_pnl:,.2f}")
    print(f"   Expected Final: ${expected_final_capital:,.2f}")
    print(f"   Actual Final: ${actual_final_capital:,.2f}")
    print(f"   Status: {'✅ PASS' if capital_match else '❌ FAIL'}")
    
    print(f"\n5. HIGH-LEVEL METRICS:")
    print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"   Win Rate: {results.get('win_rate', 0):.1f}%")
    print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    
    # Overall status
    all_pass = (position_errors == 0 and pnl_errors == 0 and 
                tp_size_errors == 0 and cumulative_pnl_errors == 0 and capital_match)
    
    print("\n" + "="*80)
    print(f"OVERALL VALIDATION: {'✅ ALL TESTS PASSED' if all_pass else '❌ ERRORS FOUND'}")
    print("="*80)
    
    # TP Distribution Analysis
    print("\n" + "="*80)
    print("TP EXIT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    tp_counts = {1: 0, 2: 0, 3: 0}
    tp_pnls = {1: [], 2: [], 3: []}
    
    for trade in trades:
        for pe in trade.partial_exits:
            if hasattr(pe, 'tp_level') and pe.tp_level > 0:
                tp_counts[pe.tp_level] = tp_counts.get(pe.tp_level, 0) + 1
                tp_pnls[pe.tp_level].append(pe.pnl)
    
    print("\nTP Hit Statistics:")
    for tp_level in [1, 2, 3]:
        count = tp_counts.get(tp_level, 0)
        pnls = tp_pnls.get(tp_level, [])
        avg_pnl = np.mean(pnls) if pnls else 0
        total_pnl = sum(pnls)
        
        print(f"  TP{tp_level}: {count} hits, Total P&L: ${total_pnl:,.2f}, Avg P&L: ${avg_pnl:,.2f}")
    
    # Exit reason breakdown
    exit_reasons = {}
    for trade in trades:
        reason = str(trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason)
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    print("\nExit Reason Breakdown:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(trades)) * 100
        print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    return all_pass

if __name__ == "__main__":
    validate_pnl_integrity()