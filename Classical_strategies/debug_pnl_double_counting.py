"""
Deep debugging of P&L calculation logic to check for double counting
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def debug_pnl_calculations():
    print("="*80)
    print("DEEP P&L DOUBLE COUNTING ANALYSIS")
    print("="*80)
    
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take last 5000 rows
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy with debug enabled
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
    
    # Track capital changes manually
    manual_capital = strategy_config.initial_capital
    capital_changes = []
    
    results = strategy.run_backtest(df)
    trades = results['trades']
    
    print(f"\nTotal trades analyzed: {len(trades)}")
    print(f"Capital changes recorded: {len(capital_changes)}")
    
    # Analyze trades with multiple exits
    multi_exit_trades = [t for t in trades if len(t.partial_exits) > 1]
    print(f"Trades with multiple exits: {len(multi_exit_trades)}")
    
    # Check for double counting
    double_count_issues = []
    
    for i, trade in enumerate(trades):
        issues = []
        
        # 1. Check if sum of partial P&Ls equals total P&L
        sum_partial_pnls = sum(pe.pnl for pe in trade.partial_exits)
        pnl_diff = abs(trade.pnl - sum_partial_pnls)
        
        if pnl_diff > 0.01:
            issues.append(f"P&L sum mismatch: Total={trade.pnl:.2f}, Sum={sum_partial_pnls:.2f}, Diff={pnl_diff:.2f}")
        
        # 2. Check if partial_pnl field matches sum of partials (excluding final)
        if hasattr(trade, 'partial_pnl'):
            # partial_pnl should be sum of all TP exits (not including final non-TP exit)
            tp_exits = [pe for pe in trade.partial_exits if hasattr(pe, 'tp_level') and pe.tp_level > 0]
            expected_partial_pnl = sum(pe.pnl for pe in tp_exits)
            
            # For trades that exit fully via TPs, partial_pnl might equal total pnl
            if len(tp_exits) == len(trade.partial_exits):
                expected_partial_pnl = trade.pnl
            
            partial_pnl_diff = abs(trade.partial_pnl - expected_partial_pnl)
            if partial_pnl_diff > 0.01 and trade.partial_pnl != trade.pnl:
                issues.append(f"Partial P&L mismatch: Field={trade.partial_pnl:.2f}, Expected={expected_partial_pnl:.2f}")
        
        # 3. Check for duplicate P&L entries at same timestamp
        exit_times = [pe.time for pe in trade.partial_exits]
        if len(exit_times) != len(set(exit_times)):
            # Find duplicates
            seen = set()
            duplicates = []
            for et in exit_times:
                if et in seen:
                    duplicates.append(et)
                seen.add(et)
            issues.append(f"Duplicate exits at same time: {duplicates}")
        
        # 4. Verify position sizing adds up
        total_exit_size = sum(pe.size for pe in trade.partial_exits)
        size_diff = abs(trade.position_size - total_exit_size)
        if size_diff > 1:  # 1 unit tolerance
            issues.append(f"Position size mismatch: Initial={trade.position_size/1e6:.2f}M, Exited={total_exit_size/1e6:.2f}M")
        
        # 5. Manual P&L calculation check
        manual_trade_pnl = 0
        for pe in trade.partial_exits:
            if trade.direction.value == 'short':
                price_change = trade.entry_price - pe.price
            else:
                price_change = pe.price - trade.entry_price
            
            pips = price_change / 0.0001
            manual_pnl = (pe.size / 1e6) * 100 * pips
            manual_trade_pnl += manual_pnl
            
            # Check individual exit P&L
            individual_diff = abs(manual_pnl - pe.pnl)
            if individual_diff > 0.01:
                issues.append(f"Exit P&L calc error at {pe.time}: Manual={manual_pnl:.2f}, Recorded={pe.pnl:.2f}")
        
        # Check total manual vs recorded
        total_manual_diff = abs(manual_trade_pnl - trade.pnl)
        if total_manual_diff > 0.01:
            issues.append(f"Total P&L calc error: Manual={manual_trade_pnl:.2f}, Recorded={trade.pnl:.2f}")
        
        if issues:
            double_count_issues.append((i, trade, issues))
    
    # Print detailed issues
    if double_count_issues:
        print(f"\n{'='*80}")
        print(f"FOUND {len(double_count_issues)} TRADES WITH P&L ISSUES:")
        print(f"{'='*80}")
        
        for idx, (i, trade, issues) in enumerate(double_count_issues[:5]):  # First 5
            print(f"\nTrade #{i+1} ({trade.entry_time}):")
            print(f"  Direction: {trade.direction.value}")
            print(f"  Entry: {trade.entry_price:.5f}")
            print(f"  Total P&L: ${trade.pnl:.2f}")
            print(f"  Partial P&L field: ${getattr(trade, 'partial_pnl', 0):.2f}")
            print(f"  Exit count: {len(trade.partial_exits)}")
            print(f"  Issues:")
            for issue in issues:
                print(f"    ❌ {issue}")
            
            # Show exit breakdown
            print(f"  Exit breakdown:")
            for j, pe in enumerate(trade.partial_exits):
                tp_str = f"TP{pe.tp_level}" if hasattr(pe, 'tp_level') and pe.tp_level > 0 else "Final"
                print(f"    {j+1}. {tp_str} at {pe.time}: {pe.size/1e6:.2f}M @ {pe.price:.5f} = ${pe.pnl:.2f}")
    else:
        print("\n✅ NO P&L DOUBLE COUNTING ISSUES FOUND!")
    
    # Capital tracking analysis
    print(f"\n{'='*80}")
    print("CAPITAL TRACKING ANALYSIS:")
    print(f"{'='*80}")
    
    # Manual capital calculation
    manual_final_capital = strategy_config.initial_capital
    for trade in trades:
        manual_final_capital += trade.pnl
    
    print(f"Initial Capital: ${strategy_config.initial_capital:,.2f}")
    print(f"Sum of Trade P&Ls: ${sum(t.pnl for t in trades):,.2f}")
    print(f"Expected Final Capital: ${manual_final_capital:,.2f}")
    print(f"Actual Final Capital: ${strategy.current_capital:,.2f}")
    print(f"Difference: ${abs(manual_final_capital - strategy.current_capital):,.2f}")
    
    if abs(manual_final_capital - strategy.current_capital) > 0.01:
        print("❌ CAPITAL TRACKING ERROR DETECTED!")
    else:
        print("✅ Capital tracking is correct")
    
    # Check for any negative position sizes or other anomalies
    print(f"\n{'='*80}")
    print("ANOMALY CHECKS:")
    print(f"{'='*80}")
    
    anomalies = []
    for i, trade in enumerate(trades):
        # Check for negative position sizes
        if trade.position_size < 0:
            anomalies.append(f"Trade #{i+1}: Negative position size: {trade.position_size}")
        
        # Check for negative remaining size
        if hasattr(trade, 'remaining_size') and trade.remaining_size < -1:  # Allow small negative due to rounding
            anomalies.append(f"Trade #{i+1}: Negative remaining size: {trade.remaining_size}")
        
        # Check for exits larger than position
        for pe in trade.partial_exits:
            if pe.size > trade.position_size * 1.01:  # 1% tolerance
                anomalies.append(f"Trade #{i+1}: Exit size ({pe.size/1e6:.2f}M) > position ({trade.position_size/1e6:.2f}M)")
    
    if anomalies:
        print("Found anomalies:")
        for anomaly in anomalies[:10]:  # First 10
            print(f"  ❌ {anomaly}")
    else:
        print("✅ No anomalies found")
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    
    total_issues = len(double_count_issues)
    print(f"Total P&L issues found: {total_issues}")
    print(f"Percentage of trades with issues: {(total_issues/len(trades)*100):.1f}%")
    
    if total_issues == 0 and abs(manual_final_capital - strategy.current_capital) < 0.01:
        print("\n✅ ALL P&L CALCULATIONS ARE CORRECT - NO DOUBLE COUNTING DETECTED")
    else:
        print("\n❌ P&L CALCULATION ISSUES DETECTED - INVESTIGATION NEEDED")
    
    return total_issues == 0

if __name__ == "__main__":
    debug_pnl_calculations()