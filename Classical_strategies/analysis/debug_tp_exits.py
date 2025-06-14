#!/usr/bin/env python3
"""
Debug script to understand why TP exits show tp_hits=0
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import os

def analyze_tp_exit_issue():
    """Analyze why tp_hits shows 0 for TP exits"""
    
    # Load data
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Filter to Feb-March 2025
    df = df[(df.index >= '2025-02-01') & (df.index <= '2025-03-31')]
    
    # Calculate indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Run strategy
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        realistic_costs=True,
        verbose=True
    )
    
    strategy = OptimizedProdStrategy(config)
    results = strategy.run_backtest(df)
    
    # Analyze TP exits
    tp_exits = []
    for trade in results['trades']:
        if trade.exit_reason and 'take_profit' in trade.exit_reason.value:
            tp_exits.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'exit_reason': trade.exit_reason.value,
                'tp_hits': trade.tp_hits,
                'partial_exits': len(trade.partial_exits),
                'position_size': trade.position_size,
                'remaining_size': trade.remaining_size,
                'pnl': trade.pnl,
                'tp_levels': trade.take_profits
            })
    
    print(f"\nFound {len(tp_exits)} trades with TP exits")
    print("\nFirst 10 TP exits:")
    for i, trade in enumerate(tp_exits[:10]):
        print(f"\nTrade {i+1}:")
        print(f"  Exit Reason: {trade['exit_reason']}")
        print(f"  TP Hits: {trade['tp_hits']}")
        print(f"  Partial Exits: {trade['partial_exits']}")
        print(f"  Position Size: {trade['position_size']/1e6:.1f}M")
        print(f"  Remaining Size: {trade['remaining_size']/1e6:.1f}M")
        print(f"  P&L: ${trade['pnl']:,.0f}")
    
    # Check for the specific pattern
    print("\n\nLooking for trades with tp_hits=0 but exit_reason=take_profit_1:")
    suspicious_trades = [t for t in tp_exits if t['tp_hits'] == 0 and t['exit_reason'] == 'take_profit_1']
    print(f"Found {len(suspicious_trades)} suspicious trades")
    
    # Analyze partial exits
    print("\n\nAnalyzing partial exits structure:")
    for trade in results['trades'][:5]:
        if len(trade.partial_exits) > 0:
            print(f"\nTrade with {len(trade.partial_exits)} partial exits:")
            for i, pe in enumerate(trade.partial_exits):
                print(f"  Partial Exit {i+1}:")
                print(f"    TP Level: {pe.tp_level}")
                print(f"    Size: {pe.size/1e6:.1f}M")
                print(f"    P&L: ${pe.pnl:,.0f}")

if __name__ == "__main__":
    analyze_tp_exit_issue()