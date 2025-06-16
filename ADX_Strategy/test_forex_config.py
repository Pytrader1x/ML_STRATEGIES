#!/usr/bin/env python3
"""
Test ADX strategy with forex-optimized configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import run_backtest
from config_forex import get_forex_config

# Test all three configurations
data_path = '../data/AUDUSD_MASTER_1H.csv'

for style in ['balanced', 'conservative', 'aggressive']:
    print(f"\n{'='*60}")
    print(f"Testing {style.upper()} Configuration")
    print('='*60)
    
    config = get_forex_config(style)
    
    results = run_backtest(
        data_path=data_path,
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
        initial_cash=config['backtest']['initial_cash'],
        commission=config['backtest']['commission'],
        plot=False,
        **config['strategy']
    )
    
    if results:
        print(f"\nFinal Results for {style.upper()}:")
        print(f"  Total Return: {results['total_return'] * 100:.2f}%")
        if results['sharpe_ratio']:
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        
        # Count trades from history file
        import pandas as pd
        if os.path.exists('trade_history.csv'):
            trades_df = pd.read_csv('trade_history.csv')
            print(f"  Total Trades: {len(trades_df)}")

# Also run with specific date range where we know there was high volatility
print(f"\n\n{'='*60}")
print("Testing on 2022 (High Volatility Period)")
print('='*60)

config = get_forex_config('balanced')
results = run_backtest(
    data_path=data_path,
    start_date='2022-01-01',
    end_date='2022-12-31',
    initial_cash=10000,
    commission=0.0002,
    plot=False,
    **config['strategy']
)

if results:
    print(f"\n2022 Results (Balanced):")
    print(f"  Total Return: {results['total_return'] * 100:.2f}%")
    if results['sharpe_ratio']:
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")