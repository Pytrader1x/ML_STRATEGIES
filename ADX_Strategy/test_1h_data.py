#!/usr/bin/env python3
"""
Test ADX strategy on 1H resampled data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import run_backtest
from config import STRATEGY_PARAMS

# Path to 1H data
data_path = '../data/AUDUSD_MASTER_1H.csv'

print("Running ADX Strategy on AUDUSD 1H data...")
print("-" * 50)

# Run backtest with logging disabled
params = STRATEGY_PARAMS.copy()
params['printlog'] = False

results = run_backtest(
    data_path=data_path,
    start_date='2020-01-01',  # More recent data
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.0002,
    plot=False,
    **params
)

if results:
    print("\n=== BACKTEST RESULTS (1H Data) ===")
    print(f"Final Portfolio Value: ${results['final_value']:.2f}")
    print(f"Total Return: {results['total_return'] * 100:.2f}%")
    if results['sharpe_ratio']:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    
    # Also try with optimized parameters for forex
    print("\n\n=== Testing with Adjusted Parameters ===")
    
    # Adjust parameters for forex 1H
    params['adx_threshold'] = 30  # Lower threshold for forex
    params['williams_period'] = 20  # Longer period
    params['sma_period'] = 40  # Shorter SMA
    
    results2 = run_backtest(
        data_path=data_path,
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_cash=10000,
        commission=0.0002,
        plot=False,
        **params
    )
    
    if results2:
        print("\nAdjusted Parameters Results:")
        print(f"Final Portfolio Value: ${results2['final_value']:.2f}")
        print(f"Total Return: {results2['total_return'] * 100:.2f}%")
        if results2['sharpe_ratio']:
            print(f"Sharpe Ratio: {results2['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results2['max_drawdown']:.2f}%")