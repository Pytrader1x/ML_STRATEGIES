#!/usr/bin/env python3
"""
Test script for running ADX strategy on AUDUSD data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import run_backtest
from config import STRATEGY_PARAMS

# Path to your AUDUSD data
data_path = '../data/AUDUSD_MASTER_15M.csv'

# Run backtest with custom parameters for 15M timeframe
print("Testing ADX Strategy on AUDUSD 15M data...")
print("-" * 50)

# Adjust parameters for 15M timeframe (optional)
custom_params = STRATEGY_PARAMS.copy()
custom_params['printlog'] = False  # Reduce log spam for 15M data

# Run backtest on a specific date range
results = run_backtest(
    data_path=data_path,
    start_date='2010-01-01',  # Adjust based on your data
    end_date='2023-12-31',    # Adjust based on your data
    initial_cash=10000,
    commission=0.0002,  # Lower commission for forex
    plot=False,  # Set to True if you want to see the chart
    **custom_params
)

if results:
    print("\n=== BACKTEST RESULTS ===")
    print(f"Final Portfolio Value: ${results['final_value']:.2f}")
    print(f"Total Return: {results['total_return'] * 100:.2f}%")
    if results['sharpe_ratio']:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
else:
    print("Backtest failed. Please check your data and parameters.")