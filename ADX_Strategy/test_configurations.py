#!/usr/bin/env python3
"""
Test different ADX strategy configurations to find the best parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import run_backtest
from config import STRATEGY_PARAMS
from config_forex import get_forex_config

# Test configurations
configs = {
    'Original (ADX > 50)': {
        'adx_threshold': 50,
        'williams_oversold': -80,
        'williams_overbought': -20,
    },
    'Moderate (ADX > 35)': {
        'adx_threshold': 35,
        'williams_oversold': -85,
        'williams_overbought': -15,
    },
    'Forex Balanced (ADX > 30)': {
        'adx_threshold': 30,
        'williams_oversold': -85,
        'williams_overbought': -15,
    },
    'Active (ADX > 25)': {
        'adx_threshold': 25,
        'williams_oversold': -90,
        'williams_overbought': -10,
    }
}

data_path = '../data/AUDUSD_MASTER_1H.csv'

print("=== TESTING DIFFERENT ADX CONFIGURATIONS ===\n")

results_summary = []

for config_name, custom_params in configs.items():
    print(f"\nTesting: {config_name}")
    print("-" * 40)
    
    # Start with base parameters
    params = STRATEGY_PARAMS.copy()
    params.update(custom_params)
    params['printlog'] = False  # Disable verbose logging
    
    results = run_backtest(
        data_path=data_path,
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_cash=10000,
        commission=0.0002,
        plot=False,
        **params
    )
    
    if results:
        # Get trade count
        import pandas as pd
        trade_count = 0
        win_rate = 0
        if os.path.exists('trade_history.csv'):
            trades_df = pd.read_csv('trade_history.csv')
            trade_count = len(trades_df)
            if trade_count > 0:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                win_rate = len(winning_trades) / trade_count * 100
        
        results_summary.append({
            'Config': config_name,
            'ADX Threshold': custom_params['adx_threshold'],
            'Total Return': f"{results['total_return'] * 100:.2f}%",
            'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}" if results['sharpe_ratio'] else 'N/A',
            'Max Drawdown': f"{results['max_drawdown']:.2f}%",
            'Trades': trade_count,
            'Win Rate': f"{win_rate:.1f}%",
            'Final Value': f"${results['final_value']:.2f}"
        })

# Print summary table
print("\n\n=== RESULTS SUMMARY ===")
print("-" * 120)
print(f"{'Config':<25} {'ADX':<5} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<8} {'Win Rate':<10} {'Final Value':<12}")
print("-" * 120)

for result in results_summary:
    print(f"{result['Config']:<25} {result['ADX Threshold']:<5} {result['Total Return']:<10} {result['Sharpe Ratio']:<8} {result['Max Drawdown']:<10} {result['Trades']:<8} {result['Win Rate']:<10} {result['Final Value']:<12}")

print("\n=== RECOMMENDATIONS ===")
print("1. Original (ADX > 50): Very selective, only 3 trades but positive returns")
print("2. Moderate (ADX > 35): Better balance of trade frequency and performance")
print("3. Lower ADX thresholds (<30): Risk of overtrading in ranging markets")
print("\nConsider using ADX > 35-40 for AUDUSD on 1H timeframe.")