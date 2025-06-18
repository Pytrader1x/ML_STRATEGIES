"""
Test script to verify that the refactored code produces the same results
"""

import pandas as pd
import numpy as np
from ultimate_optimizer import AdvancedBacktest
from multi_currency_analyzer import EnhancedBacktest

# Test with a sample dataset
def test_backtest_comparison():
    """Compare original and enhanced backtest results"""
    
    # Load a small test dataset
    data_path = '../data/AUDUSD_MASTER_15M.csv'
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    test_data = data[-1000:]  # Use last 1000 bars for quick test
    
    # Parameters
    lookback = 40
    entry_z = 1.5
    exit_z = 0.5
    
    # Test original backtest
    original_backtester = AdvancedBacktest(test_data)
    original_result = original_backtester.strategy_momentum(
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z
    )
    
    # Test enhanced backtest (without trade tracking features)
    enhanced_backtester = EnhancedBacktest(test_data)
    enhanced_result = enhanced_backtester.strategy_momentum_enhanced(
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z
    )
    
    # Compare core metrics
    print("Comparison of Core Metrics:")
    print("-" * 50)
    print(f"Original Sharpe: {original_result['sharpe']:.6f}")
    print(f"Enhanced Sharpe: {enhanced_result['sharpe']:.6f}")
    print(f"Difference: {abs(original_result['sharpe'] - enhanced_result['sharpe']):.6f}")
    print()
    print(f"Original Returns: {original_result['returns']:.6f}%")
    print(f"Enhanced Returns: {enhanced_result['returns']:.6f}%")
    print(f"Difference: {abs(original_result['returns'] - enhanced_result['returns']):.6f}%")
    print()
    print(f"Original Win Rate: {original_result['win_rate']:.6f}%")
    print(f"Enhanced Win Rate: {enhanced_result['win_rate']:.6f}%")
    print(f"Difference: {abs(original_result['win_rate'] - enhanced_result['win_rate']):.6f}%")
    print()
    print(f"Original Trades: {original_result['trades']}")
    print(f"Enhanced Trades: {enhanced_result['trades']}")
    print()
    
    # Check if differences are within acceptable tolerance
    tolerance = 0.01  # 1% tolerance
    sharpe_match = abs(original_result['sharpe'] - enhanced_result['sharpe']) < tolerance
    returns_match = abs(original_result['returns'] - enhanced_result['returns']) < tolerance
    trades_match = original_result['trades'] == enhanced_result['trades']
    
    if sharpe_match and returns_match and trades_match:
        print("✅ Test PASSED: Results match within tolerance")
    else:
        print("❌ Test FAILED: Results differ significantly")
    
    # Show additional metrics from enhanced version
    print("\nAdditional Metrics from Enhanced Version:")
    print("-" * 50)
    print(f"Average Holding Period: {enhanced_result['avg_holding_hours']:.1f} hours")
    print(f"Median Holding Period: {enhanced_result['median_holding_hours']:.1f} hours")
    if enhanced_result['top_entry_hours']:
        print(f"Top Entry Hours: {list(enhanced_result['top_entry_hours'].keys())}")
    

if __name__ == "__main__":
    test_backtest_comparison()