"""
Validation script to ensure the reorganized backtesting framework 
produces the same results as the original implementation
"""

import pandas as pd
import numpy as np
from backtesting import Backtester, MomentumStrategy

def validate_momentum_strategy():
    """Validate that our momentum strategy produces expected results"""
    
    # Load test data
    data_path = '../data/AUDUSD_MASTER_15M.csv'
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    test_data = data[-1000:]  # Small sample for validation
    
    # Test parameters
    params = {
        'lookback': 40,
        'entry_z': 1.5,
        'exit_z': 0.5
    }
    
    # Run backtest
    strategy = MomentumStrategy(**params)
    backtester = Backtester(test_data)
    result = backtester.run_backtest(strategy, track_trades=True)
    
    # Display results
    print("Validation Test Results:")
    print("-" * 50)
    print(f"Sharpe Ratio: {result['sharpe']:.6f}")
    print(f"Total Returns: {result['returns']:.6f}%")
    print(f"Win Rate: {result['win_rate']:.6f}%")
    print(f"Max Drawdown: {result['max_dd']:.6f}%")
    print(f"Total Trades: {result['trades']}")
    
    if 'avg_holding_hours' in result:
        print(f"\nTrade Analysis:")
        print(f"Average Holding: {result['avg_holding_hours']:.2f} hours")
        print(f"Trades tracked: {len(backtester.trades)}")
    
    # Expected values from previous test
    expected_sharpe = 8.696115
    expected_trades = 302
    
    # Validate
    sharpe_match = abs(result['sharpe'] - expected_sharpe) < 0.001
    trades_match = result['trades'] == expected_trades
    
    print(f"\nValidation Results:")
    print(f"Sharpe matches expected: {sharpe_match} ✓" if sharpe_match else f"Sharpe mismatch: {sharpe_match} ✗")
    print(f"Trade count matches: {trades_match} ✓" if trades_match else f"Trade count mismatch: {trades_match} ✗")
    
    if sharpe_match and trades_match:
        print("\n✅ VALIDATION PASSED: Backtesting framework working correctly")
    else:
        print("\n❌ VALIDATION FAILED: Results don't match expected values")
    
    return result

if __name__ == "__main__":
    validate_momentum_strategy()