#!/usr/bin/env python3
"""Quick test of single FX pair"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classical_strategies.extended_fx_backtest import test_single_pair, CONFIGS

# Test AUDUSD
result = test_single_pair('AUDUSD', 'config_1_ultra_tight', CONFIGS['config_1_ultra_tight'], start_year=2010)

if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"Success! Total trades: {result['total_trades']}")
    metrics = result['overall_metrics']
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")