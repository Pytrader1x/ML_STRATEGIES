"""
Quick Strategy Improvement Test
Tests key improvements with minimal data
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def test_strategy(config_name, config, df):
    """Test a single configuration"""
    strategy = OptimizedProdStrategy(config)
    result = strategy.run_backtest(df)
    
    print(f"\n{config_name}:")
    print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
    print(f"  Return: {result['total_return']:.2f}%")
    print(f"  Win Rate: {result['win_rate']:.1f}%")
    print(f"  Trades: {result['total_trades']}")
    print(f"  P&L: ${result['total_pnl']:,.0f}")
    
    return result

def main():
    # Load minimal data for quick testing
    print("Loading AUDUSD data...")
    data_path = '../data' if os.path.exists('../data') else 'data'
    df = pd.read_csv(f'{data_path}/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use only 10 days of recent data for speed
    df = df.tail(960)  # 10 days * 96 bars/day
    
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    print(f"\nTest data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print("="*60)
    
    # Test 1: Baseline (current validated strategy)
    baseline_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.005,
        base_position_size_millions=1.0,
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,
        sl_min_pips=3.0,
        sl_max_pips=10.0,
        tp_atr_multipliers=(0.15, 0.25, 0.4),
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        intelligent_sizing=False,
        realistic_costs=True
    )
    
    baseline_result = test_strategy("BASELINE (Current Validated)", baseline_config, df)
    
    # Test 2: Institutional sizing (1M relaxed, 2M standard)
    institutional_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.005,
        base_position_size_millions=2.0,  # KEY CHANGE
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,  # 2M * 0.5 = 1M
        sl_min_pips=3.0,
        sl_max_pips=10.0,
        tp_atr_multipliers=(0.15, 0.25, 0.4),
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        intelligent_sizing=False,
        realistic_costs=True
    )
    
    inst_result = test_strategy("INSTITUTIONAL SIZING (1M/2M)", institutional_config, df)
    
    # Test 3: Better partial profit logic
    better_partial_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.005,
        base_position_size_millions=2.0,
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,
        sl_min_pips=5.0,  # Wider stops
        sl_max_pips=15.0,
        tp_atr_multipliers=(0.3, 0.5, 0.8),  # Wider TPs
        partial_profit_sl_distance_ratio=0.5,  # KEY CHANGE
        partial_profit_size_percent=0.4,  # KEY CHANGE
        intelligent_sizing=False,
        realistic_costs=True
    )
    
    partial_result = test_strategy("IMPROVED PARTIAL PROFIT", better_partial_config, df)
    
    # Test 4: Intelligent sizing enabled
    intelligent_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.005,
        base_position_size_millions=2.0,
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,
        sl_min_pips=5.0,
        sl_max_pips=15.0,
        tp_atr_multipliers=(0.3, 0.5, 0.8),
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.4,
        intelligent_sizing=True,  # KEY CHANGE
        confidence_thresholds=(40.0, 60.0, 80.0),
        size_multipliers=(0.5, 0.75, 1.0, 1.5),
        realistic_costs=True
    )
    
    intelligent_result = test_strategy("INTELLIGENT SIZING ON", intelligent_config, df)
    
    # Summary comparison
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    
    improvements = {
        'Institutional Sizing': (inst_result['sharpe_ratio'] - baseline_result['sharpe_ratio']) / abs(baseline_result['sharpe_ratio']) * 100,
        'Better Partial Profit': (partial_result['sharpe_ratio'] - baseline_result['sharpe_ratio']) / abs(baseline_result['sharpe_ratio']) * 100,
        'Intelligent Sizing': (intelligent_result['sharpe_ratio'] - baseline_result['sharpe_ratio']) / abs(baseline_result['sharpe_ratio']) * 100
    }
    
    for name, improvement in improvements.items():
        print(f"{name}: {improvement:+.1f}% Sharpe improvement")
    
    # Best configuration
    all_results = {
        'Baseline': baseline_result,
        'Institutional': inst_result,
        'Better Partial': partial_result,
        'Intelligent': intelligent_result
    }
    
    best_name = max(all_results, key=lambda x: all_results[x]['sharpe_ratio'])
    best_result = all_results[best_name]
    
    print(f"\n🏆 BEST: {best_name}")
    print(f"   Sharpe: {best_result['sharpe_ratio']:.3f}")
    print(f"   P&L: ${best_result['total_pnl']:,.0f}")

if __name__ == "__main__":
    main()