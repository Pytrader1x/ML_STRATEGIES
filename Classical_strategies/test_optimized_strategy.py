"""
Test Optimized Strategy Parameters from Recursive Optimizer
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union, Any
import json
import time

warnings.filterwarnings('ignore')

# Optimized parameters from generation 10
OPTIMIZED_PARAMS = {
    'tsl_activation_pips': 13.000,
    'risk_per_trade': 0.003,
    'tp1_multiplier': 0.256,
    'tp2_multiplier': 0.330,
    'tp3_multiplier': 0.930,
    'sl_atr_multiplier': 1.561,
    'partial_profit_sl_distance_ratio': 0.367,
    'partial_profit_size_percent': 0.657,
    'tp_chop_market_multiplier': 0.486,
    'tp_range_market_multiplier': 0.473,
    'tp_trend_market_multiplier': 1.114,
    'trailing_atr_multiplier': 1.557,
    'sl_min_pips': 6.130,
    'tsl_min_profit_pips': 1.367,
    'sl_max_pips': 27.100
}

def create_optimized_strategy_config():
    """Create strategy config with optimized parameters"""
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=OPTIMIZED_PARAMS['risk_per_trade'],
        sl_min_pips=OPTIMIZED_PARAMS['sl_min_pips'],
        sl_max_pips=OPTIMIZED_PARAMS['sl_max_pips'],
        sl_atr_multiplier=OPTIMIZED_PARAMS['sl_atr_multiplier'],
        tp_atr_multipliers=(
            OPTIMIZED_PARAMS['tp1_multiplier'],
            OPTIMIZED_PARAMS['tp2_multiplier'],
            OPTIMIZED_PARAMS['tp3_multiplier']
        ),
        max_tp_percent=0.005,  # 0.5% max TP
        tsl_activation_pips=OPTIMIZED_PARAMS['tsl_activation_pips'],
        tsl_min_profit_pips=OPTIMIZED_PARAMS['tsl_min_profit_pips'],
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=OPTIMIZED_PARAMS['trailing_atr_multiplier'],
        tp_range_market_multiplier=OPTIMIZED_PARAMS['tp_range_market_multiplier'],
        tp_trend_market_multiplier=OPTIMIZED_PARAMS['tp_trend_market_multiplier'],
        tp_chop_market_multiplier=OPTIMIZED_PARAMS['tp_chop_market_multiplier'],
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=OPTIMIZED_PARAMS['partial_profit_sl_distance_ratio'],
        partial_profit_size_percent=OPTIMIZED_PARAMS['partial_profit_size_percent'],
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        relaxed_position_multiplier=0.5,
        relaxed_mode=False,
        realistic_costs=True,
        verbose=False,
        debug_decisions=True,
        use_daily_sharpe=True
    )
    return config

def load_and_prepare_data(currency_pair='AUDUSD'):
    """Load and prepare data with indicators"""
    data_path = 'data' if os.path.exists('data') else '../data'
    file_path = os.path.join(data_path, f'{currency_pair}_MASTER_15M.csv')
    
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df, ha_len=350, ha_len2=30)
    df = TIC.add_intelligent_chop(df)
    
    return df

def test_multiple_periods(df, strategy, periods):
    """Test strategy across multiple time periods"""
    results = []
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"\n📊 Testing period: {period_name} ({start_date} to {end_date})")
        
        # Filter data
        period_df = df.loc[start_date:end_date].copy()
        
        if len(period_df) < 100:
            print(f"  ⚠️ Insufficient data for period (only {len(period_df)} rows)")
            continue
        
        # Run backtest
        result = strategy.run_backtest(period_df)
        
        # Store results
        results.append({
            'period': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_return': result.get('total_return', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'sortino_ratio': result.get('sortino_ratio', 0),
            'win_rate': result.get('win_rate', 0),
            'profit_factor': result.get('profit_factor', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'total_trades': result.get('total_trades', 0),
            'avg_trade': result.get('avg_trade', 0),
            'total_pnl': result.get('total_pnl', 0)
        })
        
        # Print summary
        print(f"  Sharpe: {result.get('sharpe_ratio', 0):.3f}")
        print(f"  Return: {result.get('total_return', 0):.1f}%")
        print(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
        print(f"  Trades: {result.get('total_trades', 0)}")
        print(f"  P&L: ${result.get('total_pnl', 0):,.0f}")
    
    return results

def main():
    """Main testing function"""
    print("🚀 Testing Optimized Strategy Parameters")
    print("="*60)
    
    # Print optimized parameters
    print("\n📊 Optimized Parameters:")
    for param, value in OPTIMIZED_PARAMS.items():
        print(f"  {param}: {value:.3f}")
    
    # Load data
    df = load_and_prepare_data('AUDUSD')
    
    # Create strategy with optimized parameters
    print("\n🔧 Creating strategy with optimized parameters...")
    config = create_optimized_strategy_config()
    strategy = OptimizedProdStrategy(config)
    
    # Define test periods
    test_periods = {
        '2023 Q1': ('2023-01-01', '2023-03-31'),
        '2023 Q2': ('2023-04-01', '2023-06-30'),
        '2023 Q3': ('2023-07-01', '2023-09-30'),
        '2023 Q4': ('2023-10-01', '2023-12-31'),
        '2024 Q1': ('2024-01-01', '2024-03-31'),
        '2024 H1': ('2024-01-01', '2024-06-30'),
        'Full 2023': ('2023-01-01', '2023-12-31'),
        'Recent 6M': ('2024-01-01', '2024-06-30'),
        'Last 3M': ('2024-04-01', '2024-06-30')
    }
    
    # Test across multiple periods
    results = test_multiple_periods(df, strategy, test_periods)
    
    # Display results summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    avg_sharpe = results_df['sharpe_ratio'].mean()
    min_sharpe = results_df['sharpe_ratio'].min()
    max_sharpe = results_df['sharpe_ratio'].max()
    std_sharpe = results_df['sharpe_ratio'].std()
    
    print(f"\n📈 Sharpe Ratio Statistics:")
    print(f"  Average: {avg_sharpe:.3f}")
    print(f"  Min: {min_sharpe:.3f}")
    print(f"  Max: {max_sharpe:.3f}")
    print(f"  Std Dev: {std_sharpe:.3f}")
    print(f"  Consistency: {(1 - std_sharpe/avg_sharpe):.1%}" if avg_sharpe > 0 else "  Consistency: N/A")
    
    # Check robustness
    robust_periods = sum(1 for sharpe in results_df['sharpe_ratio'] if sharpe > 1.0)
    total_periods = len(results_df)
    robustness = robust_periods / total_periods * 100 if total_periods > 0 else 0
    
    print(f"\n🎯 Robustness Analysis:")
    print(f"  Periods with Sharpe > 1.0: {robust_periods}/{total_periods} ({robustness:.1f}%)")
    print(f"  Average Return: {results_df['total_return'].mean():.1f}%")
    print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
    print(f"  Total P&L: ${results_df['total_pnl'].sum():,.0f}")
    
    # Print detailed results
    print("\n📋 Detailed Results by Period:")
    print(results_df[['period', 'sharpe_ratio', 'total_return', 'win_rate', 'total_trades', 'total_pnl']].to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'results/optimized_strategy_test_{timestamp}.csv', index=False)
    print(f"\n💾 Results saved to: results/optimized_strategy_test_{timestamp}.csv")
    
    # Verdict
    print("\n" + "="*60)
    if robustness >= 70 and avg_sharpe > 1.0:
        print("✅ STRATEGY PASSED: Robust performance across multiple periods!")
    elif robustness >= 50 and avg_sharpe > 0.8:
        print("⚠️ STRATEGY MODERATE: Decent but needs improvement")
    else:
        print("❌ STRATEGY FAILED: Not robust enough, needs optimization")
    print("="*60)

if __name__ == "__main__":
    main()