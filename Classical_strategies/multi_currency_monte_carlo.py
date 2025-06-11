"""
Multi-Currency Monte Carlo Testing for Robust High Sharpe Strategies
Tests both configurations on GBPUSD, EURUSD, USDJPY, NZDUSD
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')


def create_config_1_ultra_tight_risk():
    """
    Configuration 1: Ultra-Tight Risk Management
    Achieved Sharpe Ratio: 1.171 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        tsl_activation_pips=3,
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        tp_range_market_multiplier=0.5,
        tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )
    return OptimizedProdStrategy(config)


def create_config_2_scalping():
    """
    Configuration 2: Scalping Strategy
    Achieved Sharpe Ratio: 1.146 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.001,
        sl_max_pips=5.0,
        sl_atr_multiplier=0.5,
        tp_atr_multipliers=(0.1, 0.2, 0.3),
        max_tp_percent=0.002,
        tsl_activation_pips=2,
        tsl_min_profit_pips=0.5,
        tsl_initial_buffer_multiplier=0.5,
        trailing_atr_multiplier=0.5,
        tp_range_market_multiplier=0.3,
        tp_trend_market_multiplier=0.5,
        tp_chop_market_multiplier=0.2,
        sl_range_market_multiplier=0.5,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=0.0,
        signal_flip_min_time_hours=0.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )
    return OptimizedProdStrategy(config)


def run_currency_test(currency_pair, n_iterations=30, sample_size=5000):
    """
    Run Monte Carlo test for a specific currency pair
    
    Parameters:
    - currency_pair: Currency pair to test (e.g., 'GBPUSD')
    - n_iterations: Number of Monte Carlo iterations
    - sample_size: Size of each sample
    
    Returns:
    - Dictionary with results for both configurations
    """
    
    print(f"\n{'='*80}")
    print(f"Testing {currency_pair}")
    print(f"{'='*80}")
    
    # Construct file path
    data_path = f'../data/{currency_pair}_MASTER_15M.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found for {currency_pair}")
        return None
    
    # Load data
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk", create_config_1_ultra_tight_risk()),
        ("Config 2: Scalping", create_config_2_scalping())
    ]
    
    currency_results = {}
    
    for config_name, strategy in configs:
        print(f"\nTesting {config_name}...")
        
        # Storage for results
        iteration_results = []
        
        for i in range(n_iterations):
            # Get random starting point
            max_start = len(df) - sample_size
            if max_start < 0:
                print(f"Insufficient data for {currency_pair} (need {sample_size} rows)")
                return None
                
            start_idx = np.random.randint(0, max_start)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            
            # Run backtest
            results = strategy.run_backtest(sample_df)
            
            iteration_results.append({
                'sharpe_ratio': results['sharpe_ratio'],
                'total_pnl': results['total_pnl'],
                'win_rate': results['win_rate'],
                'max_drawdown': results['max_drawdown'],
                'profit_factor': results['profit_factor'],
                'total_trades': results['total_trades']
            })
        
        # Calculate averages
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in iteration_results])
        avg_pnl = np.mean([r['total_pnl'] for r in iteration_results])
        avg_win_rate = np.mean([r['win_rate'] for r in iteration_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in iteration_results])
        avg_pf = np.mean([r['profit_factor'] for r in iteration_results])
        avg_trades = np.mean([r['total_trades'] for r in iteration_results])
        sharpe_above_1 = sum(1 for r in iteration_results if r['sharpe_ratio'] > 1.0) / n_iterations * 100
        
        currency_results[config_name] = {
            'avg_sharpe': avg_sharpe,
            'avg_pnl': avg_pnl,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown,
            'avg_profit_factor': avg_pf,
            'avg_trades': avg_trades,
            'sharpe_above_1_pct': sharpe_above_1,
            'all_results': iteration_results
        }
        
        print(f"  Average Sharpe: {avg_sharpe:.3f}")
        print(f"  Average P&L: ${avg_pnl:,.0f}")
        print(f"  Win Rate: {avg_win_rate:.1f}%")
        print(f"  Max DD: {avg_drawdown:.1f}%")
        print(f"  % Sharpe > 1.0: {sharpe_above_1:.1f}%")
    
    return currency_results


def main():
    """Run Monte Carlo tests on multiple currency pairs"""
    
    print("="*80)
    print("MULTI-CURRENCY MONTE CARLO TESTING")
    print("Testing Robust High Sharpe Strategies on Multiple Pairs")
    print("="*80)
    
    # Currency pairs to test
    currency_pairs = ['GBPUSD', 'EURUSD', 'USDJPY', 'NZDUSD', 'USDCAD']
    
    # Parameters
    n_iterations = 30  # 30 iterations for robust statistics
    sample_size = 5000
    
    # Store all results
    all_results = {}
    
    # Test each currency pair
    for currency in currency_pairs:
        results = run_currency_test(currency, n_iterations, sample_size)
        if results:
            all_results[currency] = results
    
    # Summary report
    print("\n" + "="*80)
    print("MULTI-CURRENCY SUMMARY REPORT")
    print("="*80)
    
    # Create summary table
    print("\nConfig 1: Ultra-Tight Risk Management")
    print("-" * 60)
    print(f"{'Currency':<10} {'Avg Sharpe':>12} {'Avg P&L':>12} {'Win Rate':>10} {'Sharpe>1':>10}")
    print("-" * 60)
    
    for currency in currency_pairs:
        if currency in all_results:
            config1 = all_results[currency].get('Config 1: Ultra-Tight Risk', {})
            print(f"{currency:<10} {config1.get('avg_sharpe', 0):>12.3f} "
                  f"${config1.get('avg_pnl', 0):>11,.0f} "
                  f"{config1.get('avg_win_rate', 0):>9.1f}% "
                  f"{config1.get('sharpe_above_1_pct', 0):>9.0f}%")
    
    print("\n\nConfig 2: Scalping Strategy")
    print("-" * 60)
    print(f"{'Currency':<10} {'Avg Sharpe':>12} {'Avg P&L':>12} {'Win Rate':>10} {'Sharpe>1':>10}")
    print("-" * 60)
    
    for currency in currency_pairs:
        if currency in all_results:
            config2 = all_results[currency].get('Config 2: Scalping', {})
            print(f"{currency:<10} {config2.get('avg_sharpe', 0):>12.3f} "
                  f"${config2.get('avg_pnl', 0):>11,.0f} "
                  f"{config2.get('avg_win_rate', 0):>9.1f}% "
                  f"{config2.get('sharpe_above_1_pct', 0):>9.0f}%")
    
    # Best performing currency for each config
    print("\n" + "="*80)
    print("BEST PERFORMING CURRENCIES")
    print("="*80)
    
    # Config 1 best
    best_sharpe_c1 = 0
    best_currency_c1 = None
    for currency, results in all_results.items():
        if 'Config 1: Ultra-Tight Risk' in results:
            sharpe = results['Config 1: Ultra-Tight Risk']['avg_sharpe']
            if sharpe > best_sharpe_c1:
                best_sharpe_c1 = sharpe
                best_currency_c1 = currency
    
    # Config 2 best
    best_sharpe_c2 = 0
    best_currency_c2 = None
    for currency, results in all_results.items():
        if 'Config 2: Scalping' in results:
            sharpe = results['Config 2: Scalping']['avg_sharpe']
            if sharpe > best_sharpe_c2:
                best_sharpe_c2 = sharpe
                best_currency_c2 = currency
    
    print(f"\nConfig 1 Best: {best_currency_c1} with Sharpe {best_sharpe_c1:.3f}")
    print(f"Config 2 Best: {best_currency_c2} with Sharpe {best_sharpe_c2:.3f}")
    
    # Save detailed results to CSV
    print("\n" + "="*80)
    print("Saving detailed results to CSV...")
    
    # Create results dataframe
    rows = []
    for currency, currency_results in all_results.items():
        for config_name, config_results in currency_results.items():
            for i, iteration in enumerate(config_results['all_results']):
                row = {
                    'currency': currency,
                    'config': config_name,
                    'iteration': i + 1,
                    'sharpe_ratio': iteration['sharpe_ratio'],
                    'total_pnl': iteration['total_pnl'],
                    'win_rate': iteration['win_rate'],
                    'max_drawdown': iteration['max_drawdown'],
                    'profit_factor': iteration['profit_factor'],
                    'total_trades': iteration['total_trades']
                }
                rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv('results/multi_currency_monte_carlo_results.csv', index=False)
    print("Results saved to results/multi_currency_monte_carlo_results.csv")
    
    print(f"\nTesting completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()