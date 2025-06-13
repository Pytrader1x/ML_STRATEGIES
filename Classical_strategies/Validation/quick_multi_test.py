"""
Quick Multi-Sample Test Script
Runs multiple random samples to test strategy robustness
"""

import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategy_code.Prod_strategy import OptimizedStrategyConfig
    from strategy_code.Prod_plotting import plot_production_results
    from real_time_strategy_simulator import RealTimeStrategySimulator
    from real_time_data_generator import RealTimeDataGenerator
    from technical_indicators_custom import TIC
    
    print("✅ All imports successful")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def run_single_test(generator, config, rows=2000, plot=False, test_num=None):
    """Run a single test with random sample"""
    
    # Get random sample period
    start_idx, end_idx = generator.get_sample_period(rows=rows)
    
    print(f"\n{'='*60}")
    if test_num:
        print(f"TEST {test_num}: Rows {start_idx:,} to {end_idx:,}")
    print(f"Date range: {generator.full_data.iloc[start_idx]['DateTime']} to {generator.full_data.iloc[end_idx-1]['DateTime']}")
    
    # Create simulator
    simulator = RealTimeStrategySimulator(config)
    
    # Run simulation
    results = simulator.run_real_time_simulation(
        currency_pair='AUDUSD',
        rows_to_simulate=rows,
        start_idx=start_idx,
        verbose=False
    )
    
    # Print results
    print(f"Results: Sharpe={results['performance_metrics']['sharpe_ratio']:.3f} | "
          f"Return={results['performance_metrics']['total_return']:.1f}% | "
          f"Trades={results['trade_statistics']['total_trades']} | "
          f"WR={results['trade_statistics']['win_rate']:.1f}%")
    
    # Plot if requested
    if plot:
        plot_results(generator, results, start_idx, end_idx, test_num)
    
    return results, start_idx, end_idx


def plot_results(generator, results, start_idx, end_idx, test_num=None):
    """Plot the trading results using the exact simulation data"""
    
    print(f"\n📊 Plotting results...")
    
    # Get the exact data slice that was simulated
    df_plot = generator.full_data.iloc[start_idx:end_idx].copy()
    
    # Ensure DateTime index
    if 'DateTime' in df_plot.columns and not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.set_index('DateTime', inplace=True)
    
    # Add indicators
    df_plot = TIC.add_neuro_trend_intelligent(df_plot)
    df_plot = TIC.add_market_bias(df_plot)
    df_plot = TIC.add_intelligent_chop(df_plot)
    
    # Format results for plotting
    formatted_results = {
        'trades': results['detailed_data']['trades'],
        'equity_curve': results['detailed_data']['capital_history'],
        'total_trades': results['trade_statistics']['total_trades'],
        'win_rate': results['trade_statistics']['win_rate'],
        'sharpe_ratio': results['performance_metrics']['sharpe_ratio'],
        'total_pnl': results['performance_metrics']['total_pnl'],
        'total_return': results['performance_metrics']['total_return'],
        'max_drawdown': results['performance_metrics']['max_drawdown']
    }
    
    # Generate plot
    title = f"Real-time Simulation"
    if test_num:
        title += f" - Test {test_num}"
    title += f"\nSharpe: {results['performance_metrics']['sharpe_ratio']:.3f} | "
    title += f"Return: {results['performance_metrics']['total_return']:.1f}% | "
    title += f"Trades: {results['trade_statistics']['total_trades']}"
    
    fig = plot_production_results(
        df=df_plot,
        results=formatted_results,
        title=title,
        show_pnl=True,
        show_position_sizes=False,
        show_chop_subplots=False,
        show=True
    )
    
    # Save plot
    if fig:
        os.makedirs('validation_charts', exist_ok=True)
        filename = f'validation_charts/test_{test_num if test_num else "single"}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Chart saved to {filename}")
    
    return fig


def run_multi_test(num_tests=10, rows_per_test=2000, plot_best=True, plot_all=False):
    """Run multiple tests with random samples"""
    
    print(f"\n{'='*80}")
    print(f"MULTI-SAMPLE VALIDATION TEST")
    print(f"Tests: {num_tests} | Rows per test: {rows_per_test:,}")
    print(f"{'='*80}")
    
    # Load data once
    print("\nLoading data...")
    generator = RealTimeDataGenerator('AUDUSD')
    print(f"✅ Data loaded: {len(generator.full_data):,} rows")
    
    # Create config once
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
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
        verbose=False,
        debug_decisions=False
    )
    
    # Run tests
    all_results = []
    
    for i in range(num_tests):
        results, start_idx, end_idx = run_single_test(
            generator, 
            config, 
            rows=rows_per_test, 
            plot=plot_all,
            test_num=i+1
        )
        
        all_results.append({
            'test_num': i+1,
            'results': results,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'sharpe': results['performance_metrics']['sharpe_ratio'],
            'return': results['performance_metrics']['total_return'],
            'trades': results['trade_statistics']['total_trades']
        })
    
    # Summary statistics
    sharpes = [r['sharpe'] for r in all_results]
    returns = [r['return'] for r in all_results]
    trades = [r['trades'] for r in all_results]
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Sharpe Ratio: Mean={np.mean(sharpes):.3f}, Std={np.std(sharpes):.3f}, "
          f"Min={np.min(sharpes):.3f}, Max={np.max(sharpes):.3f}")
    print(f"Returns:      Mean={np.mean(returns):.1f}%, Std={np.std(returns):.1f}%, "
          f"Min={np.min(returns):.1f}%, Max={np.max(returns):.1f}%")
    print(f"Trades:       Mean={np.mean(trades):.1f}, Min={np.min(trades)}, Max={np.max(trades)}")
    print(f"Profitable:   {sum(1 for r in returns if r > 0)}/{num_tests} "
          f"({sum(1 for r in returns if r > 0)/num_tests*100:.0f}%)")
    
    # Plot best result
    if plot_best and not plot_all:
        best_result = max(all_results, key=lambda x: x['sharpe'])
        print(f"\n📈 Plotting best result (Test {best_result['test_num']}, Sharpe={best_result['sharpe']:.3f})")
        
        plot_results(
            generator,
            best_result['results'],
            best_result['start_idx'],
            best_result['end_idx'],
            f"{best_result['test_num']}_best"
        )
    
    return all_results


def main():
    """Main function with options"""
    
    print("\n" + "="*60)
    print("QUICK MULTI-SAMPLE TEST")
    print("="*60)
    
    # Run options (modify as needed)
    RUN_SINGLE = False
    RUN_MULTI = True
    
    if RUN_SINGLE:
        # Single test with plot
        print("\nRunning single test...")
        generator = RealTimeDataGenerator('AUDUSD')
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            debug_decisions=False
        )
        
        results, start_idx, end_idx = run_single_test(
            generator, 
            config, 
            rows=2000, 
            plot=True
        )
    
    if RUN_MULTI:
        # Multiple tests
        print("\nRunning multiple tests...")
        all_results = run_multi_test(
            num_tests=10,        # Number of random samples
            rows_per_test=2000,  # Rows per sample
            plot_best=True,      # Plot the best result
            plot_all=False       # Plot all results (set True if you want all plots)
        )
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()