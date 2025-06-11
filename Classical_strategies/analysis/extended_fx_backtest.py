#!/usr/bin/env python3
"""
Extended FX Backtesting - Test FX strategies from 2010 to present
Comprehensive analysis across all major currency pairs with yearly breakdowns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classical_strategies.strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from technical_indicators_custom import TIC


# Currency pairs to test
CURRENCY_PAIRS = [
    'AUDUSD', 'GBPUSD', 'EURUSD', 'NZDUSD', 'USDCAD', 'USDJPY',
    'GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD', 'EURGBP'
]

# Strategy configurations
CONFIGS = {
    'config_1_ultra_tight': {
        'risk_pct': 0.001,  # 0.1% risk per trade
        'sl_pips': 15,
        'tp_pips': 30,
        'tsl_trigger_pips': 20,
        'tsl_distance_pips': 10,
        'volatility_threshold': 0.5,
        'trend_strength_threshold': 0.6
    },
    'config_2_scalping': {
        'risk_pct': 0.002,  # 0.2% risk per trade
        'sl_pips': 20,
        'tp_pips': 40,
        'tsl_trigger_pips': 25,
        'tsl_distance_pips': 12,
        'volatility_threshold': 0.4,
        'trend_strength_threshold': 0.5
    }
}


def get_pip_value(pair):
    """Get pip value for currency pair"""
    if 'JPY' in pair:
        return 0.01
    else:
        return 0.0001


def test_single_pair(pair, config_name, config, start_year=2010):
    """Test a single currency pair with given configuration"""
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'data', f'{pair}_MASTER_15M.csv')
    
    if not os.path.exists(data_path):
        return {
            'pair': pair,
            'config': config_name,
            'error': f'Data file not found: {data_path}'
        }
    
    try:
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Handle different datetime column names
        datetime_col = None
        for col in ['Time', 'DateTime', 'timestamp']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            return {
                'pair': pair,
                'config': config_name,
                'error': 'No datetime column found'
            }
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        
        # Filter data from start_year
        df = df[df.index.year >= start_year]
        
        if len(df) < 1000:
            return {
                'pair': pair,
                'config': config_name,
                'error': f'Insufficient data: only {len(df)} rows'
            }
        
        # Add required indicators
        df = TIC.add_neuro_trend_intelligent(df)
        df = TIC.add_market_bias(df)
        df = TIC.add_intelligent_chop(df)
        
        # Get pip value
        pip_value = get_pip_value(pair)
        
        # Create strategy config
        strategy_config = OptimizedStrategyConfig()
        strategy_config.initial_capital = 10000
        strategy_config.risk_per_trade = config['risk_pct']
        strategy_config.sl_max_pips = config['sl_pips']
        strategy_config.tsl_activation_pips = config['tsl_trigger_pips']
        
        # Initialize strategy with OptimizedProdStrategy
        strategy = OptimizedProdStrategy(config=strategy_config)
        
        # Run backtest
        metrics = strategy.run_backtest(df)
        trades = metrics.get('trades', [])
        
        # Yearly breakdown
        yearly_results = {}
        for year in range(start_year, datetime.now().year + 1):
            year_data = df[df.index.year == year]
            if len(year_data) > 100:  # Need sufficient data
                strategy.reset()  # Reset strategy for each year
                year_metrics = strategy.run_backtest(year_data)
                year_trades = year_metrics.get('trades', [])
                if year_metrics.get('total_trades', 0) > 0:
                    yearly_results[year] = {
                        'trades': len(year_trades),
                        'return': year_metrics.get('total_return', 0),
                        'sharpe': year_metrics.get('sharpe_ratio', 0),
                        'win_rate': year_metrics.get('win_rate', 0),
                        'max_drawdown': year_metrics.get('max_drawdown', 0)
                    }
        
        return {
            'pair': pair,
            'config': config_name,
            'success': True,
            'data_range': f"{df.index[0]} to {df.index[-1]}",
            'total_rows': len(df),
            'overall_metrics': metrics,
            'total_trades': len(trades),
            'yearly_breakdown': yearly_results,
            'trades_sample': trades[:10] if isinstance(trades, list) else (trades.head(10).to_dict('records') if not trades.empty else [])
        }
        
    except Exception as e:
        return {
            'pair': pair,
            'config': config_name,
            'error': str(e)
        }


def run_extended_fx_backtest(start_year=2010, save_results=True):
    """Run comprehensive FX backtest from 2010 to present"""
    
    print("="*80)
    print(f"EXTENDED FX BACKTESTING - {len(CURRENCY_PAIRS)} PAIRS")
    print(f"Period: {start_year} to Present")
    print("="*80)
    
    all_results = {}
    
    # Create tasks for parallel execution
    tasks = []
    for pair in CURRENCY_PAIRS:
        for config_name, config in CONFIGS.items():
            tasks.append((pair, config_name, config, start_year))
    
    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(test_single_pair, *task): task 
            for task in tasks
        }
        
        with tqdm(total=len(tasks), desc="Running backtests") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pair = result['pair']
                config = result['config']
                
                if pair not in all_results:
                    all_results[pair] = {}
                all_results[pair][config] = result
                
                pbar.update(1)
    
    # Print summary results
    print("\n" + "="*100)
    print("SUMMARY RESULTS BY CURRENCY PAIR")
    print("="*100)
    
    for pair in CURRENCY_PAIRS:
        print(f"\n{pair}:")
        print("-" * 80)
        
        if pair in all_results:
            for config_name in CONFIGS:
                if config_name in all_results[pair]:
                    result = all_results[pair][config_name]
                    
                    if 'error' in result:
                        print(f"  {config_name}: ERROR - {result['error']}")
                    else:
                        metrics = result['overall_metrics']
                        print(f"  {config_name}:")
                        print(f"    Total Return: {metrics.get('total_return', 0):.2%}")
                        print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                        print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
                        print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                        print(f"    Total Trades: {result['total_trades']}")
                        print(f"    Data Range: {result['data_range']}")
        else:
            print("  No results available")
    
    # Configuration comparison across all pairs
    print("\n" + "="*100)
    print("CONFIGURATION PERFORMANCE COMPARISON")
    print("="*100)
    
    for config_name in CONFIGS:
        print(f"\n{config_name.upper()}:")
        
        successful_pairs = []
        for pair in CURRENCY_PAIRS:
            if pair in all_results and config_name in all_results[pair]:
                result = all_results[pair][config_name]
                if 'error' not in result:
                    metrics = result['overall_metrics']
                    successful_pairs.append({
                        'pair': pair,
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'return': metrics.get('total_return', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'trades': result['total_trades']
                    })
        
        if successful_pairs:
            # Sort by Sharpe ratio
            successful_pairs.sort(key=lambda x: x['sharpe'], reverse=True)
            
            print(f"\n  Top Performers (by Sharpe Ratio):")
            for i, perf in enumerate(successful_pairs[:5]):
                print(f"    {i+1}. {perf['pair']}: Sharpe {perf['sharpe']:.3f}, "
                      f"Return {perf['return']:.2%}, Win Rate {perf['win_rate']:.2%}, "
                      f"Trades {perf['trades']}")
            
            # Average statistics
            avg_sharpe = np.mean([p['sharpe'] for p in successful_pairs])
            avg_return = np.mean([p['return'] for p in successful_pairs])
            avg_win_rate = np.mean([p['win_rate'] for p in successful_pairs])
            
            print(f"\n  Average Performance ({len(successful_pairs)} pairs):")
            print(f"    Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"    Total Return: {avg_return:.2%}")
            print(f"    Win Rate: {avg_win_rate:.2%}")
    
    # Save results
    if save_results:
        results_dir = 'Classical_strategies/results'
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'extended_fx_backtest_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n\nDetailed results saved to: {output_file}")
        
        # Also save a summary CSV
        summary_data = []
        for pair in CURRENCY_PAIRS:
            if pair in all_results:
                for config_name in CONFIGS:
                    if config_name in all_results[pair]:
                        result = all_results[pair][config_name]
                        if 'error' not in result:
                            metrics = result['overall_metrics']
                            summary_data.append({
                                'pair': pair,
                                'config': config_name,
                                'total_return': metrics.get('total_return', 0),
                                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                                'win_rate': metrics.get('win_rate', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'total_trades': result['total_trades'],
                                'data_start': result['data_range'].split(' to ')[0],
                                'data_end': result['data_range'].split(' to ')[1]
                            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(results_dir, 'extended_fx_backtest_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary CSV saved to: {summary_file}")
    
    return all_results


def analyze_correlation(all_results):
    """Analyze correlation between currency pairs"""
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Extract returns for correlation
    returns_data = {}
    
    for config_name in CONFIGS:
        returns_data[config_name] = {}
        
        for pair in CURRENCY_PAIRS:
            if pair in all_results and config_name in all_results[pair]:
                result = all_results[pair][config_name]
                if 'error' not in result and 'yearly_breakdown' in result:
                    yearly_returns = {
                        year: data['return'] 
                        for year, data in result['yearly_breakdown'].items()
                    }
                    if yearly_returns:
                        returns_data[config_name][pair] = yearly_returns
    
    # Calculate correlations
    for config_name in CONFIGS:
        if returns_data[config_name]:
            print(f"\n{config_name.upper()} - Pair Correlations:")
            
            # Create DataFrame from yearly returns
            years = set()
            for pair_returns in returns_data[config_name].values():
                years.update(pair_returns.keys())
            
            years = sorted(years)
            
            returns_matrix = []
            pair_names = []
            
            for pair, yearly in returns_data[config_name].items():
                pair_returns = [yearly.get(year, np.nan) for year in years]
                if not all(np.isnan(pair_returns)):
                    returns_matrix.append(pair_returns)
                    pair_names.append(pair)
            
            if len(returns_matrix) > 1:
                returns_df = pd.DataFrame(returns_matrix, index=pair_names, columns=years).T
                corr_matrix = returns_df.corr()
                
                # Find highest and lowest correlations
                corr_values = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_values.append({
                            'pair1': corr_matrix.index[i],
                            'pair2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
                
                if corr_values:
                    corr_values.sort(key=lambda x: x['correlation'])
                    
                    print("\n  Lowest Correlations (best for diversification):")
                    for item in corr_values[:3]:
                        if not np.isnan(item['correlation']):
                            print(f"    {item['pair1']} - {item['pair2']}: {item['correlation']:.3f}")
                    
                    print("\n  Highest Correlations:")
                    for item in corr_values[-3:]:
                        if not np.isnan(item['correlation']):
                            print(f"    {item['pair1']} - {item['pair2']}: {item['correlation']:.3f}")


if __name__ == "__main__":
    # Change to correct directory
    os.chdir('/Users/williamsmith/Python_local_Mac/Ml_Strategies')
    
    # Run extended backtest
    results = run_extended_fx_backtest(start_year=2010)
    
    # Run correlation analysis
    analyze_correlation(results)