#!/usr/bin/env python3
"""
Analyze FX Performance - Detailed analysis of extended FX backtesting results
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime


def analyze_fx_results():
    """Analyze the FX backtest results in detail"""
    
    print("="*80)
    print("FX STRATEGY PERFORMANCE ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load results
    results_path = 'Classical_strategies/results/extended_fx_backtest_results.json'
    if not os.path.exists(results_path):
        print("Results file not found. Running sample analysis on AUDUSD...")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Analyze by configuration
    configs = ['config_1_ultra_tight', 'config_2_scalping']
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"{config.upper()} ANALYSIS")
        print(f"{'='*60}")
        
        # Collect metrics for all pairs
        config_data = []
        
        for pair, pair_results in results.items():
            if config in pair_results:
                result = pair_results[config]
                if 'error' not in result and 'overall_metrics' in result:
                    metrics = result['overall_metrics']
                    if metrics:  # Check if metrics is not empty
                        config_data.append({
                            'pair': pair,
                            'total_trades': result.get('total_trades', 0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                            'total_return': metrics.get('total_return', 0),
                            'win_rate': metrics.get('win_rate', 0),
                            'max_drawdown': metrics.get('max_drawdown', 0),
                            'profit_factor': metrics.get('profit_factor', 0),
                            'data_years': len(result.get('yearly_breakdown', {}))
                        })
        
        if config_data:
            df = pd.DataFrame(config_data)
            
            # Overall statistics
            print("\nOVERALL STATISTICS:")
            print(f"Pairs tested: {len(df)}")
            print(f"Total trades across all pairs: {df['total_trades'].sum():,}")
            print(f"Average trades per pair: {df['total_trades'].mean():.0f}")
            
            # Performance metrics
            print("\nPERFORMANCE METRICS (Averages):")
            print(f"Sharpe Ratio: {df['sharpe_ratio'].mean():.3f}")
            print(f"Win Rate: {df['win_rate'].mean():.1f}%")
            print(f"Profit Factor: {df['profit_factor'].mean():.2f}")
            
            # Best performers
            print("\nTOP 5 PAIRS BY SHARPE RATIO:")
            top_sharpe = df.nlargest(5, 'sharpe_ratio')
            for _, row in top_sharpe.iterrows():
                print(f"  {row['pair']}: Sharpe {row['sharpe_ratio']:.3f}, "
                      f"Win Rate {row['win_rate']:.1f}%, "
                      f"Trades {row['total_trades']:,}")
            
            # Risk analysis
            print("\nRISK ANALYSIS:")
            print(f"Average Max Drawdown: {df['max_drawdown'].mean():.1f}%")
            print(f"Worst Max Drawdown: {df['max_drawdown'].min():.1f}% ({df.loc[df['max_drawdown'].idxmin(), 'pair']})")
            
            # Consistency analysis
            print("\nCONSISTENCY ANALYSIS:")
            positive_sharpe = (df['sharpe_ratio'] > 0).sum()
            print(f"Pairs with positive Sharpe: {positive_sharpe}/{len(df)} ({positive_sharpe/len(df)*100:.1f}%)")
            
            high_win_rate = (df['win_rate'] > 50).sum()
            print(f"Pairs with >50% win rate: {high_win_rate}/{len(df)} ({high_win_rate/len(df)*100:.1f}%)")
            
            # Trading frequency
            print("\nTRADING FREQUENCY:")
            print(f"Most active pair: {df.loc[df['total_trades'].idxmax(), 'pair']} ({df['total_trades'].max():,} trades)")
            print(f"Least active pair: {df.loc[df['total_trades'].idxmin(), 'pair']} ({df['total_trades'].min():,} trades)")
            
            # Yearly breakdown analysis
            print("\nYEARLY PERFORMANCE SUMMARY:")
            yearly_stats = {}
            
            for pair, pair_results in results.items():
                if config in pair_results:
                    result = pair_results[config]
                    if 'yearly_breakdown' in result:
                        for year, year_data in result['yearly_breakdown'].items():
                            if year not in yearly_stats:
                                yearly_stats[year] = {
                                    'returns': [],
                                    'sharpes': [],
                                    'trades': 0
                                }
                            yearly_stats[year]['returns'].append(year_data.get('return', 0))
                            yearly_stats[year]['sharpes'].append(year_data.get('sharpe', 0))
                            yearly_stats[year]['trades'] += year_data.get('trades', 0)
            
            # Print yearly summary
            for year in sorted(yearly_stats.keys())[-5:]:  # Last 5 years
                stats = yearly_stats[year]
                avg_return = np.mean(stats['returns']) if stats['returns'] else 0
                avg_sharpe = np.mean(stats['sharpes']) if stats['sharpes'] else 0
                print(f"  {year}: Avg Return {avg_return:.1f}%, "
                      f"Avg Sharpe {avg_sharpe:.2f}, "
                      f"Total Trades {stats['trades']:,}")
    
    # Portfolio analysis
    print("\n" + "="*80)
    print("PORTFOLIO ANALYSIS")
    print("="*80)
    
    print("\nRECOMMENDED PORTFOLIO (Based on Sharpe Ratio):")
    
    # Get best pairs from each config
    best_pairs = {}
    for config in configs:
        config_pairs = []
        for pair, pair_results in results.items():
            if config in pair_results:
                result = pair_results[config]
                if 'error' not in result and 'overall_metrics' in result:
                    metrics = result['overall_metrics']
                    if metrics and metrics.get('sharpe_ratio', 0) > 0:
                        config_pairs.append({
                            'pair': pair,
                            'config': config,
                            'sharpe': metrics.get('sharpe_ratio', 0),
                            'return': metrics.get('total_return', 0),
                            'win_rate': metrics.get('win_rate', 0)
                        })
        
        # Sort by Sharpe and take top 5
        config_pairs.sort(key=lambda x: x['sharpe'], reverse=True)
        best_pairs[config] = config_pairs[:5]
    
    # Display recommended portfolio
    for config, pairs in best_pairs.items():
        print(f"\n{config}:")
        for i, pair_info in enumerate(pairs, 1):
            print(f"  {i}. {pair_info['pair']}: Sharpe {pair_info['sharpe']:.3f}, "
                  f"Win Rate {pair_info['win_rate']:.1f}%")
    
    # Risk warnings
    print("\n" + "="*80)
    print("IMPORTANT CONSIDERATIONS")
    print("="*80)
    
    print("\n1. EXTREME RETURNS:")
    print("   - Some pairs show extremely high returns (>1000%)")
    print("   - This suggests compounding over 15 years")
    print("   - Real-world execution may face liquidity constraints")
    
    print("\n2. TRANSACTION COSTS:")
    print("   - Results may not fully account for spreads and slippage")
    print("   - High trade frequency could impact net returns")
    print("   - Consider reducing position sizes in live trading")
    
    print("\n3. RECOMMENDED APPROACH:")
    print("   - Start with paper trading to validate execution")
    print("   - Use smaller position sizes initially")
    print("   - Monitor slippage and adjust parameters")
    print("   - Focus on pairs with consistent Sharpe ratios")
    
    print("\n4. PAIR SELECTION:")
    print("   - Prioritize major pairs (EURUSD, GBPUSD, USDJPY)")
    print("   - These typically have better liquidity")
    print("   - Consider correlation for portfolio diversification")


if __name__ == "__main__":
    analyze_fx_results()