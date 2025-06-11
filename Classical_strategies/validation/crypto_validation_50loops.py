"""
Comprehensive Crypto Strategy Validation
50 loops of 10,000 rows each for robust statistical analysis
"""

import pandas as pd
import numpy as np
from crypto_strategy_final import FinalCryptoStrategy, create_final_conservative_config, create_final_moderate_config
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


def run_comprehensive_validation(n_loops=50, sample_size=10000):
    """
    Run comprehensive validation with 50 loops of 10k rows each
    """
    
    print("="*80)
    print("COMPREHENSIVE CRYPTO STRATEGY VALIDATION")
    print(f"Running {n_loops} loops with {sample_size:,} rows each")
    print("="*80)
    
    # Load data
    data_path = '../crypto_data/ETHUSD_MASTER_15M.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"\nData loaded: {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Check if we have enough data
    if len(df) < sample_size:
        print(f"Insufficient data. Need at least {sample_size:,} rows, have {len(df):,}")
        return None
    
    # Add indicators to full dataset once
    print("\nCalculating indicators on full dataset...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    print("Indicators calculated successfully")
    
    # Configurations to test
    configs = [
        ("Conservative", create_final_conservative_config()),
        ("Moderate", create_final_moderate_config())
    ]
    
    # Store all results
    validation_results = {}
    
    for config_name, config in configs:
        print(f"\n\n{'='*60}")
        print(f"Validating {config_name} Configuration")
        print(f"{'='*60}")
        
        loop_results = []
        
        # Progress tracking
        print(f"\nRunning {n_loops} loops...")
        
        for loop in range(n_loops):
            # Get random sample
            max_start = len(df) - sample_size
            start_idx = np.random.randint(0, max_start)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            
            # Get date range for this sample
            start_date = sample_df.index[0]
            end_date = sample_df.index[-1]
            
            # Calculate some market stats for this period
            price_change = (sample_df['Close'].iloc[-1] - sample_df['Close'].iloc[0]) / sample_df['Close'].iloc[0] * 100
            volatility = sample_df['Close'].pct_change().std() * np.sqrt(96 * 365) * 100  # Annualized
            
            # Run strategy
            strategy = FinalCryptoStrategy(config)
            
            try:
                metrics = strategy.run_backtest(sample_df)
                
                # Add additional info
                metrics['loop'] = loop + 1
                metrics['start_date'] = start_date
                metrics['end_date'] = end_date
                metrics['market_return'] = price_change
                metrics['market_volatility'] = volatility
                metrics['sample_years'] = (end_date - start_date).days / 365.25
                
                loop_results.append(metrics)
                
                # Progress update
                if (loop + 1) % 10 == 0:
                    completed = loop + 1
                    avg_sharpe = np.mean([r['sharpe_ratio'] for r in loop_results])
                    positive_sharpe = sum(1 for r in loop_results if r['sharpe_ratio'] > 0)
                    print(f"  Progress: {completed}/{n_loops} loops | "
                          f"Avg Sharpe: {avg_sharpe:.3f} | "
                          f"Positive: {positive_sharpe}/{completed}")
                    
            except Exception as e:
                print(f"  Error in loop {loop + 1}: {e}")
                continue
        
        # Store results
        validation_results[config_name] = loop_results
        
        # Calculate statistics
        if loop_results:
            print(f"\n{config_name} Validation Complete!")
            print(f"Successful loops: {len(loop_results)}/{n_loops}")
            
            # Performance metrics
            sharpe_ratios = [r['sharpe_ratio'] for r in loop_results]
            returns = [r['total_return_pct'] for r in loop_results]
            win_rates = [r['win_rate'] for r in loop_results]
            max_dds = [r['max_drawdown'] for r in loop_results]
            profit_factors = [r['profit_factor'] for r in loop_results if r['profit_factor'] != np.inf]
            
            print(f"\nPerformance Statistics:")
            print(f"  Sharpe Ratio: {np.mean(sharpe_ratios):.3f} ± {np.std(sharpe_ratios):.3f}")
            print(f"  Returns: {np.mean(returns):.1f}% ± {np.std(returns):.1f}%")
            print(f"  Win Rate: {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
            print(f"  Max Drawdown: {np.mean(max_dds):.1f}% ± {np.std(max_dds):.1f}%")
            print(f"  Profit Factor: {np.mean(profit_factors):.2f}")
            
            # Robustness metrics
            sharpe_above_0 = sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios) * 100
            sharpe_above_1 = sum(1 for s in sharpe_ratios if s > 1) / len(sharpe_ratios) * 100
            profitable_loops = sum(1 for r in returns if r > 0) / len(returns) * 100
            
            print(f"\nRobustness Metrics:")
            print(f"  Sharpe > 0: {sharpe_above_0:.1f}%")
            print(f"  Sharpe > 1: {sharpe_above_1:.1f}%")
            print(f"  Profitable loops: {profitable_loops:.1f}%")
            
            # Market condition analysis
            bull_markets = [r for r in loop_results if r['market_return'] > 10]
            bear_markets = [r for r in loop_results if r['market_return'] < -10]
            range_markets = [r for r in loop_results if -10 <= r['market_return'] <= 10]
            
            if bull_markets:
                bull_sharpe = np.mean([r['sharpe_ratio'] for r in bull_markets])
                print(f"\n  Bull Market Sharpe ({len(bull_markets)} samples): {bull_sharpe:.3f}")
            if bear_markets:
                bear_sharpe = np.mean([r['sharpe_ratio'] for r in bear_markets])
                print(f"  Bear Market Sharpe ({len(bear_markets)} samples): {bear_sharpe:.3f}")
            if range_markets:
                range_sharpe = np.mean([r['sharpe_ratio'] for r in range_markets])
                print(f"  Range Market Sharpe ({len(range_markets)} samples): {range_sharpe:.3f}")
    
    return validation_results


def analyze_validation_results(results):
    """
    Detailed analysis of validation results
    """
    
    print("\n\n" + "="*80)
    print("DETAILED VALIDATION ANALYSIS")
    print("="*80)
    
    # Create visualizations directory
    os.makedirs('validation_charts', exist_ok=True)
    
    for config_name, loop_results in results.items():
        print(f"\n{config_name} Configuration Analysis")
        print("-"*60)
        
        # Convert to DataFrame for easier analysis
        df_results = pd.DataFrame(loop_results)
        
        # 1. Distribution Analysis
        print("\n1. Distribution of Returns:")
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            value = np.percentile(df_results['sharpe_ratio'], p)
            print(f"  {p}th percentile Sharpe: {value:.3f}")
        
        # 2. Consistency Analysis
        print("\n2. Consistency Metrics:")
        consecutive_positive = 0
        max_consecutive_positive = 0
        consecutive_negative = 0
        max_consecutive_negative = 0
        
        for sharpe in df_results['sharpe_ratio']:
            if sharpe > 0:
                consecutive_positive += 1
                consecutive_negative = 0
                max_consecutive_positive = max(max_consecutive_positive, consecutive_positive)
            else:
                consecutive_negative += 1
                consecutive_positive = 0
                max_consecutive_negative = max(max_consecutive_negative, consecutive_negative)
        
        print(f"  Max consecutive positive Sharpe: {max_consecutive_positive}")
        print(f"  Max consecutive negative Sharpe: {max_consecutive_negative}")
        
        # 3. Risk Analysis
        print("\n3. Risk Analysis:")
        worst_drawdown = df_results['max_drawdown'].min()
        worst_return = df_results['total_return_pct'].min()
        var_95 = np.percentile(df_results['total_return_pct'], 5)
        
        print(f"  Worst drawdown: {worst_drawdown:.1f}%")
        print(f"  Worst return: {worst_return:.1f}%")
        print(f"  95% VaR: {var_95:.1f}%")
        
        # 4. Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{config_name} Configuration Validation Results', fontsize=16)
        
        # Sharpe ratio distribution
        axes[0, 0].hist(df_results['sharpe_ratio'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Sharpe = 0')
        axes[0, 0].axvline(x=1, color='green', linestyle='--', label='Sharpe = 1')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sharpe Ratio Distribution')
        axes[0, 0].legend()
        
        # Returns distribution
        axes[0, 1].hist(df_results['total_return_pct'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Breakeven')
        axes[0, 1].set_xlabel('Total Return (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].legend()
        
        # Sharpe vs Market Return
        axes[1, 0].scatter(df_results['market_return'], df_results['sharpe_ratio'], alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Market Return (%)')
        axes[1, 0].set_ylabel('Strategy Sharpe Ratio')
        axes[1, 0].set_title('Performance vs Market Conditions')
        
        # Win Rate vs Sharpe
        axes[1, 1].scatter(df_results['win_rate'], df_results['sharpe_ratio'], alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Win Rate (%)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_title('Win Rate vs Performance')
        
        plt.tight_layout()
        plt.savefig(f'validation_charts/crypto_{config_name.lower()}_validation.png', dpi=300)
        plt.close()
        
        # 5. Create detailed statistics table
        stats_summary = {
            'mean_sharpe': df_results['sharpe_ratio'].mean(),
            'std_sharpe': df_results['sharpe_ratio'].std(),
            'median_sharpe': df_results['sharpe_ratio'].median(),
            'mean_return': df_results['total_return_pct'].mean(),
            'std_return': df_results['total_return_pct'].std(),
            'mean_win_rate': df_results['win_rate'].mean(),
            'mean_max_dd': df_results['max_drawdown'].mean(),
            'sharpe_above_0_pct': (df_results['sharpe_ratio'] > 0).sum() / len(df_results) * 100,
            'sharpe_above_1_pct': (df_results['sharpe_ratio'] > 1).sum() / len(df_results) * 100,
            'profitable_pct': (df_results['total_return_pct'] > 0).sum() / len(df_results) * 100
        }
        
        print("\n4. Summary Statistics:")
        for key, value in stats_summary.items():
            print(f"  {key}: {value:.3f}")
    
    return True


def create_final_report(results):
    """
    Create comprehensive validation report
    """
    
    report = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_loops': 50,
        'sample_size': 10000,
        'configurations': {}
    }
    
    for config_name, loop_results in results.items():
        if loop_results:
            df = pd.DataFrame(loop_results)
            
            config_stats = {
                'n_successful_loops': len(loop_results),
                'sharpe_mean': float(df['sharpe_ratio'].mean()),
                'sharpe_std': float(df['sharpe_ratio'].std()),
                'sharpe_min': float(df['sharpe_ratio'].min()),
                'sharpe_max': float(df['sharpe_ratio'].max()),
                'return_mean': float(df['total_return_pct'].mean()),
                'return_std': float(df['total_return_pct'].std()),
                'win_rate_mean': float(df['win_rate'].mean()),
                'max_dd_mean': float(df['max_drawdown'].mean()),
                'sharpe_above_0_pct': float((df['sharpe_ratio'] > 0).sum() / len(df) * 100),
                'sharpe_above_1_pct': float((df['sharpe_ratio'] > 1).sum() / len(df) * 100),
                'profitable_loops_pct': float((df['total_return_pct'] > 0).sum() / len(df) * 100)
            }
            
            report['configurations'][config_name] = config_stats
    
    # Save report
    with open('results/crypto_validation_50loops_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Validation report saved to results/crypto_validation_50loops_report.json")
    
    return report


def main():
    """
    Run complete validation process
    """
    
    print("Starting comprehensive crypto strategy validation...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run validation
    results = run_comprehensive_validation(n_loops=50, sample_size=10000)
    
    if results:
        # Analyze results
        analyze_validation_results(results)
        
        # Create report
        report = create_final_report(results)
        
        # Final verdict
        print("\n\n" + "="*80)
        print("FINAL VALIDATION VERDICT")
        print("="*80)
        
        for config_name, stats in report['configurations'].items():
            print(f"\n{config_name} Configuration:")
            print(f"  Average Sharpe: {stats['sharpe_mean']:.3f} ± {stats['sharpe_std']:.3f}")
            print(f"  Success Rate: {stats['sharpe_above_0_pct']:.1f}% positive Sharpe")
            print(f"  Excellence Rate: {stats['sharpe_above_1_pct']:.1f}% Sharpe > 1.0")
            
            if stats['sharpe_mean'] > 1.0 and stats['sharpe_above_0_pct'] > 80:
                print(f"  ✅ HIGHLY ROBUST - Excellent for production")
            elif stats['sharpe_mean'] > 0.5 and stats['sharpe_above_0_pct'] > 70:
                print(f"  ✅ ROBUST - Good for production with monitoring")
            elif stats['sharpe_mean'] > 0 and stats['sharpe_above_0_pct'] > 60:
                print(f"  ⚠️ MODERATE - Needs optimization")
            else:
                print(f"  ❌ WEAK - Not recommended for production")
        
        print(f"\nValidation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\nValidation failed. Please check data and try again.")


if __name__ == "__main__":
    main()