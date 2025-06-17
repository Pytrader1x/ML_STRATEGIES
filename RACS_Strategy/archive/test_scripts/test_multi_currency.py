"""
Test the winning momentum strategy across multiple currency pairs
to verify it's not overfitted to AUDUSD
"""

import pandas as pd
import numpy as np
import json
from ultimate_optimizer import AdvancedBacktest
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def test_currency_pair(currency_pair, data_path, lookback=40, entry_z=1.5, exit_z=0.5, 
                      test_period='last_50000', save_details=False):
    """Test the winning strategy on a specific currency pair"""
    
    print(f"\n{'='*60}")
    print(f"Testing {currency_pair}")
    print(f"{'='*60}")
    
    try:
        # Load data
        data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Total data points: {len(data):,}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Select test period
        if test_period == 'last_50000':
            test_data = data[-50000:] if len(data) > 50000 else data
        elif test_period == 'last_20000':
            test_data = data[-20000:] if len(data) > 20000 else data
        elif test_period == 'full':
            test_data = data
        else:
            # Assume it's a tuple with start and end dates
            if isinstance(test_period, tuple):
                start_date, end_date = test_period
                test_data = data[start_date:end_date]
            else:
                test_data = data
        
        print(f"Testing on {len(test_data):,} bars")
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Run backtest
        backtester = AdvancedBacktest(test_data)
        result = backtester.strategy_momentum(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z
        )
        
        # Print results
        print(f"\nResults:")
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"Total Returns: {result['returns']:.1f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Max Drawdown: {result['max_dd']:.1f}%")
        print(f"Total Trades: {result['trades']}")
        
        # Save detailed results if requested
        if save_details:
            detail_file = f"{currency_pair}_backtest_details.json"
            with open(detail_file, 'w') as f:
                json.dump({
                    'currency_pair': currency_pair,
                    'test_period': str(test_period),
                    'data_range': f"{test_data.index[0]} to {test_data.index[-1]}",
                    'bars_tested': len(test_data),
                    'parameters': {
                        'lookback': lookback,
                        'entry_z': entry_z,
                        'exit_z': exit_z
                    },
                    'results': result,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            print(f"Detailed results saved to {detail_file}")
        
        return {
            'currency_pair': currency_pair,
            'sharpe': result['sharpe'],
            'returns': result['returns'],
            'win_rate': result['win_rate'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'bars_tested': len(test_data),
            'test_range': f"{test_data.index[0]} to {test_data.index[-1]}"
        }
        
    except Exception as e:
        print(f"Error testing {currency_pair}: {str(e)}")
        return {
            'currency_pair': currency_pair,
            'sharpe': None,
            'returns': None,
            'win_rate': None,
            'max_dd': None,
            'trades': None,
            'bars_tested': 0,
            'test_range': 'Error',
            'error': str(e)
        }


def run_multi_currency_test(test_period='last_50000', save_plots=True):
    """Run the winning strategy on multiple currency pairs"""
    
    print("="*80)
    print("MULTI-CURRENCY BACKTEST - WINNING MOMENTUM STRATEGY")
    print("="*80)
    print(f"\nStrategy Parameters (from AUDUSD optimization):")
    print(f"Lookback: 40")
    print(f"Entry Z-Score: 1.5")
    print(f"Exit Z-Score: 0.5")
    print(f"Original AUDUSD Sharpe: 1.286")
    
    # Define currency pairs to test
    currency_pairs = {
        'AUDUSD': '../data/AUDUSD_MASTER_15M.csv',
        'AUDJPY': '../data/AUDJPY_MASTER_15M.csv',
        'GBPUSD': '../data/GBPUSD_MASTER_15M.csv',
        'USDCAD': '../data/USDCAD_MASTER_15M.csv',
        'NZDUSD': '../data/NZDUSD_MASTER_15M.csv',
        'EURUSD': '../data/EURUSD_MASTER_15M.csv'
    }
    
    # Run tests
    results = []
    for pair, data_path in currency_pairs.items():
        result = test_currency_pair(
            currency_pair=pair,
            data_path=data_path,
            test_period=test_period,
            save_details=True
        )
        results.append(result)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe', ascending=False, na_position='last')
    
    # Save summary results
    results_df.to_csv('multi_currency_test_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print("\nResults sorted by Sharpe Ratio:")
    print("-"*80)
    print(f"{'Currency':<10} {'Sharpe':>8} {'Returns':>10} {'Win Rate':>10} {'Max DD':>10} {'Trades':>8}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        if row['sharpe'] is not None:
            print(f"{row['currency_pair']:<10} {row['sharpe']:>8.3f} {row['returns']:>10.1f}% "
                  f"{row['win_rate']:>10.1f}% {row['max_dd']:>10.1f}% {row['trades']:>8}")
        else:
            print(f"{row['currency_pair']:<10} {'Error':>8}")
    
    # Calculate statistics
    valid_results = results_df.dropna(subset=['sharpe'])
    if len(valid_results) > 0:
        print("\n" + "-"*80)
        print("STATISTICS:")
        print(f"Average Sharpe: {valid_results['sharpe'].mean():.3f}")
        print(f"Median Sharpe: {valid_results['sharpe'].median():.3f}")
        print(f"Std Dev Sharpe: {valid_results['sharpe'].std():.3f}")
        print(f"Min Sharpe: {valid_results['sharpe'].min():.3f}")
        print(f"Max Sharpe: {valid_results['sharpe'].max():.3f}")
        
        # Count how many have positive Sharpe
        positive_sharpe = (valid_results['sharpe'] > 0).sum()
        above_one = (valid_results['sharpe'] > 1.0).sum()
        print(f"\nPositive Sharpe: {positive_sharpe}/{len(valid_results)} ({positive_sharpe/len(valid_results)*100:.1f}%)")
        print(f"Sharpe > 1.0: {above_one}/{len(valid_results)} ({above_one/len(valid_results)*100:.1f}%)")
    
    # Create visualization
    if save_plots and len(valid_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sharpe Ratio comparison
        ax1 = axes[0, 0]
        colors = ['green' if x > 1.0 else 'orange' if x > 0 else 'red' 
                  for x in valid_results['sharpe']]
        ax1.bar(valid_results['currency_pair'], valid_results['sharpe'], color=colors)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax1.axhline(y=1.286, color='blue', linestyle='--', alpha=0.5, label='AUDUSD Original')
        ax1.set_title('Sharpe Ratio by Currency Pair', fontsize=14)
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns comparison
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in valid_results['returns']]
        ax2.bar(valid_results['currency_pair'], valid_results['returns'], color=colors)
        ax2.set_title('Total Returns by Currency Pair', fontsize=14)
        ax2.set_ylabel('Returns (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate comparison
        ax3 = axes[1, 0]
        ax3.bar(valid_results['currency_pair'], valid_results['win_rate'], color='blue')
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Win Rate by Currency Pair', fontsize=14)
        ax3.set_ylabel('Win Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk metrics scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(valid_results['max_dd'], valid_results['sharpe'], 
                            s=valid_results['trades']/10, alpha=0.6)
        for i, txt in enumerate(valid_results['currency_pair']):
            ax4.annotate(txt, (valid_results['max_dd'].iloc[i], valid_results['sharpe'].iloc[i]))
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Return Profile (size = # trades/10)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_currency_test_results.png', dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to multi_currency_test_results.png")
        plt.show()
    
    # Generate report
    generate_report(results_df, test_period)
    
    return results_df


def generate_report(results_df, test_period):
    """Generate a detailed report of the multi-currency test"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MULTI-CURRENCY STRATEGY TEST REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Period: {test_period}")
    report_lines.append("\nStrategy: Momentum (Z-Score Mean Reversion)")
    report_lines.append("Parameters:")
    report_lines.append("  - Lookback: 40 bars")
    report_lines.append("  - Entry Z-Score: 1.5")
    report_lines.append("  - Exit Z-Score: 0.5")
    report_lines.append("\nOriginal Performance (AUDUSD):")
    report_lines.append("  - Sharpe Ratio: 1.286")
    report_lines.append("  - Discovered via optimization on AUDUSD data")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("RESULTS SUMMARY")
    report_lines.append("="*80)
    
    valid_results = results_df.dropna(subset=['sharpe'])
    
    if len(valid_results) > 0:
        # Best and worst performers
        best = valid_results.iloc[0]
        worst = valid_results.iloc[-1]
        
        report_lines.append(f"\nBest Performer: {best['currency_pair']}")
        report_lines.append(f"  - Sharpe: {best['sharpe']:.3f}")
        report_lines.append(f"  - Returns: {best['returns']:.1f}%")
        report_lines.append(f"  - Win Rate: {best['win_rate']:.1f}%")
        
        report_lines.append(f"\nWorst Performer: {worst['currency_pair']}")
        report_lines.append(f"  - Sharpe: {worst['sharpe']:.3f}")
        report_lines.append(f"  - Returns: {worst['returns']:.1f}%")
        report_lines.append(f"  - Win Rate: {worst['win_rate']:.1f}%")
        
        # Overall statistics
        report_lines.append("\nOverall Statistics:")
        report_lines.append(f"  - Average Sharpe: {valid_results['sharpe'].mean():.3f}")
        report_lines.append(f"  - Sharpe Std Dev: {valid_results['sharpe'].std():.3f}")
        report_lines.append(f"  - Currencies with Sharpe > 0: {(valid_results['sharpe'] > 0).sum()}/{len(valid_results)}")
        report_lines.append(f"  - Currencies with Sharpe > 1.0: {(valid_results['sharpe'] > 1.0).sum()}/{len(valid_results)}")
        
        # Robustness assessment
        report_lines.append("\n" + "="*80)
        report_lines.append("ROBUSTNESS ASSESSMENT")
        report_lines.append("="*80)
        
        avg_sharpe = valid_results['sharpe'].mean()
        if avg_sharpe > 0.5 and (valid_results['sharpe'] > 0).sum() / len(valid_results) > 0.7:
            report_lines.append("\nConclusion: Strategy shows GOOD robustness")
            report_lines.append("  - Positive performance across most currency pairs")
            report_lines.append("  - Not overfitted to AUDUSD")
        elif avg_sharpe > 0.2 and (valid_results['sharpe'] > 0).sum() / len(valid_results) > 0.5:
            report_lines.append("\nConclusion: Strategy shows MODERATE robustness")
            report_lines.append("  - Mixed performance across currency pairs")
            report_lines.append("  - Some degree of generalization beyond AUDUSD")
        else:
            report_lines.append("\nConclusion: Strategy shows POOR robustness")
            report_lines.append("  - Performance highly dependent on currency pair")
            report_lines.append("  - Likely overfitted to AUDUSD characteristics")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        if avg_sharpe > 0.5:
            report_lines.append("  1. Strategy is suitable for multi-currency trading")
            report_lines.append("  2. Consider position sizing based on individual pair performance")
            report_lines.append("  3. Monitor performance regularly across all pairs")
        else:
            report_lines.append("  1. Use strategy primarily on best-performing pairs")
            report_lines.append("  2. Consider pair-specific parameter optimization")
            report_lines.append("  3. Implement additional filters for poor performers")
    
    # Save report
    report_content = '\n'.join(report_lines)
    with open('multi_currency_test_report.txt', 'w') as f:
        f.write(report_content)
    
    print(f"\n\nDetailed report saved to multi_currency_test_report.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test winning strategy on multiple currency pairs')
    parser.add_argument('--period', default='last_50000', 
                       help='Test period: last_50000, last_20000, full, or date range')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    
    args = parser.parse_args()
    
    # Run multi-currency test
    results = run_multi_currency_test(
        test_period=args.period,
        save_plots=not args.no_plots
    )
    
    print("\n" + "="*80)
    print("Multi-Currency Test Complete!")
    print("="*80)
    print("\nFiles created:")
    print("  - multi_currency_test_results.csv")
    print("  - multi_currency_test_report.txt")
    print("  - [CURRENCY]_backtest_details.json (for each pair)")
    if not args.no_plots:
        print("  - multi_currency_test_results.png")