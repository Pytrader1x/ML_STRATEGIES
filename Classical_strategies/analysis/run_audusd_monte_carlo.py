"""
Run Monte Carlo simulation specifically for AUDUSD
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')
from robust_sharpe_both_configs_monte_carlo import run_monte_carlo_test_both_configs

def main():
    print("=" * 80)
    print("RUNNING AUDUSD MONTE CARLO SIMULATION")
    print("=" * 80)
    
    # Load AUDUSD data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    print(f"Loaded AUDUSD data: {len(df):,} rows")
    
    # Run Monte Carlo simulation
    print("\nRunning 30 iterations of Monte Carlo simulation...")
    results = run_monte_carlo_test_both_configs(
        n_iterations=30,
        sample_size=10000,
        plot_last=False,
        save_plots=False,
        symbol='AUDUSD'
    )
    
    print("\n" + "="*80)
    print("AUDUSD MONTE CARLO RESULTS SUMMARY")
    print("="*80)
    
    # Display results
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        
        sharpe_ratios = [r['sharpe_ratio'] for r in config_results]
        total_pnls = [r['total_pnl'] for r in config_results]
        win_rates = [r['win_rate'] for r in config_results]
        
        print(f"- Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f} ± {np.std(sharpe_ratios):.3f}")
        print(f"- Sharpe > 1.0: {sum(s > 1.0 for s in sharpe_ratios)}/30 ({sum(s > 1.0 for s in sharpe_ratios)/30*100:.1f}%)")
        print(f"- Average P&L: ${np.mean(total_pnls):,.0f} ± ${np.std(total_pnls):,.0f}")
        print(f"- Average Win Rate: {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
        
        # Check for suspicious patterns
        all_positive = all(s > 0 for s in sharpe_ratios)
        if all_positive:
            print("  ⚠️ All iterations positive (requires scrutiny)")
        else:
            print(f"  ✓ Positive iterations: {sum(s > 0 for s in sharpe_ratios)}/30")

if __name__ == "__main__":
    main()