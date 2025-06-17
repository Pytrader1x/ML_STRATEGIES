"""
Optimize risk management parameters for the advanced momentum strategy
"""

import pandas as pd
import numpy as np
from advanced_momentum_strategy import AdvancedMomentumStrategy
import json
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def test_parameters(params):
    """Test a single parameter combination"""
    data, sl_atr, tp_atr, trail_atr, risk_pct = params
    
    try:
        strategy = AdvancedMomentumStrategy(
            data.copy(),
            sl_atr_multiplier=sl_atr,
            tp_atr_multiplier=tp_atr,
            trailing_sl_atr=trail_atr,
            risk_per_trade=risk_pct
        )
        
        df = strategy.run_backtest()
        metrics = strategy.calculate_metrics(df)
        
        return {
            'sl_atr': sl_atr,
            'tp_atr': tp_atr,
            'trail_atr': trail_atr,
            'risk_pct': risk_pct,
            'sharpe': metrics['sharpe'],
            'returns': metrics['returns'],
            'win_rate': metrics['win_rate'],
            'max_dd': metrics['max_dd'],
            'trades': metrics['trades'],
            'profit_factor': metrics['profit_factor'],
            'exit_analysis': metrics['exit_analysis']
        }
    except Exception as e:
        print(f"Error with params SL={sl_atr}, TP={tp_atr}: {str(e)}")
        return None


def optimize_risk_parameters(data_path='../data/AUDUSD_MASTER_15M.csv',
                           test_size=20000,
                           n_jobs=4):
    """Optimize risk management parameters"""
    
    print("="*60)
    print("Risk Parameter Optimization")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}")
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    data = data[-test_size:]
    print(f"Testing on {len(data):,} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Define parameter ranges
    sl_atr_range = [2.0, 2.5, 3.0, 3.5, 4.0]  # Wider stops
    tp_atr_range = [3.0, 4.0, 5.0, 6.0, 8.0]  # Wider targets
    trail_atr_range = [1.5, 2.0, 2.5, 3.0]    # Various trailing stops
    risk_pct_range = [0.01, 0.02, 0.03]       # 1-3% risk per trade
    
    # Create all combinations
    all_combinations = list(itertools.product(
        sl_atr_range, tp_atr_range, trail_atr_range, risk_pct_range
    ))
    
    # Filter valid combinations (TP > SL)
    valid_combinations = [(sl, tp, trail, risk) 
                         for sl, tp, trail, risk in all_combinations 
                         if tp > sl]
    
    print(f"\nTesting {len(valid_combinations)} parameter combinations...")
    
    # Prepare parameter sets for parallel processing
    param_sets = [(data, sl, tp, trail, risk) 
                  for sl, tp, trail, risk in valid_combinations]
    
    # Run optimization in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(test_parameters, params): params 
                  for params in param_sets}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"Completed {completed}/{len(valid_combinations)} tests...")
    
    # Convert to DataFrame and sort by Sharpe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    # Display top 10 results
    print("\n" + "="*80)
    print("TOP 10 PARAMETER COMBINATIONS:")
    print("="*80)
    
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  SL: {row['sl_atr']}x ATR, TP: {row['tp_atr']}x ATR, Trail: {row['trail_atr']}x ATR")
        print(f"  Risk per trade: {row['risk_pct']*100:.1f}%")
        print(f"  Sharpe: {row['sharpe']:.3f}, Returns: {row['returns']:.1f}%")
        print(f"  Win Rate: {row['win_rate']:.1f}%, Max DD: {row['max_dd']:.1f}%")
        print(f"  Trades: {row['trades']}, PF: {row['profit_factor']:.2f}")
    
    # Save best parameters
    best_params = top_10.iloc[0]
    
    # Check if this beats our original strategy
    original_sharpe = 1.286
    if best_params['sharpe'] > original_sharpe:
        print("\n" + "="*60)
        print("🎉 SUCCESS! New parameters beat original strategy!")
        print(f"Original Sharpe: {original_sharpe:.3f}")
        print(f"New Sharpe: {best_params['sharpe']:.3f}")
        print("="*60)
        
        # Save winning parameters
        success_data = {
            'success': True,
            'best_sharpe': float(best_params['sharpe']),
            'best_params': {
                'sl_atr_multiplier': float(best_params['sl_atr']),
                'tp_atr_multiplier': float(best_params['tp_atr']),
                'trailing_sl_atr': float(best_params['trail_atr']),
                'risk_per_trade': float(best_params['risk_pct'])
            },
            'comparison': {
                'original_sharpe': original_sharpe,
                'improvement': float(best_params['sharpe'] - original_sharpe),
                'improvement_pct': float((best_params['sharpe'] / original_sharpe - 1) * 100)
            },
            'metrics': {
                'returns': float(best_params['returns']),
                'win_rate': float(best_params['win_rate']),
                'max_dd': float(best_params['max_dd']),
                'trades': int(best_params['trades']),
                'profit_factor': float(best_params['profit_factor'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('ADVANCED_SUCCESS_PARAMS.json', 'w') as f:
            json.dump(success_data, f, indent=2)
        
        print("\nParameters saved to ADVANCED_SUCCESS_PARAMS.json")
    
    # Save all results
    results_df.to_csv('risk_optimization_results.csv', index=False)
    print("\nAll results saved to risk_optimization_results.csv")
    
    # Create visualization
    plot_optimization_results(results_df)
    
    return results_df, best_params


def plot_optimization_results(results_df):
    """Visualize optimization results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Sharpe vs SL multiplier
    ax1 = axes[0, 0]
    for tp in results_df['tp_atr'].unique():
        subset = results_df[results_df['tp_atr'] == tp]
        ax1.scatter(subset['sl_atr'], subset['sharpe'], alpha=0.6, label=f'TP={tp}')
    ax1.set_xlabel('Stop Loss (ATR multiplier)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio vs Stop Loss Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Win rate vs Sharpe
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['win_rate'], results_df['sharpe'], 
                         c=results_df['profit_factor'], cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe vs Win Rate (colored by Profit Factor)')
    plt.colorbar(scatter, ax=ax2, label='Profit Factor')
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk per trade impact
    ax3 = axes[1, 0]
    for risk in results_df['risk_pct'].unique():
        subset = results_df[results_df['risk_pct'] == risk]
        ax3.scatter(subset['returns'], subset['max_dd'], 
                   alpha=0.6, label=f'{risk*100:.0f}% risk')
    ax3.set_xlabel('Total Returns (%)')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Returns vs Drawdown by Risk Level')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Exit analysis for top 5
    ax4 = axes[1, 1]
    top_5 = results_df.head(5)
    exit_types = ['Stop Loss', 'Take Profit', 'Trailing Stop', 'Momentum Exit']
    
    for i, (idx, row) in enumerate(top_5.iterrows()):
        if row['exit_analysis']:
            values = [row['exit_analysis'].get(exit_type, 0) for exit_type in exit_types]
            total = sum(values)
            if total > 0:
                values_pct = [v/total*100 for v in values]
                ax4.bar([x + i*0.15 for x in range(len(exit_types))], values_pct, 
                       width=0.15, alpha=0.7, 
                       label=f"#{i+1} (Sharpe={row['sharpe']:.2f})")
    
    ax4.set_xticks(range(len(exit_types)))
    ax4.set_xticklabels(exit_types, rotation=45)
    ax4.set_ylabel('Exit Type (%)')
    ax4.set_title('Exit Type Distribution (Top 5 Strategies)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_optimization_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_best_parameters(data_path='../data/AUDUSD_MASTER_15M.csv', 
                       test_period='last_50000'):
    """Run strategy with the best optimized parameters"""
    
    # Load best parameters
    try:
        with open('ADVANCED_SUCCESS_PARAMS.json', 'r') as f:
            params = json.load(f)
            best_params = params['best_params']
    except:
        print("No optimized parameters found. Run optimization first!")
        return
    
    print("="*60)
    print("Running Advanced Strategy with Optimized Parameters")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Stop Loss: {best_params['sl_atr_multiplier']}x ATR")
    print(f"  Take Profit: {best_params['tp_atr_multiplier']}x ATR")
    print(f"  Trailing Stop: {best_params['trailing_sl_atr']}x ATR")
    print(f"  Risk per Trade: {best_params['risk_per_trade']*100:.1f}%")
    
    # Load data
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    if test_period == 'last_50000':
        data = data[-50000:]
    elif test_period == 'last_20000':
        data = data[-20000:]
    
    print(f"\nTesting on {len(data):,} bars")
    
    # Run strategy
    strategy = AdvancedMomentumStrategy(
        data,
        sl_atr_multiplier=best_params['sl_atr_multiplier'],
        tp_atr_multiplier=best_params['tp_atr_multiplier'],
        trailing_sl_atr=best_params['trailing_sl_atr'],
        risk_per_trade=best_params['risk_per_trade']
    )
    
    df = strategy.run_backtest()
    metrics = strategy.calculate_metrics(df)
    
    print("\nResults:")
    print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"Total Returns: {metrics['returns']:.1f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
    print(f"Total Trades: {metrics['trades']}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    if metrics['exit_analysis']:
        print("\nExit Analysis:")
        for reason, count in metrics['exit_analysis'].items():
            print(f"  {reason}: {count}")
    
    # Save detailed trade log
    if strategy.trades:
        trade_df = pd.DataFrame(strategy.trades)
        trade_df.to_csv('optimized_strategy_trades.csv', index=False)
        print("\nTrade log saved to optimized_strategy_trades.csv")
    
    return df, metrics, strategy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize risk parameters for advanced momentum strategy')
    parser.add_argument('--run-best', action='store_true', help='Run with best parameters')
    parser.add_argument('--test-size', type=int, default=20000, help='Number of bars for optimization')
    parser.add_argument('--n-jobs', type=int, default=4, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    if args.run_best:
        run_best_parameters()
    else:
        optimize_risk_parameters(
            test_size=args.test_size,
            n_jobs=args.n_jobs
        )