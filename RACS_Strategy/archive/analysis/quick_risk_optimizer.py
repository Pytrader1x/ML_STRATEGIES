"""
Quick risk parameter optimizer - faster version
"""

import pandas as pd
import numpy as np
from advanced_momentum_strategy import AdvancedMomentumStrategy
import json
from datetime import datetime

def quick_optimize(data_path='../data/AUDUSD_MASTER_15M.csv'):
    """Quick optimization with fewer parameter combinations"""
    
    print("="*60)
    print("Quick Risk Parameter Optimization")
    print("="*60)
    
    # Load data
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    data = data[-10000:]  # Last 10k bars
    print(f"Testing on {len(data):,} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Focused parameter ranges based on market analysis
    sl_atr_range = [2.5, 3.0, 3.5]      # Moderate stops
    tp_atr_range = [4.0, 5.0, 6.0]      # Good risk/reward ratios
    trail_atr_range = [2.0, 2.5]        # Reasonable trailing stops
    
    best_sharpe = -999
    best_params = None
    results = []
    
    print("\nTesting parameter combinations...")
    
    for sl in sl_atr_range:
        for tp in tp_atr_range:
            if tp <= sl:  # Skip invalid combinations
                continue
                
            for trail in trail_atr_range:
                print(f"\nTesting: SL={sl}, TP={tp}, Trail={trail}")
                
                try:
                    strategy = AdvancedMomentumStrategy(
                        data.copy(),
                        sl_atr_multiplier=sl,
                        tp_atr_multiplier=tp,
                        trailing_sl_atr=trail,
                        risk_per_trade=0.02  # Fixed 2% risk
                    )
                    
                    df = strategy.run_backtest()
                    metrics = strategy.calculate_metrics(df)
                    
                    result = {
                        'sl_atr': sl,
                        'tp_atr': tp,
                        'trail_atr': trail,
                        'sharpe': metrics['sharpe'],
                        'returns': metrics['returns'],
                        'win_rate': metrics['win_rate'],
                        'max_dd': metrics['max_dd'],
                        'trades': metrics['trades'],
                        'profit_factor': metrics['profit_factor'],
                        'exit_analysis': metrics['exit_analysis']
                    }
                    
                    results.append(result)
                    
                    print(f"  Sharpe: {metrics['sharpe']:.3f}, Returns: {metrics['returns']:.1f}%")
                    print(f"  Win Rate: {metrics['win_rate']:.1f}%, Trades: {metrics['trades']}")
                    
                    if metrics['sharpe'] > best_sharpe:
                        best_sharpe = metrics['sharpe']
                        best_params = result
                        
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
    
    # Display best result
    if best_params:
        print("\n" + "="*60)
        print("BEST PARAMETERS FOUND:")
        print("="*60)
        print(f"Stop Loss: {best_params['sl_atr']}x ATR")
        print(f"Take Profit: {best_params['tp_atr']}x ATR")
        print(f"Trailing Stop: {best_params['trail_atr']}x ATR")
        print(f"Sharpe Ratio: {best_params['sharpe']:.3f}")
        print(f"Total Returns: {best_params['returns']:.1f}%")
        print(f"Win Rate: {best_params['win_rate']:.1f}%")
        print(f"Max Drawdown: {best_params['max_dd']:.1f}%")
        print(f"Total Trades: {best_params['trades']}")
        print(f"Profit Factor: {best_params['profit_factor']:.2f}")
        
        if best_params['exit_analysis']:
            print("\nExit Breakdown:")
            total_exits = sum(best_params['exit_analysis'].values())
            for reason, count in best_params['exit_analysis'].items():
                pct = (count / total_exits * 100) if total_exits > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Compare with original
        original_sharpe = 1.286
        if best_sharpe > original_sharpe:
            print("\n" + "="*60)
            print("🎉 SUCCESS! Beat original strategy!")
            print(f"Original Sharpe: {original_sharpe:.3f}")
            print(f"New Sharpe: {best_sharpe:.3f}")
            print(f"Improvement: +{best_sharpe - original_sharpe:.3f} ({(best_sharpe/original_sharpe - 1)*100:.1f}%)")
            print("="*60)
        
        # Save results
        save_data = {
            'success': bool(best_sharpe > original_sharpe),
            'best_sharpe': float(best_sharpe),
            'best_params': {
                'sl_atr_multiplier': float(best_params['sl_atr']),
                'tp_atr_multiplier': float(best_params['tp_atr']),
                'trailing_sl_atr': float(best_params['trail_atr']),
                'risk_per_trade': 0.02
            },
            'metrics': {
                'returns': float(best_params['returns']),
                'win_rate': float(best_params['win_rate']),
                'max_dd': float(best_params['max_dd']),
                'trades': int(best_params['trades']),
                'profit_factor': float(best_params['profit_factor'])
            },
            'exit_analysis': best_params['exit_analysis'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('quick_optimization_results.json', 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print("\nResults saved to quick_optimization_results.json")
        
        # Save all results
        results_df = pd.DataFrame(results)
        results_df.to_csv('quick_optimization_details.csv', index=False)
        print("Details saved to quick_optimization_details.csv")
    
    return best_params


def test_on_different_periods(best_params):
    """Test best parameters on different time periods"""
    
    print("\n" + "="*60)
    print("Testing Best Parameters on Different Periods")
    print("="*60)
    
    data_path = '../data/AUDUSD_MASTER_15M.csv'
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    periods = {
        'Recent 5k': data[-5000:],
        'Recent 20k': data[-20000:],
        'Recent 50k': data[-50000:],
        'Mid 2024': data['2024-04-01':'2024-09-30'],
        'Early 2024': data['2024-01-01':'2024-03-31']
    }
    
    for period_name, period_data in periods.items():
        if len(period_data) < 1000:
            continue
            
        print(f"\n{period_name} ({len(period_data):,} bars):")
        
        strategy = AdvancedMomentumStrategy(
            period_data,
            sl_atr_multiplier=best_params['sl_atr'],
            tp_atr_multiplier=best_params['tp_atr'],
            trailing_sl_atr=best_params['trail_atr'],
            risk_per_trade=0.02
        )
        
        df = strategy.run_backtest()
        metrics = strategy.calculate_metrics(df)
        
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Returns: {metrics['returns']:.1f}%")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%, Max DD: {metrics['max_dd']:.1f}%")


if __name__ == "__main__":
    # Run quick optimization
    best = quick_optimize()
    
    # Test on different periods if we found good parameters
    if best and best['sharpe'] > 0.5:
        test_on_different_periods(best)