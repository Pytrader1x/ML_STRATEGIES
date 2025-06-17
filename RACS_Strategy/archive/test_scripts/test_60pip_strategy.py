"""
Test the advanced momentum strategy with 60 pip stop loss
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_momentum_strategy import AdvancedMomentumStrategy, run_advanced_backtest
import pandas as pd

def test_60pip_strategy():
    """Test strategy with 60 pip fixed stop loss"""
    
    print("="*60)
    print("Testing Advanced Momentum Strategy with 60 Pip Stop Loss")
    print("="*60)
    
    # Load data
    data_path = '../data/AUDUSD_MASTER_15M.csv'
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Use last 50,000 bars for testing
    data = data[-50000:]
    
    print(f"\nTesting on {len(data):,} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create strategy with 60 pip stop loss
    strategy = AdvancedMomentumStrategy(
        data,
        # Keep original momentum parameters
        lookback=40,
        entry_z=1.5,
        exit_z=0.5,
        # Use fixed 60 pip stop loss
        fixed_sl_pips=60,
        use_fixed_sl=True,
        # Keep ATR-based take profit
        tp_atr_multiplier=3.0,
        # 2% risk per trade
        risk_per_trade=0.02,
        initial_capital=10000
    )
    
    # Run backtest
    df = strategy.run_backtest()
    metrics = strategy.calculate_metrics(df)
    
    print("\n" + "-"*40)
    print("BACKTEST RESULTS (60 PIP STOP LOSS):")
    print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"Total Returns: {metrics['returns']:.1f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
    print(f"Total Trades: {metrics['trades']}")
    print(f"Average Win: {metrics['avg_win']:.2f}%")
    print(f"Average Loss: {metrics['avg_loss']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    if metrics['exit_analysis']:
        print("\nExit Analysis:")
        for reason, count in metrics['exit_analysis'].items():
            print(f"  {reason}: {count}")
    
    # Save trade log
    if strategy.trades:
        trade_df = pd.DataFrame(strategy.trades)
        trade_df.to_csv('60pip_strategy_trades.csv', index=False)
        print("\nTrade log saved to 60pip_strategy_trades.csv")
    
    # Compare different stop loss levels
    print("\n" + "="*40)
    print("Testing Different Stop Loss Levels:")
    print("="*40)
    
    sl_levels = [30, 40, 50, 60, 70, 80, 100]
    results = []
    
    for sl_pips in sl_levels:
        strategy = AdvancedMomentumStrategy(
            data,
            lookback=40,
            entry_z=1.5,
            exit_z=0.5,
            fixed_sl_pips=sl_pips,
            use_fixed_sl=True,
            tp_atr_multiplier=3.0,
            risk_per_trade=0.02,
            initial_capital=10000
        )
        
        df = strategy.run_backtest()
        metrics = strategy.calculate_metrics(df)
        
        results.append({
            'sl_pips': sl_pips,
            'sharpe': metrics['sharpe'],
            'returns': metrics['returns'],
            'win_rate': metrics['win_rate'],
            'max_dd': metrics['max_dd'],
            'trades': metrics['trades'],
            'profit_factor': metrics['profit_factor']
        })
        
        print(f"\nSL: {sl_pips} pips - Sharpe: {metrics['sharpe']:.3f}, "
              f"Returns: {metrics['returns']:.1f}%, Win Rate: {metrics['win_rate']:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('stop_loss_comparison.csv', index=False)
    print("\nStop loss comparison saved to stop_loss_comparison.csv")
    
    # Find optimal stop loss
    best_idx = results_df['sharpe'].idxmax()
    best_sl = results_df.iloc[best_idx]
    
    print("\n" + "="*40)
    print("OPTIMAL STOP LOSS:")
    print(f"Stop Loss: {best_sl['sl_pips']} pips")
    print(f"Sharpe Ratio: {best_sl['sharpe']:.3f}")
    print(f"Total Returns: {best_sl['returns']:.1f}%")
    print(f"Win Rate: {best_sl['win_rate']:.1f}%")
    print("="*40)

if __name__ == "__main__":
    test_60pip_strategy()