#!/usr/bin/env python3
"""
Extended Crypto Backtesting - Test crypto strategy from 2015 to present
Comprehensive analysis with yearly breakdowns and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classical_strategies.crypto_strategy_final import FinalCryptoStrategy
from technical_indicators_custom import TIC


def run_extended_crypto_backtest(start_year=2015, save_results=True):
    """Run comprehensive crypto backtest from 2015 to present"""
    
    print("="*80)
    print("EXTENDED CRYPTO BACKTESTING - ETH/USD")
    print("="*80)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'crypto_data/ETHUSD_MASTER_15M.csv')
    
    print(f"\nLoading ETH data from: {data_path}")
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    
    # Filter data from start_year
    df = df[df.index.year >= start_year]
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total rows: {len(df):,}")
    
    # Add required indicators
    print("\nPreparing data with indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    print("Indicators added successfully")
    
    # Test both configurations - matching FinalCryptoStrategy format
    configs = {
        'conservative': {
            'initial_capital': 10000,
            'risk_per_trade': 0.01,  # 1% risk
            'max_position_pct': 0.5,  # Max 50% of capital
            'min_stop_pct': 0.05,  # 5% minimum stop
            'atr_multiplier_sl': 3.0,
            'risk_reward_ratio': 3.0,  # 3:1 RR ratio
            'use_trailing_stop': True,
            'trailing_activation_pct': 0.08,  # 8% profit before trailing
            'trailing_distance_pct': 0.03,  # 3% trailing distance
            'trailing_lock_profit_pct': 0.02,  # Lock 2% profit
            'min_trend_score': 3,  # Require strong trend (out of 5)
            'min_bars_between_trades': 4  # 1 hour between trades
        },
        'moderate': {
            'initial_capital': 10000,
            'risk_per_trade': 0.015,  # 1.5% risk
            'max_position_pct': 0.6,  # Max 60% of capital
            'min_stop_pct': 0.04,  # 4% minimum stop
            'atr_multiplier_sl': 2.5,
            'risk_reward_ratio': 2.5,  # 2.5:1 RR ratio
            'use_trailing_stop': True,
            'trailing_activation_pct': 0.06,  # 6% profit before trailing
            'trailing_distance_pct': 0.025,  # 2.5% trailing distance
            'trailing_lock_profit_pct': 0.015,  # Lock 1.5% profit
            'min_trend_score': 2,  # Moderate trend requirement
            'min_bars_between_trades': 2  # 30 mins between trades
        }
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing {config_name.upper()} Configuration")
        print(f"{'='*60}")
        
        # Initialize strategy
        strategy = FinalCryptoStrategy(config)
        
        # Run full backtest
        print("\nRunning full period backtest...")
        metrics = strategy.run_backtest(df)
        trades = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'pnl_percent': t.pnl_pct,
            'exit_reason': t.exit_reason,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600 if t.exit_time else 0
        } for t in strategy.trades if t.exit_time is not None])
        
        # Store overall results
        all_results[config_name] = {
            'full_period': {
                'metrics': metrics,
                'total_trades': len(trades),
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1])
            },
            'yearly_breakdown': {},
            'trades': trades.to_dict('records') if not trades.empty else []
        }
        
        # Print full period results
        print(f"\nFull Period Results ({start_year}-Present):")
        print(f"Total Return: {metrics.get('total_return_pct', 0)/100:.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Win Rate: {metrics.get('win_rate', 0)/100:.2%}")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0)/100:.2%}")
        
        # Yearly breakdown
        print("\nYearly Performance Breakdown:")
        print("-" * 60)
        print(f"{'Year':<6} {'Trades':<8} {'Return':<10} {'Sharpe':<8} {'Win Rate':<10} {'Max DD':<10}")
        print("-" * 60)
        
        for year in range(start_year, datetime.now().year + 1):
            year_data = df[df.index.year == year]
            if len(year_data) > 0:
                # Reset strategy for each year
                strategy.trades = []
                strategy.current_capital = config['initial_capital']
                year_metrics = strategy.run_backtest(year_data)
                if year_metrics['total_trades'] > 0:
                    
                    all_results[config_name]['yearly_breakdown'][year] = {
                        'metrics': year_metrics,
                        'total_trades': year_metrics['total_trades'],
                        'start_date': str(year_data.index[0]),
                        'end_date': str(year_data.index[-1])
                    }
                    
                    print(f"{year:<6} {year_metrics.get('total_trades', 0):<8} "
                          f"{year_metrics.get('total_return_pct', 0):>8.1f}% "
                          f"{year_metrics.get('sharpe_ratio', 0):>8.2f} "
                          f"{year_metrics.get('win_rate', 0):>8.1f}% "
                          f"{year_metrics.get('max_drawdown', 0):>8.1f}%")
                else:
                    print(f"{year:<6} {'0':<8} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
        
        # Additional analysis
        if not trades.empty:
            print(f"\nAdditional Statistics ({config_name}):")
            print(f"Average Trade Duration: {trades['duration_hours'].mean():.1f} hours")
            print(f"Best Trade: {trades['pnl_percent'].max():.2%}")
            print(f"Worst Trade: {trades['pnl_percent'].min():.2%}")
            print(f"Average Win: {trades[trades['pnl_percent'] > 0]['pnl_percent'].mean():.2%}")
            print(f"Average Loss: {trades[trades['pnl_percent'] < 0]['pnl_percent'].mean():.2%}")
            
            # Exit reason analysis
            print(f"\nExit Reasons:")
            exit_counts = trades['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")
    
    # Save results
    if save_results:
        results_dir = 'Classical_strategies/results'
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'extended_crypto_backtest_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n\nResults saved to: {output_file}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    
    for config_name in configs:
        metrics = all_results[config_name]['full_period']['metrics']
        print(f"\n{config_name.upper()}:")
        print(f"  Total Return: {metrics.get('total_return_pct', 0)/100:.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0)/100:.2%}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)/100:.2%}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    return all_results


def analyze_market_conditions(df, trades):
    """Analyze performance under different market conditions"""
    
    # Calculate market metrics
    df['returns'] = df['Close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(20*96).std() * np.sqrt(365*96)  # Annualized
    df['trend_20d'] = (df['Close'] / df['Close'].shift(20*96) - 1)
    
    # Classify market conditions
    conditions = []
    
    for _, trade in trades.iterrows():
        entry_time = trade['entry_time']
        if entry_time in df.index:
            vol = df.loc[entry_time, 'volatility_20d']
            trend = df.loc[entry_time, 'trend_20d']
            
            if pd.notna(vol) and pd.notna(trend):
                if vol < 0.5:
                    vol_regime = 'Low Vol'
                elif vol < 1.0:
                    vol_regime = 'Med Vol'
                else:
                    vol_regime = 'High Vol'
                
                if trend < -0.1:
                    trend_regime = 'Bear'
                elif trend > 0.1:
                    trend_regime = 'Bull'
                else:
                    trend_regime = 'Range'
                
                conditions.append({
                    'trade_id': trade.name,
                    'volatility_regime': vol_regime,
                    'trend_regime': trend_regime,
                    'pnl_percent': trade['pnl_percent']
                })
    
    # Analyze by condition
    if conditions:
        conditions_df = pd.DataFrame(conditions)
        
        print("\nPerformance by Market Condition:")
        print("-" * 50)
        
        # By volatility
        print("\nBy Volatility Regime:")
        for vol in ['Low Vol', 'Med Vol', 'High Vol']:
            subset = conditions_df[conditions_df['volatility_regime'] == vol]
            if len(subset) > 0:
                avg_pnl = subset['pnl_percent'].mean()
                win_rate = (subset['pnl_percent'] > 0).mean()
                print(f"  {vol}: {len(subset)} trades, {avg_pnl:.2%} avg PnL, {win_rate:.1%} win rate")
        
        # By trend
        print("\nBy Trend Regime:")
        for trend in ['Bear', 'Range', 'Bull']:
            subset = conditions_df[conditions_df['trend_regime'] == trend]
            if len(subset) > 0:
                avg_pnl = subset['pnl_percent'].mean()
                win_rate = (subset['pnl_percent'] > 0).mean()
                print(f"  {trend}: {len(subset)} trades, {avg_pnl:.2%} avg PnL, {win_rate:.1%} win rate")


if __name__ == "__main__":
    # Run extended backtest
    results = run_extended_crypto_backtest(start_year=2015)
    
    # Additional market condition analysis
    print("\n" + "="*80)
    print("MARKET CONDITION ANALYSIS")
    print("="*80)
    
    # Load data again for analysis
    df = pd.read_csv('../crypto_data/ETHUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    df = df[df.index.year >= 2015]
    
    # Run analysis for moderate config
    moderate_config = {
        'initial_capital': 10000,
        'risk_per_trade': 0.015,
        'max_position_pct': 0.6,
        'min_stop_pct': 0.04,
        'atr_multiplier_sl': 2.5,
        'risk_reward_ratio': 2.5,
        'use_trailing_stop': True,
        'trailing_activation_pct': 0.06,
        'trailing_distance_pct': 0.025,
        'trailing_lock_profit_pct': 0.015,
        'min_trend_score': 2,
        'min_bars_between_trades': 2
    }
    strategy = FinalCryptoStrategy(moderate_config)
    
    metrics = strategy.run_backtest(df)
    trades = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'pnl_percent': t.pnl_pct,
        'exit_reason': t.exit_reason
    } for t in strategy.trades if t.exit_time is not None])
    
    if not trades.empty:
        analyze_market_conditions(df, trades)