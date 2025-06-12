"""
Trade Pattern Analyzer - Deep dive into what makes trades successful
Focus on specific improvements rather than wholesale changes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig

def create_detailed_test_data(n_periods: int = 3000) -> pd.DataFrame:
    """Create test data with varied market conditions"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2010-01-01', periods=n_periods, freq='1H')
    
    # Create different market regimes
    regime_length = n_periods // 6
    
    # Trending up period
    trend_up = np.cumsum(np.random.normal(0.0005, 0.001, regime_length))
    
    # Trending down period  
    trend_down = np.cumsum(np.random.normal(-0.0005, 0.001, regime_length))
    
    # Ranging periods (low drift)
    range1 = np.cumsum(np.random.normal(0, 0.002, regime_length))
    range2 = np.cumsum(np.random.normal(0, 0.002, regime_length))
    
    # Volatile periods
    volatile1 = np.cumsum(np.random.normal(0, 0.004, regime_length))
    volatile2 = np.cumsum(np.random.normal(0, 0.003, regime_length))
    
    # Combine all regimes
    combined_returns = np.concatenate([trend_up, range1, trend_down, volatile1, range2, volatile2])
    
    # Generate prices
    prices = np.cumprod(1 + combined_returns) * 0.75
    
    # Create OHLC data
    df = pd.DataFrame(index=dates[:len(prices)])
    df['Close'] = prices
    
    # Generate realistic OHLC
    for i in range(len(prices)):
        daily_range = abs(np.random.normal(0, 0.002))
        df.loc[df.index[i], 'Open'] = prices[i] + np.random.normal(0, daily_range/4)
        df.loc[df.index[i], 'High'] = prices[i] + abs(np.random.normal(daily_range/2, daily_range/4))
        df.loc[df.index[i], 'Low'] = prices[i] - abs(np.random.normal(daily_range/2, daily_range/4))
    
    # Fix OHLC consistency
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
        df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])
    
    # Add realistic indicators
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_8'] = df['Close'].ewm(span=8).mean()
    
    # NTI Direction (more nuanced)
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['SMA_Slope'] = df['SMA_20'].pct_change(5)
    
    df['NTI_Direction'] = 0
    # Strong long signals
    df.loc[(df['Price_vs_SMA20'] > 0.003) & (df['SMA_Slope'] > 0.001), 'NTI_Direction'] = 1
    # Strong short signals  
    df.loc[(df['Price_vs_SMA20'] < -0.003) & (df['SMA_Slope'] < -0.001), 'NTI_Direction'] = -1
    
    # NTI Strength
    df['NTI_Strength'] = np.clip(abs(df['Price_vs_SMA20']) * 100 + abs(df['SMA_Slope']) * 50, 0.1, 1.0)
    
    # MB Bias (momentum)
    df['ROC_5'] = df['Close'].pct_change(5)
    df['ROC_10'] = df['Close'].pct_change(10)
    df['RSI'] = 50 + 50 * df['ROC_5']  # Simplified RSI proxy
    
    df['MB_Bias'] = 0
    df.loc[(df['ROC_10'] > 0.002) & (df['RSI'] > 55), 'MB_Bias'] = 1
    df.loc[(df['ROC_10'] < -0.002) & (df['RSI'] < 45), 'MB_Bias'] = -1
    
    # IC Regime (more sophisticated)
    df['Volatility'] = df['Close'].rolling(20).std()
    df['Trend_Strength'] = abs(df['SMA_20'].pct_change(20))
    
    df['IC_Regime'] = 2  # Default
    # Strong trend = low volatility + strong trend
    df.loc[(df['Volatility'] < df['Volatility'].quantile(0.3)) & 
           (df['Trend_Strength'] > df['Trend_Strength'].quantile(0.7)), 'IC_Regime'] = 1
    # Range = high volatility + weak trend
    df.loc[(df['Volatility'] > df['Volatility'].quantile(0.7)) & 
           (df['Trend_Strength'] < df['Trend_Strength'].quantile(0.3)), 'IC_Regime'] = 3
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    df['IC_ATR_Normalized'] = np.clip((df['ATR'] / df['Close'] * 10000), 10, 100)
    
    # Regime names
    regime_map = {1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'}
    df['IC_RegimeName'] = df['IC_Regime'].map(regime_map)
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def analyze_trade_patterns(strategy, df):
    """Analyze individual trade patterns in detail"""
    
    results = strategy.run_backtest(df)
    trades = strategy.trades
    
    if len(trades) == 0:
        print("No trades to analyze!")
        return {}
    
    print(f"\\nTRADE PATTERN ANALYSIS ({len(trades)} trades)")
    print("="*50)
    
    # Separate winning and losing trades
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    
    # Analyze winning trade characteristics
    if winning_trades:
        avg_win_pnl = np.mean([t.pnl for t in winning_trades])
        avg_win_duration = np.mean([(t.exit_time - t.entry_time).total_seconds()/3600 for t in winning_trades])
        
        print(f"\\nWINNING TRADE CHARACTERISTICS:")
        print(f"  Average P&L: ${avg_win_pnl:.2f}")
        print(f"  Average Duration: {avg_win_duration:.1f} hours")
        
        # Exit reasons for winners
        win_exit_reasons = [t.exit_reason.value for t in winning_trades]
        exit_counts = pd.Series(win_exit_reasons).value_counts()
        print(f"  Exit reasons: {exit_counts.to_dict()}")
        
        # Direction analysis
        long_wins = [t for t in winning_trades if t.direction.value == 'long']
        short_wins = [t for t in winning_trades if t.direction.value == 'short']
        print(f"  Long wins: {len(long_wins)}, Short wins: {len(short_wins)}")
    
    # Analyze losing trade characteristics
    if losing_trades:
        avg_loss_pnl = np.mean([t.pnl for t in losing_trades])
        avg_loss_duration = np.mean([(t.exit_time - t.entry_time).total_seconds()/3600 for t in losing_trades])
        
        print(f"\\nLOSING TRADE CHARACTERISTICS:")
        print(f"  Average P&L: ${avg_loss_pnl:.2f}")
        print(f"  Average Duration: {avg_loss_duration:.1f} hours")
        
        # Exit reasons for losers
        loss_exit_reasons = [t.exit_reason.value for t in losing_trades]
        exit_counts = pd.Series(loss_exit_reasons).value_counts()
        print(f"  Exit reasons: {exit_counts.to_dict()}")
    
    # Risk-reward analysis
    if winning_trades and losing_trades:
        risk_reward = abs(avg_win_pnl / avg_loss_pnl)
        print(f"\\nRISK-REWARD RATIO: {risk_reward:.2f}")
    
    # Market regime analysis
    print(f"\\nMARKET REGIME ANALYSIS:")
    regime_performance = {}
    
    for trade in trades:
        # Find market regime at entry time
        entry_idx = df.index.get_indexer([trade.entry_time], method='nearest')[0]
        if entry_idx < len(df):
            regime = df.iloc[entry_idx]['IC_Regime']
            if regime not in regime_performance:
                regime_performance[regime] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            regime_performance[regime]['trades'] += 1
            regime_performance[regime]['total_pnl'] += trade.pnl
            if trade.pnl > 0:
                regime_performance[regime]['wins'] += 1
    
    for regime, perf in regime_performance.items():
        win_rate = perf['wins'] / perf['trades'] * 100 if perf['trades'] > 0 else 0
        avg_pnl = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
        regime_name = {1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'}[regime]
        print(f"  {regime_name}: {perf['trades']} trades, {win_rate:.1f}% win rate, avg P&L: ${avg_pnl:.2f}")
    
    return {
        'total_trades': len(trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_win': avg_win_pnl if winning_trades else 0,
        'avg_loss': avg_loss_pnl if losing_trades else 0,
        'risk_reward': risk_reward if winning_trades and losing_trades else 0,
        'regime_performance': regime_performance,
        'sharpe_ratio': results.get('sharpe_ratio', 0)
    }

def identify_optimization_opportunities(analysis):
    """Identify specific areas for optimization based on trade analysis"""
    
    print(f"\\n" + "="*60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("="*60)
    
    opportunities = []
    
    # 1. Risk-reward optimization
    if analysis['risk_reward'] < 1.5:
        opportunities.append(f"IMPROVE RISK-REWARD: Current {analysis['risk_reward']:.2f}, target 1.5+")
        print(f"📈 Risk-reward ratio too low: {analysis['risk_reward']:.2f}")
        print("   - Consider tighter stop losses")
        print("   - Wider take profit targets")
        print("   - Better entry timing")
    
    # 2. Win rate analysis
    if analysis['win_rate'] < 0.6:
        opportunities.append(f"IMPROVE WIN RATE: Current {analysis['win_rate']*100:.1f}%, target 60%+")
        print(f"🎯 Win rate could be improved: {analysis['win_rate']*100:.1f}%")
        print("   - Better signal filtering")
        print("   - Market regime awareness")
        print("   - Entry confirmation")
    
    # 3. Trade frequency
    if analysis['total_trades'] < 50:
        opportunities.append("INCREASE TRADE FREQUENCY: For higher Sharpe ratios")
        print(f"⚡ Low trade frequency: {analysis['total_trades']} trades")
        print("   - Relax entry conditions slightly")
        print("   - Add more timeframes")
        print("   - Reduce filtering")
    
    # 4. Regime-specific performance
    if 'regime_performance' in analysis:
        for regime, perf in analysis['regime_performance'].items():
            win_rate = perf['wins'] / perf['trades'] * 100 if perf['trades'] > 0 else 0
            avg_pnl = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
            
            regime_name = {1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'}[regime]
            
            if win_rate < 50 or avg_pnl < 0:
                opportunities.append(f"AVOID {regime_name.upper()} markets")
                print(f"❌ Poor performance in {regime_name}: {win_rate:.1f}% win rate")
                print("   - Consider filtering out these conditions")
    
    # 5. Sharpe ratio assessment
    if analysis['sharpe_ratio'] < 2.0:
        gap = 2.0 - analysis['sharpe_ratio']
        opportunities.append(f"SHARPE IMPROVEMENT NEEDED: {gap:.2f} points to reach 2.0")
        print(f"🎯 Sharpe ratio gap: {analysis['sharpe_ratio']:.2f} (need {gap:.2f} more)")
    
    return opportunities

def main():
    """Run comprehensive trade pattern analysis"""
    
    print("Creating detailed test data with varied market conditions...")
    df = create_detailed_test_data(3000)
    
    print(f"Data created: {len(df)} periods")
    print(f"Signal distribution:")
    print(f"  NTI_Direction: {df['NTI_Direction'].value_counts().to_dict()}")
    print(f"  MB_Bias: {df['MB_Bias'].value_counts().to_dict()}")
    print(f"  IC_Regime: {df['IC_Regime'].value_counts().to_dict()}")
    
    # Test multiple configurations
    configs = {
        'Standard': OptimizedStrategyConfig(verbose=False),
        'Tight Risk': OptimizedStrategyConfig(
            risk_per_trade=0.015,
            sl_atr_multiplier=1.5,
            tp_atr_multipliers=(1.0, 2.0, 3.0),
            verbose=False
        ),
        'High Frequency': OptimizedStrategyConfig(
            relaxed_mode=True,
            risk_per_trade=0.015,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=3.0,
            verbose=False
        )
    }
    
    best_config = None
    best_sharpe = 0
    
    for config_name, config in configs.items():
        print(f"\\n{'='*60}")
        print(f"TESTING: {config_name.upper()}")
        print('='*60)
        
        strategy = OptimizedProdStrategy(config)
        analysis = analyze_trade_patterns(strategy, df)
        
        print(f"\\nSUMMARY:")
        print(f"  Sharpe Ratio: {analysis['sharpe_ratio']:.3f}")
        print(f"  Total Trades: {analysis['total_trades']}")
        print(f"  Win Rate: {analysis['win_rate']*100:.1f}%")
        print(f"  Risk-Reward: {analysis['risk_reward']:.2f}")
        
        if analysis['sharpe_ratio'] > best_sharpe:
            best_sharpe = analysis['sharpe_ratio']
            best_config = config_name
        
        # Identify improvements
        opportunities = identify_optimization_opportunities(analysis)
    
    print(f"\\n{'='*60}")
    print(f"BEST CONFIGURATION: {best_config} (Sharpe: {best_sharpe:.3f})")
    print('='*60)
    
    if best_sharpe >= 2.0:
        print("🎯 TARGET ACHIEVED! Sharpe ratio ≥ 2.0")
    else:
        gap = 2.0 - best_sharpe
        print(f"📈 Need {gap:.3f} more Sharpe points to reach target")
        print("Key focus areas:")
        print("  1. Increase trade frequency")
        print("  2. Improve risk-reward ratio")
        print("  3. Filter poor market conditions")
        print("  4. Optimize exit timing")

if __name__ == "__main__":
    main()