"""
Focused Strategy Optimization Based on Analysis
Target: Increase Sharpe > 2.0 frequency from 10% to 25%+

Key insights from high performers:
- Need +332 more trades (1209 vs 877)
- Better profit factor (2.94 vs 2.23)
- Better risk-reward (2.16 vs 1.40)
- All high performers were in 2011
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from dataclasses import dataclass


@dataclass
class HighPerformanceConfig(OptimizedStrategyConfig):
    """Configuration optimized for high performance periods"""
    
    # TRADE FREQUENCY BOOST (Target: +332 trades)
    relaxed_mode: bool = True  # Enable more entry opportunities
    
    # PROFIT FACTOR OPTIMIZATION (Target: 2.94+)
    tp_atr_multipliers: tuple = (1.0, 2.0, 3.2)  # Wider TPs for better RR
    sl_atr_multiplier: float = 1.8  # Tighter stops
    
    # ENHANCED EXIT LOGIC
    exit_on_signal_flip: bool = True
    signal_flip_min_profit_pips: float = 3.0  # Lower threshold for faster exits
    signal_flip_min_time_hours: float = 1.0   # Reduced time requirement
    signal_flip_partial_exit_percent: float = 0.4  # Take 40% profits on flip
    
    # RISK MANAGEMENT (maintain low drawdowns)
    risk_per_trade: float = 0.01  # Slightly lower risk
    
    # INTELLIGENT SIZING (boost for high confidence)
    intelligent_sizing: bool = True
    size_multipliers: tuple = (0.8, 1.0, 1.5, 2.0)  # Conservative to aggressive
    confidence_thresholds: tuple = (25.0, 45.0, 65.0)


def test_high_performance_optimization():
    """Test configurations targeting high performance characteristics"""
    
    print("HIGH PERFORMANCE OPTIMIZATION TEST")
    print("="*60)
    print("Target: Achieve characteristics of Sharpe > 2.0 periods")
    print("- Trade frequency: 1209+ (vs 877 baseline)")
    print("- Profit factor: 2.94+ (vs 2.23 baseline)")
    print("- Risk-reward: 2.16+ (vs 1.40 baseline)")
    print()
    
    # Configuration variations
    configs = {
        'Baseline': OptimizedStrategyConfig(
            relaxed_mode=False,
            risk_per_trade=0.01,
            verbose=False
        ),
        
        'Frequency_Boost': OptimizedStrategyConfig(
            relaxed_mode=True,
            risk_per_trade=0.01,
            verbose=False
        ),
        
        'Better_Exits': OptimizedStrategyConfig(
            relaxed_mode=True,
            tp_atr_multipliers=(1.0, 2.0, 3.2),
            sl_atr_multiplier=1.8,
            signal_flip_min_profit_pips=3.0,
            signal_flip_min_time_hours=1.0,
            risk_per_trade=0.01,
            verbose=False
        ),
        
        'High_Performance': HighPerformanceConfig(verbose=False)
    }
    
    # Run tests
    results = {}
    
    print("Testing on realistic market data...")
    print()
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        try:
            # Test with multiple periods to get average performance
            period_results = []
            
            for seed in [42, 123, 456, 789, 999]:
                # Generate realistic test data
                np.random.seed(seed)
                dates = pd.date_range(start='2011-09-01', periods=3000, freq='h')
                
                # Create trending market similar to 2011 high performers
                trend_returns = np.random.normal(0.0001, 0.0008, 3000)  # Slight upward bias
                prices = np.cumprod(1 + trend_returns) * 0.75
                
                df = pd.DataFrame({
                    'Open': prices + np.random.normal(0, 0.00005, 3000),
                    'High': prices + abs(np.random.normal(0, 0.0001, 3000)),
                    'Low': prices - abs(np.random.normal(0, 0.0001, 3000)),
                    'Close': prices
                }, index=dates)
                
                # Fix OHLC
                for i in range(len(df)):
                    df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
                    df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])
                
                # Add indicators
                df['EMA_10'] = df['Close'].ewm(span=10).mean()
                df['EMA_20'] = df['Close'].ewm(span=20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                
                # Strong directional signals (like 2011)
                df['Price_Mom'] = df['Close'].pct_change(10)
                df['EMA_Cross'] = (df['EMA_10'] > df['EMA_20']).astype(int) * 2 - 1
                
                df['NTI_Direction'] = 0
                strong_up = (df['Price_Mom'] > 0.001) & (df['EMA_Cross'] == 1)
                strong_down = (df['Price_Mom'] < -0.001) & (df['EMA_Cross'] == -1)
                df.loc[strong_up, 'NTI_Direction'] = 1
                df.loc[strong_down, 'NTI_Direction'] = -1
                
                # Momentum confirmation
                df['ROC_5'] = df['Close'].pct_change(5)
                df['ROC_15'] = df['Close'].pct_change(15)
                df['MB_Bias'] = 0
                
                momentum_up = (df['ROC_5'] > 0.0005) & (df['ROC_15'] > 0.0002)
                momentum_down = (df['ROC_5'] < -0.0005) & (df['ROC_15'] < -0.0002)
                df.loc[momentum_up, 'MB_Bias'] = 1
                df.loc[momentum_down, 'MB_Bias'] = -1
                
                # Favorable regime (trending like 2011)
                df['Volatility'] = df['Close'].rolling(20).std()
                df['Trend_Strength'] = abs(df['Close'].pct_change(20))
                
                vol_median = df['Volatility'].median()
                trend_median = df['Trend_Strength'].median()
                
                df['IC_Regime'] = 1  # Default to strong trend
                # Weaker periods
                df.loc[(df['Volatility'] > vol_median * 1.2) | (df['Trend_Strength'] < trend_median * 0.8), 'IC_Regime'] = 2
                
                # ATR
                df['TR'] = np.maximum(
                    df['High'] - df['Low'],
                    np.maximum(
                        abs(df['High'] - df['Close'].shift(1)),
                        abs(df['Low'] - df['Close'].shift(1))
                    )
                )
                df['ATR'] = df['TR'].rolling(14).mean()
                df['IC_ATR_Normalized'] = np.clip((df['ATR'] / df['Close'] * 10000), 15, 80)
                
                # Required fields
                df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend'})
                df = df.fillna(method='bfill').fillna(method='ffill')
                
                # Run backtest
                strategy = OptimizedProdStrategy(config)
                result = strategy.run_backtest(df)
                period_results.append(result)
            
            # Average results across periods
            avg_result = {}
            for key in ['sharpe_ratio', 'total_trades', 'win_rate', 'profit_factor', 'total_return']:
                values = [r[key] for r in period_results if key in r]
                avg_result[key] = np.mean(values) if values else 0
            
            results[config_name] = avg_result
            
            # Display results
            print(f"  Sharpe: {avg_result['sharpe_ratio']:.3f}")
            print(f"  Trades: {avg_result['total_trades']:.0f}")
            print(f"  Win Rate: {avg_result['win_rate']:.1f}%")
            print(f"  Profit Factor: {avg_result['profit_factor']:.3f}")
            print(f"  Return: {avg_result['total_return']:.1f}%")
            
            # Check targets
            if avg_result['sharpe_ratio'] > 2.0:
                print(f"  🎯 SHARPE TARGET HIT!")
            if avg_result['total_trades'] > 1100:
                print(f"  📈 TRADE FREQUENCY TARGET APPROACHING!")
            if avg_result['profit_factor'] > 2.8:
                print(f"  💰 PROFIT FACTOR TARGET APPROACHING!")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Analysis
    print("PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    if 'Baseline' in results:
        baseline = results['Baseline']
        
        for config_name, result in results.items():
            if config_name == 'Baseline':
                continue
                
            sharpe_diff = result['sharpe_ratio'] - baseline['sharpe_ratio']
            trade_diff = result['total_trades'] - baseline['total_trades']
            pf_diff = result['profit_factor'] - baseline['profit_factor']
            
            print(f"{config_name}:")
            print(f"  Sharpe: {sharpe_diff:+.3f} vs baseline")
            print(f"  Trades: {trade_diff:+.0f} vs baseline")
            print(f"  Profit Factor: {pf_diff:+.3f} vs baseline")
            
            # Score the configuration
            score = 0
            if sharpe_diff > 0.3: score += 3
            elif sharpe_diff > 0.1: score += 1
            
            if trade_diff > 200: score += 2
            elif trade_diff > 100: score += 1
            
            if pf_diff > 0.5: score += 2
            elif pf_diff > 0.2: score += 1
            
            if score >= 5:
                print(f"  ⭐ EXCELLENT improvement (score: {score})")
            elif score >= 3:
                print(f"  ✅ Good improvement (score: {score})")
            elif score >= 1:
                print(f"  📈 Some improvement (score: {score})")
            else:
                print(f"  ❌ Limited improvement (score: {score})")
            
            print()
    
    return results


if __name__ == "__main__":
    results = test_high_performance_optimization()