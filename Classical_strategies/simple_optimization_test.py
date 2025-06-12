"""
Simple Test: Small tweaks to existing strategy parameters
Focus on the key finding: +124 more trades for Sharpe > 2.0
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig

def test_parameter_tweaks():
    """Test simple parameter adjustments to increase trade frequency"""
    
    print("SIMPLE PARAMETER OPTIMIZATION TEST")
    print("="*50)
    print("Goal: Increase trade frequency by ~124 trades")
    print("Based on: High performers had 1024 vs 900 avg trades")
    print()
    
    # Test configurations with small tweaks
    configs = {
        'Baseline': OptimizedStrategyConfig(
            relaxed_mode=False,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            verbose=False
        ),
        
        'More_Trades_v1': OptimizedStrategyConfig(
            relaxed_mode=True,  # Enable relaxed mode for more trades
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            verbose=False
        ),
        
        'More_Trades_v2': OptimizedStrategyConfig(
            relaxed_mode=True,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=3.0,  # Lower threshold for faster exits
            signal_flip_min_time_hours=1.0,   # Reduced from 2 hours
            verbose=False
        ),
        
        'Better_RR': OptimizedStrategyConfig(
            relaxed_mode=True,
            tp_atr_multipliers=(1.2, 2.4, 3.6),  # Wider TPs
            sl_atr_multiplier=1.6,                # Tighter stops  
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=3.0,
            verbose=False
        ),
        
        'Optimized': OptimizedStrategyConfig(
            relaxed_mode=True,
            tp_atr_multipliers=(1.2, 2.4, 3.6),
            sl_atr_multiplier=1.6,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=3.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=0.6,  # More aggressive partial exits
            risk_per_trade=0.015,  # Slightly lower risk
            verbose=False
        )
    }
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=2500, freq='1H')
    
    # Generate more realistic trending/ranging periods
    regime_length = len(dates) // 4
    
    # Create 4 different market regimes
    trend_up = np.cumsum(np.random.normal(0.0002, 0.001, regime_length))
    range_period = np.cumsum(np.random.normal(0, 0.002, regime_length))  
    trend_down = np.cumsum(np.random.normal(-0.0002, 0.001, regime_length))
    volatile_period = np.cumsum(np.random.normal(0, 0.003, len(dates) - 3*regime_length))
    
    returns = np.concatenate([trend_up, range_period, trend_down, volatile_period])
    prices = np.cumprod(1 + returns) * 0.75
    
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 0.0001, len(prices)),
        'High': prices + abs(np.random.normal(0, 0.0002, len(prices))),
        'Low': prices - abs(np.random.normal(0, 0.0002, len(prices))),
        'Close': prices
    }, index=dates)
    
    # Fix OHLC consistency
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
        df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])
    
    # Add realistic indicators
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Create stronger, more realistic signals
    df['Price_vs_EMA'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['EMA_Slope'] = df['EMA_20'].pct_change(5)
    
    # NTI Direction - based on multiple timeframe alignment
    df['NTI_Direction'] = 0
    long_condition = (df['Price_vs_EMA'] > 0.001) & (df['EMA_Slope'] > 0.0005) & (df['EMA_10'] > df['EMA_20'])
    short_condition = (df['Price_vs_EMA'] < -0.001) & (df['EMA_Slope'] < -0.0005) & (df['EMA_10'] < df['EMA_20'])
    
    df.loc[long_condition, 'NTI_Direction'] = 1
    df.loc[short_condition, 'NTI_Direction'] = -1
    
    # MB Bias - momentum confirmation  
    df['ROC_10'] = df['Close'].pct_change(10)
    df['ROC_20'] = df['Close'].pct_change(20)
    df['MB_Bias'] = 0
    
    strong_momentum_up = (df['ROC_10'] > 0.003) & (df['ROC_20'] > 0.001)
    strong_momentum_down = (df['ROC_10'] < -0.003) & (df['ROC_20'] < -0.001)
    
    df.loc[strong_momentum_up, 'MB_Bias'] = 1
    df.loc[strong_momentum_down, 'MB_Bias'] = -1
    
    # IC Regime - based on volatility and trend strength
    df['Volatility'] = df['Close'].rolling(20).std()
    df['Trend_Strength'] = abs(df['EMA_20'].pct_change(20))
    
    vol_median = df['Volatility'].median()
    trend_median = df['Trend_Strength'].median()
    
    df['IC_Regime'] = 2  # Default weak trend
    # Strong trend = low vol + strong trend  
    df.loc[(df['Volatility'] < vol_median * 0.7) & (df['Trend_Strength'] > trend_median), 'IC_Regime'] = 1
    # Range = high vol + weak trend
    df.loc[(df['Volatility'] > vol_median * 1.3) & (df['Trend_Strength'] < trend_median * 0.5), 'IC_Regime'] = 3
    
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
    
    # Other required fields
    df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'})
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Test each configuration
    results = {}
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        try:
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(df)
            results[config_name] = result
            
            trades = result.get('total_trades', 0)
            sharpe = result.get('sharpe_ratio', 0)
            win_rate = result.get('win_rate', 0)
            profit_factor = result.get('profit_factor', 0)
            returns = result.get('total_return', 0)
            
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  Trades: {trades}")
            print(f"  Win Rate: {win_rate:.1f}%") 
            print(f"  Profit Factor: {profit_factor:.3f}")
            print(f"  Return: {returns:.1f}%")
            
            if sharpe > 2.0:
                print(f"  🎯 TARGET ACHIEVED!")
            elif sharpe > 1.5:
                print(f"  📈 Strong performance")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Analysis
    print("ANALYSIS:")
    print("-" * 20)
    
    if 'Baseline' in results:
        baseline_sharpe = results['Baseline']['sharpe_ratio']
        baseline_trades = results['Baseline']['total_trades']
        
        best_config = None
        best_sharpe = 0
        best_improvement = 0
        
        for config_name, result in results.items():
            if config_name == 'Baseline':
                continue
                
            sharpe = result['sharpe_ratio']
            trades = result['total_trades']
            trade_increase = trades - baseline_trades
            sharpe_improvement = sharpe - baseline_sharpe
            
            print(f"{config_name}:")
            print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
            print(f"  Trade increase: {trade_increase:+d}")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = config_name
                best_improvement = sharpe_improvement
            
            print()
        
        print(f"BEST CONFIGURATION: {best_config}")
        print(f"Best Sharpe: {best_sharpe:.3f} (+{best_improvement:.3f})")
        
        if best_sharpe > 2.0:
            print("🎯 SUCCESS! Target achieved with simple optimizations")
        elif best_improvement > 0.1:
            print("📈 Significant improvement - continue refinement")
        else:
            print("⚠️ Limited improvement - may need different approach")
    
    return results

if __name__ == "__main__":
    results = test_parameter_tweaks()