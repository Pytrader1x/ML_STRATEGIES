"""
Quick Strategy Test - Simplified version to validate enhancements
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from enhanced_strategy import EnhancedStrategy, EnhancedStrategyConfig

def create_test_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create test data with proper signal structure"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2010-01-01', periods=n_periods, freq='1H')
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.001, n_periods)
    prices = np.cumprod(1 + returns) * 0.75  # Start around 0.75
    
    # Generate OHLC
    noise = np.random.normal(0, 0.0001, (n_periods, 4))
    df = pd.DataFrame({
        'Open': prices + noise[:, 0],
        'High': prices + abs(noise[:, 1]),
        'Low': prices - abs(noise[:, 2]),
        'Close': prices + noise[:, 3]
    }, index=dates)
    
    # Fix OHLC consistency
    for i in range(n_periods):
        df.loc[df.index[i], 'High'] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
        df.loc[df.index[i], 'Low'] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])
    
    # Add indicators with stronger signals
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Create clearer directional signals
    df['NTI_Direction'] = 0
    df['NTI_Direction'] = np.where(df['SMA_20'] > df['SMA_50'] * 1.001, 1, df['NTI_Direction'])
    df['NTI_Direction'] = np.where(df['SMA_20'] < df['SMA_50'] * 0.999, -1, df['NTI_Direction'])
    
    # Add NTI_Strength
    df['NTI_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['Close'] * 100
    df['NTI_Strength'] = np.clip(df['NTI_Strength'], 0.1, 1.0)
    
    # MB_Bias based on momentum
    df['ROC_10'] = df['Close'].pct_change(10)
    df['MB_Bias'] = 0
    df['MB_Bias'] = np.where(df['ROC_10'] > 0.002, 1, df['MB_Bias'])
    df['MB_Bias'] = np.where(df['ROC_10'] < -0.002, -1, df['MB_Bias'])
    
    # IC_Regime - simpler classification
    df['Volatility'] = df['Close'].rolling(20).std()
    vol_mean = df['Volatility'].mean()
    df['IC_Regime'] = 2  # Default weak trend
    df.loc[df['Volatility'] < vol_mean * 0.7, 'IC_Regime'] = 1  # Strong trend
    df.loc[df['Volatility'] > vol_mean * 1.3, 'IC_Regime'] = 3  # Range
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    df['IC_ATR_Normalized'] = (df['ATR'] / df['Close'] * 10000)  # In pips
    
    # Regime names
    regime_map = {1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'}
    df['IC_RegimeName'] = df['IC_Regime'].map(regime_map)
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def test_strategies():
    """Test the strategies with clear comparison"""
    
    print("Creating test data...")
    df = create_test_data(2000)
    
    print(f"Data created: {len(df)} periods from {df.index[0]} to {df.index[-1]}")
    
    # Check signal distribution
    print(f"NTI_Direction distribution: {df['NTI_Direction'].value_counts().to_dict()}")
    print(f"MB_Bias distribution: {df['MB_Bias'].value_counts().to_dict()}")
    print(f"IC_Regime distribution: {df['IC_Regime'].value_counts().to_dict()}")
    
    # Test original strategy
    print("\\nTesting original strategy...")
    original_config = OptimizedStrategyConfig(verbose=False)
    original_strategy = OptimizedProdStrategy(original_config)
    original_results = original_strategy.run_backtest(df)
    
    # Test enhanced strategy with relaxed thresholds
    print("Testing enhanced strategy...")
    enhanced_config = EnhancedStrategyConfig(
        verbose=False,
        signal_quality_threshold=30.0,  # Much lower threshold
        mtf_confluence_threshold=0.5,   # Lower confluence requirement
        use_mtf_confluence=False,       # Disable for now
        dynamic_position_sizing=True,
        risk_per_trade=0.02
    )
    enhanced_strategy = EnhancedStrategy(enhanced_config)
    enhanced_results = enhanced_strategy.run_enhanced_backtest(df)
    
    # Compare results
    print("\\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    strategies = {
        'Original Strategy': original_results,
        'Enhanced Strategy': enhanced_results
    }
    
    for name, results in strategies.items():
        print(f"\\n{name}:")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        print(f"  Total Return: {results.get('total_return', 0):.2f}%")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.3f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        
        # Enhanced metrics if available
        if 'avg_signal_quality' in results:
            print(f"  Avg Signal Quality: {results['avg_signal_quality']:.1f}")
        if 'actual_risk_reward_ratio' in results:
            print(f"  Risk-Reward Ratio: {results['actual_risk_reward_ratio']:.2f}")
    
    # Improvement analysis
    if enhanced_results.get('sharpe_ratio', 0) > original_results.get('sharpe_ratio', 0):
        improvement = enhanced_results['sharpe_ratio'] - original_results['sharpe_ratio']
        print(f"\\n✅ Enhanced strategy improved Sharpe by {improvement:.3f}")
        
        if enhanced_results['sharpe_ratio'] > 2.0:
            print("🎯 TARGET ACHIEVED: Sharpe > 2.0!")
        elif enhanced_results['sharpe_ratio'] > 1.5:
            print("📈 Good progress toward Sharpe > 2.0 target")
        else:
            print("⚠️ Still needs work to reach Sharpe > 2.0")
    else:
        print("\\n⚠️ Enhanced strategy did not improve performance")
    
    return original_results, enhanced_results

if __name__ == "__main__":
    original, enhanced = test_strategies()