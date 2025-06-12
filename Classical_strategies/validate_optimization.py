"""
Validate the optimized strategy using the same Monte Carlo framework as run_Strategy.py
Compare against original configs to measure improvement
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy
from dataclasses import dataclass
from focused_optimization import HighPerformanceConfig
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load AUDUSD data similar to run_Strategy.py"""
    
    # Try to load real data, fallback to synthetic
    try:
        # This mimics the data loading from run_Strategy.py
        data_file = 'data/FX/AUDUSD_15MIN.csv'
        df = pd.read_csv(data_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        print(f"Loaded real AUDUSD data: {len(df)} rows")
    except:
        print("Creating synthetic AUDUSD-like data...")
        
        # Create comprehensive synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2010-01-01', end='2025-01-01', freq='15min')
        
        # Generate realistic forex price movements
        returns = np.random.normal(0, 0.0003, len(dates))
        
        # Add some trending periods (like 2011)
        trend_periods = [
            (5000, 8000),   # Strong trend period 1
            (15000, 18000), # Strong trend period 2
            (25000, 28000)  # Strong trend period 3
        ]
        
        for start, end in trend_periods:
            if end < len(returns):
                returns[start:end] += np.random.normal(0.00005, 0.0002, end-start)
        
        prices = np.cumprod(1 + returns) * 0.85  # Start around realistic level
        
        df = pd.DataFrame({
            'Open': prices + np.random.normal(0, 0.00001, len(prices)),
            'High': prices + abs(np.random.normal(0, 0.00003, len(prices))),
            'Low': prices - abs(np.random.normal(0, 0.00003, len(prices))),
            'Close': prices
        }, index=dates)
        
        # Fix OHLC consistency
        for i in range(0, len(df), 1000):  # Sample every 1000 rows for efficiency
            end_idx = min(i + 1000, len(df))
            for j in range(i, end_idx):
                df.iloc[j, df.columns.get_loc('High')] = max(df.iloc[j][['Open', 'High', 'Low', 'Close']])
                df.iloc[j, df.columns.get_loc('Low')] = min(df.iloc[j][['Open', 'High', 'Low', 'Close']])
    
    # Add comprehensive indicators
    print("Calculating indicators...")
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Price momentum and trends
    df['ROC_5'] = df['Close'].pct_change(5)
    df['ROC_10'] = df['Close'].pct_change(10)
    df['ROC_20'] = df['Close'].pct_change(20)
    
    # NeuroTrend Direction (NTI_Direction)
    df['NTI_Direction'] = 0
    
    # Strong trend conditions
    strong_up = (df['SMA_10'] > df['SMA_20']) & (df['ROC_10'] > 0.0008) & (df['EMA_12'] > df['EMA_26'])
    strong_down = (df['SMA_10'] < df['SMA_20']) & (df['ROC_10'] < -0.0008) & (df['EMA_12'] < df['EMA_26'])
    
    # Weaker trend conditions
    weak_up = (df['SMA_10'] > df['SMA_20']) & (df['ROC_10'] > 0.0003)
    weak_down = (df['SMA_10'] < df['SMA_20']) & (df['ROC_10'] < -0.0003)
    
    df.loc[strong_up, 'NTI_Direction'] = 1
    df.loc[strong_down, 'NTI_Direction'] = -1
    df.loc[weak_up & (df['NTI_Direction'] == 0), 'NTI_Direction'] = 1
    df.loc[weak_down & (df['NTI_Direction'] == 0), 'NTI_Direction'] = -1
    
    # Momentum Bias (MB_Bias)
    df['MB_Bias'] = 0
    momentum_up = (df['ROC_5'] > 0.001) & (df['ROC_20'] > 0.0005)
    momentum_down = (df['ROC_5'] < -0.001) & (df['ROC_20'] < -0.0005)
    df.loc[momentum_up, 'MB_Bias'] = 1
    df.loc[momentum_down, 'MB_Bias'] = -1
    
    # Intelligent Chop Regime (IC_Regime)
    df['Volatility'] = df['Close'].rolling(20).std()
    df['Trend_Strength'] = abs(df['Close'].pct_change(20))
    
    vol_med = df['Volatility'].median()
    trend_med = df['Trend_Strength'].median()
    
    df['IC_Regime'] = 1  # Default strong trend
    
    # Weak trend
    weak_trend = (df['Volatility'] > vol_med * 0.8) & (df['Trend_Strength'] < trend_med * 1.2)
    df.loc[weak_trend, 'IC_Regime'] = 2
    
    # Range/chop
    choppy = (df['Volatility'] > vol_med * 1.2) | (df['Trend_Strength'] < trend_med * 0.6)
    df.loc[choppy, 'IC_Regime'] = 3
    
    # ATR calculation
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    df['IC_ATR_Normalized'] = np.clip((df['ATR'] / df['Close'] * 10000), 8, 120)
    
    # Required fields
    df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'})
    
    # Clean data
    df = df.fillna(method='bfill').fillna(method='ffill')
    df = df.dropna()
    
    print(f"Data prepared: {len(df)} rows")
    return df


def run_monte_carlo_validation(config, config_name, df, iterations=25):
    """Run Monte Carlo validation similar to run_Strategy.py"""
    
    print(f"\\nTesting {config_name} ({iterations} iterations)...")
    
    sample_size = 8000  # Same as run_Strategy.py
    results = []
    
    for i in range(iterations):
        # Random sample
        if len(df) > sample_size:
            start_idx = np.random.randint(0, len(df) - sample_size)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
        else:
            sample_df = df.copy()
        
        # Run backtest
        try:
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(sample_df)
            
            if result['total_trades'] > 0:  # Only count periods with trades
                results.append(result)
        
        except Exception as e:
            print(f"    Error in iteration {i+1}: {e}")
            continue
        
        # Progress update
        if (i + 1) % 10 == 0:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_trades = np.mean([r['total_trades'] for r in results])
            print(f"    [{i+1:2d}/{iterations}] Avg Sharpe: {avg_sharpe:.3f} | Avg Trades: {avg_trades:.0f}")
    
    return results


def analyze_results(results, config_name):
    """Analyze Monte Carlo results"""
    
    if not results:
        print(f"No valid results for {config_name}")
        return {}
    
    metrics = {}
    
    # Calculate statistics
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    total_trades = [r['total_trades'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    profit_factors = [r['profit_factor'] for r in results]
    returns = [r['total_return'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    
    metrics = {
        'sharpe_mean': np.mean(sharpe_ratios),
        'sharpe_std': np.std(sharpe_ratios),
        'trades_mean': np.mean(total_trades),
        'trades_std': np.std(total_trades),
        'win_rate_mean': np.mean(win_rates),
        'profit_factor_mean': np.mean(profit_factors),
        'return_mean': np.mean(returns),
        'drawdown_mean': np.mean(drawdowns),
        'sharpe_above_2': sum(1 for s in sharpe_ratios if s > 2.0),
        'sharpe_above_1_5': sum(1 for s in sharpe_ratios if s > 1.5),
        'profitable_periods': sum(1 for r in returns if r > 0),
        'total_periods': len(results)
    }
    
    # Display results
    print(f"\\n{config_name} Results:")
    print("-" * 50)
    print(f"Sharpe Ratio:     {metrics['sharpe_mean']:.3f} ± {metrics['sharpe_std']:.3f}")
    print(f"Total Trades:     {metrics['trades_mean']:.0f} ± {metrics['trades_std']:.0f}")
    print(f"Win Rate:         {metrics['win_rate_mean']:.1f}%")
    print(f"Profit Factor:    {metrics['profit_factor_mean']:.3f}")
    print(f"Total Return:     {metrics['return_mean']:.1f}%")
    print(f"Max Drawdown:     {metrics['drawdown_mean']:.1f}%")
    print()
    print(f"Performance Distribution:")
    print(f"  Sharpe > 2.0:   {metrics['sharpe_above_2']}/{metrics['total_periods']} ({metrics['sharpe_above_2']/metrics['total_periods']*100:.1f}%)")
    print(f"  Sharpe > 1.5:   {metrics['sharpe_above_1_5']}/{metrics['total_periods']} ({metrics['sharpe_above_1_5']/metrics['total_periods']*100:.1f}%)")
    print(f"  Profitable:     {metrics['profitable_periods']}/{metrics['total_periods']} ({metrics['profitable_periods']/metrics['total_periods']*100:.1f}%)")
    
    return metrics


def main():
    """Main validation function"""
    
    print("OPTIMIZED STRATEGY VALIDATION")
    print("="*60)
    print("Comparing optimized config against original using Monte Carlo")
    
    # Load data
    df = load_and_prepare_data()
    
    # Define configurations
    from strategy_code.Prod_strategy import OptimizedStrategyConfig
    
    configs = {
        'Original_Scalping': OptimizedStrategyConfig(
            relaxed_mode=False,
            risk_per_trade=0.01,
            tp_atr_multipliers=(0.8, 1.5, 2.5),
            sl_atr_multiplier=2.0,
            verbose=False
        ),
        
        'Optimized_High_Performance': HighPerformanceConfig(verbose=False)
    }
    
    # Run validation
    all_results = {}
    all_metrics = {}
    
    for config_name, config in configs.items():
        results = run_monte_carlo_validation(config, config_name, df, iterations=25)
        metrics = analyze_results(results, config_name)
        all_results[config_name] = results
        all_metrics[config_name] = metrics
    
    # Final comparison
    print("\\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    if len(all_metrics) >= 2:
        original = all_metrics['Original_Scalping']
        optimized = all_metrics['Optimized_High_Performance']
        
        improvements = {
            'sharpe': optimized['sharpe_mean'] - original['sharpe_mean'],
            'trades': optimized['trades_mean'] - original['trades_mean'],
            'profit_factor': optimized['profit_factor_mean'] - original['profit_factor_mean'],
            'sharpe_2plus': optimized['sharpe_above_2'] - original['sharpe_above_2'],
            'win_rate': optimized['win_rate_mean'] - original['win_rate_mean']
        }
        
        print(f"Sharpe Ratio improvement:  {improvements['sharpe']:+.3f}")
        print(f"Trade frequency increase:  {improvements['trades']:+.0f}")
        print(f"Profit Factor improvement: {improvements['profit_factor']:+.3f}")
        print(f"Win Rate improvement:      {improvements['win_rate']:+.1f}%")
        print(f"Periods with Sharpe > 2.0: {improvements['sharpe_2plus']:+d}")
        
        # Calculate success score
        success_score = 0
        if improvements['sharpe'] > 0.1: success_score += 1
        if improvements['trades'] > 50: success_score += 1
        if improvements['profit_factor'] > 0.2: success_score += 1
        if improvements['sharpe_2plus'] > 0: success_score += 2
        
        print(f"\\nSUCCESS EVALUATION:")
        if success_score >= 4:
            print("🎯 EXCELLENT - Optimization clearly successful!")
        elif success_score >= 3:
            print("✅ GOOD - Significant improvements achieved")
        elif success_score >= 2:
            print("📈 MODERATE - Some improvements achieved")
        else:
            print("❌ LIMITED - Minimal improvement")
        
        # Target analysis
        high_perf_target_trades = 1209
        high_perf_target_pf = 2.94
        high_perf_target_sharpe_freq = 0.25  # 25% of periods should achieve Sharpe > 2.0
        
        current_sharpe_freq = optimized['sharpe_above_2'] / optimized['total_periods']
        
        print(f"\\nHIGH PERFORMANCE TARGETS:")
        print(f"Trade frequency: {optimized['trades_mean']:.0f} / {high_perf_target_trades} target ({optimized['trades_mean']/high_perf_target_trades*100:.1f}%)")
        print(f"Profit factor: {optimized['profit_factor_mean']:.3f} / {high_perf_target_pf:.3f} target ({optimized['profit_factor_mean']/high_perf_target_pf*100:.1f}%)")
        print(f"Sharpe > 2.0 frequency: {current_sharpe_freq:.1%} / {high_perf_target_sharpe_freq:.1%} target")


if __name__ == "__main__":
    main()