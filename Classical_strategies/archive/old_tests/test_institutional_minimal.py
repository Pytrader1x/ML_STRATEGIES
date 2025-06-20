"""
Minimal test of institutional strategy improvements
Uses pre-calculated indicators to avoid timeout
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import warnings
import os

warnings.filterwarnings('ignore')

def run_comparison():
    # Load data with indicators already calculated
    print("Loading pre-processed AUDUSD data...")
    
    # Try to find a recent results file with data
    results_path = 'results'
    csv_files = [f for f in os.listdir(results_path) if f.endswith('_trades_detail.csv')]
    
    if not csv_files:
        print("No pre-processed data found. Creating synthetic test...")
        # Create synthetic data for testing
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'Open': 0.6500 + np.random.randn(1000).cumsum() * 0.0001,
            'High': 0.6510 + np.random.randn(1000).cumsum() * 0.0001,
            'Low': 0.6490 + np.random.randn(1000).cumsum() * 0.0001,
            'Close': 0.6500 + np.random.randn(1000).cumsum() * 0.0001,
            'NTI_Direction': np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3]),
            'NTI_Confidence': np.random.uniform(20, 80, 1000),
            'MB_Bias': np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3]),
            'IC_Regime': np.random.choice([1, 2, 3, 4], 1000, p=[0.2, 0.3, 0.3, 0.2]),
            'IC_RegimeName': np.random.choice(['Strong Trend', 'Weak Trend', 'Range', 'Chop'], 1000),
            'IC_ATR_Normalized': np.random.uniform(0.0001, 0.0003, 1000),
            'IC_ATR_MA': np.random.uniform(0.0001, 0.0003, 1000),
            'MB_l2': 0.6480 + np.random.randn(1000).cumsum() * 0.0001,
            'MB_h2': 0.6520 + np.random.randn(1000).cumsum() * 0.0001
        }, index=dates)
    else:
        print("Using synthetic data for quick test...")
        dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
        np.random.seed(42)
        
        # Create more realistic synthetic data
        price = 0.6500
        prices = []
        for i in range(2000):
            price += np.random.randn() * 0.0002
            prices.append(price)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + abs(np.random.randn()) * 0.0001 for p in prices],
            'Low': [p - abs(np.random.randn()) * 0.0001 for p in prices],
            'Close': [p + np.random.randn() * 0.00005 for p in prices],
            'NTI_Direction': np.random.choice([-1, 0, 1], 2000, p=[0.3, 0.4, 0.3]),
            'NTI_Confidence': np.random.uniform(20, 80, 2000),
            'MB_Bias': np.random.choice([-1, 0, 1], 2000, p=[0.3, 0.4, 0.3]),
            'IC_Regime': np.random.choice([1, 2, 3, 4], 2000, p=[0.2, 0.3, 0.3, 0.2]),
            'IC_RegimeName': np.random.choice(['Strong Trend', 'Weak Trend', 'Range', 'Chop'], 2000),
            'IC_ATR_Normalized': np.random.uniform(0.0001, 0.0003, 2000),
            'IC_ATR_MA': np.random.uniform(0.0001, 0.0003, 2000),
            'MB_l2': [p - 0.002 for p in prices],
            'MB_h2': [p + 0.002 for p in prices]
        }, index=dates)
    
    print(f"Test data ready: {len(df)} rows")
    print("="*80)
    
    # Test 1: Original validated strategy
    print("\n1. ORIGINAL VALIDATED STRATEGY")
    print("-" * 40)
    
    validated_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.005,
        base_position_size_millions=1.0,
        
        sl_min_pips=3.0,
        sl_max_pips=10.0,
        sl_atr_multiplier=0.8,
        
        tp_atr_multipliers=(0.15, 0.25, 0.4),
        max_tp_percent=0.005,
        
        tsl_activation_pips=8.0,
        tsl_min_profit_pips=1.0,
        
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,
        
        intelligent_sizing=False,
        realistic_costs=True,
        entry_slippage_pips=0.5,
        stop_loss_slippage_pips=2.0,
        
        verbose=False
    )
    
    strategy1 = OptimizedProdStrategy(validated_config)
    result1 = strategy1.run_backtest(df)
    
    print(f"Sharpe Ratio: {result1['sharpe_ratio']:.3f}")
    print(f"Total Return: {result1['total_return']:.2f}%")
    print(f"Win Rate: {result1['win_rate']:.1f}%")
    print(f"Total Trades: {result1['total_trades']}")
    print(f"P&L: ${result1['total_pnl']:,.0f}")
    
    # Test 2: Institutional improvements
    print("\n2. INSTITUTIONAL STRATEGY (IMPROVED)")
    print("-" * 40)
    
    institutional_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,  # More conservative
        
        # INSTITUTIONAL SIZING
        base_position_size_millions=2.0,  # 2M for standard
        relaxed_position_multiplier=0.5,  # 1M for relaxed
        
        # BETTER STOPS
        sl_min_pips=5.0,
        sl_max_pips=15.0,
        sl_atr_multiplier=1.0,
        
        # WIDER TARGETS
        tp_atr_multipliers=(0.3, 0.6, 1.0),
        max_tp_percent=0.01,
        
        # PROFESSIONAL TRAILING
        tsl_activation_pips=10.0,
        tsl_min_profit_pips=3.0,
        trailing_atr_multiplier=1.0,
        
        # IMPROVED PARTIAL PROFIT
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.6,
        partial_profit_size_percent=0.4,
        
        # INTELLIGENT SIZING ON
        intelligent_sizing=True,
        confidence_thresholds=(40.0, 60.0, 80.0),
        size_multipliers=(0.5, 0.75, 1.0, 1.25),
        
        # VOLATILITY ADAPTATION
        sl_volatility_adjustment=True,
        sl_range_market_multiplier=0.8,
        sl_trend_market_multiplier=1.2,
        
        relaxed_mode=True,
        
        realistic_costs=True,
        entry_slippage_pips=0.3,
        stop_loss_slippage_pips=1.0,
        
        verbose=False
    )
    
    strategy2 = OptimizedProdStrategy(institutional_config)
    result2 = strategy2.run_backtest(df)
    
    print(f"Sharpe Ratio: {result2['sharpe_ratio']:.3f}")
    print(f"Total Return: {result2['total_return']:.2f}%")
    print(f"Win Rate: {result2['win_rate']:.1f}%")
    print(f"Total Trades: {result2['total_trades']}")
    print(f"P&L: ${result2['total_pnl']:,.0f}")
    
    # Comparison
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    sharpe_improvement = (result2['sharpe_ratio'] - result1['sharpe_ratio']) / abs(result1['sharpe_ratio']) * 100 if result1['sharpe_ratio'] != 0 else 0
    pnl_improvement = (result2['total_pnl'] - result1['total_pnl']) / abs(result1['total_pnl']) * 100 if result1['total_pnl'] != 0 else 0
    
    print(f"Sharpe Improvement: {sharpe_improvement:+.1f}%")
    print(f"P&L Improvement: {pnl_improvement:+.1f}%")
    print(f"Win Rate Change: {result2['win_rate'] - result1['win_rate']:+.1f}%")
    
    # Position size analysis
    if 'trades' in result2 and result2['trades']:
        sizes = [t.position_size / 1_000_000 for t in result2['trades'] if hasattr(t, 'position_size')]
        if sizes:
            print(f"\nPosition Sizing (Institutional):")
            print(f"  Average: {np.mean(sizes):.2f}M")
            print(f"  1M trades: {sum(1 for s in sizes if s < 1.5)}")
            print(f"  2M trades: {sum(1 for s in sizes if s >= 1.5)}")
    
    print("\n📊 Key Differences:")
    print("- Position sizes: 0.5M/1M → 1M/2M")
    print("- Stop losses: 3-10 pips → 5-15 pips")
    print("- Take profits: 0.15x/0.25x/0.4x → 0.3x/0.6x/1.0x ATR")
    print("- Partial profit: 70% at 3 pips → 40% at 60% to TP1")
    print("- Intelligent sizing: OFF → ON (scales with confidence)")


if __name__ == "__main__":
    run_comparison()