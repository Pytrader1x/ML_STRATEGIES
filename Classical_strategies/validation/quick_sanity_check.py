"""
Quick Sanity Check for Backtesting Implementation
Run this after fixing issues to verify corrections
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import random
import warnings

warnings.filterwarnings('ignore')

def run_sanity_checks():
    """Run quick sanity checks to verify backtesting is realistic"""
    
    print("="*80)
    print("BACKTESTING SANITY CHECK")
    print("="*80)
    
    # Load data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Test on recent data
    test_df = df['2023-01-01':'2023-12-31'].copy()
    test_df = TIC.add_neuro_trend_intelligent(test_df)
    test_df = TIC.add_market_bias(test_df)
    test_df = TIC.add_intelligent_chop(test_df)
    
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Total bars: {len(test_df):,}")
    
    results = {}
    
    # Check 1: Weekend data
    print("\n1. WEEKEND DATA CHECK")
    weekend_bars = test_df[test_df.index.dayofweek >= 5]
    results['weekend_bars'] = len(weekend_bars)
    print(f"   Weekend bars found: {len(weekend_bars)}")
    print(f"   ✅ PASS" if len(weekend_bars) == 0 else f"   ❌ FAIL - Should be 0")
    
    # Check 2: Random strategy performance
    print("\n2. RANDOM STRATEGY CHECK")
    random_sharpes = []
    
    for i in range(10):
        # Create pure random strategy
        class RandomStrategy(OptimizedProdStrategy):
            def generate_signal(self, row, prev_row=None):
                return random.choice([-1, 0, 0, 0, 0, 1])  # 20% chance of signal
        
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=False
        )
        
        strategy = RandomStrategy(config)
        result = strategy.run_backtest(test_df.iloc[-5000:])  # Last 5000 bars
        random_sharpes.append(result['sharpe_ratio'])
    
    avg_random_sharpe = np.mean(random_sharpes)
    results['avg_random_sharpe'] = avg_random_sharpe
    print(f"   Average random Sharpe: {avg_random_sharpe:.3f}")
    print(f"   ✅ PASS" if -0.2 < avg_random_sharpe < 0.2 else f"   ❌ FAIL - Should be near 0")
    
    # Check 3: Transaction cost impact
    print("\n3. TRANSACTION COST CHECK")
    print("   Testing if spread reduces profits...")
    
    # This would need proper implementation in the strategy
    # For now, just flag it
    print("   ⚠️  Manual verification needed")
    print("   Expected: Adding 1 pip spread reduces profit by ~$100/trade")
    
    # Check 4: Position sizing consistency
    print("\n4. POSITION SIZING CHECK")
    print("   Checking if position sizes are fixed...")
    
    # Run a quick backtest and check position sizes
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        verbose=True
    )
    strategy = OptimizedProdStrategy(config)
    
    # Note: Would need to capture and analyze position sizes
    print("   ⚠️  Manual verification needed")
    print("   Expected: All trades should be 1M units")
    
    # Check 5: Buy and hold baseline
    print("\n5. BUY AND HOLD BASELINE")
    initial_price = test_df['Close'].iloc[0]
    final_price = test_df['Close'].iloc[-1]
    bh_return = (final_price - initial_price) / initial_price * 100
    results['buy_hold_return'] = bh_return
    print(f"   Buy & Hold return: {bh_return:.2f}%")
    print(f"   Market direction: {'Bullish' if bh_return > 0 else 'Bearish'}")
    
    # Summary
    print("\n" + "="*80)
    print("SANITY CHECK SUMMARY")
    print("="*80)
    
    pass_count = 0
    total_checks = 2  # Only counting automated checks
    
    if results['weekend_bars'] == 0:
        pass_count += 1
    if -0.2 < results['avg_random_sharpe'] < 0.2:
        pass_count += 1
    
    print(f"\nAutomated checks passed: {pass_count}/{total_checks}")
    
    if pass_count == total_checks:
        print("✅ Basic sanity checks PASSED")
        print("⚠️  Still need manual verification of:")
        print("   - Transaction costs properly reduce profits")
        print("   - Position sizes are truly fixed")
        print("   - Spread is applied correctly on entry")
    else:
        print("❌ Sanity checks FAILED")
        print("🚨 DO NOT proceed until all checks pass")
    
    return results


def main():
    """Run sanity checks"""
    results = run_sanity_checks()
    
    print("\n💡 After fixing issues, all checks should pass")
    print("📊 Random strategies should have Sharpe between -0.2 and 0.2")
    print("🚫 Weekend bars should be 0")
    print("💰 Transaction costs should reduce profits, not increase them")


if __name__ == "__main__":
    main()