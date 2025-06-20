"""
Focused Standard Mode Validation
Tests ONLY the conservative entry logic with proper configuration
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC

def create_standard_mode_config():
    """Create configuration for STANDARD MODE ONLY"""
    return OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.001,  # 0.1% risk
        base_position_size_millions=1.0,
        
        # CRITICAL: Standard Mode Settings
        relaxed_mode=False,  # THIS ENSURES STANDARD MODE
        relaxed_position_multiplier=0.5,  # Not used in standard mode
        
        # Conservative stop loss
        sl_min_pips=5.0,
        sl_max_pips=20.0,
        sl_atr_multiplier=1.5,
        
        # Standard take profit levels
        tp_atr_multipliers=(1.0, 2.0, 3.0),
        tp_min_pips=(15.0, 25.0, 35.0),
        
        # Realistic costs
        realistic_costs=True,
        entry_slippage_pips=0.1,
        stop_loss_slippage_pips=0.5,
        
        # Other settings
        intelligent_sizing=False,
        exit_on_signal_flip=True,
        use_daily_sharpe=True,
        verbose=False,
        debug_decisions=False
    )

def test_standard_mode():
    """Test standard mode entry conditions"""
    print("=" * 80)
    print("STANDARD MODE VALIDATION")
    print("=" * 80)
    print("\nEntry Requirements:")
    print("LONG:  NTI_Direction == 1 AND MB_Bias == 1 AND IC_Regime ∈ [1,2]")
    print("SHORT: NTI_Direction == -1 AND MB_Bias == -1 AND IC_Regime ∈ [1,2]")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    from run_validated_strategy import ValidatedStrategyRunner
    runner = ValidatedStrategyRunner('AUDUSD', initial_capital=1_000_000, position_size_millions=1.0)
    runner.load_data()
    df = runner.df.copy()
    print(f"   Data loaded: {len(df)} bars")
    
    # Test on recent data
    test_start = '2023-01-01'
    test_end = '2023-12-31'
    test_df = df.loc[test_start:test_end].copy()
    print(f"   Test period: {test_start} to {test_end}")
    print(f"   Test bars: {len(test_df)}")
    
    # Count standard mode signals
    print("\n2. Analyzing standard mode signals...")
    long_signals = 0
    short_signals = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Check LONG conditions
        if (row['NTI_Direction'] == 1 and 
            row['MB_Bias'] == 1 and 
            row['IC_Regime'] in [1, 2]):
            long_signals += 1
            
        # Check SHORT conditions
        elif (row['NTI_Direction'] == -1 and 
              row['MB_Bias'] == -1 and 
              row['IC_Regime'] in [1, 2]):
            short_signals += 1
    
    total_signals = long_signals + short_signals
    signal_rate = (total_signals / len(test_df)) * 100 if len(test_df) > 0 else 0
    
    print(f"\n   Signal Analysis:")
    print(f"   LONG signals:  {long_signals}")
    print(f"   SHORT signals: {short_signals}")
    print(f"   Total signals: {total_signals}")
    print(f"   Signal rate:   {signal_rate:.2f}% of bars")
    
    # Run backtest with standard mode
    print("\n3. Running backtest with STANDARD MODE...")
    config = create_standard_mode_config()
    strategy = OptimizedProdStrategy(config)
    
    # Verify config
    print(f"\n   Config verification:")
    print(f"   Relaxed mode: {config.relaxed_mode} (should be False)")
    print(f"   Position size: {config.base_position_size_millions}M")
    print(f"   Risk per trade: {config.risk_per_trade*100:.1f}%")
    
    # Run backtest
    result = strategy.run_backtest(test_df)
    
    # Display results
    print(f"\n4. STANDARD MODE Results:")
    print(f"   Sharpe Ratio:  {result.get('sharpe_ratio', 0):.3f}")
    print(f"   Total Return:  {result.get('total_return', 0):.2f}%")
    print(f"   Max Drawdown:  {result.get('max_drawdown', 0):.2f}%")
    print(f"   Total Trades:  {result.get('total_trades', 0)}")
    print(f"   Win Rate:      {result.get('win_rate', 0):.1f}%")
    print(f"   Profit Factor: {result.get('profit_factor', 0):.2f}")
    print(f"   Total P&L:     ${result.get('total_pnl', 0):,.0f}")
    
    # Check if trades match expected signals
    if result.get('total_trades', 0) > 0:
        trade_efficiency = (result['total_trades'] / total_signals) * 100 if total_signals > 0 else 0
        print(f"\n   Signal Efficiency: {trade_efficiency:.1f}%")
        print(f"   (Trades taken vs signals generated)")
    
    # Validate trade entries
    print("\n5. Validating trade entries...")
    if 'trades' in result and result['trades']:
        standard_trades = 0
        relaxed_trades = 0
        
        for trade in result['trades']:
            if trade.get('is_relaxed', False):
                relaxed_trades += 1
            else:
                standard_trades += 1
        
        print(f"   Standard mode trades: {standard_trades}")
        print(f"   Relaxed mode trades:  {relaxed_trades}")
        
        if relaxed_trades == 0:
            print("\n   ✅ VALIDATION PASSED: All trades use standard mode entry")
        else:
            print("\n   ❌ VALIDATION FAILED: Found relaxed mode trades!")
    
    # Test multiple years
    print("\n6. Testing multiple years...")
    years = ['2020', '2021', '2022', '2023', '2024']
    results_summary = []
    
    for year in years:
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        
        # Skip if data not available
        if year_start not in df.index or (year == '2024' and year_end not in df.index):
            year_end = f'{year}-06-30' if year == '2024' else year_end
        
        year_df = df.loc[year_start:year_end].copy()
        if len(year_df) == 0:
            continue
            
        # Run backtest
        year_result = strategy.run_backtest(year_df)
        
        results_summary.append({
            'year': year,
            'sharpe': year_result.get('sharpe_ratio', 0),
            'return': year_result.get('total_return', 0),
            'trades': year_result.get('total_trades', 0),
            'win_rate': year_result.get('win_rate', 0)
        })
        
        print(f"\n   {year}: Sharpe={year_result.get('sharpe_ratio', 0):.3f}, "
              f"Return={year_result.get('total_return', 0):.1f}%, "
              f"Trades={year_result.get('total_trades', 0)}")
    
    # Summary statistics
    if results_summary:
        avg_sharpe = np.mean([r['sharpe'] for r in results_summary])
        avg_return = np.mean([r['return'] for r in results_summary])
        avg_trades = np.mean([r['trades'] for r in results_summary])
        
        print(f"\n7. Multi-year Summary:")
        print(f"   Average Sharpe:  {avg_sharpe:.3f}")
        print(f"   Average Return:  {avg_return:.1f}%")
        print(f"   Average Trades:  {avg_trades:.0f} per year")
        
        positive_years = sum(1 for r in results_summary if r['sharpe'] > 0)
        consistency = (positive_years / len(results_summary)) * 100
        print(f"   Consistency:     {positive_years}/{len(results_summary)} positive years ({consistency:.0f}%)")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_standard_mode()