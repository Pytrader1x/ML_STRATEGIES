"""Verify fixes for duplicate markers and SL calculation"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection
from strategy_code.Prod_plotting import ProductionPlotter

# Create test configuration
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.3, 0.6, 1.0),
    realistic_costs=False,
    verbose=False
)

# Create test data
dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
base_price = 0.7500

# Initialize data
data = pd.DataFrame({
    'Open': [base_price] * 100,
    'High': [base_price + 0.0002] * 100,
    'Low': [base_price - 0.0002] * 100,
    'Close': [base_price] * 100,
    'NTI_Direction': [0] * 100,
    'MB_Bias': [0] * 100,
    'IC_Regime': [0] * 100,
    'IC_RegimeName': ['quiet_range'] * 100,
    'IC_ATR_Normalized': [0.0010] * 100,
    'MB_o2': [base_price] * 100,
    'MB_c2': [base_price] * 100,
    'MB_h2': [base_price + 0.0002] * 100,
    'MB_l2': [base_price - 0.0002] * 100,
    'NTI_Value': [0] * 100,
    'NTI_Confidence': [0.5] * 100,
    'IC_Value': [0] * 100
}, index=dates)

# Test 1: Long trade that hits multiple TPs at same time
data.loc[dates[10], 'NTI_Direction'] = 1
data.loc[dates[10], 'MB_Bias'] = 1
data.loc[dates[10], 'IC_Regime'] = 1
data.loc[dates[10], 'IC_RegimeName'] = 'strong_trend'

# Big move up to hit multiple TPs
data.loc[dates[20], 'High'] = 0.7515
data.loc[dates[20], 'Close'] = 0.7514

# Test 2: Short trade that hits SL
data.loc[dates[40], 'NTI_Direction'] = -1
data.loc[dates[40], 'MB_Bias'] = -1
data.loc[dates[40], 'IC_Regime'] = 1
data.loc[dates[40], 'IC_RegimeName'] = 'strong_trend'

# Move up to hit SL
data.loc[dates[50], 'High'] = 0.7515
data.loc[dates[50], 'Close'] = 0.7513

print("=== Testing Fixes ===")
print("1. Duplicate TP markers should be vertically offset")
print("2. SL should show correct P&L for full position size")
print("="*60)

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(data)

print(f"\nTotal trades: {len(results['trades'])}")

# Analyze trades
for i, trade in enumerate(results['trades']):
    print(f"\n{'='*60}")
    print(f"Trade {i+1}: {trade.direction.value.upper()}")
    print(f"  Entry: {trade.entry_price:.5f}")
    print(f"  Position size: {trade.position_size/1e6:.2f}M")
    print(f"  Exit reason: {trade.exit_reason}")
    
    if trade.exit_reason == 'stop_loss':
        # Calculate expected SL loss
        if trade.direction == TradeDirection.LONG:
            sl_pips = (trade.stop_loss - trade.entry_price) * 10000
        else:
            sl_pips = (trade.entry_price - trade.stop_loss) * 10000
            
        # Get remaining size at SL
        remaining_size = trade.position_size / 1e6
        if hasattr(trade, 'partial_exits'):
            for pe in trade.partial_exits:
                remaining_size -= pe.size / 1e6
                
        expected_loss = abs(sl_pips) * remaining_size * 100
        
        print(f"\n  SL Analysis:")
        print(f"    SL level: {trade.stop_loss:.5f}")
        print(f"    SL distance: {abs(sl_pips):.1f} pips")
        print(f"    Remaining size at SL: {remaining_size:.2f}M")
        print(f"    Expected loss: ${-expected_loss:.2f}")
        print(f"    Actual P&L: ${trade.pnl:.2f}")
        
        if abs(abs(trade.pnl) - expected_loss) > 10:
            print(f"    ⚠️  ISSUE: P&L mismatch!")

# Create plot
print("\n=== Creating verification plot ===")
plotter = ProductionPlotter()

plot_results = {
    'trades': results['trades'],
    'equity_curve': results['equity_curve'],
    'sharpe_ratio': 1.0,
    'total_pnl': sum(t.pnl for t in results['trades'])
}

try:
    fig = plotter.plot_strategy_results(
        data, 
        plot_results,
        title="Fix Verification: TP Offsets & SL Calculation",
        show=False
    )
    
    fig.savefig('verify_fixes_plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved as verify_fixes_plot.png")
    print("\nCheck the plot:")
    print("1. Multiple TP markers at same time should be slightly offset vertically")
    print("2. SL marker should show correct P&L based on remaining position size")
    
except Exception as e:
    print(f"Error creating plot: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Verification complete ===")