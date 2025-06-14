"""Test script to verify TP display fix"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection
from strategy_code.Prod_plotting import ProductionPlotter

# Create test configuration with tight TPs
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.3, 0.6, 1.0),  # Tight TPs to ensure multiple hits
    realistic_costs=False,
    verbose=False,
    debug_decisions=True
)

# Create test data
dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
base_price = 0.7500

data = pd.DataFrame({
    'Open': [base_price] * 50,
    'High': [base_price + 0.0001] * 50,
    'Low': [base_price - 0.0001] * 50,
    'Close': [base_price] * 50,
    'NTI_Direction': [0] * 50,
    'MB_Bias': [0] * 50,
    'IC_Regime': [0] * 50,
    'IC_RegimeName': ['quiet_range'] * 50,
    'IC_ATR_Normalized': [0.0010] * 50,
    'MB_o2': [base_price] * 50,  # Add missing columns for plotting
    'MB_c2': [base_price] * 50,
}, index=dates)

# Set up entry signal
data.loc[dates[5], 'NTI_Direction'] = 1
data.loc[dates[5], 'MB_Bias'] = 1
data.loc[dates[5], 'IC_Regime'] = 1
data.loc[dates[5], 'IC_RegimeName'] = 'strong_trend'

# Create big move that hits multiple TPs
data.loc[dates[20], 'High'] = 0.7515  # Should hit TP1 (0.7503), TP2 (0.7506), close to TP3
data.loc[dates[20], 'Close'] = 0.7514

print("=== Running strategy to test TP display fix ===")

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(data)

print(f"\nTotal trades: {len(results['trades'])}")

# Check for trades with multiple TP hits
multi_tp_trades = [t for t in results['trades'] if t.tp_hits > 0]
print(f"Trades with TP hits: {len(multi_tp_trades)}")

if multi_tp_trades:
    trade = multi_tp_trades[0]
    print(f"\nTrade with {trade.tp_hits} TP hits:")
    print(f"  Entry: {trade.entry_price:.5f}")
    print(f"  Position size: {trade.position_size/1e6:.2f}M")
    
    print(f"\n  Partial exits:")
    for i, pe in enumerate(trade.partial_exits):
        print(f"\n  Exit {i+1} (TP{pe.tp_level}):")
        print(f"    Price: {pe.price:.5f}")
        print(f"    Size: {pe.size/1e6:.2f}M")
        print(f"    P&L: ${pe.pnl:.2f}")

    # Create plot
    print("\n=== Creating plot to verify display ===")
    plotter = ProductionPlotter()
    
    # Plot results
    plot_results = {
        'trades': [trade],
        'equity_curve': results['equity_curve'],
        'sharpe_ratio': results.get('sharpe_ratio', 0),
        'total_pnl': trade.pnl
    }
    
    try:
        fig = plotter.plot_strategy_results(
            data[dates[0]:dates[30]], 
            plot_results,
            title="TP Display Fix Test",
            show=False
        )
        fig.savefig('test_tp_display_fix.png', dpi=150, bbox_inches='tight')
        print("Plot saved as test_tp_display_fix.png")
        print("\nCheck the plot to verify that all TP exits show proper P&L and size values.")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Test complete ===")