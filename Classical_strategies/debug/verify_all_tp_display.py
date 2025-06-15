"""Verify that all TP levels (including TP2) are displayed with P&L and size"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import ProductionPlotter
import matplotlib.pyplot as plt

print("=== Testing TP Display Fix - All TP Levels ===")

# Create test data
dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
base_price = 0.7500

# Create synthetic data
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
    # Add all required columns
    'MB_o2': [base_price] * 50,
    'MB_c2': [base_price] * 50,
    'MB_h2': [base_price + 0.0001] * 50,
    'MB_l2': [base_price - 0.0001] * 50,
    'NTI_Value': [0] * 50,
    'NTI_Confidence': [0.5] * 50,
    'IC_Value': [0] * 50
}, index=dates)

# Create entry signal
data.loc[dates[5], 'NTI_Direction'] = 1
data.loc[dates[5], 'MB_Bias'] = 1
data.loc[dates[5], 'IC_Regime'] = 1
data.loc[dates[5], 'IC_RegimeName'] = 'strong_trend'

# Create a big move that hits all TPs
data.loc[dates[20], 'High'] = 0.7520
data.loc[dates[20], 'Close'] = 0.7518

# Configuration with tight TPs to ensure all are hit
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.2, 0.4, 0.6),  # Tight TPs
    realistic_costs=False,
    verbose=False
)

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(data)

print(f"\nTotal trades: {len(results['trades'])}")

# Find trades with TP hits
for trade in results['trades']:
    if hasattr(trade, 'partial_exits') and len(trade.partial_exits) > 0:
        print(f"\nTrade with {len(trade.partial_exits)} exits:")
        print(f"  Entry: {trade.entry_price:.5f}")
        print(f"  Position size: {trade.position_size/1e6:.2f}M")
        
        # Check each exit
        tp_levels_found = []
        for i, pe in enumerate(trade.partial_exits):
            tp_level = getattr(pe, 'tp_level', 'N/A')
            if tp_level != 'N/A' and tp_level > 0:
                tp_levels_found.append(tp_level)
            print(f"\n  Exit {i+1} (TP{tp_level}):")
            print(f"    Price: {pe.price:.5f}")
            print(f"    Size: {pe.size/1e6:.2f}M")
            print(f"    P&L: ${pe.pnl:.2f}")
        
        print(f"\n  TP levels hit: {sorted(tp_levels_found)}")
        
        # Create plot
        print("\nCreating plot...")
        plotter = ProductionPlotter()
        
        plot_results = {
            'trades': [trade],
            'equity_curve': results['equity_curve'],
            'sharpe_ratio': 1.0,
            'total_pnl': trade.pnl
        }
        
        try:
            fig = plotter.plot_strategy_results(
                data[dates[0]:dates[30]], 
                plot_results,
                title="All TP Levels Display Test - TP1, TP2, TP3",
                show=False
            )
            
            filename = 'verify_all_tp_levels_display.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename}")
            
            # Visual verification
            print("\n" + "="*60)
            print("VISUAL VERIFICATION CHECKLIST:")
            print("="*60)
            print("✓ TP1 should show: TP1|+X.Xp|$XXX|X.XXM")
            print("✓ TP2 should show: TP2|+X.Xp|$XXX|X.XXM (NOT $0|0.00M)")
            print("✓ TP3 should show: TP3|+X.Xp|$XXX|X.XXM")
            print("\nAll TP levels should display actual P&L and size values.")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()
        
        break  # Only need first trade

print("\n=== Test Complete ===")