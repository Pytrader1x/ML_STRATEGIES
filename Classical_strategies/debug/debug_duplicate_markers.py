"""Debug script to identify duplicate TP markers and incorrect SL calculations"""

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
    verbose=False,
    debug_decisions=True
)

# Create test data
dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
base_price = 0.7500

data = pd.DataFrame({
    'Open': [base_price] * 50,
    'High': [base_price + 0.0002] * 50,
    'Low': [base_price - 0.0002] * 50,
    'Close': [base_price] * 50,
    'NTI_Direction': [0] * 50,
    'MB_Bias': [0] * 50,
    'IC_Regime': [0] * 50,
    'IC_RegimeName': ['quiet_range'] * 50,
    'IC_ATR_Normalized': [0.0010] * 50,
    'MB_o2': [base_price] * 50,
    'MB_c2': [base_price] * 50,
    'MB_h2': [base_price + 0.0002] * 50,
    'MB_l2': [base_price - 0.0002] * 50,
    'NTI_Value': [0] * 50,
    'NTI_Confidence': [0.5] * 50,
    'IC_Value': [0] * 50
}, index=dates)

# Create entry signal
data.loc[dates[10], 'NTI_Direction'] = 1
data.loc[dates[10], 'MB_Bias'] = 1
data.loc[dates[10], 'IC_Regime'] = 1
data.loc[dates[10], 'IC_RegimeName'] = 'strong_trend'

# Test 1: Create move that hits multiple TPs
data.loc[dates[20], 'High'] = 0.7510
data.loc[dates[20], 'Close'] = 0.7509

# Test 2: Create move that hits stop loss
data.loc[dates[30], 'NTI_Direction'] = -1
data.loc[dates[30], 'MB_Bias'] = -1
data.loc[dates[30], 'IC_Regime'] = 1
data.loc[dates[30], 'IC_RegimeName'] = 'strong_trend'

# Create downward move for SL
data.loc[dates[35], 'Low'] = 0.7485
data.loc[dates[35], 'Close'] = 0.7487

print("=== Debugging Duplicate Markers and SL Calculations ===")

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(data)

print(f"\nTotal trades: {len(results['trades'])}")

# Analyze trades
for i, trade in enumerate(results['trades']):
    print(f"\n{'='*60}")
    print(f"Trade {i+1}:")
    print(f"  Entry: {trade.entry_price:.5f}")
    print(f"  Direction: {trade.direction.value}")
    print(f"  Position size: {trade.position_size/1e6:.2f}M")
    print(f"  Exit reason: {trade.exit_reason}")
    print(f"  Exit price: {trade.exit_price:.5f}")
    
    # Calculate expected P&L
    if trade.direction == TradeDirection.LONG:
        pips = (trade.exit_price - trade.entry_price) * 10000
    else:
        pips = (trade.entry_price - trade.exit_price) * 10000
    
    print(f"  Exit pips: {pips:.1f}")
    
    # Check partial exits
    if hasattr(trade, 'partial_exits') and len(trade.partial_exits) > 0:
        print(f"\n  Partial exits ({len(trade.partial_exits)}):")
        
        # Check for duplicates
        tp_times = {}
        for j, pe in enumerate(trade.partial_exits):
            tp_level = getattr(pe, 'tp_level', 'N/A')
            pe_time = pe.time
            
            # Track times for each TP level
            if tp_level not in tp_times:
                tp_times[tp_level] = []
            tp_times[tp_level].append(pe_time)
            
            print(f"\n    Exit {j+1} (TP{tp_level}):")
            print(f"      Time: {pe_time}")
            print(f"      Price: {pe.price:.5f}")
            print(f"      Size: {pe.size/1e6:.2f}M")
            print(f"      P&L: ${pe.pnl:.2f}")
        
        # Check for duplicates
        for tp_level, times in tp_times.items():
            if len(times) > 1:
                print(f"\n  ⚠️  WARNING: TP{tp_level} has {len(times)} exits!")
                for t in times:
                    print(f"      - {t}")
    
    # For SL exits, verify P&L calculation
    if trade.exit_reason == 'stop_loss':
        print(f"\n  SL Exit Analysis:")
        print(f"    Total P&L: ${trade.pnl:.2f}")
        
        # Calculate expected P&L
        remaining_size = trade.position_size / 1e6  # in millions
        
        # Subtract partial exit sizes
        if hasattr(trade, 'partial_exits'):
            for pe in trade.partial_exits:
                remaining_size -= pe.size / 1e6
        
        expected_pnl = pips * remaining_size * 100  # $100 per pip per million
        print(f"    Remaining size at SL: {remaining_size:.2f}M")
        print(f"    Expected P&L: ${expected_pnl:.2f}")
        print(f"    Pips lost: {pips:.1f}")
        
        if abs(expected_pnl - trade.pnl) > 1:
            print(f"    ⚠️  P&L MISMATCH: Expected ${expected_pnl:.2f}, got ${trade.pnl:.2f}")

# Create plot to verify
print("\n=== Creating debug plot ===")
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
        title="Debug: Duplicate Markers & SL Calculation",
        show=False
    )
    
    fig.savefig('debug_markers_and_sl.png', dpi=150, bbox_inches='tight')
    print("Plot saved as debug_markers_and_sl.png")
    
except Exception as e:
    print(f"Error creating plot: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug complete ===")