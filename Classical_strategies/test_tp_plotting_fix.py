"""Test script to verify TP plotting with actual trade data"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, Trade, PartialExit, TradeDirection, ExitReason
from strategy_code.Prod_plotting import ProductionPlotter, PlotConfig, plot_production_results
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Create test data
dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
prices = 0.7500 + np.random.randn(100).cumsum() * 0.0001

df = pd.DataFrame({
    'Open': prices + np.random.randn(100) * 0.00005,
    'High': prices + abs(np.random.randn(100) * 0.0001),
    'Low': prices - abs(np.random.randn(100) * 0.0001),
    'Close': prices
}, index=dates)

# Add required indicator columns
df['NTI_Direction'] = np.random.choice([1, -1], size=100)
df['MB_Bias'] = np.random.choice([1, -1], size=100)
df['IC_Regime'] = np.random.choice([1, 2, 3], size=100)
df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Quiet Range'})

# Add Market Bias overlay columns (required by plotting)
df['MB_o2'] = df['Open'] + np.random.randn(100) * 0.00002
df['MB_h2'] = df['High'] + np.random.randn(100) * 0.00002
df['MB_l2'] = df['Low'] - np.random.randn(100) * 0.00002
df['MB_c2'] = df['Close'] + np.random.randn(100) * 0.00002

# Add NeuroTrend EMAs
df['NTI_FastEMA'] = df['Close'].ewm(span=8).mean()
df['NTI_SlowEMA'] = df['Close'].ewm(span=21).mean()

# Create test trades with TP exits
trades = []

# Trade 1: Short trade with TP1 and TP2 hits
trade1 = Trade(
    entry_time=dates[10],
    entry_price=0.7520,
    direction=TradeDirection.SHORT,
    position_size=1_000_000,
    stop_loss=0.7530,
    take_profits=[0.7510, 0.7500, 0.7490],
    exit_time=dates[30],
    exit_price=0.7505,
    exit_reason=ExitReason.TRAILING_STOP,
    confidence=75.0
)

# Add partial exits for TP1 and TP2
trade1.partial_exits = [
    PartialExit(
        time=dates[15],
        price=0.7510,
        size=500_000,
        tp_level=1,
        pnl=500.0
    ),
    PartialExit(
        time=dates[20],
        price=0.7500,
        size=165_000,
        tp_level=2,
        pnl=330.0
    )
]
trade1.tp_hits = 2
trade1.partial_pnl = 830.0
trade1.remaining_size = 335_000
trade1.pnl = 1200.0  # Total P&L including final exit

trades.append(trade1)

# Trade 2: Long trade with TP1 hit
trade2 = Trade(
    entry_time=dates[40],
    entry_price=0.7480,
    direction=TradeDirection.LONG,
    position_size=1_000_000,
    stop_loss=0.7470,
    take_profits=[0.7490, 0.7500, 0.7510],
    exit_time=dates[50],
    exit_price=0.7485,
    exit_reason=ExitReason.STOP_LOSS,
    confidence=60.0
)

# Add partial exit for TP1
trade2.partial_exits = [
    PartialExit(
        time=dates[45],
        price=0.7490,
        size=500_000,
        tp_level=1,
        pnl=500.0
    )
]
trade2.tp_hits = 1
trade2.partial_pnl = 500.0
trade2.remaining_size = 500_000
trade2.pnl = 250.0  # Total P&L

trades.append(trade2)

# Create results dict
results = {
    'total_trades': len(trades),
    'winning_trades': 1,
    'losing_trades': 1,
    'win_rate': 50.0,
    'total_pnl': 1450.0,
    'total_return': 0.145,
    'sharpe_ratio': 1.5,
    'profit_factor': 1.2,
    'max_drawdown': 0.5,
    'trades': trades
}

print("Test trades created:")
for i, trade in enumerate(trades):
    print(f"\nTrade {i+1}:")
    print(f"  Direction: {trade.direction.value}")
    print(f"  Entry: {trade.entry_time} @ {trade.entry_price}")
    print(f"  Exit: {trade.exit_time} @ {trade.exit_price} ({trade.exit_reason.value})")
    print(f"  TP hits: {trade.tp_hits}")
    print(f"  Partial exits: {len(trade.partial_exits)}")
    for j, pe in enumerate(trade.partial_exits):
        print(f"    - TP{pe.tp_level} at {pe.time} @ {pe.price}")

# Now create the plot
print("\nCreating plot...")
fig = plot_production_results(df, results)

# Save the plot
plt.savefig('test_tp_plotting.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'test_tp_plotting.png'")

# Also test with the plotter directly
print("\nTesting PlotConfig settings...")
config = PlotConfig()
plotter = ProductionPlotter(config)

# Check if the plotting function correctly identifies TP exits
for trade in trades:
    trade_dict = {
        'entry_time': trade.entry_time,
        'exit_time': trade.exit_time,
        'entry_price': trade.entry_price,
        'exit_price': trade.exit_price,
        'direction': trade.direction.value,
        'exit_reason': trade.exit_reason.value,
        'take_profits': trade.take_profits,
        'stop_loss': trade.stop_loss,
        'partial_exits': trade.partial_exits
    }
    
    # Test the TP exit filtering logic
    partial_exits = trade_dict.get('partial_exits', [])
    tp_exits = [p for p in partial_exits 
               if (hasattr(p, 'tp_level') and p.tp_level > 0) or 
                  (isinstance(p, dict) and p.get('tp_level', 0) > 0)]
    
    print(f"\nTrade direction: {trade_dict['direction']}")
    print(f"Partial exits found: {len(partial_exits)}")
    print(f"TP exits found: {len(tp_exits)}")

print("\nIf TP exit markers are not visible in the plot, the issue is in the plotting code.")
print("If they are visible, the issue is that the strategy is not generating trades with TP hits.")