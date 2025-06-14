"""Debug script to check why TP2 shows $0 and 0.00M in the plot"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection
from strategy_code.Prod_plotting import ProductionPlotter, PlotConfig

# Create test configuration
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.5, 1.0, 1.5),  # Close TPs for testing
    realistic_costs=False,  # Disable for cleaner testing
    verbose=False,
    debug_decisions=True
)

# Create test data that will trigger multiple TPs in one candle
dates = pd.date_range(start='2024-01-01 00:00', periods=20, freq='15min')
base_price = 0.7500

# Create data
data = {
    'Open': [base_price] * 20,
    'High': [base_price + 0.0001] * 20,
    'Low': [base_price - 0.0001] * 20,
    'Close': [base_price] * 20,
    'NTI_Direction': [0] * 20,
    'MB_Bias': [0] * 20,
    'IC_Regime': [0] * 20,
    'IC_RegimeName': ['quiet_range'] * 20,
    'IC_ATR_Normalized': [0.0010] * 20
}

# Set up entry signal at index 5
data['NTI_Direction'][5] = 1
data['MB_Bias'][5] = 1
data['IC_Regime'][5] = 1
data['IC_RegimeName'][5] = 'strong_trend'

# Create a candle that hits TP1 and TP2
data['High'][10] = 0.7512  # Should hit TP1 (0.7505) and TP2 (0.7510)
data['Close'][10] = 0.7511

df = pd.DataFrame(data, index=dates)

print("=== Running strategy to debug TP P&L display ===")

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(df)

print(f"\nTotal trades: {len(results['trades'])}")

if len(results['trades']) > 0:
    trade = results['trades'][0]
    print(f"\nTrade details:")
    print(f"  Entry: {trade.entry_price:.5f}")
    print(f"  Position size: {trade.position_size/1e6:.2f}M")
    print(f"  TP hits: {trade.tp_hits}")
    print(f"  Exit count: {trade.exit_count}")
    
    if trade.partial_exits:
        print(f"\nPartial exits ({len(trade.partial_exits)}):")
        for i, pe in enumerate(trade.partial_exits):
            print(f"\n  Exit {i+1}:")
            print(f"    Type: TP{pe.tp_level}")
            print(f"    Price: {pe.price:.5f}")
            print(f"    Size: {pe.size/1e6:.2f}M")
            print(f"    P&L: ${pe.pnl:.2f}")
            print(f"    Has 'size' attr: {hasattr(pe, 'size')}")
            print(f"    Has 'pnl' attr: {hasattr(pe, 'pnl')}")
            print(f"    Size value: {pe.size if hasattr(pe, 'size') else 'NO SIZE ATTR'}")
            print(f"    P&L value: {pe.pnl if hasattr(pe, 'pnl') else 'NO PNL ATTR'}")
    
    # Now test plotting
    print("\n=== Testing plot data extraction ===")
    
    # Convert trade to dict as plotting does
    trade_dict = {
        'entry_time': trade.entry_time,
        'exit_time': trade.exit_time,
        'entry_price': trade.entry_price,
        'exit_price': trade.exit_price,
        'direction': trade.direction.value if hasattr(trade.direction, 'value') else trade.direction,
        'exit_reason': trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason,
        'take_profits': trade.take_profits,
        'stop_loss': trade.stop_loss,
        'partial_exits': trade.partial_exits,
        'position_size': trade.position_size,
        'pnl': trade.pnl
    }
    
    print(f"\nTrade dict keys: {list(trade_dict.keys())}")
    print(f"Partial exits in dict: {len(trade_dict.get('partial_exits', []))}")
    
    # Check how partial exits are accessed in plotting
    partial_exits = trade_dict.get('partial_exits', [])
    for i, pe in enumerate(partial_exits):
        print(f"\nPartial exit {i+1} in trade_dict:")
        print(f"  Type: {type(pe)}")
        print(f"  Has tp_level: {hasattr(pe, 'tp_level')}")
        print(f"  Has size: {hasattr(pe, 'size')}")
        print(f"  Has pnl: {hasattr(pe, 'pnl')}")
        
        if hasattr(pe, 'tp_level'):
            print(f"  TP level: {pe.tp_level}")
        if hasattr(pe, 'size'):
            size_m = pe.size / 1000000
            print(f"  Size (M): {size_m:.2f}")
        if hasattr(pe, 'pnl'):
            print(f"  P&L: ${pe.pnl:.2f}")
            
    # Test plotting
    print("\n=== Creating plot ===")
    plotter = ProductionPlotter()
    
    # Create a minimal results dict for plotting
    plot_results = {
        'trades': [trade],
        'equity_curve': results['equity_curve'],
        'sharpe_ratio': 1.0,
        'total_pnl': trade.pnl
    }
    
    try:
        fig = plotter.plot_strategy_results(df, plot_results)
        fig.savefig('debug_tp_pnl_plot.png', dpi=150, bbox_inches='tight')
        print("Plot saved as debug_tp_pnl_plot.png")
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Debug complete ===")