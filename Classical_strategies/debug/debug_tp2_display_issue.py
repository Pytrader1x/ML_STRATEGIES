"""Debug script to identify why TP2 shows $0|0.00M in plots"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import ProductionPlotter
import matplotlib.pyplot as plt

# Load recent results to check
print("=== Loading recent trade results ===")

# Check if we have detailed CSV files
csv_files = [
    '../results/AUDUSD_config_1_ultra-tight_risk_management_trades_detail_20250614_165616.csv',
    '../results/AUDUSD_config_2_scalping_strategy_trades_detail_20250614_165620.csv'
]

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        print(f"\n{csv_file}:")
        print(f"Total trades: {len(df)}")
        
        # Find trades with multiple TP hits
        multi_tp_trades = df[df['tp_hits'] > 1]
        print(f"Trades with multiple TP hits: {len(multi_tp_trades)}")
        
        if len(multi_tp_trades) > 0:
            # Look at first multi-TP trade
            trade = multi_tp_trades.iloc[0]
            print(f"\nExample multi-TP trade:")
            print(f"  Trade ID: {trade['trade_id']}")
            print(f"  TP hits: {trade['tp_hits']}")
            print(f"  Exit reason: {trade['exit_reason']}")
            
            # Check partial exit columns
            for i in range(1, 4):
                exit_type = trade.get(f'partial_exit_{i}_type', '')
                exit_size = trade.get(f'partial_exit_{i}_size_m', 0)
                exit_pnl = trade.get(f'partial_exit_{i}_pnl', 0)
                
                if pd.notna(exit_type) and exit_type:
                    print(f"\n  Partial Exit {i}:")
                    print(f"    Type: {exit_type}")
                    print(f"    Size (M): {exit_size}")
                    print(f"    P&L: ${exit_pnl}")
        
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

# Now let's trace through a live run with debugging
print("\n\n=== Running live test with TP debugging ===")

# Load some real data
data_file = '../../ML_Backtesting_Data_15min/15min_AUDUSD.csv'
try:
    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take a slice where we know there are trades
    test_df = df['2015-10-19':'2015-10-30'].copy()
    
    print(f"Test data period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Rows: {len(test_df)}")
    
    # Create config with settings that will hit multiple TPs
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        tp_atr_multipliers=(0.3, 0.6, 1.0),  # Closer TPs for testing
        realistic_costs=False,
        verbose=False,
        debug_decisions=True
    )
    
    # Run strategy
    strategy = OptimizedProdStrategy(config)
    results = strategy.run_backtest(test_df)
    
    print(f"\nTotal trades: {len(results['trades'])}")
    
    # Find trades with multiple TP hits
    multi_tp_trades = [t for t in results['trades'] if t.tp_hits > 1]
    print(f"Trades with multiple TP hits: {len(multi_tp_trades)}")
    
    if multi_tp_trades:
        # Analyze first multi-TP trade
        trade = multi_tp_trades[0]
        print(f"\nAnalyzing trade with {trade.tp_hits} TP hits:")
        print(f"  Entry: {trade.entry_price:.5f}")
        print(f"  Direction: {trade.direction.value}")
        print(f"  Position size: {trade.position_size/1e6:.2f}M")
        print(f"  Take profits: {[f'{tp:.5f}' for tp in trade.take_profits]}")
        
        print(f"\n  Partial exits ({len(trade.partial_exits)}):")
        for i, pe in enumerate(trade.partial_exits):
            print(f"\n  Exit {i+1}:")
            print(f"    TP Level: {pe.tp_level}")
            print(f"    Price: {pe.price:.5f}")
            print(f"    Size: {pe.size/1e6:.2f}M")
            print(f"    P&L: ${pe.pnl:.2f}")
            
            # Check object attributes
            print(f"    Attributes: {list(vars(pe).keys())}")
            
        # Now test plotting this specific trade
        print("\n=== Testing plot generation ===")
        
        # Create a smaller dataframe around this trade
        trade_start = trade.entry_time - pd.Timedelta(hours=2)
        trade_end = trade.exit_time + pd.Timedelta(hours=2)
        plot_df = test_df[trade_start:trade_end].copy()
        
        # Create plotter and plot
        plotter = ProductionPlotter()
        plot_results = {
            'trades': [trade],
            'equity_curve': pd.Series([1000000, 1000000 + trade.pnl], 
                                    index=[plot_df.index[0], plot_df.index[-1]]),
            'sharpe_ratio': 1.0,
            'total_pnl': trade.pnl
        }
        
        try:
            fig = plotter.plot_strategy_results(
                plot_df, 
                plot_results,
                title=f"Debug: Trade with {trade.tp_hits} TP hits",
                show=False
            )
            
            # Save plot
            fig.savefig('debug_multi_tp_plot.png', dpi=150, bbox_inches='tight')
            print("Plot saved as debug_multi_tp_plot.png")
            plt.close(fig)
            
            # Let's also manually check what the plotting function sees
            print("\n=== Manual plot data check ===")
            
            # Simulate what happens in the plotting function
            for j, partial_exit in enumerate(trade.partial_exits):
                if hasattr(partial_exit, 'tp_level') and partial_exit.tp_level > 0:
                    tp_level = partial_exit.tp_level
                    partial_price = partial_exit.price if hasattr(partial_exit, 'price') else partial_exit.get('price')
                    
                    # Calculate pips
                    if trade.direction.value == 'long':
                        partial_pips = (partial_price - trade.entry_price) * 10000
                    else:
                        partial_pips = (trade.entry_price - partial_price) * 10000
                    
                    # Get size and P&L
                    partial_size = partial_exit.size if hasattr(partial_exit, 'size') else partial_exit.get('size', 0)
                    partial_size_m = partial_size / 1000000
                    partial_pnl = partial_exit.pnl if hasattr(partial_exit, 'pnl') else partial_exit.get('pnl', 0)
                    
                    # Format P&L
                    if partial_pnl is not None and abs(partial_pnl) >= 1000:
                        pnl_text = f"${partial_pnl/1000:.1f}k"
                    elif partial_pnl is not None:
                        pnl_text = f"${partial_pnl:.0f}"
                    else:
                        pnl_text = "$0"
                    
                    text = f'TP{tp_level}|+{partial_pips:.1f}p|{pnl_text}|{partial_size_m:.2f}M'
                    print(f"  TP{tp_level} annotation text: {text}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()
            
except FileNotFoundError:
    print(f"Could not find data file: {data_file}")
    print("Please ensure you have the 15min AUDUSD data file")
    
print("\n=== Debug complete ===")