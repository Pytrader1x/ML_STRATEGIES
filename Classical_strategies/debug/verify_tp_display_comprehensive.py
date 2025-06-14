"""Comprehensive test to verify TP display fix with real data"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import os
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import ProductionPlotter

# Check if we have real data
data_paths = [
    '../../ML_Backtesting_Data_15min/15min_AUDUSD.csv',
    '../../../ML_Backtesting_Data_15min/15min_AUDUSD.csv',
    '../../../../ML_Backtesting_Data_15min/15min_AUDUSD.csv'
]

df = None
for path in data_paths:
    if os.path.exists(path):
        print(f"Found data at: {path}")
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        break

if df is None:
    print("Could not find real data. Creating synthetic test...")
    
    # Create synthetic data with specific TP scenarios
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    base_price = 0.7500
    
    # Create data with trending moves
    prices = []
    for i in range(100):
        if i < 20:
            price = base_price
        elif i < 40:
            # Uptrend
            price = base_price + (i - 20) * 0.0003
        elif i < 60:
            # Consolidation
            price = base_price + 0.006 + np.sin((i - 40) * 0.3) * 0.0002
        else:
            # Downtrend
            price = base_price + 0.006 - (i - 60) * 0.0002
        prices.append(price)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 0.0002 for p in prices],
        'Low': [p - 0.0002 for p in prices],
        'Close': [p + 0.0001 for p in prices],
        'NTI_Direction': [0] * 100,
        'MB_Bias': [0] * 100,
        'IC_Regime': [0] * 100,
        'IC_RegimeName': ['quiet_range'] * 100,
        'IC_ATR_Normalized': [0.0010] * 100,
        # Add all required columns for plotting
        'MB_o2': prices,
        'MB_c2': [p + 0.0001 for p in prices],
        'MB_h2': [p + 0.0002 for p in prices],
        'MB_l2': [p - 0.0002 for p in prices],
        'NTI_Value': [0] * 100,
        'NTI_Confidence': [0.5] * 100,
        'IC_Value': [0] * 100
    }, index=dates)
    
    # Add entry signals
    df.loc[dates[10], 'NTI_Direction'] = 1
    df.loc[dates[10], 'MB_Bias'] = 1
    df.loc[dates[10], 'IC_Regime'] = 1
    df.loc[dates[10], 'IC_RegimeName'] = 'strong_trend'
    
    df.loc[dates[50], 'NTI_Direction'] = -1
    df.loc[dates[50], 'MB_Bias'] = -1
    df.loc[dates[50], 'IC_Regime'] = 1
    df.loc[dates[50], 'IC_RegimeName'] = 'strong_trend'

# Test multiple configurations
configs = [
    # Config 1: Ultra-tight with close TPs
    {
        'name': 'Ultra-Tight Multi-TP Test',
        'config': OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            tp_atr_multipliers=(0.2, 0.4, 0.6),  # Very close TPs
            realistic_costs=False,
            verbose=False,
            debug_decisions=False
        )
    },
    # Config 2: Scalping with tight TPs
    {
        'name': 'Scalping Multi-TP Test',
        'config': OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            tp_atr_multipliers=(0.1, 0.2, 0.3),  # Even tighter
            realistic_costs=False,
            verbose=False,
            debug_decisions=False
        )
    }
]

# Test each configuration
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {cfg['name']}")
    print('='*60)
    
    strategy = OptimizedProdStrategy(cfg['config'])
    results = strategy.run_backtest(df)
    
    print(f"Total trades: {len(results['trades'])}")
    
    # Find trades with multiple TP hits
    multi_tp_trades = []
    for trade in results['trades']:
        if hasattr(trade, 'tp_hits') and trade.tp_hits > 1:
            multi_tp_trades.append(trade)
        elif hasattr(trade, 'partial_exits') and len(trade.partial_exits) > 2:
            multi_tp_trades.append(trade)
    
    print(f"Trades with multiple exits: {len(multi_tp_trades)}")
    
    if multi_tp_trades:
        # Analyze first multi-exit trade
        trade = multi_tp_trades[0]
        print(f"\nExample trade analysis:")
        print(f"  Entry: {trade.entry_price:.5f}")
        print(f"  Direction: {trade.direction.value}")
        print(f"  Position size: {trade.position_size/1e6:.2f}M")
        print(f"  TP hits: {getattr(trade, 'tp_hits', 'N/A')}")
        print(f"  Total P&L: ${trade.pnl:.2f}")
        
        print(f"\n  Partial exits ({len(trade.partial_exits)}):")
        for i, pe in enumerate(trade.partial_exits):
            tp_level = getattr(pe, 'tp_level', 'N/A')
            print(f"\n  Exit {i+1} (TP{tp_level}):")
            print(f"    Price: {pe.price:.5f}")
            print(f"    Size: {pe.size/1e6:.2f}M")
            print(f"    P&L: ${pe.pnl:.2f}")
            
            # Verify values are not zero
            if pe.size == 0:
                print("    ⚠️  WARNING: Size is zero!")
            if pe.pnl == 0:
                print("    ⚠️  WARNING: P&L is zero!")
        
        # Create plot
        print(f"\nCreating plot for {cfg['name']}...")
        plotter = ProductionPlotter()
        
        # Get data around the trade
        trade_start_idx = df.index.get_loc(trade.entry_time)
        trade_end_idx = df.index.get_loc(trade.exit_time) if trade.exit_time else len(df) - 1
        
        start_idx = max(0, trade_start_idx - 20)
        end_idx = min(len(df), trade_end_idx + 20)
        
        plot_df = df.iloc[start_idx:end_idx].copy()
        
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
                title=f"{cfg['name']} - TP Display Verification",
                show=False
            )
            
            filename = f"verify_tp_display_{cfg['name'].replace(' ', '_').lower()}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Plot saved as {filename}")
            
        except Exception as e:
            print(f"  Error creating plot: {e}")
    else:
        print("  No trades with multiple exits found in this test")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nThe fix has been applied. When multiple TPs are hit:")
print("1. Each TP exit should show the correct size in millions (e.g., 0.33M, 1.00M)")
print("2. Each TP exit should show the correct P&L in dollars") 
print("3. The annotation format is: TP[level]|[pips]p|$[P&L]|[size]M")
print("\nCheck the generated plots to verify all values are displayed correctly.")