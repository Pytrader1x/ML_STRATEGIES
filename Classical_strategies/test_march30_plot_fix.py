"""
Test the March 30 21:15 trade plotting fix
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("TEST: March 30 21:15 Trade Plot Fix")
    print("="*80)
    
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take last 5000 rows
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000, risk_per_trade=0.002, sl_max_pips=10.0,
        sl_atr_multiplier=1.0, tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003, tsl_activation_pips=15, tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0, trailing_atr_multiplier=1.2,
        tp_range_market_multiplier=0.5, tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3, sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False, partial_profit_before_sl=False,
        debug_decisions=False, use_daily_sharpe=True
    )
    
    strategy = OptimizedProdStrategy(strategy_config)
    results = strategy.run_backtest(df)
    
    # Find and display the March 30 trade details
    target = pd.Timestamp('2025-03-30 21:15:00')
    found_trade = None
    
    for trade in results['trades']:
        if trade.entry_time == target:
            found_trade = trade
            break
    
    if found_trade:
        print(f"\nFound March 30 21:15 trade:")
        print(f"  Direction: {found_trade.direction.value}")
        print(f"  Entry: {found_trade.entry_price:.5f}")
        print(f"  Exit Reason: {found_trade.exit_reason}")
        print(f"  Total P&L: ${found_trade.pnl:.2f}")
        
        print("\n  Partial Exits:")
        for i, pe in enumerate(found_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'Final'
            print(f"    {i+1}. {'TP' + str(tp_level) if tp_level != 'Final' and tp_level != 0 else 'Final'}: {pe.size/1e6:.2f}M @ {pe.price:.5f} = ${pe.pnl:.2f}")
        
        # Filter to show only around March 30 trade
        start_idx = None
        end_idx = None
        for i, ts in enumerate(df.index):
            if ts.date() == pd.Timestamp('2025-03-29').date():
                if start_idx is None:
                    start_idx = i
            elif ts.date() == pd.Timestamp('2025-04-01').date():
                end_idx = i
                break
        
        if start_idx is not None and end_idx is not None:
            df_subset = df.iloc[start_idx:end_idx].copy()
            
            # Create plot focused on this trade
            print("\n  Creating focused plot...")
            try:
                fig = plot_production_results(
                    df_subset, 
                    {'trades': [found_trade]}, 
                    title="March 30 21:15 Trade - Fixed Display",
                    figsize=(14, 8),
                    show_pnl=False,
                    show_position_sizes=False
                )
                
                # Save the plot
                plt.savefig('test_march30_fix.png', dpi=150, bbox_inches='tight')
                print("  Plot saved as 'test_march30_fix.png'")
                
                plt.close()
            except Exception as e:
                print(f"  Error creating plot: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n  Expected Final Marker Display:")
        print("    TP1 PB|+9.4p|$236|Total $1180|0.25M")
        print("\n  (Should show TP1 pullback with correct P&L values)")
    else:
        print("Trade not found!")

if __name__ == "__main__":
    main()