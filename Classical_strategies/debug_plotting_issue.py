"""
Debug the plotting issue with TP markers
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
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
        exit_on_signal_flip=False, signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0, signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=False, partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5, intelligent_sizing=False,
        sl_volatility_adjustment=True, relaxed_position_multiplier=0.5,
        relaxed_mode=False, realistic_costs=True, verbose=False,
        debug_decisions=False, use_daily_sharpe=True
    )

    strategy = OptimizedProdStrategy(strategy_config)
    results = strategy.run_backtest(df)

    # Find a trade with multiple TP exits
    target_trade = None
    for trade in results['trades']:
        if len(trade.partial_exits) >= 2:  # Find trade with multiple partial exits
            target_trade = trade
            break

    if target_trade:
        print("="*60)
        print("ANALYZING TRADE WITH MULTIPLE TP EXITS")
        print("="*60)
        
        print(f"Entry: {target_trade.entry_time} at {target_trade.entry_price}")
        print(f"Final Exit: {target_trade.exit_time} at {target_trade.exit_price}")
        print(f"Final Exit Reason: {target_trade.exit_reason}")
        print(f"TP Hits: {target_trade.tp_hits}")
        
        print(f"\nPartial Exits ({len(target_trade.partial_exits)}):")
        for i, pe in enumerate(target_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            print(f"  {i+1}. Time: {pe.time}")
            print(f"     Price: {pe.price}")
            print(f"     TP Level: {tp_level}")
            print(f"     Size: {pe.size/1e6:.2f}M")
            print(f"     P&L: ${pe.pnl:.2f}")
            
            # Check if this partial exit time matches final exit time
            matches_final = pe.time == target_trade.exit_time
            print(f"     Matches Final Exit Time: {'YES' if matches_final else 'NO'}")
            print()
        
        print("PLOTTING LOGIC SIMULATION:")
        print("-" * 40)
        
        final_exit_time = target_trade.exit_time
        print(f"Final exit time: {final_exit_time}")
        
        for i, pe in enumerate(target_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 0
            partial_time = pe.time
            
            # Simulate the skip logic
            will_skip = final_exit_time and partial_time == final_exit_time
            
            print(f"Partial exit {i+1} (TP{tp_level}):")
            print(f"  Time: {partial_time}")
            print(f"  Will be skipped: {'YES' if will_skip else 'NO'}")
            print(f"  Reason: {'Same as final exit time' if will_skip else 'Different time, will plot'}")
            print()
        
        # Check what would be shown on chart
        print("CHART MARKERS THAT SHOULD APPEAR:")
        print("-" * 40)
        
        # Partial exit markers (excluding final exit time)
        partial_markers = []
        for pe in target_trade.partial_exits:
            if not (final_exit_time and pe.time == final_exit_time):
                tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 0
                partial_markers.append(f"TP{tp_level} at {pe.time}")
        
        print("Partial exit markers:")
        for marker in partial_markers:
            print(f"  • {marker}")
        
        # Final exit marker
        print(f"\nFinal exit marker:")
        print(f"  • {target_trade.exit_reason} at {target_trade.exit_time}")
        
        # Test if there's an issue with TP level detection
        print("\nTP LEVEL VALIDATION:")
        print("-" * 40)
        for i, pe in enumerate(target_trade.partial_exits):
            print(f"Partial exit {i+1}:")
            print(f"  hasattr(pe, 'tp_level'): {hasattr(pe, 'tp_level')}")
            if hasattr(pe, 'tp_level'):
                print(f"  pe.tp_level: {pe.tp_level}")
            else:
                print(f"  pe.get('tp_level', 0): {pe.get('tp_level', 0) if hasattr(pe, 'get') else 'No get method'}")
    else:
        print("No trade found with multiple partial exits")
        print(f"Total trades: {len(results['trades'])}")
        
        # Show summary of all trades
        for i, trade in enumerate(results['trades'][:5]):  # First 5 trades
            print(f"Trade {i+1}: {len(trade.partial_exits)} partial exits")

if __name__ == "__main__":
    main()