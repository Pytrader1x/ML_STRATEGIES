"""
Debug why final exit marker shows wrong values
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("DEBUG: Final Exit Marker Display Issue")
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
    
    # Find trades with TP1 pullback exits
    tp1_pb_trades = []
    for trade in results['trades']:
        if str(trade.exit_reason) == 'ExitReason.TP1_PULLBACK':
            tp1_pb_trades.append(trade)
    
    print(f"\nFound {len(tp1_pb_trades)} trades with TP1 pullback exits")
    
    # Analyze first few
    for i, trade in enumerate(tp1_pb_trades[:3]):
        print(f"\n{'='*60}")
        print(f"Trade #{i+1}: {trade.entry_time}")
        print(f"Direction: {trade.direction.value}")
        print(f"Entry: {trade.entry_price:.5f}")
        print(f"Position: {trade.position_size/1e6:.2f}M")
        print(f"Exit Reason: {trade.exit_reason}")
        print(f"Total P&L: ${trade.pnl:.2f}")
        
        print(f"\nPartial Exits ({len(trade.partial_exits)}):")
        remaining = trade.position_size
        for j, pe in enumerate(trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            remaining -= pe.size
            print(f"  {j+1}. TP{tp_level}: {pe.size/1e6:.2f}M @ {pe.price:.5f} = ${pe.pnl:.2f}")
            print(f"     Time: {pe.time}")
            print(f"     Remaining after: {remaining/1e6:.2f}M")
        
        # Find the final exit
        final_exit = None
        for pe in trade.partial_exits:
            if pe.time == trade.exit_time:
                final_exit = pe
                print(f"\n  FINAL EXIT FOUND:")
                print(f"    Time: {pe.time}")
                print(f"    TP Level: {pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'}")
                print(f"    Size: {pe.size/1e6:.2f}M")
                print(f"    P&L: ${pe.pnl:.2f}")
        
        if not final_exit:
            print(f"\n  ❌ NO FINAL EXIT FOUND AT {trade.exit_time}")
        
        # Check what the plotting code would see
        print(f"\nPLOTTING LOGIC CHECK:")
        print(f"  Trade exit time: {trade.exit_time}")
        print(f"  Trade exit price: {trade.exit_price:.5f}")
        print(f"  Trade total P&L: ${trade.pnl:.2f}")
        
        # Calculate what should be displayed
        if trade.direction.value == 'short':
            final_pips = (trade.entry_price - trade.exit_price) / 0.0001
        else:
            final_pips = (trade.exit_price - trade.entry_price) / 0.0001
        
        print(f"  Expected final pips: {final_pips:.1f}")
        
        # Find actual final exit size and P&L
        actual_final_size = 0
        actual_final_pnl = 0
        for pe in trade.partial_exits:
            if pe.time == trade.exit_time and hasattr(pe, 'tp_level') and pe.tp_level == 0:
                actual_final_size = pe.size / 1e6
                actual_final_pnl = pe.pnl
                break
        
        print(f"\n  EXPECTED DISPLAY:")
        print(f"    Text: TP1 PB|+{final_pips:.1f}p|${actual_final_pnl:.0f}|Total ${trade.pnl:.0f}|{actual_final_size:.2f}M")

if __name__ == "__main__":
    main()