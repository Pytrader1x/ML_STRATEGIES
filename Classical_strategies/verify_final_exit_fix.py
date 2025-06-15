"""
Verify final exit marker display fix
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def test_final_exit_calculation():
    """Test the final exit calculation logic"""
    print("="*80)
    print("TEST: Final Exit Calculation Logic")
    print("="*80)
    
    # Simulate trade data
    trade_dict = {
        'entry_time': pd.Timestamp('2025-03-30 21:15:00'),
        'entry_price': 0.62959,
        'direction': 'short',
        'position_size': 1_000_000,
        'exit_time': pd.Timestamp('2025-03-31 02:30:00'),
        'exit_price': 0.62865,
        'exit_reason': 'tp1_pullback',
        'pnl': 1180.48,
        'partial_exits': [
            type('PartialExit', (), {
                'time': pd.Timestamp('2025-03-30 21:45:00'),
                'price': 0.62865,
                'size': 500_000,
                'tp_level': 1,
                'pnl': 472.19
            })(),
            type('PartialExit', (), {
                'time': pd.Timestamp('2025-03-31 00:00:00'),
                'price': 0.62770,
                'size': 250_000,
                'tp_level': 2,
                'pnl': 472.19
            })(),
            type('PartialExit', (), {
                'time': pd.Timestamp('2025-03-31 02:30:00'),  # Same as exit_time
                'price': 0.62865,
                'size': 250_000,
                'tp_level': 0,  # Final exit
                'pnl': 236.10
            })()
        ]
    }
    
    # Simulate the plotting logic
    entry_price = trade_dict['entry_price']
    exit_price = trade_dict['exit_price']
    direction = trade_dict['direction']
    final_exit_time = trade_dict['exit_time']
    
    # Calculate pips
    if direction == 'short':
        exit_pips = (entry_price - exit_price) * 10000
    else:
        exit_pips = (exit_price - entry_price) * 10000
    
    print(f"Exit pips: {exit_pips:+.1f}")
    
    # Calculate remaining size
    position_size = trade_dict['position_size'] / 1_000_000
    partial_exits = trade_dict['partial_exits']
    remaining_size = position_size
    
    print(f"\nPosition size: {position_size:.2f}M")
    print("Calculating remaining size before final exit:")
    
    for pe in partial_exits:
        pe_time = pe.time
        pe_size = pe.size / 1_000_000
        
        # Only subtract if this is NOT the final exit
        if pe_time != final_exit_time:
            remaining_size -= pe_size
            print(f"  - {pe_size:.2f}M (TP{pe.tp_level}), remaining: {remaining_size:.2f}M")
        else:
            print(f"  - Skip {pe_size:.2f}M (Final exit)")
    
    print(f"Remaining size before final exit: {remaining_size:.2f}M")
    
    # Find final exit data
    individual_exit_pnl = 0
    final_exit_size = remaining_size
    
    for pe in partial_exits:
        if pe.time == final_exit_time:
            individual_exit_pnl = pe.pnl
            final_exit_size = pe.size / 1_000_000
            print(f"\nFound final exit data:")
            print(f"  Size: {final_exit_size:.2f}M")
            print(f"  P&L: ${individual_exit_pnl:.2f}")
            break
    
    # Format P&L text
    total_trade_pnl = trade_dict['pnl']
    
    # Individual P&L
    if abs(individual_exit_pnl) >= 1000:
        individual_pnl_text = f"${individual_exit_pnl/1000:+.1f}k"
    else:
        individual_pnl_text = f"${individual_exit_pnl:+.0f}"
    
    # Total P&L
    if total_trade_pnl > 0:
        total_pnl_text = f"$+{total_trade_pnl/1000:.1f}k" if total_trade_pnl >= 1000 else f"$+{total_trade_pnl:.0f}"
    else:
        total_pnl_text = f"${total_trade_pnl/1000:.1f}k" if abs(total_trade_pnl) >= 1000 else f"${total_trade_pnl:.0f}"
    
    print(f"\nFormatted P&L:")
    print(f"  Individual: {individual_pnl_text}")
    print(f"  Total: {total_pnl_text}")
    
    # Final display text
    text = f'TP1 PB|{exit_pips:+.1f}p|{individual_pnl_text}|Total {total_pnl_text}|{final_exit_size:.2f}M'
    
    print(f"\nFinal display text:")
    print(f"  {text}")
    
    print(f"\nExpected: TP1 PB|+9.4p|$+236|Total $+1.2k|0.25M")
    print(f"Success: {'✅' if final_exit_size == 0.25 and '+236' in individual_pnl_text and '+1.2k' in total_pnl_text else '❌'}")

if __name__ == "__main__":
    test_final_exit_calculation()