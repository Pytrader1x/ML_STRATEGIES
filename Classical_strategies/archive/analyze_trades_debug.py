"""
Analyze existing trade results and create debug JSON
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def analyze_trade_results(csv_file: str):
    """Analyze trade results from CSV and create debug JSON"""
    
    # Load trade results
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*60}")
    print(f"Trade Debug Analysis")
    print(f"{'='*60}")
    print(f"File: {csv_file}")
    print(f"Total trades: {len(df)}")
    print(f"{'='*60}\n")
    
    # Create debug entries for each trade
    debug_trades = []
    validation_errors = []
    
    for idx, trade in df.iterrows():
        # Parse partial exits
        partial_exits = []
        ppt_info = None
        tp1_info = None
        tp2_info = None
        tp3_info = None
        
        # Check for PPT (Partial Profit Taking)
        if pd.notna(trade.get('partial_exit_1_type')) and trade['partial_exit_1_type'] == 'PPT':
            ppt_info = {
                'size_millions': trade.get('partial_exit_1_size_m', 0),
                'pnl': trade.get('partial_exit_1_pnl', 0)
            }
        
        # Check for TP exits
        for i in range(1, 4):
            pe_type = trade.get(f'partial_exit_{i}_type')
            if pd.notna(pe_type):
                if pe_type == 'TP1':
                    tp1_info = {
                        'size_millions': trade.get(f'partial_exit_{i}_size_m', 0),
                        'pnl': trade.get(f'partial_exit_{i}_pnl', 0)
                    }
                elif pe_type == 'TP2':
                    tp2_info = {
                        'size_millions': trade.get(f'partial_exit_{i}_size_m', 0),
                        'pnl': trade.get(f'partial_exit_{i}_pnl', 0)
                    }
                elif pe_type == 'TP3':
                    tp3_info = {
                        'size_millions': trade.get(f'partial_exit_{i}_size_m', 0),
                        'pnl': trade.get(f'partial_exit_{i}_pnl', 0)
                    }
        
        # Calculate sizes after each exit
        initial_size = trade['initial_size_millions']
        remaining_size = initial_size
        
        position_after_ppt = initial_size
        if ppt_info:
            position_after_ppt = initial_size - ppt_info['size_millions']
            remaining_size = position_after_ppt
        
        position_after_tp1 = remaining_size
        if tp1_info:
            position_after_tp1 = remaining_size - tp1_info['size_millions']
            remaining_size = position_after_tp1
        
        position_after_tp2 = remaining_size
        if tp2_info:
            position_after_tp2 = remaining_size - tp2_info['size_millions']
            remaining_size = position_after_tp2
        
        position_after_tp3 = remaining_size
        if tp3_info:
            position_after_tp3 = remaining_size - tp3_info['size_millions']
            remaining_size = position_after_tp3
        
        # Calculate final exit PnL
        final_exit_pnl = trade['final_pnl']
        if pd.notna(trade.get('partial_pnl_total')):
            final_exit_pnl = trade['final_pnl'] - trade['partial_pnl_total']
        
        # Create debug entry
        debug_entry = {
            'trade_id': trade['trade_id'],
            'entry_time': trade['entry_time'],
            'entry_price': trade['entry_price'],
            'direction': trade['direction'],
            'initial_size_millions': initial_size,
            'confidence': trade['confidence'],
            'is_relaxed': trade['is_relaxed'],
            'entry_logic': trade['entry_logic'],
            'sl_price': trade['sl_price'],
            'sl_distance_pips': trade['sl_distance_pips'],
            'tp1_price': trade['tp1_price'],
            'tp2_price': trade['tp2_price'],
            'tp3_price': trade['tp3_price'],
            
            # PPT info
            'ppt_active': ppt_info is not None,
            'ppt_size_closed_millions': ppt_info['size_millions'] if ppt_info else 0,
            'ppt_pnl_dollars': ppt_info['pnl'] if ppt_info else 0,
            'position_after_ppt_millions': position_after_ppt,
            
            # TP1 info
            'tp1_hit': tp1_info is not None or trade['tp_hits'] >= 1,
            'tp1_size_closed_millions': tp1_info['size_millions'] if tp1_info else 0,
            'tp1_pnl_dollars': tp1_info['pnl'] if tp1_info else 0,
            'position_after_tp1_millions': position_after_tp1,
            
            # TP2 info
            'tp2_hit': tp2_info is not None or trade['tp_hits'] >= 2,
            'tp2_size_closed_millions': tp2_info['size_millions'] if tp2_info else 0,
            'tp2_pnl_dollars': tp2_info['pnl'] if tp2_info else 0,
            'position_after_tp2_millions': position_after_tp2,
            
            # TP3 info
            'tp3_hit': tp3_info is not None or trade['tp_hits'] >= 3,
            'tp3_size_closed_millions': tp3_info['size_millions'] if tp3_info else 0,
            'tp3_pnl_dollars': tp3_info['pnl'] if tp3_info else 0,
            'position_after_tp3_millions': position_after_tp3,
            
            # Final exit
            'exit_time': trade['exit_time'],
            'exit_price': trade['exit_price'],
            'exit_reason': trade['exit_reason'],
            'final_position_size_millions': remaining_size,
            'final_exit_pnl_dollars': final_exit_pnl,
            'final_exit_pips': trade.get('final_exit_pips', 0),
            
            # Trade summary
            'total_pnl_dollars': trade['final_pnl'],
            'trade_duration_hours': trade['trade_duration_hours']
        }
        
        # Validate the trade
        errors = validate_trade_debug(debug_entry)
        if errors:
            validation_errors.append({
                'trade_id': trade['trade_id'],
                'errors': errors
            })
        
        debug_entry['validation_errors'] = errors
        debug_entry['is_valid'] = len(errors) == 0
        
        debug_trades.append(debug_entry)
    
    # Create summary statistics
    summary = {
        'total_trades': len(debug_trades),
        'valid_trades': sum(1 for t in debug_trades if t['is_valid']),
        'invalid_trades': sum(1 for t in debug_trades if not t['is_valid']),
        'total_pnl': sum(t['total_pnl_dollars'] for t in debug_trades),
        'winning_trades': sum(1 for t in debug_trades if t['total_pnl_dollars'] > 0),
        'losing_trades': sum(1 for t in debug_trades if t['total_pnl_dollars'] < 0),
        'tp1_hits': sum(1 for t in debug_trades if t['tp1_hit']),
        'tp2_hits': sum(1 for t in debug_trades if t['tp2_hit']),
        'tp3_hits': sum(1 for t in debug_trades if t['tp3_hit']),
        'ppt_triggers': sum(1 for t in debug_trades if t['ppt_active'])
    }
    
    # Create output
    output = {
        'timestamp': datetime.now().isoformat(),
        'source_file': csv_file,
        'summary': summary,
        'validation_errors': validation_errors,
        'trades': debug_trades
    }
    
    # Save to JSON
    output_dir = Path('debug')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'trade_analysis_debug_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Debug analysis saved to: {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Valid trades: {summary['valid_trades']}")
    print(f"  Invalid trades: {summary['invalid_trades']}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"  Win rate: {summary['winning_trades'] / summary['total_trades'] * 100:.1f}%")
    print(f"  TP1 hit rate: {summary['tp1_hits'] / summary['total_trades'] * 100:.1f}%")
    print(f"  TP2 hit rate: {summary['tp2_hits'] / summary['total_trades'] * 100:.1f}%")
    print(f"  TP3 hit rate: {summary['tp3_hits'] / summary['total_trades'] * 100:.1f}%")
    print(f"  PPT trigger rate: {summary['ppt_triggers'] / summary['total_trades'] * 100:.1f}%")
    
    if validation_errors:
        print("\nValidation Errors Found:")
        for error in validation_errors[:5]:  # Show first 5
            print(f"\nTrade {error['trade_id']}:")
            for e in error['errors']:
                print(f"  - {e}")
        if len(validation_errors) > 5:
            print(f"\n... and {len(validation_errors) - 5} more trades with errors")
    
    return output


def validate_trade_debug(trade: dict) -> list:
    """Validate trade debug entry for consistency"""
    errors = []
    
    # Check size tracking
    total_closed = (
        trade['ppt_size_closed_millions'] +
        trade['tp1_size_closed_millions'] +
        trade['tp2_size_closed_millions'] +
        trade['tp3_size_closed_millions'] +
        trade['final_position_size_millions']
    )
    
    if abs(total_closed - trade['initial_size_millions']) > 0.001:
        errors.append(
            f"Size mismatch: closed {total_closed:.3f}M vs initial {trade['initial_size_millions']:.3f}M"
        )
    
    # Check PnL components (if we have partial PnL data)
    component_pnl = (
        trade['ppt_pnl_dollars'] +
        trade['tp1_pnl_dollars'] +
        trade['tp2_pnl_dollars'] +
        trade['tp3_pnl_dollars'] +
        trade['final_exit_pnl_dollars']
    )
    
    # Only validate if we have meaningful partial PnL data
    if any([trade['ppt_pnl_dollars'], trade['tp1_pnl_dollars'], 
            trade['tp2_pnl_dollars'], trade['tp3_pnl_dollars']]):
        if abs(component_pnl - trade['total_pnl_dollars']) > 1.0:
            errors.append(
                f"PnL mismatch: components sum to ${component_pnl:.2f} vs total ${trade['total_pnl_dollars']:.2f}"
            )
    
    # Check TP hit sequence
    if trade['tp3_hit'] and not trade['tp2_hit']:
        errors.append("TP3 hit without TP2 being hit")
    if trade['tp2_hit'] and not trade['tp1_hit']:
        errors.append("TP2 hit without TP1 being hit")
    
    # Check position tracking
    if trade['tp1_hit'] and trade['tp1_size_closed_millions'] == 0:
        errors.append("TP1 hit but no size closed")
    if trade['tp2_hit'] and trade['tp2_size_closed_millions'] == 0:
        errors.append("TP2 hit but no size closed")
    if trade['tp3_hit'] and trade['tp3_size_closed_millions'] == 0:
        errors.append("TP3 hit but no size closed")
    
    return errors


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trade results and create debug JSON')
    parser.add_argument('csv_file', help='Trade results CSV file to analyze')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return
    
    analyze_trade_results(args.csv_file)


if __name__ == "__main__":
    main()