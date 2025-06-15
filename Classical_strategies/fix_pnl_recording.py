"""
Fix for P&L recording issue - ensure all exits are recorded as partial exits
"""

def fix_execute_full_exit():
    """
    Show the fix needed in _execute_full_exit method to record all exits
    """
    print("="*80)
    print("FIX FOR P&L RECORDING ISSUE")
    print("="*80)
    
    print("\nPROBLEM:")
    print("- Stop loss and other non-TP exits don't create PartialExit records")
    print("- This causes position and P&L mismatches in validation")
    
    print("\nSOLUTION:")
    print("Add the following code after line 1457 in Prod_strategy.py:")
    
    fix_code = '''
            # Record the final exit as a partial exit (for consistency)
            trade.partial_exits.append(PartialExit(
                time=exit_time,
                price=exit_price,
                size=trade.remaining_size,
                tp_level=0,  # 0 indicates non-TP exit
                pnl=remaining_pnl
            ))
    '''
    
    print(fix_code)
    
    print("\nThis ensures:")
    print("1. All exits are recorded in partial_exits list")
    print("2. Sum of partial exit sizes equals initial position size")
    print("3. Sum of partial exit P&Ls equals total trade P&L")
    print("4. Consistent tracking for all exit types")

def show_current_code():
    """Show the current problematic code section"""
    print("\nCURRENT CODE (lines 1453-1460):")
    print("-"*60)
    current = '''
        # Calculate final P&L for full exits (non-TP exits)
        if trade.remaining_size > 0:
            remaining_pnl, pips = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, trade.remaining_size, trade.direction
            )
            trade.pnl = trade.partial_pnl + remaining_pnl
            self.current_capital += remaining_pnl
            trade.exit_count += 1  # Increment exit counter for full exits
            
            # [MISSING: No PartialExit record created here!]
    '''
    print(current)

def show_fixed_code():
    """Show the fixed code"""
    print("\nFIXED CODE:")
    print("-"*60)
    fixed = '''
        # Calculate final P&L for full exits (non-TP exits)
        if trade.remaining_size > 0:
            remaining_pnl, pips = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, trade.remaining_size, trade.direction
            )
            trade.pnl = trade.partial_pnl + remaining_pnl
            self.current_capital += remaining_pnl
            trade.exit_count += 1  # Increment exit counter for full exits
            
            # Record the final exit as a partial exit (for consistency)
            trade.partial_exits.append(PartialExit(
                time=exit_time,
                price=exit_price,
                size=trade.remaining_size,
                tp_level=0,  # 0 indicates non-TP exit
                pnl=remaining_pnl
            ))
    '''
    print(fixed)

if __name__ == "__main__":
    fix_execute_full_exit()
    show_current_code()
    show_fixed_code()
    
    print("\n" + "="*80)
    print("IMPLEMENTATION:")
    print("="*80)
    print("1. Add the PartialExit record after line 1459")
    print("2. This ensures ALL exits are recorded, not just TP exits")
    print("3. tp_level=0 indicates a non-TP exit (SL, TSL, signal flip, etc.)")
    print("4. This will fix all position and P&L validation errors")