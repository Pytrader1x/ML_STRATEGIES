"""
Simplified script to demonstrate the TP exit size calculation bug
"""

def simulate_exits():
    """Simulate the exit sequence to show the bug"""
    
    print("="*60)
    print("DEMONSTRATING TP EXIT SIZE BUG")
    print("="*60)
    
    # Initial trade setup
    original_position_size = 1_000_000  # 1M units
    remaining_size = original_position_size
    exit_count = 0
    
    print(f"\nInitial Position: {original_position_size:,} units")
    print(f"Remaining Size: {remaining_size:,} units")
    
    # Step 1: Partial profit exit at 50%
    print("\n--- PARTIAL PROFIT EXIT (50%) ---")
    partial_exit_percent = 0.5
    partial_exit_size = remaining_size * partial_exit_percent
    remaining_size -= partial_exit_size
    print(f"Exit Size: {partial_exit_size:,} units ({partial_exit_percent*100}% of remaining)")
    print(f"Remaining After: {remaining_size:,} units")
    
    # Step 2: TP1 Exit (current buggy logic)
    print("\n--- TP1 EXIT (BUGGY LOGIC) ---")
    print("Current code: desired_exit = trade.position_size / 3.0")
    desired_exit = original_position_size / 3.0  # Bug: uses original size
    tp1_exit_size = min(desired_exit, remaining_size)
    print(f"Desired Exit: {desired_exit:,} units (1/3 of ORIGINAL {original_position_size:,})")
    print(f"Actual Exit: {tp1_exit_size:,} units (limited by remaining)")
    remaining_size -= tp1_exit_size
    exit_count += 1
    print(f"Remaining After: {remaining_size:,} units")
    print(f"Exit Count: {exit_count}")
    
    # Step 3: TP2 Exit (current buggy logic)
    print("\n--- TP2 EXIT (BUGGY LOGIC) ---")
    desired_exit = original_position_size / 3.0  # Bug: still uses original size
    tp2_exit_size = min(desired_exit, remaining_size)
    print(f"Desired Exit: {desired_exit:,} units (1/3 of ORIGINAL {original_position_size:,})")
    print(f"Actual Exit: {tp2_exit_size:,} units (limited by remaining)")
    remaining_size -= tp2_exit_size
    exit_count += 1
    print(f"Remaining After: {remaining_size:,} units")
    print(f"Exit Count: {exit_count}")
    
    print("\n" + "="*60)
    print("ISSUE IDENTIFIED:")
    print("="*60)
    print("After 50% partial profit exit, only 500k remains")
    print("But TP exits calculate 1/3 of ORIGINAL position (333k each)")
    print("This causes:")
    print("- TP1 exits 333k (leaving 167k)")
    print("- TP2 tries to exit 333k but only 167k remains")
    print("- Duplicate markers appear because intended vs actual exit sizes differ")
    
    print("\n" + "="*60)
    print("PROPOSED FIX:")
    print("="*60)
    
    # Reset for fixed logic demo
    remaining_size = 500_000  # After 50% partial profit
    exit_count = 0
    
    print(f"\nAfter Partial Profit Exit: {remaining_size:,} units remain")
    
    # Fixed TP1 logic
    print("\n--- TP1 EXIT (FIXED LOGIC) ---")
    print("Fixed code: exit 50% of REMAINING position for TP1")
    tp1_exit_percent = 0.5  # Exit 50% of remaining
    tp1_exit_size = remaining_size * tp1_exit_percent
    remaining_size -= tp1_exit_size
    exit_count += 1
    print(f"Exit Size: {tp1_exit_size:,} units ({tp1_exit_percent*100}% of remaining)")
    print(f"Remaining After: {remaining_size:,} units")
    print(f"Exit Count: {exit_count}")
    
    # Fixed TP2 logic
    print("\n--- TP2 EXIT (FIXED LOGIC) ---")
    print("Fixed code: exit 100% of REMAINING position for TP2 (or 50% if TP3 exists)")
    tp2_exit_percent = 0.5  # Exit 50% of remaining (assuming TP3 exists)
    tp2_exit_size = remaining_size * tp2_exit_percent
    remaining_size -= tp2_exit_size
    exit_count += 1
    print(f"Exit Size: {tp2_exit_size:,} units ({tp2_exit_percent*100}% of remaining)")
    print(f"Remaining After: {remaining_size:,} units")
    print(f"Exit Count: {exit_count}")
    
    # Fixed TP3 logic
    print("\n--- TP3 EXIT (FIXED LOGIC) ---")
    print("Fixed code: exit ALL remaining position for TP3")
    tp3_exit_size = remaining_size
    remaining_size -= tp3_exit_size
    exit_count += 1
    print(f"Exit Size: {tp3_exit_size:,} units (100% of remaining)")
    print(f"Remaining After: {remaining_size:,} units")
    print(f"Exit Count: {exit_count}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("The bug is that TP exits use 1/3 of ORIGINAL position size")
    print("instead of a percentage of REMAINING position size.")
    print("\nThis causes duplicate markers when the intended exit size")
    print("differs from the actual exit size (limited by remaining position).")

if __name__ == "__main__":
    simulate_exits()