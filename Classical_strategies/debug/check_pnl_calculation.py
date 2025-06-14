"""Debug script to check P&L calculation discrepancy"""

# Test the P&L calculation
# From the image: TP1 exit shows 11.0 pips, 0.3M size, but $368 P&L

# Expected calculation:
position_size_m = 0.3  # 0.3M
pips = 11.0
pip_value_per_million = 100  # $100 per pip per million

expected_pnl = position_size_m * pip_value_per_million * pips
print(f"Expected P&L for {position_size_m}M at {pips} pips: ${expected_pnl:.2f}")

# Reverse calculation from displayed P&L
displayed_pnl = 368
actual_pips = displayed_pnl / (position_size_m * pip_value_per_million)
print(f"Actual pips from ${displayed_pnl} P&L: {actual_pips:.2f}")

# Check if this is a display issue
print(f"\nIf pips were {actual_pips:.2f}, formatted as '.1f' it would show: {actual_pips:.1f}p")

# The issue might be that:
# 1. The partial exit size shown (0.3M) is rounded
# 2. The pips shown (11.0) might be calculated differently

# Let's check what size would give us $368 at 11.0 pips
size_for_368 = displayed_pnl / (pip_value_per_million * pips)
print(f"\nSize needed for ${displayed_pnl} at {pips} pips: {size_for_368:.3f}M")

# Check if standard position sizing could explain this
standard_position = 1.0  # 1M standard
one_third = standard_position / 3
print(f"\nStandard 1/3 position: {one_third:.6f}M")
print(f"If position was actually {one_third:.6f}M:")
print(f"  P&L at 11.0 pips: ${one_third * pip_value_per_million * 11.0:.2f}")

# Another possibility: the position size might not be exactly 1M
# If original position was larger
for orig_size in [1.0, 1.1, 1.2]:
    third = orig_size / 3
    pnl_11_pips = third * pip_value_per_million * 11.0
    print(f"\nIf original position was {orig_size}M:")
    print(f"  1/3 = {third:.3f}M")
    print(f"  P&L at 11.0 pips = ${pnl_11_pips:.2f}")