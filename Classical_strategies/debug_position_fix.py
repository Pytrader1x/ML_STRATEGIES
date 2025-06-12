"""
Debug and fix position sizing calculation
"""

# Manual calculation
initial_capital = 100_000
risk_per_trade = 0.02
entry_price = 0.75
stop_loss = 0.748

sl_distance_pips = abs(entry_price - stop_loss) / 0.0001
risk_amount = initial_capital * risk_per_trade

print(f"Stop loss distance: {sl_distance_pips} pips")
print(f"Risk amount: ${risk_amount}")

# Formula: position_size = (risk_amount * min_lot_size) / (sl_distance_pips * pip_value_per_million)
min_lot_size = 1_000_000
pip_value_per_million = 100

calculated_position_size = (risk_amount * min_lot_size) / (sl_distance_pips * pip_value_per_million)
print(f"Calculated position size: {calculated_position_size}")

# This should be about 1M units for a 20 pip stop with 2% risk
# Let's verify: 20 pips * $100 per pip per 1M units = $2000 risk ✓

# The issue is in the rounding logic
position_size_millions = calculated_position_size / min_lot_size
print(f"Position size in millions: {position_size_millions}")

# Round to 0.1M
rounded_millions = round(position_size_millions, 1)
print(f"Rounded to 0.1M: {rounded_millions}")

final_position_size = rounded_millions * min_lot_size
print(f"Final position size: {final_position_size}")

# Verify final risk
final_risk = (sl_distance_pips * pip_value_per_million * final_position_size / min_lot_size)
final_risk_pct = final_risk / initial_capital * 100
print(f"Final risk: ${final_risk} ({final_risk_pct:.2f}%)")

# This should be very close to 2%