"""
Debug position sizing calculation
"""

from strategy_code.Prod_strategy import OptimizedStrategyConfig, RiskManager

# Create config
config = OptimizedStrategyConfig(
    initial_capital=100_000,
    risk_per_trade=0.02,
    min_lot_size=1_000_000,
    pip_value_per_million=100
)

# Create risk manager
risk_manager = RiskManager(config)

# Test calculation
entry_price = 0.75000
stop_loss = 0.74800  # 20 pip stop loss
current_capital = 100_000

print("DEBUG POSITION SIZING:")
print(f"Entry price: {entry_price}")
print(f"Stop loss: {stop_loss}")
print(f"Current capital: ${current_capital}")
print(f"Risk per trade: {config.risk_per_trade}")

# Manual calculation
sl_distance_pips = abs(entry_price - stop_loss) / 0.0001
risk_amount = current_capital * config.risk_per_trade
print(f"Stop loss distance: {sl_distance_pips} pips")
print(f"Risk amount: ${risk_amount}")

base_position_size = (risk_amount * config.min_lot_size) / (sl_distance_pips * config.pip_value_per_million)
print(f"Base position size: {base_position_size}")

# Test the actual function
try:
    position_size = risk_manager.calculate_position_size(
        entry_price, stop_loss, current_capital
    )
    print(f"Calculated position size: {position_size}")
    
    # Verify risk
    actual_risk = (sl_distance_pips * config.pip_value_per_million * position_size / config.min_lot_size)
    print(f"Actual risk: ${actual_risk}")
    print(f"Risk as % of capital: {actual_risk / current_capital * 100:.2f}%")
    
except Exception as e:
    print(f"Error: {e}")