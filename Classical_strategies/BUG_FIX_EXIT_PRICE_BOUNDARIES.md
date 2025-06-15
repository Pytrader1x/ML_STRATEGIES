# Bug Fix: Exit Prices Exceeding Candle Boundaries

## Issue Description
Exit points (X markers) were being plotted above/below the actual price candles in the chart, which is impossible in real trading. This occurred specifically with stop loss and trailing stop exits when slippage was applied.

## Root Cause
The bug was in the `_get_exit_price` method in `Prod_strategy.py`:

1. When `intrabar_stop_on_touch=True`, the strategy correctly detects stop losses when the candle's Low (for long trades) or High (for short trades) touches the stop level.

2. However, when determining the exit price, it used the stop loss price and applied slippage, which could push the exit price beyond the candle's boundaries:
   - For **short trades**: When High touches stop loss, slippage was added (making price even higher)
   - For **long trades**: When Low touches stop loss, slippage was subtracted (making price even lower)

3. This resulted in exit prices that were impossible in real trading (e.g., exiting above the High of a candle).

## Solution Implemented

### 1. Primary Fix
Added boundary checking in `_get_exit_price` method to ensure exit prices respect candle limits:

```python
# For stop losses and trailing stops triggered by intrabar touch,
# the exit price cannot exceed the candle's High/Low
if self.config.intrabar_stop_on_touch and exit_reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
    if trade.direction == TradeDirection.LONG:
        # For long trades, stop loss exit cannot be lower than the Low
        exit_price = max(exit_price, row['Low'])
    else:
        # For short trades, stop loss exit cannot be higher than the High
        exit_price = min(exit_price, row['High'])
```

### 2. Universal Safety Check
Added an additional safety check to ensure ALL exit types stay within candle boundaries:

```python
# Additional safety: Ensure all exits are within the candle's range
# This prevents any exit from being plotted outside the candle
exit_price = max(min(exit_price, row['High']), row['Low'])
```

### 3. Debug Logging
Added debug logging to track when prices are adjusted:

```python
if self.config.debug_decisions and original_exit_price != exit_price:
    print(f"  ⚠️  EXIT PRICE ADJUSTED to respect candle boundaries:")
    print(f"     Original exit price: {original_exit_price:.5f}")
    print(f"     Adjusted exit price: {exit_price:.5f}")
    print(f"     Candle High: {row['High']:.5f}, Low: {row['Low']:.5f}")
```

## Testing
Created `test_exit_price_fix.py` to verify the fix:
- Generates synthetic data with scenarios where stop losses would be hit
- Runs backtest with debug mode enabled
- Verifies all exit prices are within candle boundaries
- Test result: ✅ SUCCESS - All exits now respect candle boundaries

## Impact
- **Visual**: Exit markers (X) now always appear within the candle range on charts
- **Realism**: Exit prices now accurately reflect what would happen in real trading
- **Slippage**: Still applied but constrained to realistic values within the candle

## Files Modified
1. `strategy_code/Prod_strategy.py` - Added boundary checking in `_get_exit_price` method
2. `test_exit_price_fix.py` - Created test script to verify the fix

## Verification
To verify the fix is working:
1. Run any backtest with `debug_decisions=True`
2. Look for "EXIT PRICE ADJUSTED" messages in the output
3. Check charts to ensure all X markers are within candle boundaries