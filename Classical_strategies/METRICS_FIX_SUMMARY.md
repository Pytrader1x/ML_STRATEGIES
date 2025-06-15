# Performance Metrics Fix Summary

## Issue
Several performance metrics were displaying as 0.00 in the validated strategy runner when they should have had actual values.

## Root Cause
The missing metrics were not being calculated in the `_calculate_performance_metrics()` method in `Prod_strategy.py`. The method was only calculating basic metrics but missing several important ones that were being displayed in the results output.

## Metrics Added
The following metrics were added to the calculation:

1. **Sortino Ratio** - Downside deviation-based risk metric
2. **Average Trade** - Simple average P&L per trade
3. **Win/Loss Ratio** - Ratio of average win to average loss
4. **Expectancy** - Expected value per trade
5. **Best Trade** - Largest winning trade
6. **Worst Trade** - Largest losing trade
7. **SQN (System Quality Number)** - Van Tharp's system quality metric
8. **Trades per Day** - Average number of trades per trading day
9. **Recovery Factor** - Total return divided by max drawdown

## Implementation Details

### Sortino Ratio
- Uses downside returns only (negative returns) for volatility calculation
- Supports both daily and bar-level calculations
- Falls back to Sharpe ratio if no downside volatility exists

### SQN Score
- Calculated as: (mean trade P&L / std dev of trades) * sqrt(number of trades)
- Provides a normalized measure of system quality

### Trades per Day
- Calculates based on actual trading days in the data
- If less than a day of data, extrapolates from hourly rate

### Win/Loss Ratio
- Handles edge cases (no losses, zero average loss)
- Returns infinity if only winning trades exist

## Verification
All metrics now display correctly in the strategy output:
- Sortino Ratio: -3.731 (calculated from downside deviation)
- Average Trade: $-43.52 (total P&L / number of trades)
- Win/Loss Ratio: 0.41 (average win / average loss)
- Expectancy: $-43.52 (probability-weighted expected value)
- Best Trade: $815.19
- Worst Trade: $-1,144.49
- SQN Score: -0.78 (system quality metric)
- Trades per Day: 7.8
- Recovery Factor: 0.41 (return / drawdown)

## Files Modified
- `strategy_code/Prod_strategy.py`:
  - Updated `_calculate_performance_metrics()` method
  - Updated `_empty_metrics()` method to include new metrics

The fix ensures all performance metrics are properly calculated and displayed, providing a complete picture of strategy performance.