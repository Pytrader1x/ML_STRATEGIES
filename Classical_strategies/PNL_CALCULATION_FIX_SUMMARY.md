# P&L Calculation Fix Summary

## Issues Identified

### 1. Partial Exit Size Calculation (CRITICAL)
**Problem**: Partial exits were calculated as a percentage of the ORIGINAL position size, not the REMAINING size.

**Example**:
- Start with 1.0M position
- Exit 0.333M at TP1 (leaving 0.667M)
- Exit 50% → Old logic: 0.5M of original = 0.5M (leaving 0.167M) ❌
- Exit 50% → New logic: 50% of remaining 0.667M = 0.333M (leaving 0.333M) ✅

**Fix**: Changed `exit_size = trade.position_size * exit_percent` to `exit_size = trade.remaining_size * exit_percent`

### 2. P&L Display Calculation in Plotting
**Problem**: P&L calculation in plotting was multiplying by 1,000,000 unnecessarily.

**Fix**: Corrected formula to: `pip_change * remaining_size * 100` where:
- `pip_change` = price difference in pips
- `remaining_size` = position size in millions
- `100` = pip value per million for AUDUSD

## P&L Calculation Formula

For AUDUSD:
- **1 pip = 0.0001 price movement**
- **1M position, 1 pip = $100**
- **Formula**: P&L = Position_Size_In_Millions × Pips × $100

### Examples:
- 1.0M × 10 pips × $100 = $1,000
- 0.5M × 10 pips × $100 = $500
- 0.333M × 6 pips × $100 = $200
- 0.17M × 1 pip × $100 = $17

## Files Modified

1. **strategy_code/Prod_strategy.py**
   - Fixed `_execute_partial_exit()` to use remaining size for percentage calculations

2. **strategy_code/Prod_plotting.py**
   - Fixed P&L calculation for remaining positions in exit annotations
   - Ensured TSL and SL exits show correct P&L

## Impact

These fixes ensure:
1. **Position sizes add up correctly**: Entry size = Sum of all exit sizes
2. **P&L calculations are accurate**: Based on actual position sizes exited
3. **Chart annotations are consistent**: Entry and exit sizes match reality
4. **No over-exiting**: Can't exit more than the remaining position

## Verification

Before fix:
- 1M entry → 0.33M (TP1) → 0.50M (partial) → 0.17M (final) = 1.0M ✅ but wrong distribution

After fix:
- 1M entry → 0.33M (TP1) → 0.33M (partial) → 0.33M (final) = 1.0M ✅ correct distribution