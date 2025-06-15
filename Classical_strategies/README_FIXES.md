# Strategy Fixes and Improvements

## Date: 2025-06-15

This document outlines the major fixes and improvements made to the Classical Trading Strategy system.

## 1. Stop Loss Direction Fix for Short Positions

### Problem
Stop losses for SHORT positions were incorrectly placed BELOW the entry price instead of ABOVE, causing immediate exits and losses.

### Root Cause
In `Prod_strategy.py` line 449, the code was using `min()` instead of `max()` when selecting between ATR-based and Market Bias-based stop losses for short positions.

### Fix
```python
# Before (WRONG):
stop_loss = min(atr_stop, mb_stop)

# After (CORRECT):
stop_loss = max(atr_stop, mb_stop)  # For shorts, we want the HIGHER value
```

### Additional Improvement
Added minimum 5-pip stop loss distance requirement to ensure stops are never too close to entry:
```python
min_sl_distance = 5.0 * FOREX_PIP_SIZE  # Minimum 5 pips
sl_distance = max(min(sl_distance, max_sl_distance), min_sl_distance)
```

## 2. TP Exit Size Calculation Fix

### Problem
After partial profit exits, TP exits were calculating sizes based on the ORIGINAL position size instead of REMAINING position size, causing:
- Incorrect exit sizes
- Duplicate exit markers on charts
- Position management errors

### Example
1. Entry: 1M units
2. Partial profit exit: 500k (50%)
3. TP1 wanted to exit 333k (1/3 of original 1M) but only 500k remained
4. TP2 wanted to exit 333k again but only 167k remained → duplicate markers

### Fix
Replaced fixed fraction logic with percentage-based exits from remaining position:

```python
# Before (BUGGY):
desired_exit = trade.position_size / 3.0  # Always 1/3 of original

# After (FIXED):
# Calculate exit percentage based on TP strategy
if total_tps == 3:  # 3 TPs
    if current_tp_level == 1:
        exit_percent = 0.5      # TP1: 50% of remaining
    elif current_tp_level == 2:
        exit_percent = 0.5      # TP2: 50% of remaining
    else:
        exit_percent = 1.0      # TP3: 100% of remaining

exit_size = trade.remaining_size * exit_percent
```

## 3. Partial Profit Exit Visualization

### Problem
Partial profit exits (50% position exit when price reaches 50% of distance to stop loss) were not visible on charts despite being executed.

### Root Cause
The plotting code in `Prod_plotting.py` was filtering out exits with `tp_level == 0` (partial profit exits) and only showing TP exits.

### Fix
```python
# Before:
tp_exits = [p for p in partial_exits 
            if (hasattr(p, 'tp_level') and p.tp_level > 0)]

# After:
# Plot all partial exits (including partial profit and TP exits)
for i, partial_exit in enumerate(partial_exits):
```

### Visual Improvements
- Added gold color (#FFD700) for partial profit markers
- Display format: "PP|+6.7p|$335|0.50M" (type|pips|P&L|size)
- Added "● Partial Profit" to legend

## 4. Interactive Plot Cursor Format

### Problem
X-axis cursor values showed raw numbers instead of readable datetime format.

### Fix
Added custom cursor formatter to display:
- X-axis: Formatted datetime (YYYY-MM-DD HH:MM)
- Y-axis: 4 decimal places for price

```python
def format_coord(x, y):
    if x >= 0 and x < len(df):
        idx = int(round(x))
        date_str = df.index[idx].strftime('%Y-%m-%d %H:%M')
    else:
        date_str = ''
    
    if ax == axes[0]:  # Price axis
        y_str = f'{y:.4f}'
    else:  # Other axes
        y_str = f'{y:,.2f}'
    
    return f'({date_str}, {y_str})'
```

## 5. Partial Profit Statistics Tracking

### Problem
The strategy was not tracking how many trades exited via partial profit (PP) mechanisms, making it difficult to assess this feature's effectiveness.

### Implementation
Added comprehensive partial profit statistics to the strategy metrics:

```python
# In _calculate_performance_metrics():
# Count trades with partial profit exits (tp_level=0)
has_pp_exit = any(
    hasattr(pe, 'tp_level') and pe.tp_level == 0 
    for pe in trade.partial_exits
)
if has_pp_exit:
    pp_stats['pp_trades'] += 1

# Calculate percentage
if len(self.trades) > 0:
    pp_stats['pp_percentage'] = (pp_stats['pp_trades'] / len(self.trades)) * 100
```

### Display Format
The statistics are now displayed in run_strategy_single.py:
```
💰 PARTIAL PROFIT STATISTICS:
  PP Exits: 42 trades (20.2%)
```

## Summary

These fixes ensure:
1. ✅ Correct stop loss placement for both long and short trades
2. ✅ Proper position sizing for all TP exits after partial exits
3. ✅ Complete visualization of all exit types on charts
4. ✅ User-friendly interactive plot experience
5. ✅ Comprehensive tracking of partial profit exit statistics

The strategy now correctly manages positions through complex exit sequences and provides clear visual feedback of all trading actions.