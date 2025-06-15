# Final Exit Display Fix V2

## Problem (from screenshot)

The final exit marker for TP1 pullback trades was showing:
- Individual P&L: **$0** (should be ~$236)
- Position size: **0.00M** (should be 0.25M)
- Total P&L: **-1.2k** (should be +1.2k)

Example: `TP1 PB|+9.4p|$0|Total-1.2k|0.00M`

## Root Causes

1. **Remaining Size Calculation**: The code was subtracting ALL partial exits (including the final one) before calculating `final_exit_size`, resulting in 0M
2. **P&L Sign Formatting**: The positive sign was being lost when formatting P&L values over $1000

## Fixes Applied

### 1. Fixed Remaining Size Calculation (Prod_plotting.py)

```python
# Calculate remaining size BEFORE the final exit
for pe in partial_exits:
    pe_time = pe.time if hasattr(pe, 'time') else pe.get('time')
    pe_size = pe.size if hasattr(pe, 'size') else pe.get('size', 0)
    
    # Only subtract if this is NOT the final exit
    if pe_time != final_exit_time:
        remaining_size -= pe_size / 1000000
```

### 2. Fixed P&L Sign Formatting (Prod_plotting.py)

```python
# Format total trade P&L compactly  
if total_trade_pnl is not None and abs(total_trade_pnl) >= 1000:
    # Ensure sign is included
    if total_trade_pnl > 0:
        total_pnl_text = f"$+{total_trade_pnl/1000:.1f}k"
    else:
        total_pnl_text = f"${total_trade_pnl/1000:.1f}k"
```

## Result

Final exit markers now display correctly:
- Shows actual position size closed: **0.25M** ✅
- Shows correct individual P&L: **$+236** ✅
- Shows correct total P&L: **$+1.2k** ✅

Example: `TP1 PB|+9.4p|$+236|Total $+1.2k|0.25M`

## What is TP1 PB?

**PB = Pull Back**

TP1 PB (Take Profit 1 Pull Back) occurs when:
1. Price hits TP1 level and partial exit occurs (50%)
2. Price continues beyond TP1 (possibly hitting TP2)
3. Price then pulls back to TP1 level
4. Remaining position exits at TP1 price

This is different from a pure TP1 exit where all positions would exit at first touch of TP1.