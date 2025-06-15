# Final Exit Marker Display Fix

## Problem

The final exit marker for trades ending with TP1 pullback was showing incorrect information:
- Displayed as "TP3" instead of "TP1 PB" (TP1 pullback)
- Showed 0M position size and $0 P&L for the exit
- Only the total P&L was correct

Example from March 30 21:15 trade:
- Entry: Short 1M @ 0.62959
- TP1: 0.5M @ +9.4 pips = $472
- TP2: 0.25M @ +18.9 pips = $472
- Final: 0.25M @ +9.4 pips = $236 (TP1 pullback)
- Total: $1,180

## Root Causes

1. **Exit Reason Detection**: The code wasn't properly handling 'tp1_pullback' exit reason
2. **P&L Calculation**: The final exit P&L lookup was failing to find the partial exit record with tp_level=0

## Fixes Applied

### 1. Exit Reason Display (Prod_plotting.py)

Added specific handling for TP1 pullback exits:

```python
elif exit_reason == 'tp1_pullback' or 'tp1_pullback' in exit_reason_str:
    # TP1 pullback is the final exit at TP1 price
    text = f'TP1 PB|{exit_pips:+.1f}p|{individual_pnl_text}|Total {total_pnl_text}|{exit_size_m:.2f}M'
```

### 2. Final Exit P&L Lookup (Prod_plotting.py)

Improved the logic to find the correct partial exit record for final exits:

```python
# For final exits, look for the partial exit with tp_level=0 (non-TP exit)
if is_final_exit and final_exit_time:
    for pe in partial_exits:
        pe_time = pe.time if hasattr(pe, 'time') else pe.get('time')
        if pe_time == final_exit_time:
            pe_tp_level = pe.tp_level if hasattr(pe, 'tp_level') else pe.get('tp_level', -1)
            # For final exits, we want tp_level=0
            if pe_tp_level == 0 or ('take_profit' in str(exit_reason) and pe_tp_level > 0):
                individual_exit_pnl = pe.pnl if hasattr(pe, 'pnl') else pe.get('pnl', 0)
                pe_size = pe.size if hasattr(pe, 'size') else pe.get('size', 0)
                final_exit_size = pe_size / 1000000  # Convert to millions
                break
```

## Expected Result

Final exit markers now display correctly:
- TP1 pullback shows as "TP1 PB" (not TP3)
- Shows actual position size closed (0.25M)
- Shows correct P&L for that exit ($236)
- Shows total trade P&L ($1,180)

Format: `TP1 PB|+9.4p|$236|Total $1180|0.25M`

## Verification

The March 30 21:15 trade now displays:
- TP1: 0.50M @ 0.62865 = $472.19 ✅
- TP2: 0.25M @ 0.62770 = $472.19 ✅
- Final (TP1 PB): 0.25M @ 0.62865 = $236.10 ✅
- Total P&L: $1,180.48 ✅