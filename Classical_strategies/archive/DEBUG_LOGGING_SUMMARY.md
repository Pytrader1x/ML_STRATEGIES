# Trade Debug Logging System

## Overview
A comprehensive debugging system has been implemented to track all trade lifecycle events and validate trade consistency.

## Components

### 1. `debug_trade_logger.py`
- **TradeDebugEntry**: Dataclass that tracks complete trade lifecycle including:
  - Entry details (price, size, direction, confidence, logic)
  - Risk management setup (SL, TP levels)
  - Partial exit tracking (PPT, TP1, TP2, TP3)
  - Final exit details
  - Trade validation
  
- **TradeDebugLogger**: Main logging class that:
  - Creates trade entries
  - Updates partial exits (PPT, TP hits)
  - Records final exits
  - Validates trade consistency
  - Saves to JSON with summary statistics

### 2. `run_strategy_with_debug.py`
- Enhanced strategy runner with integrated debug logging
- **DebugEnabledStrategy**: Extends OptimizedProdStrategy with:
  - Debug logging on trade entry
  - Partial exit tracking
  - Final exit logging
  - Maximum excursion tracking
  
### 3. `analyze_trades_debug.py`
- Analyzes existing CSV trade results
- Creates debug JSON from historical trades
- Validates trade consistency
- Generates summary statistics

## Debug JSON Structure

```json
{
  "timestamp": "2025-06-14T22:41:20",
  "source_file": "trade_results.csv",
  "summary": {
    "total_trades": 73,
    "valid_trades": 73,
    "invalid_trades": 0,
    "total_pnl": 6915.22,
    "winning_trades": 45,
    "losing_trades": 28,
    "tp1_hits": 40,
    "tp2_hits": 2,
    "tp3_hits": 0,
    "ppt_triggers": 41
  },
  "validation_errors": [],
  "trades": [
    {
      "trade_id": 1,
      "entry_time": "2023-07-30 22:00:00",
      "entry_price": 0.667152,
      "direction": "long",
      "initial_size_millions": 1.0,
      "confidence": 40.63,
      "entry_logic": "Standard (NTI+MB+IC)",
      
      // Risk management
      "sl_price": 0.666652,
      "sl_distance_pips": 5.0,
      "tp1_price": 0.667752,
      "tp2_price": 0.668352,
      "tp3_price": 0.669153,
      
      // Partial exits
      "ppt_active": true,
      "ppt_size_closed_millions": 0.7,
      "ppt_pnl_dollars": 271.90,
      "position_after_ppt_millions": 0.3,
      
      "tp1_hit": true,
      "tp1_size_closed_millions": 0.3,
      "tp1_pnl_dollars": 180.13,
      "position_after_tp1_millions": 0.0,
      
      // Final exit
      "exit_time": "2023-07-30 22:15:00",
      "exit_price": 0.667752,
      "exit_reason": "take_profit_1",
      "final_position_size_millions": 0.0,
      "total_pnl_dollars": 452.03,
      
      "validation_errors": [],
      "is_valid": true
    }
  ]
}
```

## Validation Checks

The system performs several validation checks:

1. **Size Tracking**: Ensures all partial exits + final position = initial size
2. **P&L Components**: Verifies sum of partial P&Ls + final P&L = total P&L
3. **TP Sequence**: Validates TP2 only hit after TP1, TP3 only after TP2
4. **Exit Logic**: Ensures size is closed when TP is marked as hit

## Usage

### Analyze Existing Results
```bash
python analyze_trades_debug.py results/AUDUSD_config_2_scalping_strategy_trades_detail.csv
```

### Run Strategy with Debug Logging
```bash
python run_strategy_with_debug.py --config scalping --currency AUDUSD --start 2023-07-30 --end 2023-08-07
```

## Key Insights from Analysis

From the scalping strategy analysis:
- **Total trades**: 73
- **Win rate**: 61.6%
- **TP1 hit rate**: 54.8% 
- **TP2 hit rate**: 2.7%
- **TP3 hit rate**: 0.0%
- **PPT trigger rate**: 56.2%

All trades passed validation, indicating the strategy is correctly tracking position sizes and P&L through all partial exits.

## Benefits

1. **Complete Trade Tracking**: Every aspect of the trade lifecycle is recorded
2. **Automatic Validation**: Catches inconsistencies in size/P&L tracking
3. **JSON Format**: Easy to analyze programmatically
4. **Summary Statistics**: Quick overview of strategy performance
5. **Debug Integration**: Can be integrated into live strategy execution