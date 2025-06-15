# Classical Trading Strategies

A production-ready trading strategy implementation with comprehensive backtesting, plotting, and Monte Carlo simulation capabilities.

## Recent Fixes & Improvements (June 15, 2025)

### 1. TP2 Marker Visibility Fix
- **Issue**: TP2 markers were not displaying when TP2 and TP3 exits occurred at the same timestamp
- **Root Cause**: Duplicate detection logic was incorrectly skipping all partial exits at the final exit time
- **Solution**: Modified logic to only skip partial exits that match the final TP level number

### 2. P&L Recording & Integrity
- **Issue**: Stop loss exits weren't creating PartialExit records, causing position/P&L tracking mismatches
- **Solution**: Added PartialExit record creation for ALL exit types (not just TPs) with tp_level=0 for non-TP exits
- **Validation**: Created comprehensive validation scripts confirming no double counting across 206 trades

### 3. Final Exit Marker Display
- **Issues**: 
  - Showed 0M position and $0 P&L for final exits
  - Total P&L displayed negative values incorrectly
  - TP1 pullback exits showed as generic exits
- **Solutions**:
  - Fixed remaining size calculation to exclude the final exit itself
  - Added explicit positive sign formatting for P&L values over $1000
  - Added proper exit reason detection for enums (ExitReason.TP1_PULLBACK)

### 4. Total P&L Display Accuracy
- **Issue**: Total P&L showing incorrect values (e.g., +1.4k instead of +1.2k)
- **Root Cause**: Missing 'pnl' field when converting Trade objects to dictionaries for plotting
- **Solution**: Added 'pnl' and 'position_size' fields to trade dictionary conversion

### 5. TP1 Pullback Logic Fix
- **Issue**: TP1 pullback could trigger in the same candle as a TP2 exit when candle high reached TP2 and low pulled back to TP1
- **Solution**: Modified check_exit_conditions to track exits within the current candle and prevent TP1 pullback if any TP exit already occurred

### 6. Intrabar Stop-Loss Feature (NEW)
- **Feature**: Added optional intrabar stop-loss checking
- **Configuration**: New `intrabar_stop_on_touch` parameter (default: False)
- **Behavior**: 
  - When False (default): Stop loss only triggers if candle closes beyond stop level
  - When True: Stop loss triggers if candle high/low touches stop level
- **Benefits**: More realistic stop-loss execution for strategies that use hard stops

## Project Structure

```
Classical_strategies/
├── strategy_code/
│   ├── Prod_strategy.py          # Main strategy implementation
│   ├── Prod_plotting.py          # Advanced plotting with trade markers
│   └── Prod_strategy_multi_tp.py # Multi-TP variant
├── run_strategy_single.py        # Single backtest runner
├── run_strategy_oop.py          # OOP-style runner with Monte Carlo
├── run_Strategy.py              # Original strategy runner
├── results/                     # Backtest results and trade details
├── charts/                      # Generated charts and visualizations
└── archive/                     # Archived code and analysis
```

## Key Features

- **Advanced Entry Logic**: Three-confluence system (NeuroTrend + Market Bias + IC Regime)
- **Multi-Level Take Profits**: Dynamic TP levels with partial exits (33% at each TP)
- **Intelligent Stop Loss**: ATR-based with market regime adjustments
- **Trailing Stop Loss**: Adaptive trailing with guaranteed minimum profit
- **TP1 Pullback Logic**: Captures additional profits when price returns to TP1 after hitting TP2
- **Position Tracking**: Accurate partial exit recording with P&L calculations
- **Monte Carlo Simulation**: Statistical validation with multiple random runs
- **Production Plotting**: Professional charts with trade markers and P&L annotations

## Usage

### Single Backtest
```bash
python run_strategy_single.py
```

### Monte Carlo Simulation
```bash
python run_strategy_oop.py
```

### Custom Configuration
```python
strategy_config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.2, 0.3, 0.5),
    tsl_activation_pips=15,
    tsl_min_profit_pips=1,
    intrabar_stop_on_touch=False  # Set to True for intrabar stop-loss
)
```

## Performance Metrics

The strategy tracks comprehensive metrics including:
- Total Return & CAGR
- Sharpe Ratio (daily & annualized)
- Maximum Drawdown
- Win Rate & Profit Factor
- Average Trade Duration
- Risk-Reward Ratios

## Validation & Testing

All fixes have been thoroughly validated:
- P&L calculations verified across 206 trades
- No double counting or metric inflation
- Proper intra-candle price movement handling
- Accurate position size tracking through all exits

## Dependencies

- pandas
- numpy
- matplotlib
- technical_indicators_custom (TIC)
- pathlib
- datetime