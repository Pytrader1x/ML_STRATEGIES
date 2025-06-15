# 🚀 Quick Start Guide - Validated Strategy

## How to Run and Test the Strategy

### Basic Usage (No Plots)
```bash
# Test recent performance (2024 H1) - default
python run_validated_strategy.py

# Test specific years
python run_validated_strategy.py --period 2023
python run_validated_strategy.py --period 2024

# Test custom date range
python run_validated_strategy.py --start-date 2023-06-01 --end-date 2023-12-31
```

### With Charts/Plots
```bash
# Show interactive charts
python run_validated_strategy.py --show-plots

# Save charts to PNG files (charts/ directory)
python run_validated_strategy.py --save-plots

# Both show AND save
python run_validated_strategy.py --show-plots --save-plots

# Test with specific period and charts
python run_validated_strategy.py --period 2023 --show-plots --save-plots
```

### Advanced Options
```bash
# Different currency (if you have other data files)
python run_validated_strategy.py --currency GBPUSD --show-plots

# Different capital amount
python run_validated_strategy.py --capital 100000 --show-plots

# Recent quarter with charts
python run_validated_strategy.py --period recent --show-plots
```

## What You'll See

### 1. Performance Metrics
- **Sharpe Ratio** (target: >0.7, excellent: >2.0)
- **Total Return** (%)
- **Win Rate** (typically 65-75%)
- **Max Drawdown** (typically <2%)
- **Trade Statistics** (trades per day, profit factor, etc.)

### 2. Exit Analysis
- Exit reason breakdown (signal_flip, take_profit_1, etc.)
- Take profit hit statistics (TP1, TP2, TP3)
- Detailed trading statistics

### 3. Interactive Charts (with --show-plots)
- **Price action** with candlesticks
- **Entry/exit points** marked
- **Take profit levels** displayed
- **P&L curve** overlay
- **Indicator signals** (NTI, Market Bias, IC)

### 4. Saved Files
- **Charts**: `charts/validated_strategy_AUDUSD_*.png`
- **Config**: `validated_strategy_config.json` (for live trading)

## Expected Results (Based on Validation)

For AUDUSD on most periods:
- **Sharpe Ratio**: 4.0 - 7.0 (exceptional)
- **Win Rate**: 65% - 75%
- **Return**: 15% - 40% per period
- **Max Drawdown**: <2%
- **Trades/Day**: 10-15

## Quick Test Commands

```bash
# Quick test with charts (recommended first run)
python run_validated_strategy.py --period recent --show-plots

# Full year test with saved charts
python run_validated_strategy.py --period 2023 --save-plots

# Multiple tests
python run_validated_strategy.py --period 2023 --show-plots
python run_validated_strategy.py --period 2024 --show-plots
```

### Monte Carlo Analysis
```bash
# Run Monte Carlo with 25 random samples and 2M position size
python run_validated_strategy.py --position-size 2 --monte-carlo 25

# Run with different sample counts
python run_validated_strategy.py --position-size 2 --monte-carlo 10   # Quick test
python run_validated_strategy.py --position-size 2 --monte-carlo 50   # Thorough test

# Monte Carlo with 1M position size
python run_validated_strategy.py --position-size 1 --monte-carlo 25

# Monte Carlo with plots (shows chart for last simulation)
python run_validated_strategy.py --position-size 2 --monte-carlo 25 --show-plots

# Monte Carlo saving plot of last simulation
python run_validated_strategy.py --position-size 2 --monte-carlo 25 --save-plots
```

Monte Carlo mode will:
- Randomly select 25 contiguous 90-day samples from historical data
- Run backtests on each sample
- Show summary statistics for each run
- Calculate yearly averages and performance metrics
- Save detailed results to CSV file
- With --show-plots: Display chart for the last simulation showing all trades
- With --save-plots: Save the chart for the last simulation

## File Structure After Running

```
Classical_strategies/
├── run_validated_strategy.py         # Main runner
├── validated_strategy_config.json    # Generated config
├── charts/                          # Generated charts (if --save-plots)
│   └── validated_strategy_AUDUSD_*.png
└── results/                         # Other result files
```

## Troubleshooting

### No Charts Showing
- Add `--show-plots` flag
- Check if matplotlib backend supports GUI
- Try `--save-plots` to save to files instead

### Slow Performance
- Use smaller date ranges
- The first run calculates indicators (slower)
- Subsequent runs are faster

### Data Issues
- Ensure `../data/AUDUSD_MASTER_15M.csv` exists
- Check data path in error messages

### Memory Issues
- Use shorter test periods
- Close previous chart windows
- Use `--save-plots` instead of `--show-plots`

## Live Trading Setup

After successful backtests:
1. Use generated `validated_strategy_config.json`
2. Implement in MT4/MT5 or your trading platform
3. Start with minimum position sizes
4. Monitor performance vs backtest expectations

Remember: **Backtest performance ≠ Live performance**