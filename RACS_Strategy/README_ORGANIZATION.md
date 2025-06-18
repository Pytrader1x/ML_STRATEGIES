# RACS Strategy Directory Organization

## Directory Structure

```
RACS_Strategy/
├── backtesting.py          # Core backtesting framework (custom, not backtrader)
├── multi_currency_analyzer.py  # Main analysis script
├── charts/                 # All generated charts and visualizations
├── results/                # CSV results and text reports
├── archive/                # Old versions and test files
├── pine/                   # PineScript related files
└── data/                   # (parent directory) Currency pair data files
```

## Core Files

### backtesting.py
- Object-oriented backtesting framework
- Strategy base classes and implementations
- Trade tracking and metrics calculation
- NOT using backtrader - custom implementation for better control

### multi_currency_analyzer.py
- Main script for running analysis
- Handles multiple currency pairs
- Generates reports and visualizations
- Command-line interface

## Usage

```bash
# Basic usage - analyze all currencies, show AUDUSD plots
python multi_currency_analyzer.py

# Show plots for different currency
python multi_currency_analyzer.py --show-plots GBPUSD

# No plots, just analysis
python multi_currency_analyzer.py --no-plots

# Test on different periods
python multi_currency_analyzer.py --test-period last_50000
python multi_currency_analyzer.py --test-period last_20000
python multi_currency_analyzer.py --test-period full
```

## Output Files

### charts/
- `{CURRENCY}_detailed_analysis.png` - 4-panel analysis charts
  - P&L distribution
  - Holding period distribution
  - Entry hour distribution
  - Cumulative performance

### results/
- `comprehensive_currency_test_results.csv` - Detailed metrics for all currencies
- `comprehensive_currency_test_report.txt` - Summary report with statistics

## Key Metrics Tracked

1. **Performance Metrics**
   - Sharpe Ratio
   - Total Returns
   - Win Rate
   - Maximum Drawdown
   - Number of Trades

2. **Trade Timing Metrics**
   - Average holding period (hours)
   - Median holding period
   - Long vs Short holding periods
   - Entry hour distribution
   - Time-of-day bias analysis

## Strategy Parameters

Current winning parameters (from AUDUSD optimization):
- Lookback: 40 bars
- Entry Z-Score: 1.5
- Exit Z-Score: 0.5