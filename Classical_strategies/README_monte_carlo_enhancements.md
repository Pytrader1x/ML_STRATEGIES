# Monte Carlo Results Enhancement Summary

## Overview
Enhanced the Monte Carlo simulation results display in `run_strategy_oop.py` to provide cleaner, more readable output with detailed tables and yearly performance breakdowns.

## Key Enhancements

### 1. Pretty Table Display
- Added `tabulate` library for professional table formatting
- Created detailed iteration-by-iteration results table showing:
  - Iteration number
  - Sharpe ratio with quality rating (Excellent/Very Good/Good/Moderate/Poor)
  - Return percentage
  - Win rate percentage
  - Trade counts (total and W/L breakdown)
  - Profit factor
  - Average win/loss amounts
  - Maximum drawdown
  - Date period for each sample

### 2. Summary Statistics Table
- Statistical analysis across all iterations:
  - Mean, Median, Std Dev, Min, Max, Range for key metrics
  - Sharpe Ratio, Return %, Win Rate %, Max Drawdown %, Profit Factor
- Distribution analysis showing percentage of iterations in each quality tier

### 3. Yearly Performance Breakdown
- Groups results by calendar year
- Shows aggregated metrics per year:
  - Number of samples in that year
  - Total and average P&L
  - Total trades and average trades per sample
  - Average win rate
  - Total wins/losses ratio

### 4. Visual Improvements
- Progress bar during simulation: `[████████████████████] 5/5 (100%)`
- Quality indicators for Sharpe ratios
- Emoji indicators for sections (📊, 📈, 📅, 🎯, etc.)
- Star ratings for overall performance

### 5. CSV Export
- Automatically saves Monte Carlo results to CSV files
- Filename format: `{currency}_{config}_monte_carlo.csv`
- Includes timestamp, config name, and currency columns

### 6. Condensed Summary
- Performance rating system (⭐ to ⭐⭐⭐⭐⭐)
- Consistency score calculation
- Clear target indicators (e.g., "Target: >1.0" for Sharpe)

## Usage Example
```bash
# Run with table display
python run_strategy_oop.py --iterations 50 --sample-size 8000

# Quick test with 5 iterations
python run_strategy_oop.py --iterations 5 --sample-size 2000 --no-plots
```

## Sample Output Structure
```
📊 DETAILED MONTE CARLO RESULTS - Config 1: Ultra-Tight Risk Management
======================================================================================
| Iter | Sharpe | Quality   | Return% | Win Rate% | Trades | W/L   | PF   | ... |
|======|========|===========|=========|===========|========|=======|======|=====|
| 1    | 3.573  | Excellent | 0.8     | 66.7      | 27     | 17/10 | 1.98 | ... |
| 2    | 2.461  | Excellent | 0.6     | 75.0      | 36     | 27/9  | 1.80 | ... |
...

📈 SUMMARY STATISTICS
================================================================================
Metric          Mean    Median    Std Dev    Min    Max    Range
Sharpe Ratio    2.821   3.316     1.641     -0.086  4.839  4.925
Return %        0.5     0.6       0.3       -0.0    0.8    0.8
...

📅 YEARLY PERFORMANCE BREAKDOWN
========================================================
| Year | Samples | Total P&L | Avg P&L/Sample | ... |
|======|=========|===========|================|=====|
| 2023 | 1       | $-175     | $-175          | ... |
| 2024 | 1       | $6,371    | $6,371         | ... |
...
```

## Benefits
1. **Better Readability**: Tables make it easy to compare iterations at a glance
2. **Comprehensive Analysis**: Summary statistics provide quick insights
3. **Time-based Analysis**: Yearly breakdown helps identify temporal patterns
4. **Professional Presentation**: Clean formatting suitable for reports
5. **Data Export**: CSV files enable further analysis in Excel or other tools