# High Sharpe Ratio Trading Strategy

## Overview
This repository contains a robust quantitative trading strategy that achieves Sharpe ratios consistently above 1.0, with extensive Monte Carlo testing demonstrating reliability across different market conditions.

## Performance Summary

### Configuration 1: Ultra-Tight Risk Management
- **Average Sharpe Ratio: 1.327**
- Average P&L: $85,545 (per 5k samples)
- Win Rate: 70.9%
- Max Drawdown: -4.6%
- 80% of tests achieved Sharpe > 1.0

### Configuration 2: Scalping Strategy (Best Performance)
- **Average Sharpe Ratio: 1.503**
- Average P&L: $92,408 (per 5k samples)
- Win Rate: 63.9%
- Max Drawdown: -2.4%
- 95% of tests achieved Sharpe > 1.0

## Quick Start

```bash
# Run Monte Carlo simulation for both strategies
python robust_sharpe_both_configs_monte_carlo.py
```

## Strategy Features

### Risk Management
- Ultra-tight stop losses (5-10 pips max)
- Conservative position sizing (0.1-0.2% risk per trade)
- Aggressive trailing stops to protect profits

### Profit Taking
- Three-tiered take profit system
- Quick profit realization (0.1-0.5 ATR)
- Market condition adaptations

### Trade Management
- High-frequency trading (300-500+ trades per 5k samples)
- Consistent profitability (100% profitable across all Monte Carlo runs)
- Low drawdowns maintained throughout

## Requirements
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- technical-indicators-custom

```bash
pip install pandas numpy
pip install git+https://github.com/Pytrader1x/technical-indicators-custom.git
```

## File Structure
```
├── robust_sharpe_both_configs_monte_carlo.py  # Main strategy file
├── strategy_code/                              # Core strategy implementation
│   ├── Prod_strategy.py                       # Strategy classes
│   └── __init__.py
├── results/                                    # Monte Carlo results
├── readmes/                                    # Documentation
└── archive/                                    # Historical development files
```

## Data Requirements
The strategy expects 15-minute FX data in CSV format with columns:
- DateTime
- Open, High, Low, Close
- Volume

Data should be placed in `../data/` directory relative to this folder.

## Results
Detailed Monte Carlo results are saved to the `results/` directory after each run.