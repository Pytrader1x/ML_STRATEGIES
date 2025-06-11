# Quick Start Guide

## Run the Strategy

```bash
python robust_sharpe_both_configs_monte_carlo.py
```

## What It Does
- Tests 2 high-performance trading strategies
- Runs 20 Monte Carlo simulations per strategy
- Each simulation uses 5,000 random contiguous data points
- Outputs detailed performance metrics

## Expected Results
- Config 1: ~1.3 Sharpe Ratio, 70% win rate
- Config 2: ~1.5 Sharpe Ratio, 64% win rate (recommended)
- Both achieve 100% profitability across tests

## Output Files
Results are automatically saved to:
- `results/monte_carlo_results_config_1_ultra-tight_risk_management.csv`
- `results/monte_carlo_results_config_2_scalping_strategy.csv`