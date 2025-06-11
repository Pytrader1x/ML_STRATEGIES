# Monte Carlo Update Summary

**Date: 2025-06-11**

## Key Updates Completed

### 1. Enhanced Monte Carlo Script
- **File**: `robust_sharpe_both_configs_monte_carlo.py`
- Added total return tracking per iteration (now shows `Return=XXX%` in output)
- Integrated comprehensive calendar year analysis directly into the script
- Added 4-panel visualization showing:
  - Yearly average Sharpe ratios comparison
  - Return distribution by year
  - Win rate trends over time
  - Consistency heatmap (% iterations with Sharpe > 1.0)

### 2. Calendar Year Analysis Results
- **Config 1 (Ultra-Tight Risk)**: Average Sharpe 1.279
- **Config 2 (Scalping)**: Average Sharpe 1.437 ⭐ Superior
- Both strategies show >95% iterations with Sharpe > 1.0
- Config 2 outperformed in 12 out of 15 years

### 3. Project Organization
- Moved all analysis scripts to `analysis/` directory
- Removed old `readmes/` directory
- Consolidated documentation in `results/`
- Created comprehensive performance reports

### 4. Files Structure
```
Classical_strategies/
├── analysis/              # All analysis and test scripts
├── charts/               # Generated visualizations
├── results/              # All output files and reports
├── strategy_code/        # Core strategy implementation
└── validation/           # Validation scripts and reports
```

## How to Run Updated Monte Carlo

```bash
# Basic run (50 iterations, 60k sample size)
python robust_sharpe_both_configs_monte_carlo.py

# With plots
python robust_sharpe_both_configs_monte_carlo.py --plot

# Save plots to file
python robust_sharpe_both_configs_monte_carlo.py --save-plots
```

## Key Insights from Calendar Year Analysis

1. **Best Years**: 2011 (post-GFC recovery), 2012, 2022
2. **Challenging Years**: 2024 (current conditions), 2018
3. **Consistency**: Both strategies profitable in 100% of iterations
4. **Risk**: Config 2 has lower drawdowns (-2.7% vs -4.4%)

## Git Repository
All changes have been pushed to: https://github.com/Pytrader1x/ML_STRATEGIES.git