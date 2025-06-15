# Monte Carlo Implementation Summary

## Overview
Added Monte Carlo simulation capability to `run_validated_strategy.py` that randomly samples contiguous data segments for robust strategy validation.

## Usage
```bash
# Run with 25 random samples and 2M position size
python run_validated_strategy.py --position-size 2 --monte-carlo 25

# Quick test with 10 samples
python run_validated_strategy.py --position-size 2 --monte-carlo 10

# Thorough test with 50 samples
python run_validated_strategy.py --position-size 2 --monte-carlo 50
```

## Implementation Details

### 1. Command Line Argument
- Added `--monte-carlo N` argument to specify number of simulations
- Works with all existing arguments like `--position-size`

### 2. Monte Carlo Method
```python
def run_monte_carlo(self, n_simulations=25, sample_size_days=90):
```
- Default: 25 simulations with 90-day samples
- Randomly selects contiguous data segments
- Uses fixed random seed (42) for reproducibility

### 3. Sample Selection
- Each sample is 90 days (8,640 bars at 15-minute intervals)
- Ensures sufficient buffer for indicator calculation
- Samples can overlap but starting points are random

### 4. Output for Each Simulation
```
Simulation 1/25: 2014-12-03 to 2015-04-10
  Sharpe:   0.80 | Return:   5.2% | P&L: $  10,450 | Trades:  725 | Win Rate:  68.5%
```

### 5. Comprehensive Analysis
After all simulations, provides:

#### Performance Statistics Table
```
Metric               Mean    Std Dev         Min         Max      Median
Sharpe Ratio         0.85       0.45       -0.20        2.10        0.75
Return %             4.50       3.20       -2.10       12.50        4.20
Annual Return %     18.25      12.96       -8.52       50.68       17.03
P&L ($)          9,000.00    6,400.00   -4,200.00   25,000.00    8,400.00
Annual P&L ($)  36,500.00   25,947.00  -17,031.00  101,370.00   34,051.00
```

#### Distribution Analysis
- Profitable Samples (Sharpe > 0): X%
- Good Performance (Sharpe > 0.7): X%
- Excellent (Sharpe > 1.0): X%
- Exceptional (Sharpe > 2.0): X%

#### Consistency Metrics
- Coefficient of Variation
- Success Rate (positive returns)
- Risk-Adjusted Annual Return

#### Time Period Coverage
- Shows date range of all samples
- Ensures broad market condition coverage

#### Overall Assessment
- Average Sharpe Ratio
- Expected Annual Return
- Expected Annual P&L
- Strategy Rating (EXCELLENT/VERY GOOD/GOOD/etc.)

### 6. Results Storage
- Saves detailed CSV file: `results/monte_carlo_25_samples_2M_YYYYMMDD_HHMMSS.csv`
- Contains all metrics for each simulation
- Useful for further analysis

## Benefits

1. **Robustness Testing**: Validates strategy across different market conditions
2. **Statistical Significance**: Multiple samples provide confidence intervals
3. **Yearly Projections**: Scales 90-day results to annual expectations
4. **Risk Assessment**: Identifies worst-case scenarios
5. **Consistency Check**: Measures performance variability

## Example Output
```
🏆 OVERALL ASSESSMENT:
================================================================================
Average Sharpe Ratio: 0.850
Expected Annual Return: 18.3%
Expected Annual P&L: $36,500
Position Size: 2.0M AUD
Strategy Rating: GOOD - Meets minimum requirements
```

## Integration with Existing Features
- Works with `--position-size` (1 or 2)
- Compatible with all strategy parameters
- Uses same realistic execution settings
- Maintains position sizing logic

## Next Steps
1. Run full 25-sample Monte Carlo for production validation
2. Compare results between 1M and 2M position sizes
3. Use results to set realistic performance expectations
4. Monitor live performance against Monte Carlo projections