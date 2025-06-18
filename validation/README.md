# Strategy Validation Framework

A comprehensive validation framework for quantitative trading strategies, implementing institutional-grade testing standards.

## Overview

This framework provides a complete validation pipeline including:

1. **Data Integrity Checks** - QC for missing bars, duplicates, spikes
2. **Walk-Forward Validation** - Rolling window out-of-sample testing
3. **Robustness Analysis** - Parameter sensitivity, noise injection, execution delays
4. **Capacity Analysis** - Market impact modeling and scalability limits
5. **Performance Benchmarking** - Ensure backtesting meets speed requirements
6. **Automated Reporting** - PDF/HTML reports with all results

## Quick Start

```bash
# Run complete validation
python run_validation.py

# Quick validation (3 pairs only)
python run_validation.py --quick

# Test specific pairs
python run_validation.py --pairs AUDUSD EURUSD GBPUSD

# Skip certain steps
python run_validation.py --skip-qc --skip-benchmarks
```

## Directory Structure

```
validation/
├── quantlab/                 # Core backtesting library
│   ├── signals.py           # Signal generation (momentum, MA crossover)
│   ├── portfolio.py         # Backtesting engine with costs
│   └── costs.py             # FX transaction cost models
├── data_integrity/          # Data quality control
│   └── data_qc.py          # QC checks and data cleaning
├── single_pair_validation/  # Walk-forward analysis
│   └── walk_forward.py     # WF validation with Optuna
├── robustness/             # Robustness testing
│   └── sensitivity_analysis.py
├── capacity/               # Capacity & impact analysis
│   └── impact_analysis.py
├── benchmarks/             # Performance benchmarking
│   └── performance_benchmark.py
├── reporting/              # Report generation
│   └── report_generator.py
└── run_validation.py       # Main orchestrator
```

## Validation Pipeline

### 1. Data Quality Control

```python
from data_integrity.data_qc import DataQualityChecker

qc = DataQualityChecker()
results = qc.check_data_integrity(df, 'EURUSD')
```

Checks:
- Missing bars (excluding weekends)
- Duplicate timestamps
- Price spikes (> 8σ moves)
- Spread sanity checks
- OHLC consistency
- UTC timezone compliance

### 2. Walk-Forward Validation

```python
from single_pair_validation.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(data, 'EURUSD')
results = validator.run_walk_forward(optimize=True, n_trials=100)
```

Features:
- 3-year train → 1-year test windows
- Optuna hyperparameter optimization
- Purged K-fold with embargo
- Statistical significance tests (Ledoit-Wolf adjusted)

### 3. Robustness Analysis

```python
from robustness.sensitivity_analysis import RobustnessAnalyzer

analyzer = RobustnessAnalyzer(data, 'EURUSD')
results = analyzer.create_robustness_report(output_dir)
```

Tests:
- Parameter heatmaps (±50% variation)
- Bootstrap confidence intervals
- Trade execution delays (1-3 bars)
- Price noise injection (0.5-2 pips)
- Crisis period performance

### 4. Capacity Analysis

```python
from capacity.impact_analysis import CapacityAnalyzer

analyzer = CapacityAnalyzer(data, 'EURUSD')
results = analyzer.create_capacity_report(output_dir)
```

Analysis:
- Slippage curves vs trade size
- Maximum viable size for Sharpe ≥ 1.0
- Session liquidity analysis
- Trade clustering impact

### 5. Performance Benchmarks

```python
from benchmarks.performance_benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.create_benchmark_report(output_dir)
```

Targets:
- 5 years of 1-minute data < 120 seconds
- Signal generation > 1M bars/second
- Memory usage < 500MB per million bars
- Parallel efficiency > 70%

## Output Reports

### PDF Report Contents
- Executive summary with key metrics
- Multi-currency performance matrix
- Walk-forward Sharpe evolution
- Robustness test results
- Capacity analysis with max tradeable size
- Statistical significance tests

### HTML Report
- Interactive charts
- Detailed results tables
- Session-by-session analysis
- Downloadable data

### Archive
- All results in JSON format
- Git SHA and environment info
- Data file hashes for reproducibility

## Example Results

```
VALIDATION SUMMARY
================================================================================
Multi-Currency Results (8 pairs):
Average Sharpe: 1.665
Best: AUDNZD (Sharpe: 4.358)
Worst: AUDJPY (Sharpe: 0.622)

Walk-Forward Validation:
Mean OOS Sharpe: 1.126
P-value: 0.0234

Capacity Analysis:
Max viable size: $35.2M

================================================================================
✅ VERDICT: Strategy is GOOD - suitable for live trading
================================================================================
```

## CI/CD Integration

```yaml
# .github/workflows/validation.yml
- name: Run Strategy Validation
  run: |
    python validation/run_validation.py --quick
    
- name: Upload Reports
  uses: actions/upload-artifact@v2
  with:
    name: validation-reports
    path: validation_output/reports/
```

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
optuna>=2.10.0
reportlab>=3.6.0
jinja2>=3.0.0
psutil>=5.8.0
```

## Best Practices

1. **Always run full validation before deploying any strategy**
2. **Set up automated validation in CI/CD pipeline**
3. **Archive all validation reports for audit trail**
4. **Re-validate quarterly or after significant market events**
5. **Monitor live performance against validation metrics**

## Extending the Framework

To add a new validation test:

1. Create a new module in the appropriate directory
2. Implement the test class with a `create_report()` method
3. Add the test to `run_validation.py`
4. Update report generator to include new results

## License

This validation framework is for internal use only.