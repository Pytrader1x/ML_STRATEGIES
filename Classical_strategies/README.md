# Classical Trading Strategies

Production-ready algorithmic trading system with institutional-grade features for forex trading.

## Quick Start

```bash
# Basic backtest (single period)
python run_validated_strategy.py

# Monte Carlo validation (50 samples)
python run_validated_strategy.py --validate

# Monte Carlo simulation (custom samples)
python run_validated_strategy.py --monte-carlo 100

# With visualization
python run_validated_strategy.py --monte-carlo 50 --show-plots --save-plots
```

## Features

- **Three-indicator confluence system**: NeuroTrend, Market Bias, Intelligent Chop
- **Institutional position sizing**: 1-2M units with dynamic adjustment
- **Multi-level take profits**: 3 levels with partial exits
- **Dynamic ATR-based risk management**: Adaptive to market volatility
- **Monte Carlo validation**: Robust out-of-sample testing
- **Sequential analysis**: Year-by-year and quarter-by-quarter breakdowns
- **Realistic execution costs**: Slippage modeling for all order types
- **Zero lookahead bias**: Production-safe implementation

## Performance Targets

- **Sharpe Ratio**: > 0.7 (with realistic costs)
- **Win Rate**: 65-75%
- **Max Drawdown**: < 15%
- **Risk per Trade**: 0.5%
- **Profit Factor**: > 1.2

## Directory Structure

```
Classical_strategies/
├── run_validated_strategy.py      # Main production runner
├── validated_strategy_config.json # Strategy configuration
├── strategy_code/                 # Core strategy implementation
│   ├── Prod_strategy.py          # Strategy logic
│   ├── Prod_plotting.py          # Visualization tools
│   └── __init__.py
├── charts/                        # Output charts
├── results/                       # Backtest results
└── archive/                       # Legacy code and tests
```

## Usage Examples

### Basic Commands

```bash
# Test recent performance (2024 H1)
python run_validated_strategy.py --period 2024

# Test full year 2023
python run_validated_strategy.py --period 2023

# Custom date range
python run_validated_strategy.py --start-date 2023-01-01 --end-date 2023-12-31
```

### Monte Carlo Simulation

```bash
# Standard Monte Carlo (25 samples, 90-day periods)
python run_validated_strategy.py --monte-carlo 25

# Extended Monte Carlo (100 samples)
python run_validated_strategy.py --monte-carlo 100 --save-plots

# Validation mode (50 samples with strict criteria)
python run_validated_strategy.py --validate
```

### Sequential Analysis

```bash
# Year-by-year performance
python run_validated_strategy.py --sequential yearly --start-year 2020 --end-year 2024

# Quarter-by-quarter analysis
python run_validated_strategy.py --sequential quarterly --save-plots
```

### Currency and Position Sizing

```bash
# Different currency pairs
python run_validated_strategy.py --currency GBPUSD --monte-carlo 50
python run_validated_strategy.py --currency EURUSD --monte-carlo 50

# Different position sizes (1M or 2M)
python run_validated_strategy.py --position-size 2 --monte-carlo 50

# Custom capital
python run_validated_strategy.py --capital 2000000 --position-size 2
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--currency` | Currency pair to test | AUDUSD |
| `--capital` | Initial capital | 1,000,000 |
| `--position-size` | Position size in millions (1 or 2) | 1 |
| `--show-plots` | Display charts interactively | False |
| `--save-plots` | Save charts to PNG files | False |
| `--start-date` | Backtest start date (YYYY-MM-DD) | - |
| `--end-date` | Backtest end date (YYYY-MM-DD) | - |
| `--period` | Predefined period (2024, 2023, recent, last-quarter) | - |
| `--monte-carlo` | Run Monte Carlo with N samples | - |
| `--validate` | Run validation mode (50 samples, strict criteria) | False |
| `--sequential` | Run sequential analysis (yearly or quarterly) | - |
| `--start-year` | Start year for sequential analysis | - |
| `--end-year` | End year for sequential analysis | - |

## Strategy Configuration

The strategy uses the following key parameters (from `validated_strategy_config.json`):

```json
{
  "risk_per_trade": 0.005,          // 0.5% risk per trade
  "sl_min_pips": 3.0,               // Minimum stop loss
  "sl_max_pips": 10.0,              // Maximum stop loss
  "tp_multipliers": [0.15, 0.25, 0.4],  // Take profit levels
  "tsl_activation_pips": 8.0,       // Trailing stop activation
  "relaxed_mode": true,             // More trading opportunities
  "realistic_costs": true           // Include slippage
}
```

## Validation Criteria

When running with `--validate`, the strategy must pass:

1. **Average Sharpe Ratio**: ≥ 0.7
2. **Success Rate**: ≥ 60% profitable samples
3. **Worst Drawdown**: ≤ 15%
4. **Average Profit Factor**: ≥ 1.2

## Output Files

- **Charts**: Saved to `charts/` directory with timestamp
- **Monte Carlo Results**: Saved to `results/monte_carlo_*.csv`
- **Sequential Analysis**: Saved to `results/sequential_*.csv`
- **Configuration**: `validated_strategy_config.json`

## Performance Notes

- All results include realistic execution costs (slippage)
- Monte Carlo uses random 90-day contiguous samples
- No lookahead bias - safe for production use
- Validated on 5+ years of 15-minute forex data

## Troubleshooting

1. **Missing Data**: Ensure forex data files are in `../data/` directory
2. **Import Errors**: Check that `technical_indicators_custom.py` is in parent directory
3. **Memory Issues**: Reduce Monte Carlo samples or use smaller date ranges
4. **Plot Issues**: Install matplotlib if not available

## Latest Updates

- Improved Monte Carlo progress indicator (shows every 5 samples)
- Added validation mode with strict pass/fail criteria
- Reduced console output during Monte Carlo runs
- Enhanced performance analysis with validation metrics
- Cleaned directory structure with archived legacy code

---

*Version 3.0 - Production Ready with Validation*