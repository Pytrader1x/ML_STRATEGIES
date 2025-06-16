# Classical Trading Strategies - Comprehensive Guide

> Production-ready algorithmic trading system with institutional-grade features, comprehensive backtesting, and Monte Carlo simulation capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Strategy Logic](#strategy-logic)
4. [Risk Management](#risk-management)
5. [Technical Implementation](#technical-implementation)
6. [Performance Metrics](#performance-metrics)
7. [Running Strategies](#running-strategies)
8. [Live Trading Setup](#live-trading-setup)
9. [Recent Updates](#recent-updates)
10. [Troubleshooting](#troubleshooting)

## Overview

The Classical Trading Strategies system is a sophisticated trading framework that combines three powerful technical indicators to generate high-probability trading signals. The system has been thoroughly validated for institutional use and includes comprehensive risk management features.

### Key Features

- **Multi-indicator confluence system** using NeuroTrend Intelligent (NTI), Market Bias (MB), and Intelligent Chop (IC)
- **Institutional-scale position sizing** (1M-2M AUD units)
- **Multi-level take profits** with partial position exits
- **Dynamic risk management** with ATR-based stops and regime adjustments
- **Monte Carlo simulation** for robust performance analysis
- **Zero lookahead bias** with proper point-in-time data handling
- **Realistic execution modeling** with slippage and spread costs

### Performance Summary

| Metric | Typical Range | Target |
|--------|--------------|--------|
| Sharpe Ratio | 0.7 - 2.5 | > 1.0 |
| Win Rate | 65% - 75% | > 65% |
| Max Drawdown | 1% - 3% | < 5% |
| SQN | 3.0 - 6.0 | > 3.0 |
| Recovery Factor | 5.0 - 15.0 | > 5.0 |

## Quick Start

### 1. Basic Backtest

```bash
# Run a simple backtest with default settings
python run_strategy_single.py

# Run with specific date range
python run_strategy_single.py --start_date "2020-01-01" --end_date "2023-12-31"

# Run with charts enabled
python run_strategy_single.py --show_chart
```

### 2. Monte Carlo Analysis

```bash
# Run Monte Carlo simulation (1000 iterations)
python run_strategy_oop.py

# With custom parameters
python run_strategy_oop.py --iterations 2000 --processes 8 --validate_first
```

### 3. Validated Strategy (Recommended for Production)

```bash
# Run the thoroughly validated strategy
python run_validated_strategy.py

# With all features enabled
python run_validated_strategy.py --show_chart --save_full_metrics --date_range "2023-01-01,2024-01-01"
```

### Expected Results

- **Initial backtest**: ~2 minutes for 5 years of data
- **Monte Carlo (1000 runs)**: ~5-10 minutes on 8-core machine
- **Output files**: Charts, plots, metrics CSV, and trade logs

## Strategy Logic

### Three-Indicator Confluence System

1. **NeuroTrend Intelligent (NTI)**
   - Primary trend detector with dynamic adaptation
   - Provides direction, confidence, and reversal risk metrics
   - Entry requires NTI_Confidence > 0.7

2. **Market Bias (MB)**
   - Market structure and momentum indicator
   - Confirms overall market direction
   - Entry requires MB_Bias alignment with NTI

3. **Intelligent Chop (IC)**
   - Market regime classifier (trending vs ranging)
   - Filters out choppy market conditions
   - Entry requires IC_Regime != 0 (not ranging)

### Entry Conditions

**Long Entry:**
```python
if (NTI_Direction > 0 and 
    NTI_Confidence > 0.7 and 
    MB_Bias > 0.2 and 
    IC_Regime == 1):
    enter_long()
```

**Short Entry:**
```python
if (NTI_Direction < 0 and 
    NTI_Confidence > 0.7 and 
    MB_Bias < -0.2 and 
    IC_Regime == -1):
    enter_short()
```

### Exit Strategy

#### Multi-Level Take Profits
- **TP1**: 15 pips (50% position exit)
- **TP2**: 25 pips (25% position exit)
- **TP3**: 35 pips (25% position exit)

#### TP1 Pullback Logic
After TP1 is hit, if price pulls back 60% toward the stop loss, the remaining 50% position is closed with a guaranteed minimum profit.

#### Dynamic Stop Loss
- Initial: 2.0 × ATR (trending) or 1.5 × ATR (ranging)
- Maximum: 20 pips hard limit
- Trailing: Activates after 10 pips profit
- Minimum trailing distance: 5 pips

## Risk Management

### Position Sizing

```python
# Risk-based position sizing
risk_per_trade = account_balance * 0.001  # 0.1% risk
stop_distance_pips = calculate_stop_distance()
position_size = risk_per_trade / (stop_distance_pips * pip_value)

# With institutional constraints
position_size = min(position_size, 2_000_000)  # Max 2M units
position_size = round(position_size / 1_000_000) * 1_000_000  # Round to millions
```

### Risk Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk per trade | 0.1% | Maximum account risk per position |
| Max position size | 2M AUD | Institutional liquidity constraint |
| Max stop loss | 20 pips | Hard limit for risk control |
| Min stop loss | 5 pips | Minimum viable stop distance |
| Spread | 0.2 pips | Realistic AUDUSD institutional spread |
| Slippage | 0.1 pips | Conservative execution slippage |

## Technical Implementation

### Data Requirements

- **Format**: CSV with OHLC data
- **Timeframe**: 15-minute bars
- **Minimum history**: 1 year for proper indicator calculation
- **Required columns**: DateTime, Open, High, Low, Close, Volume

### File Structure

```
Classical_strategies/
├── data/
│   └── AUDUSD_MASTER_15M.csv
├── run_strategy_single.py      # Single backtest runner
├── run_strategy_oop.py         # Monte Carlo runner
├── run_validated_strategy.py   # Production-ready runner
├── strategy_audusd_15m_v2.py  # Core strategy logic
├── backtest_engine.py          # Backtesting framework
└── performance_monitor.py      # Metrics calculation
```

### Key Components

1. **Strategy Class**: Implements entry/exit logic and position management
2. **Backtest Engine**: Handles order execution, P&L tracking, and realistic fills
3. **Performance Monitor**: Calculates comprehensive metrics and generates reports
4. **Data Manager**: Ensures proper data handling without lookahead bias

## Performance Metrics

### Core Metrics

- **Sharpe Ratio**: Risk-adjusted returns (daily, annualized to yearly)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses
- **SQN (System Quality Number)**: Van Tharp's system quality metric
- **Recovery Factor**: Net profit / Max drawdown

### Advanced Metrics

- **Calmar Ratio**: Annual return / Max drawdown
- **Expectancy**: Average profit per trade
- **Payoff Ratio**: Average win / Average loss
- **Ulcer Index**: Downside volatility measure
- **Maximum Consecutive Losses**: Risk assessment metric
- **Time in Market**: Percentage of time with open positions

### Monte Carlo Analysis

The Monte Carlo simulation provides:
- **Confidence intervals** for all metrics
- **Probability of ruin** calculations
- **Expected drawdown distributions**
- **Robust performance expectations**

## Running Strategies

### Single Backtest (Development)

```python
# Basic usage
python run_strategy_single.py

# With custom parameters
python run_strategy_single.py \
    --symbol AUDUSD \
    --risk_per_trade 0.001 \
    --position_sizes "1000000,2000000" \
    --start_date "2023-01-01" \
    --show_chart
```

### Monte Carlo Simulation (Validation)

```python
# Standard Monte Carlo run
python run_strategy_oop.py

# High-precision analysis
python run_strategy_oop.py \
    --iterations 5000 \
    --processes 16 \
    --confidence_levels "90,95,99"
```

### Production Runner (Recommended)

```bash
# Basic usage with default settings
python run_validated_strategy.py

# With charts displayed
python run_validated_strategy.py --show-plots

# Save charts to PNG files
python run_validated_strategy.py --save-plots

# Custom currency pair and capital
python run_validated_strategy.py --currency GBPUSD --capital 2000000

# Custom position size (1 or 2 million)
python run_validated_strategy.py --position-size 2

# Custom date range
python run_validated_strategy.py --start-date 2023-01-01 --end-date 2023-12-31

# Predefined periods
python run_validated_strategy.py --period 2024        # Full year 2024
python run_validated_strategy.py --period 2023        # Full year 2023
python run_validated_strategy.py --period recent      # Recent months
python run_validated_strategy.py --period last-quarter # Last quarter

# With Monte Carlo simulation
python run_validated_strategy.py --monte-carlo 1000

# Full example with all options
python run_validated_strategy.py \
    --show-plots \
    --save-plots \
    --currency AUDUSD \
    --capital 1000000 \
    --position-size 1 \
    --start-date 2024-01-01 \
    --end-date 2024-06-30
```

#### Command-Line Flags for run_validated_strategy.py

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--show-plots` | Display interactive charts | False | `--show-plots` |
| `--save-plots` | Save charts as PNG files | False | `--save-plots` |
| `--currency` | Currency pair to test | AUDUSD | `--currency GBPUSD` |
| `--capital` | Initial capital in USD | 1000000 | `--capital 2000000` |
| `--position-size` | Position size in millions (1 or 2) | 1 | `--position-size 2` |
| `--start-date` | Backtest start date (YYYY-MM-DD) | None | `--start-date 2023-01-01` |
| `--end-date` | Backtest end date (YYYY-MM-DD) | None | `--end-date 2023-12-31` |
| `--period` | Predefined test period | None | `--period 2024` |
| `--monte-carlo` | Run Monte Carlo with N samples | None | `--monte-carlo 1000` |
| `--monte-carlo-all-years` | Run Monte Carlo across all years | False | `--monte-carlo-all-years` |
| `--sequential` | Run sequential analysis mode | None | `--sequential yearly` |
| `--start-year` | Start year for sequential analysis | None | `--start-year 2020` |
| `--end-year` | End year for sequential analysis | None | `--end-year 2024` |

#### Advanced Analysis Examples

##### 1. Monte Carlo Analysis

```bash
# Standard Monte Carlo (samples from recent data)
python run_validated_strategy.py --monte-carlo 1000

# Monte Carlo across all available years
python run_validated_strategy.py --monte-carlo-all-years

# Monte Carlo with custom sample size and plots
python run_validated_strategy.py --monte-carlo 500 --show-plots --save-plots
```

##### 2. Sequential Year-by-Year Analysis

```bash
# Analyze performance year by year for all available data
python run_validated_strategy.py --sequential yearly --show-plots

# Analyze specific year range
python run_validated_strategy.py --sequential yearly --start-year 2020 --end-year 2024

# Save results with plots
python run_validated_strategy.py --sequential yearly --save-plots
```

##### 3. Sequential Quarter-by-Quarter Analysis

```bash
# Analyze performance quarter by quarter
python run_validated_strategy.py --sequential quarterly --show-plots

# Analyze specific quarters
python run_validated_strategy.py --sequential quarterly --start-year 2023 --end-year 2024

# Full analysis with all options
python run_validated_strategy.py \
    --sequential quarterly \
    --start-year 2022 \
    --end-year 2024 \
    --position-size 2 \
    --show-plots \
    --save-plots
```

##### 4. Combined Analysis Workflow

```bash
# Step 1: Run year-by-year analysis to identify patterns
python run_validated_strategy.py --sequential yearly --save-plots

# Step 2: Deep dive into specific years with quarterly analysis
python run_validated_strategy.py --sequential quarterly --start-year 2023 --end-year 2024

# Step 3: Run Monte Carlo to validate robustness
python run_validated_strategy.py --monte-carlo-all-years

# Step 4: Test specific profitable periods
python run_validated_strategy.py --start-date 2023-04-01 --end-date 2023-09-30 --show-plots
```

The sequential analysis features provide:
- **Year-by-year performance**: See how strategy performs across different market conditions
- **Quarter-by-quarter granularity**: Identify seasonal patterns or specific market regimes
- **Statistical summaries**: Average performance, best/worst periods, consistency metrics
- **Visual analysis**: Automated plots showing performance evolution over time
- **CSV exports**: All results saved for further analysis

## Live Trading Setup

### 1. System Requirements

- **Python 3.8+** with required packages
- **Reliable data feed** (15-minute OHLC)
- **Order execution API** (broker integration)
- **VPS or dedicated server** (recommended)

### 2. Configuration

```python
# config/strategy_config.json
{
    "symbol": "AUDUSD",
    "timeframe": "15m",
    "risk_per_trade": 0.001,
    "max_position_size": 2000000,
    "indicators": {
        "nti_fast": 12,
        "nti_slow": 26,
        "mb_len1": 20,
        "mb_len2": 50,
        "ic_period": 20
    }
}
```

### 3. Monitoring

- **Real-time P&L tracking**
- **Position status dashboard**
- **Alert system for anomalies**
- **Daily performance reports**

### 4. Risk Controls

- **Maximum daily loss limit**
- **Position exposure limits**
- **Correlation risk monitoring**
- **Emergency stop functionality**

## Recent Updates

### Latest Fixes (2024)

1. **TP2 Marker Visibility**
   - Fixed TP2 exit markers not showing on charts
   - Improved trade visualization

2. **P&L Recording Integrity**
   - Enhanced partial exit handling
   - Accurate multi-level TP accounting

3. **Exit Price Boundaries**
   - Added min/max price constraints
   - Prevents unrealistic exit prices

4. **Performance Metrics**
   - Added Recovery Factor
   - Added Expectancy calculations
   - Improved Sharpe Ratio computation

5. **Intrabar Stop Loss**
   - Optional high-frequency stop checking
   - More realistic stop execution

### Validation Improvements

- **Zero lookahead bias** verified
- **Realistic execution** with appropriate slippage
- **Institutional constraints** properly implemented
- **Risk management** thoroughly tested

## Troubleshooting

### Common Issues

1. **"No trades found" Error**
   - Check date range has sufficient data
   - Verify indicators are calculating properly
   - Ensure data quality (no gaps)

2. **Poor Performance**
   - Confirm using validated parameter sets
   - Check spread/slippage settings are realistic
   - Verify risk parameters are appropriate

3. **Memory Issues (Monte Carlo)**
   - Reduce number of parallel processes
   - Use smaller date ranges
   - Enable memory-efficient mode

### Debug Mode

```python
# Enable detailed logging
python run_strategy_single.py --debug

# Save all calculation details
python run_strategy_single.py --save_indicators
```

### Performance Optimization

1. **Use validated parameter sets** from optimization results
2. **Trade liquid sessions** (London/NY overlap preferred)
3. **Monitor correlation** with other positions
4. **Regular reoptimization** (quarterly recommended)

## Best Practices

1. **Always run Monte Carlo** before live trading
2. **Start with minimum position sizes** in live testing
3. **Monitor slippage** and adjust models accordingly
4. **Keep detailed logs** of all trades and modifications
5. **Regular backtesting** on recent data
6. **Risk management** is paramount - never exceed limits

## Support and Development

### Future Enhancements

- Machine learning signal filtering
- Multi-timeframe confirmation
- Correlation-based position sizing
- Advanced market regime detection
- Real-time parameter adaptation

### Contributing

For bug reports or feature requests, please document:
- Exact reproduction steps
- Data samples if relevant
- Expected vs actual behavior
- System configuration

---

*Last Updated: January 2024*
*Version: 2.0 (Production Ready)*