# ML_Strategies

Machine Learning Trading Strategies Repository

## Overview

This repository contains various machine learning and classical approaches for trading strategies:

### Classical Strategies
- **Production Strategy**: A sophisticated quantitative trading system combining three technical indicators:
  - NeuroTrend Intelligent (NTI) - Advanced trend detection
  - Market Bias (MB) - Market structure analysis
  - Intelligent Chop (IC) - Regime classification

### Machine Learning Strategies
- Dueling DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- XGBoost and Random Forest models

## Classical Strategy Features

The production-ready classical strategy (`Classical_strategies/`) includes:

### Core Features
- **Three-tiered partial take profit system** (33% at each TP level)
- **Intelligent trailing stop loss** with pip-based activation
- **Market Bias-based stop loss placement**
- **Early exit on signal flips**
- **Confidence-based position sizing** (1M, 3M, 5M lots)
- **Relaxed mode** for NeuroTrend-only entries
- **TP1 pullback logic** for optimized exits

### Risk Management
- Minimum position size: 1M AUD (1 million units)
- $100 per pip per million for AUDUSD
- ATR-based dynamic TP/SL levels
- Maximum TP distance capped at 1% from entry
- Trailing stop activation at 15 pips profit

### Performance Metrics
- **Win Rate**: 56-68% (varies by market conditions)
- **Sharpe Ratio**: 0.3-0.8
- **Profit Factor**: 1.3-1.5
- **Average Trade Duration**: 4-8 hours
- **Processing Speed**: 5,000+ bars/second

## Data

The `data/` directory contains historical FX data for multiple currency pairs:
- 1-minute interval data (raw)
- 15-minute interval data (resampled)

### Available Currency Pairs
- AUD/JPY, AUD/NZD, AUD/USD
- CAD/JPY, CHF/JPY
- EUR/GBP, EUR/JPY, EUR/USD
- GBP/JPY, GBP/USD
- NZD/USD, USD/CAD

### Data Download

To download or update FX data:
```bash
cd data
python download_fx_data.py
```

## Project Structure

```
ML_Strategies/
├── Classical_strategies/  # Production-ready classical strategy
│   ├── Prod_strategy.py  # Clean OOP strategy implementation
│   ├── Prod_plotting.py  # Enhanced visualization with data stats
│   └── example_usage.py  # Usage demonstration
├── Dueling_DQN/          # Dueling Deep Q-Network implementation
├── PPO/                  # Proximal Policy Optimization implementation
├── XG_boost_RForest/     # XGBoost and Random Forest models
└── data/                 # Historical FX data and download scripts
```

## Requirements

### Core Dependencies
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.5.0
- technical-indicators-custom (custom indicators library)
  ```bash
  pip install git+https://github.com/Pytrader1x/technical-indicators-custom.git
  ```

### Additional Dependencies
- fx_data_downloader (for data download)
- scikit-learn (for ML strategies)
- tensorflow/pytorch (for deep learning strategies)

## Getting Started

### Classical Strategy Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_Strategies.git
   cd ML_Strategies
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib
   pip install git+https://github.com/Pytrader1x/technical-indicators-custom.git
   ```

3. Run the example:
   ```bash
   cd Classical_strategies
   python example_usage.py
   ```

### Using the Production Strategy

```python
from Prod_strategy import create_strategy
from Prod_plotting import plot_production_results
from technical_indicators_custom import TIC

# Load and prepare data
df = pd.read_csv('your_data.csv')
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df)
df = TIC.add_intelligent_chop(df)

# Create and run strategy
strategy = create_strategy(
    initial_capital=100_000,
    intelligent_sizing=True,
    exit_on_signal_flip=True
)
results = strategy.run_backtest(df)

# Plot results with data statistics
plot_production_results(df, results, show_pnl=True)
```

## Recent Updates (2025)

- **Production-Ready Classical Strategy**: Complete refactor with OOP design
- **Enhanced Plotting**: Added data statistics display (rows, period, timeframe)
- **Performance Optimizations**: 5000+ bars/second processing speed
- **Intelligent Position Sizing**: Confidence-based sizing (1M, 3M, 5M lots)
- **Advanced Exit Logic**: TP1 pullback and trailing stop improvements
- **Clean Architecture**: Separated concerns with dedicated components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[To be determined]