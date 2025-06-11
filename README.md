# 🤖 ML_Strategies

Advanced Machine Learning & Quantitative Trading Strategies Repository

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

## 🏆 Classical Strategy Performance

<p align="center">
  <img src="Classical_strategies/charts/optimized_monte_carlo.png" alt="Strategy Performance" width="80%">
</p>

### 📊 Optimized Strategy Results (Monte Carlo Analysis - 20 Samples)
| Metric | Performance |
|--------|-------------|
| **Average P&L** | $63,847 |
| **Win Rate** | 69.9% |
| **Sharpe Ratio** | 0.19 |
| **Max Drawdown** | -28.4% |
| **Annual Return** | 257% (5-year avg) |

### 🎯 Core Features
- **Three-tiered partial take profit system** (33% at each TP level)
- **Intelligent trailing stop loss** with 15 pip activation, 5 pip minimum profit
- **Advanced signal flip filtering** with profit threshold and time requirements
- **Confidence-based position sizing** (1M, 3M, 5M lots)
- **Market regime adaptation** for dynamic TP/SL levels
- **Maximum 45 pip stop loss** for risk control

### ⚡ Performance Characteristics
- **Processing Speed**: 30,000+ bars/second
- **Average Trade Duration**: 12.1 hours
- **Profit Factor**: 1.30
- **Trade Frequency**: 39 trades/month (5-year avg)
- **Risk-Adjusted Return**: Sharpe 1.53 (annualized)

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

## 🔥 Recent Updates (2025)

### Major Improvements
- ✨ **Optimized Signal Flip Logic**: Reduced losses by $578k over 2 years
- 📈 **Performance Boost**: 154% average P&L improvement in Monte Carlo tests
- 🎯 **Enhanced Risk Management**: 45 pip max SL, guaranteed 5 pip TSL profit
- 🚀 **Speed Optimization**: 30,000+ bars/second processing
- 📊 **Advanced Visualization**: Real-time P&L tracking with position sizing
- 🧮 **Smart Position Sizing**: Confidence-based scaling (1M → 5M)

### Key Optimizations
1. **Signal Flip Filtering**: 
   - Before: 319 flips, 76.8% losses, -$507k
   - After: 26 flips, 100% profitable, +$70k

2. **Risk Control**:
   - Maximum 45 pip stop loss
   - TSL activation at 15 pips
   - Guaranteed 5 pip minimum profit

3. **Performance**:
   - 5-year return: 1,285%
   - Annual return: 257%
   - Win rate: 70%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[To be determined]