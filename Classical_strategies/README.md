# Classical Trading Strategies

Production-ready algorithmic trading system with institutional-grade features for forex trading.

## 🚀 Quick Start

```bash
# Basic backtest
python run_validated_strategy.py

# With visualization
python run_validated_strategy.py --show-plots

# Monte Carlo simulation
python run_validated_strategy.py --monte-carlo 1000

# Year-by-year analysis
python run_validated_strategy.py --sequential yearly --show-plots
```

## 📋 Features

- **Three-indicator confluence system** (NeuroTrend, Market Bias, Intelligent Chop)
- **Institutional position sizing** (1-2M units)
- **Multi-level take profits** with partial exits
- **Dynamic ATR-based risk management**
- **Monte Carlo simulation** for robust validation
- **Sequential analysis** (yearly/quarterly)
- **Zero lookahead bias**

## 📊 Performance

- **Target Sharpe Ratio**: > 1.0
- **Typical Win Rate**: 65-75%
- **Max Drawdown**: < 5%
- **Risk per Trade**: 0.1%

## 📁 Main Files

- `run_validated_strategy.py` - Production runner with all features
- `run_strategy_single.py` - Simple single backtest
- `run_strategy_oop.py` - Monte Carlo simulation
- `run_optimizer.py` - Parameter optimization
- `STRATEGIES_GUIDE.md` - Comprehensive documentation

## 🔧 Advanced Usage

### Sequential Analysis
```bash
# Year-by-year performance
python run_validated_strategy.py --sequential yearly --start-year 2020 --end-year 2024

# Quarter-by-quarter analysis
python run_validated_strategy.py --sequential quarterly --save-plots
```

### Monte Carlo Validation
```bash
# Full historical Monte Carlo
python run_validated_strategy.py --monte-carlo-all-years

# Custom samples
python run_validated_strategy.py --monte-carlo 2000 --show-plots
```

## 📚 Documentation

See [STRATEGIES_GUIDE.md](STRATEGIES_GUIDE.md) for complete documentation including:
- Detailed strategy logic
- Risk management rules
- Command-line options
- Live trading setup
- Troubleshooting

## ⚡ Latest Updates

- Added sequential year-by-year and quarter-by-quarter analysis
- Monte Carlo simulation across all historical data
- Enhanced command-line interface
- Performance visualization improvements
- Consolidated documentation

---

*Version 2.0 - Production Ready*