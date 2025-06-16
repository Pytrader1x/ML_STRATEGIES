# ADX Trend-Following Strategy

A sophisticated trend-following scalping strategy that combines the Directional Movement Index (DMI/ADX) with Williams %R to identify high-probability trade entries in strong trends.

## Strategy Overview

This strategy operates on the principle of "buying dips in uptrends" and "selling rallies in downtrends". It uses:

1. **ADX/DMI** - To identify strong, established trends
2. **Williams %R** - To pinpoint oversold/overbought pullbacks for optimal entries
3. **Dynamic Stop-Loss** - Using 50-period SMA
4. **Smart Take-Profit** - Based on recent price extremes

## Key Features

- **Trend Strength Filter**: Only trades when ADX > 50 (strong trend)
- **Pullback Entry**: Waits for temporary retracements for better risk/reward
- **Position Invalidation**: Exits immediately if trend direction changes
- **Risk Management**: 3% risk per trade with dynamic position sizing
- **Backtesting Framework**: Complete backtesting and optimization tools using Backtrader

## Installation

```bash
# Clone the repository
cd ADX_Strategy

# Install required packages
pip install backtrader pandas numpy yfinance matplotlib
```

## Quick Start

### 1. Run a Simple Backtest

```bash
python main.py backtest
```

### 2. Download Historical Data

```bash
python main.py download
```

### 3. Run Parameter Optimization

```bash
python main.py optimize
```

### 4. Custom Backtest with Specific Data

```bash
python main.py backtest --data data/SPY_1h.csv --start 2023-01-01 --end 2023-12-31 --cash 25000
```

## Strategy Rules

### Long Entry Conditions
All conditions must be true on the same candle:
- DI+ > DI- (Uptrend)
- ADX > 50 (Strong trend)
- Williams %R < -80 (Oversold pullback)

### Short Entry Conditions
All conditions must be true on the same candle:
- DI- > DI+ (Downtrend)
- ADX > 50 (Strong trend)
- Williams %R > -20 (Overbought pullback)

### Exit Conditions
- **Stop-Loss**: Price touches 50-period SMA
- **Take-Profit**: 
  - Long: Highest high of last 30 candles
  - Short: Lowest low of last 30 candles
- **Position Invalidation**: Opposite entry signal appears

## Configuration

Edit `config.py` to adjust strategy parameters:

```python
STRATEGY_PARAMS = {
    'adx_period': 14,           # ADX calculation period
    'adx_threshold': 50,        # Minimum ADX for strong trend
    'williams_period': 14,      # Williams %R period
    'williams_oversold': -80,   # Oversold threshold
    'williams_overbought': -20, # Overbought threshold
    'sma_period': 50,          # SMA period for stop-loss
    'tp_lookback': 30,         # Lookback for take-profit
    'risk_percent': 0.03,      # Risk 3% per trade
}
```

## File Structure

```
ADX_Strategy/
├── ADX_Strategy.py    # Core strategy implementation
├── backtest.py        # Backtesting engine
├── utils.py           # Helper functions and utilities
├── config.py          # Configuration parameters
├── main.py            # Command-line interface
├── README.md          # This file
└── data/              # Historical data storage (created on download)
```

## Performance Metrics

The strategy tracks and reports:
- Total Return & Sharpe Ratio
- Maximum Drawdown
- Win Rate & Profit Factor
- Average Win/Loss
- Trade Duration Statistics
- Consecutive Win/Loss Streaks

## Advanced Usage

### Parameter Optimization

The optimization module tests different parameter combinations:

```python
# In main.py or custom script
optimization_results = optimize_strategy(
    start_date='2022-01-01',
    end_date='2023-12-31'
)
```

### Custom Indicators

Extend the strategy by adding indicators in `ADX_Strategy.py`:

```python
# Add RSI filter
self.rsi = btind.RelativeStrengthIndex(self.datas[0])

# Modify entry condition
def check_long_entry(self):
    return (
        self.plus_di[0] > self.minus_di[0] and
        self.adx[0] > self.params.adx_threshold and
        self.williams[0] < self.params.williams_oversold and
        self.rsi[0] < 70  # Additional RSI filter
    )
```

## Risk Disclaimer

This strategy is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly and understand the risks before using any trading strategy with real capital.

## Tips for Success

1. **Market Selection**: Works best on trending markets (indices, trending forex pairs)
2. **Timeframe**: Optimized for 1H and 15M charts
3. **Market Conditions**: Avoid ranging/choppy markets
4. **Risk Management**: Never risk more than 3% per trade
5. **Backtesting**: Always backtest on your specific market and timeframe

## Troubleshooting

### Common Issues

1. **No trades executed**: Check if ADX threshold is too high for your market
2. **Too many false signals**: Increase ADX threshold or add additional filters
3. **Data issues**: Ensure your CSV has columns: Open, High, Low, Close, Volume

### Debug Mode

Enable detailed logging in `config.py`:

```python
'printlog': True  # Shows all entry/exit signals
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.