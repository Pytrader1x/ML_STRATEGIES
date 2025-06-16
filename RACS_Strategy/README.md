# Regime-Adaptive Confluence Strategy (RACS)

## Overview

RACS is a sophisticated multi-regime trading strategy that adapts its tactics based on market conditions. It achieves high Sharpe ratios by:

1. **Adapting tactics to market conditions** - Different strategies for different regimes
2. **Using simple, clear entry/exit rules** - No overfitting or complex logic
3. **Avoiding lossy environments entirely** - No trading in volatile chop
4. **Maximizing winners through intelligent exits** - Partial exits and dynamic stops

## Market Regimes

The strategy identifies four distinct market regimes using the Intelligent Chop indicator:

| Background | Label | Strategy | Position Size | New Trades? |
|------------|-------|----------|---------------|-------------|
| 🟢 **Green** | Strong Trend | Trend-Following | 100% | ✅ YES |
| 🟡 **Yellow** | Weak Trend | Selective Trend | 50% | ⚠️ MAYBE |
| 🔵 **Blue** | Quiet Range | Range Reversion | 50% | ✅ YES |
| 🔴 **Red** | Volatile Chop | Protection Mode | 0% | ❌ NO |

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the TIC indicators module in the same directory.

## Usage

### Basic Backtest

```python
python backtest_racs.py
```

### Custom Backtest

```python
from RACS_Strategy.backtest_racs import prepare_data, run_backtest, print_performance_report

# Load and prepare data
df = prepare_data('path/to/your/data.csv', start_date='2020-01-01')

# Run backtest with custom parameters
results = run_backtest(df, initial_cash=10000, commission=0.001)

# Print results
print_performance_report(results)
```

### Using the Strategy with Backtrader

```python
import backtrader as bt
from RACS_Strategy.racs_strategy import RACSStrategy

cerebro = bt.Cerebro()
cerebro.addstrategy(RACSStrategy, 
                   base_risk_pct=0.01,      # 1% risk per trade
                   max_positions=3,         # Maximum concurrent positions
                   min_confidence=60.0)     # Minimum IC confidence

# Add your data and run
```

## Strategy Parameters

### Risk Management
- `base_risk_pct`: Base risk per trade (default: 1%)
- `max_positions`: Maximum concurrent positions (default: 3)

### Regime Thresholds
- `min_confidence`: Minimum IC confidence for any trade (default: 60%)
- `yellow_confidence`: Minimum confidence for weak trends (default: 70%)

### Entry Filters
- `min_nti_confidence`: Minimum NeuroTrend confidence (default: 70%)
- `min_slope_power`: Minimum slope power for trends (default: 20.0)

### Range Trading
- `range_penetration`: Allowed penetration into range (default: 2%)
- `range_target_pct`: Target percentage of range (default: 80%)

### Position Sizing
- `yellow_size_factor`: Size multiplier for weak trends (default: 0.5)
- `blue_size_factor`: Size multiplier for ranges (default: 0.5)
- `golden_setup_bonus`: Bonus for high-probability setups (default: 1.5x)

## Entry Rules

### Trend Following (Green/Yellow Regimes)

**Long Entry:**
1. Regime = Green (or Yellow with IC_Confidence ≥ 70%)
2. NTI_Direction = +1
3. MB_Bias = +1 (or neutral for 3+ bars)
4. Price > MB_ha_avg
5. SuperTrend flips bullish
6. Enter with stop order above bar high

**Short Entry:**
- Mirror image of long rules
- SuperTrend flips bearish
- Stop order below bar low

### Range Trading (Blue Regime Only)

**Range Long:**
1. Background = Blue for 5+ bars
2. Price near lower boundary (2% penetration allowed)
3. MB_Bias turns positive
4. IC_ChoppinessIndex > 50

**Range Short:**
- Mirror image at upper boundary

## Exit Rules

### Immediate Exits
- Regime changes to Red (Volatile Chop)
- NTI_Direction reverses
- SuperTrend flips against position

### Partial Exits
- NTI_TrendPhase shifts to "Cooling"
- Exit 50% of position
- Move stop to breakeven

### Dynamic Exits
- Trail stop with SuperTrend line
- Time stop after 4x average holding period
- Range trades exit at 80% of range

## Performance Targets

- **Sharpe Ratio**: > 2.0
- **Win Rate**: > 60% (trends), > 65% (ranges)
- **Max Drawdown**: < 10%
- **Risk/Reward**: > 1.5
- **Time in Market**: 30-40%

## Golden Setups

High-probability setups (150% position size) require:
- IC_Confidence > 80%
- NTI_Confidence > 85%
- All indicators aligned
- IC_EfficiencyRatio > 0.3
- Recent test of S/R level

## Avoid Trading When

- NTI_StallDetected = True
- IC_BandWidth > 5% (too volatile)
- More than 2 direction changes in last 10 bars
- Friday afternoon (weekend risk)

## Files

- `racs_strategy.py`: Main strategy implementation
- `backtest_racs.py`: Backtesting framework
- `tic.py`: Technical Indicators Custom module
- `indicators.py`: Individual indicator implementations
- `requirements.txt`: Python dependencies

## License

This strategy is for educational purposes. Always test thoroughly before live trading.