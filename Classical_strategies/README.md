# Classical Trading Strategy - Production Ready

A high-performance algorithmic trading strategy with comprehensive position tracking, advanced risk management, and proven profitability across multiple market conditions.

## 🎯 Strategy Overview

This strategy combines three sophisticated indicators to identify high-probability trading opportunities:
- **Neuro Trend Intelligent (NTI)**: Advanced trend detection
- **Market Bias (MB)**: Market structure analysis  
- **Intelligent Chop (IC)**: Market regime identification

## 📊 Performance Summary

**Monte Carlo Analysis (50 iterations, 20K samples each):**

| Configuration | Avg Sharpe | Avg Return | Win Rate | Profitable Runs | Max Drawdown |
|---------------|------------|------------|----------|-----------------|--------------|
| **Ultra-Tight Risk** | 2.165 ± 1.081 | 2.52% ± 2.29% | 72.3% ± 9.0% | 98% | 0.66% ± 0.29% |
| **Scalping Strategy** | 2.485 ± 1.321 | 2.81% ± 2.52% | 57.4% ± 5.0% | 98% | 0.51% ± 0.21% |

**✅ 100% Position Integrity** - Zero over-exiting issues across all 100 Monte Carlo runs

## 🚀 Quick Start

```bash
# Run single currency analysis
python run_Strategy.py

# Run with specific date range
python run_Strategy.py --date-range 2024-01-01 2024-12-31

# Run multiple currencies
python run_Strategy.py --mode multi

# Show interactive charts
python run_Strategy.py --show-plots

# Monte Carlo with custom parameters
python run_Strategy.py --iterations 100 --sample-size 10000
```

## 📈 Trading Logic

### Entry Conditions

**Standard Entry (Both indicators must align):**
- NTI Direction: +1 (Long) or -1 (Short)
- Market Bias: +1 (Long) or -1 (Short)  
- IC Regime: 1 or 2 (Trending market)

**Relaxed Entry (Relaxed mode only):**
- NTI Direction: +1 (Long) or -1 (Short)
- Used in choppy market conditions

### Position Sizing

**Base Position Sizes:**
- Default: 1M units (1 million currency units)
- Intelligent Sizing (if enabled):
  - Very Low Confidence: 1M units
  - Low Confidence: 1M units  
  - Medium Confidence: 3M units
  - High Confidence: 5M units

**Risk Management:**
- Risk per trade: 0.1-0.2% of capital
- Maximum stop loss: 45 pips (Ultra-Tight) / 5 pips (Scalping)
- Margin requirement: 1% of position size

### Take Profit (TP) System

**Three-Level TP Structure:**
1. **TP1**: 33.33% of position at first target
2. **TP2**: 50% of remaining (33.33% of original) at second target  
3. **TP3**: 100% of remaining (33.33% of original) at third target

**TP Level Calculation:**
```
TP Distance = ATR × TP_Multiplier × Market_Regime_Adjustment
TP1 Multiplier: 0.8 (Ultra-Tight: 0.2)
TP2 Multiplier: 1.5 (Ultra-Tight: 0.3)  
TP3 Multiplier: 2.5 (Ultra-Tight: 0.5)
```

**Market Regime Adjustments:**
- **Trending Markets**: 1.0× (normal TPs)
- **Ranging Markets**: 0.7× (tighter TPs)
- **Choppy Markets**: 0.5× (very tight TPs)

### Stop Loss (SL) System

**Initial Stop Loss Calculation:**
```
SL Distance = min(ATR × SL_Multiplier, Max_Pips × Pip_Size)
```

**SL Parameters:**
- **Ultra-Tight**: 1.0× ATR, max 10 pips
- **Scalping**: 0.5× ATR, max 5 pips

**Market Bias Integration:**
- Uses Market Bias structural levels when available
- Adds 0.5 pip buffer to MB levels
- Takes more conservative of ATR-based or MB-based stop

### Trailing Stop Loss (TSL)

**Activation Conditions:**
- **Ultra-Tight**: Activates after 3 pips profit
- **Scalping**: Activates after 2 pips profit

**TSL Calculation:**
```
TSL Distance = ATR × 1.2 (trailing_atr_multiplier)
Min Profit Guarantee = Entry ± 5 pips (Ultra-Tight) / 0.5 pips (Scalping)
```

**Trailing Logic:**
1. **Initial Buffer**: 2.0× ATR distance on first activation
2. **Subsequent Updates**: 1.2× ATR distance 
3. **Profit Protection**: Guarantees minimum profit once activated
4. **Conservative Update**: Only moves in favorable direction

### Exit Scenarios

**1. Pure Take Profit (All 3 TPs Hit):**
- Exit 33.33% at each TP level
- Most profitable scenario

**2. Partial TP + Stop Loss:**
- Hit TP1/TP2, remainder stopped out
- Mixed outcome depending on remaining position PnL

**3. Pure Stop Loss:**
- No TPs hit, exit via initial stop loss
- Can be loss, breakeven, or small profit

**4. Trailing Stop Loss:**
- Exit via TSL after profit target reached
- Typically profitable (16-29% of SL exits)

**5. Signal Flip:**
- Exit when entry signals reverse
- Requires minimum profit and time thresholds

## 💰 PnL Calculation

**FOREX Standard:**
```
Price Change (Pips) = (Exit Price - Entry Price) × 10,000
Position Size (Millions) = Position Size / 1,000,000
PnL ($) = Millions × $100/pip × Price Change (Pips)
```

**Example:**
- Position: 3M AUDUSD Long @ 0.6500
- Exit: 0.6515 (TP1: 1M at +15 pips)
- PnL = 1 × $100 × 15 = $1,500

## 🛡️ Risk Management

### Position Tracking
- **Enhanced Trade Class**: Tracks initial size, total exited, remaining
- **Exit History**: Detailed log of every partial/full exit
- **Safety Checks**: Prevents over-exiting beyond position size
- **Verification**: Entry size = Total exit size for every trade

### Slippage Modeling (Realistic Costs Mode)
- **Market Entry**: 0-0.5 pips random slippage
- **Stop Loss**: 0-2.0 pips slippage (worse fill)
- **Trailing Stop**: 0-1.0 pips slippage  
- **Take Profit**: 0 pips (limit orders, perfect fill)

### Dynamic Adjustments
- **Volatility**: Widen stops in high volatility, tighten in low
- **Market Regime**: Tighter parameters in ranging/choppy markets
- **Time Filters**: Prevent immediate re-entry or signal flip exits

## 📁 Project Structure

```
Classical_strategies/
├── run_Strategy.py           # Main strategy runner
├── strategy_code/           # Core strategy implementation
│   ├── Prod_strategy_fixed.py  # Fixed strategy (USE THIS)
│   ├── Prod_strategy.py        # Original (has bugs)
│   └── Prod_plotting.py        # Visualization tools
├── analysis/               # Analysis and research scripts
├── results/               # CSV outputs and analysis results  
├── charts/               # Generated visualizations
├── Validation/          # Validation and testing scripts
└── archive/            # Historical versions and experiments
```

## 🔧 Configuration Options

### Ultra-Tight Risk Management
```python
Config 1: Ultra-Tight Risk Management
- Risk per trade: 0.2%
- Max stop loss: 10 pips
- TP multipliers: (0.2, 0.3, 0.5)  
- TSL activation: 3 pips
- Focus: Capital preservation, high win rate
```

### Scalping Strategy  
```python
Config 2: Scalping Strategy
- Risk per trade: 0.1%
- Max stop loss: 5 pips
- TP multipliers: (0.1, 0.2, 0.3)
- TSL activation: 2 pips  
- Focus: Frequent small profits, tight risk
```

## 📊 Supported Instruments

**Primary Focus:** AUDUSD (extensively tested and optimized)

**Multi-Currency Support:**
- GBPUSD, EURUSD, USDJPY, NZDUSD, USDCAD
- EURJPY, GBPJPY, AUDJPY, CADJPY, CHFJPY
- EURGBP, AUDNZD

**Data Requirements:**
- 15-minute timeframe (384 MB per currency pair)
- OHLC + timestamp format
- Minimum 1 year of data recommended

## 🧪 Validation & Testing

The strategy has undergone comprehensive validation:

- **✅ Position Integrity**: 100% pass rate across 100 Monte Carlo runs
- **✅ Mathematical Verification**: All PnL and sizing calculations verified
- **✅ Backtesting Accuracy**: No look-ahead bias or data leakage
- **✅ Real Market Conditions**: Includes slippage and execution costs
- **✅ Multiple Timeframes**: Tested across 15+ years of market data

## 🚨 Critical Notes

1. **Use Fixed Strategy**: Always use `Prod_strategy_fixed.py` - the original had position sizing bugs
2. **Position Verification**: The fixed version includes comprehensive position tracking
3. **Realistic Expectations**: Returns are measured over ~10-month periods in Monte Carlo analysis
4. **Market Adaptation**: Strategy adapts to different market regimes automatically
5. **Risk Management**: Never risk more than you can afford to lose

## 📚 Additional Resources

- **Detailed Analysis**: See `analysis/` directory for comprehensive research
- **Bug Fix Documentation**: `analysis/POSITION_SIZING_FIX_SUMMARY.md`
- **Trading Logic Deep Dive**: `analysis/TP_EXIT_BUG_ANALYSIS.md`  
- **Performance Reports**: `analysis/FINAL_ANALYSIS_SUMMARY.md`

## 🤝 Contributing

This is a production trading system. All modifications should be thoroughly tested using the validation framework in `Validation/` directory before deployment.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade first and never risk more than you can afford to lose.

---

**Last Updated:** June 2025  
**Version:** 2.1 (Fixed)  
**Status:** Production Ready ✅