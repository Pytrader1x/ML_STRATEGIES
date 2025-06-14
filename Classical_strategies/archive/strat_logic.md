# Trading Strategy Logic Documentation

## Overview

This document details the entry and exit logic for two trading strategies implemented in the production trading system:

1. **Strategy 1: Ultra-Tight Risk Management** - Conservative approach with small, frequent profits
2. **Strategy 2: Scalping Strategy** - High-frequency trading with tight stops and quick exits

Both strategies use three technical indicators:
- **NTI (Neuro Trend Intelligent)** - Trend direction indicator
- **MB (Market Bias)** - Market sentiment indicator
- **IC (Intelligent Chop)** - Market regime classifier

---

## Strategy 1: Ultra-Tight Risk Management

### Configuration Parameters
```python
initial_capital = $1,000,000
risk_per_trade = 0.2% ($2,000)
position_size = 1.0M units (fixed)
```

### Entry Criteria

#### Standard Entry (3-Way Confluence Required)
All three indicators must align for entry:

**LONG Entry:**
- NTI_Direction = 1 (bullish)
- MB_Bias = 1 (bullish)
- IC_Regime = 1 or 2 (trending market)

**SHORT Entry:**
- NTI_Direction = -1 (bearish)
- MB_Bias = -1 (bearish)
- IC_Regime = 1 or 2 (trending market)

> **Note:** Relaxed mode is disabled (`relaxed_mode=False`), so NTI alone cannot trigger entries.

### Exit Criteria

#### 1. Stop Loss (SL)
- **Base SL:** 1.0 × ATR
- **Maximum:** 10 pips
- **Adjustments:**
  - Range markets: 0.7× multiplier (tighter stops)
  - Volatility adjustment: Dynamic based on current vs recent ATR
  - Market Bias protection: Uses MB high/low as alternative SL levels

#### 2. Take Profit Levels (TP)
Three TP levels with dynamic adjustments:
- **TP1:** 0.2 × ATR
- **TP2:** 0.3 × ATR  
- **TP3:** 0.5 × ATR

**Market Regime Adjustments:**
- Trending markets: 0.7× multiplier
- Ranging markets: 0.5× multiplier
- Choppy markets: 0.3× multiplier

**Exit Sizing:**
- TP1: Exit 33% of position
- TP2: Exit 33% of position
- TP3: Exit remaining 34% of position

#### 3. Trailing Stop Loss (TSL)
- **Activation:** 3 pips profit
- **Minimum Profit:** 1 pip guaranteed
- **Trail Distance:** 0.8 × ATR
- **Initial Buffer:** 1.0× multiplier on first activation

#### 4. Partial Profit Taking (PPT)
- **Trigger:** When price reaches 50% of distance to SL
- **Exit Size:** 50% of position
- **Purpose:** Lock in profits before reaching TP1

#### 5. Signal Flip Exit
- **Disabled** for this strategy (`exit_on_signal_flip=False`)

### Exit Priority Order
1. Take Profit levels (if reached)
2. Stop Loss / Trailing Stop
3. Partial Profit Taking (PPT)
4. End of data

### Risk Management
- **Slippage (Realistic Costs Mode):**
  - Entry: 0-0.5 pips random slippage
  - Stop Loss: 0-2.0 pips slippage
  - Trailing Stop: 0-1.0 pips slippage
  - Take Profit: 0 pips (limit orders)

---

## Strategy 2: Scalping Strategy

### Configuration Parameters
```python
initial_capital = $1,000,000
risk_per_trade = 0.1% ($1,000)
position_size = 1.0M units (fixed)
```

### Entry Criteria

#### Standard Entry (3-Way Confluence Required)
Same as Strategy 1 - all three indicators must align:

**LONG Entry:**
- NTI_Direction = 1 (bullish)
- MB_Bias = 1 (bullish)
- IC_Regime = 1 or 2 (trending market)

**SHORT Entry:**
- NTI_Direction = -1 (bearish)
- MB_Bias = -1 (bearish)
- IC_Regime = 1 or 2 (trending market)

### Exit Criteria

#### 1. Stop Loss (SL)
- **Base SL:** 0.5 × ATR (tighter than Strategy 1)
- **Maximum:** 5 pips (half of Strategy 1)
- **Adjustments:**
  - Range markets: 0.5× multiplier
  - Volatility adjustment: Enabled
  - Market Bias protection: Active

#### 2. Take Profit Levels (TP)
Tighter TP levels for quick scalping:
- **TP1:** 0.1 × ATR
- **TP2:** 0.2 × ATR
- **TP3:** 0.3 × ATR

**Market Regime Adjustments:**
- Trending markets: 0.5× multiplier
- Ranging markets: 0.3× multiplier
- Choppy markets: 0.2× multiplier

**Exit Sizing:** Same as Strategy 1 (33%/33%/34%)

#### 3. Trailing Stop Loss (TSL)
- **Activation:** 2 pips profit (faster than Strategy 1)
- **Minimum Profit:** 0.5 pips guaranteed
- **Trail Distance:** 0.5 × ATR (tighter trail)
- **Initial Buffer:** 0.5× multiplier

#### 4. Partial Profit Taking (PPT)
- **Trigger:** When price reaches 30% of distance to SL (more aggressive)
- **Exit Size:** 70% of position (larger partial)
- **Purpose:** Aggressively lock in small profits

#### 5. Signal Flip Exit
- **Enabled** for this strategy
- **Minimum Profit:** 0 pips (immediate exit allowed)
- **Minimum Time:** 0 hours (no time restriction)
- **Exit Percent:** 100% (full exit on signal flip)
- **Momentum Threshold:** 0.7 (for full vs partial exit decision)

### Exit Priority Order
1. Signal Flip (if enabled)
2. Take Profit levels
3. Stop Loss / Trailing Stop
4. Partial Profit Taking (PPT)
5. End of data

---

## Key Differences Between Strategies

| Feature | Strategy 1 (Ultra-Tight) | Strategy 2 (Scalping) |
|---------|-------------------------|----------------------|
| Risk per Trade | 0.2% | 0.1% |
| Max Stop Loss | 10 pips | 5 pips |
| SL ATR Multiplier | 1.0× | 0.5× |
| TP Levels (ATR) | 0.2, 0.3, 0.5 | 0.1, 0.2, 0.3 |
| TSL Activation | 3 pips | 2 pips |
| TSL Min Profit | 1 pip | 0.5 pips |
| PPT Trigger | 50% to SL | 30% to SL |
| PPT Exit Size | 50% | 70% |
| Signal Flip Exit | Disabled | Enabled |

---

## Typical Trade Flow

### Strategy 1 Example:
1. **Entry:** All 3 indicators align → Enter 1M position
2. **PPT:** Price moves 5 pips → Exit 0.5M (50%)
3. **TSL Activation:** Price moves 3+ pips → TSL activated
4. **Exit:** TSL hit → Exit remaining 0.5M

**Result:** Two exits, small consistent profits, rarely hits full TP levels

### Strategy 2 Example:
1. **Entry:** All 3 indicators align → Enter 1M position
2. **PPT:** Price moves 1.5 pips → Exit 0.7M (70%)
3. **Signal Flip:** Indicators reverse → Exit remaining 0.3M

**Result:** Quick exits, high turnover, minimal exposure time

---

## Performance Characteristics

### Strategy 1:
- **Win Rate:** Typically 65-75%
- **Average Win:** ~$700-900
- **Average Loss:** ~$1,000
- **Sharpe Ratio:** 3.0-4.0 (excellent)
- **Trade Frequency:** Lower
- **Hold Time:** Hours to days

### Strategy 2:
- **Win Rate:** Typically 60-70%
- **Average Win:** ~$400-600
- **Average Loss:** ~$500-600
- **Sharpe Ratio:** 1.5-2.5 (good)
- **Trade Frequency:** Higher
- **Hold Time:** Minutes to hours

---

## Important Notes

1. **No Actual TP Hits:** Due to aggressive PPT and tight TSL, trades rarely reach full TP levels
2. **PPT Dominance:** Most profits come from partial profit taking before TP1
3. **TSL Exits:** Majority of final exits are via trailing stop
4. **Risk Control:** Both strategies prioritize capital preservation over maximum profit
5. **Market Conditions:** Best performance in trending markets (IC Regime 1-2)

This design philosophy focuses on consistent small wins rather than home runs, which explains why the charts show many TSL exits and PPT events rather than full TP completions.