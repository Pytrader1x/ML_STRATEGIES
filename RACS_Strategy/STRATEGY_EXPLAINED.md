# RACS Strategy Explained - Plain English Guide

## What Does This Strategy Do?

The RACS Strategy is like a "rubber band" trading system. It watches for when prices move too fast in one direction and bets they'll snap back to normal - just like a stretched rubber band returns to its original shape.

## The Basic Idea

Imagine you're watching a pendulum. When it swings too far to one side, you know it's going to swing back. That's exactly what this strategy does with prices:

1. **Normal Movement**: Prices usually move within a typical range
2. **Extreme Movement**: Sometimes prices move way too fast (panic selling or euphoric buying)
3. **The Opportunity**: When prices move extremely fast, they often reverse direction

## How It Works - Step by Step

### Step 1: Measuring Price Speed (Momentum)

The strategy looks at how much the price has changed over the last 10 hours (40 fifteen-minute bars):
- If EURUSD was 1.1000 ten hours ago and is now 1.1050, that's a 0.45% increase
- We call this the "momentum"

### Step 2: Comparing to Normal (Z-Score)

We then ask: "Is this momentum normal or extreme?"
- We look at the average momentum over the last 50 bars
- We measure how far today's momentum is from that average
- This measurement is called a "Z-score"

Think of it like measuring temperature:
- Normal day: 70°F (Z-score = 0)
- Hot day: 85°F (Z-score = 1.5)
- Extremely hot: 95°F (Z-score = 3.0)

### Step 3: Entry Signals

#### BUY Signal (Going Long)
When Z-score < -1.5:
- Price has fallen extremely fast
- Like a rubber band stretched too far down
- We BUY expecting it to bounce back up

Example: EURUSD drops 0.5% in 10 hours when it normally moves 0.1%

#### SELL Signal (Going Short)
When Z-score > 1.5:
- Price has risen extremely fast
- Like a rubber band stretched too far up
- We SELL expecting it to fall back down

Example: EURUSD rises 0.5% in 10 hours when it normally moves 0.1%

### Step 4: Exit Signals

We close the position when Z-score returns between -0.5 and 0.5:
- This means the extreme condition has passed
- The rubber band is returning to normal
- Time to take our profit (or loss)

## Real Trading Example

Let's walk through an actual trade:

**Day 1, 10:00 AM**
- EURUSD = 1.1000
- Momentum over last 10 hours = +0.1% (normal)
- Z-score = 0.2
- Action: No trade

**Day 1, 2:00 PM**
- EURUSD = 1.0950 (big drop!)
- Momentum = -0.45% (extreme negative)
- Z-score = -2.1
- Action: BUY signal! Open long position

**Day 1, 3:30 PM**
- EURUSD = 1.0965 (recovering)
- Momentum = -0.25%
- Z-score = -0.4
- Action: EXIT signal! Close position
- Profit: Bought at 1.0950, sold at 1.0965 = +0.14% profit

## Why It Works

1. **Market Psychology**: When prices move too fast, traders often overreact
2. **Mean Reversion**: Markets tend to return to average behavior
3. **High Probability**: Small moves happen more often than large ones

## Performance Summary

Testing across 8 major currency pairs:
- **Win Rate**: About 52% (slightly more wins than losses)
- **Average Trade**: Lasts about 1 hour
- **Risk-Adjusted Returns**: 5 out of 8 currencies showed excellent performance
- **Best Performer**: AUDNZD with Sharpe ratio of 4.36

## Important Notes

### What Makes a Good Trade
- Clear extreme movement (high Z-score)
- Normal market conditions (not during major news)
- Sufficient liquidity

### What to Avoid
- Trading during major economic announcements
- Holding positions overnight (increased risk)
- Using too much leverage

### Risk Management
- Each trade risks a small amount (typically 1-2% of capital)
- Multiple small wins add up over time
- Losses are cut quickly when momentum doesn't normalize

## Visual Guide to the Strategy

```
Price Movement:
    ↑ 
    | Extreme High (+1.5 Z-score)
    | → SELL HERE
    |
----+---- Normal Range (-0.5 to +0.5)
    |
    | → BUY HERE
    | Extreme Low (-1.5 Z-score)
    ↓

Exit when price returns to normal range
```

## Common Questions

**Q: How often does it trade?**
A: About 7,000 trades per year per currency pair (roughly 20-30 per day)

**Q: What's the typical profit per trade?**
A: Small - usually 0.1% to 0.3% - but they add up

**Q: Does it work in all markets?**
A: Best in ranging markets, struggles in strong trends

**Q: Can I trade this manually?**
A: Difficult - it requires monitoring 24/5 and quick execution

## The Bottom Line

This strategy is like being a "market shock absorber" - when prices move too violently in one direction, we bet on them returning to normal. It's not about predicting the future, but about exploiting temporary extremes in market behavior.