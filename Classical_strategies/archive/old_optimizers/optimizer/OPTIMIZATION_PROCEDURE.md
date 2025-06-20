# OPTIMIZATION PROCEDURE AND TRACKING

## Overview
This document tracks the iterative optimization process for trading strategies, maintaining a record of what works and what doesn't.

## Procedure Steps

1. **Run Optimization** (max 2 minutes per iteration)
   - Use multiprocessing with limited iterations (5-10 per run)
   - Small sample sizes (2000-3000) for speed
   - Focus on specific parameter subsets based on previous results

2. **Analyze Results**
   - Review Sharpe ratio, win rate, max drawdown
   - Identify which parameters correlate with good performance
   - Note patterns in successful configurations

3. **Adjust Focus**
   - Narrow parameter ranges around successful values
   - Explore related parameters that might improve results
   - Avoid parameter combinations that consistently fail

4. **Document Progress**
   - Update this file after each iteration
   - Track best configurations found
   - Note insights and patterns

## Optimization Runs

### Run 1 - Baseline (2024-06-15 12:50)
**Focus**: Initial broad search
**Parameters**: All parameters with default ranges
**Results**: 
- Best Sharpe: 1.536
- Best params: risk=0.32%, sl_min=5.5, sl_max=25, tp1=0.225, tp2=0.25, tp3=0.95
- Insights: Wider stops work better, smaller TP1 helps hit rate

### Run 2 - [PENDING]
**Focus**: Refine around best parameters from Run 1
**Parameters**: 
- risk_per_trade: 0.25-0.35%
- sl_min_pips: 5-8
- sl_max_pips: 20-30
- tp1_multiplier: 0.15-0.3
**Results**: [TO BE FILLED]

### Run 3 - [PENDING]
### Run 4 - [PENDING]
### Run 5 - [PENDING]
### Run 6 - [PENDING]
### Run 7 - [PENDING]
### Run 8 - [PENDING]
### Run 9 - [PENDING]
### Run 10 - [PENDING]

## Best Configurations Found

### Strategy 1 - Ultra-Tight Risk
1. **Config 1** (Sharpe: 1.536)
   - risk_per_trade: 0.32%
   - sl_min_pips: 5.5
   - sl_max_pips: 25.0
   - tp_multipliers: (0.225, 0.25, 0.95)
   - partial_profit_ratio: 0.4 @ 70%

### Strategy 2 - Scalping
[TO BE FILLED]

## Key Insights

1. **Stop Loss**: Wider stops (20-25 pips max) reduce premature exits
2. **Take Profits**: Closer TP1 (0.2-0.3 ATR) improves hit rate
3. **Risk**: 0.2-0.4% per trade seems optimal
4. **Partial Profits**: 40-70% position reduction at 40-70% to SL works well

## Failed Approaches

1. Very tight stops (< 5 pips) lead to excessive stop-outs
2. Large TP1 multipliers (> 0.5) reduce profitability
3. Risk > 0.5% increases drawdowns significantly