# Metrics Calculation Files Summary

This document summarizes the files that contain metrics calculations, particularly consecutive wins/losses calculations.

## Primary Files with Metrics Calculations

### 1. `/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/run_Strategy.py`
- **Function**: `calculate_trade_statistics()` (lines 126-202)
- **Key Metrics Calculated**:
  - Maximum consecutive wins
  - Average consecutive wins
  - Maximum consecutive losses
  - Average consecutive losses
  - Number of wins and losses
- **Implementation**: 
  - Iterates through trade P&L values
  - Tracks win/loss streaks
  - Handles both Trade objects and dictionaries
  - Returns comprehensive statistics dictionary

### 2. `/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/strategy_code/Prod_strategy.py`
- **Function**: `_calculate_performance_metrics()` (lines 881-937)
- **Key Metrics Calculated**:
  - Total trades, winning trades, losing trades
  - Win rate percentage
  - Total P&L and return
  - Average win/loss amounts
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio
  - Exit reason breakdown
- **Returns**: Dictionary with 'trades' key containing list of Trade objects

### 3. `/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/run_Strategy.py` (Crypto Section)
- **Function**: `_calculate_performance_metrics()` for FinalCryptoStrategy (lines 981-1055)
- **Key Metrics Calculated**:
  - Similar to Prod_strategy but for crypto trades
  - Adjusted Sharpe ratio calculation for crypto (365 * 96 periods)
  - Returns dictionary with trade results for compatibility

### 4. `/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/validation/crypto_validation_50loops.py`
- **Lines**: 200-210
- **Calculates**: Consecutive positive/negative Sharpe ratios across validation loops
- **Purpose**: Validation of strategy consistency

### 5. `/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/analysis/deep_audusd_validation.py`
- Validates Monte Carlo results statistically
- Checks for suspicious patterns in consecutive results
- Uses runs test for randomness validation

## Trade Object Structure

The Trade dataclass (in Prod_strategy.py) contains:
```python
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: TradeDirection
    position_size: float
    stop_loss: float
    take_profits: List[float]
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    exit_reason: Optional[ExitReason]
    pnl: Optional[float]  # Dollar P&L
    pnl_percent: Optional[float]  # Percentage P&L
    # ... other fields
```

## Consecutive Wins/Losses Algorithm

The algorithm in `calculate_trade_statistics()`:

1. **Extract P&L values** from trades (handles both objects and dicts)
2. **Create binary arrays** for wins (P&L > 0) and losses (P&L < 0)
3. **Track streaks**:
   - Iterate through wins/losses
   - Increment current streak when consecutive
   - Save streak when it ends
   - Handle final streak if still active
4. **Calculate statistics**:
   - Max consecutive: `max(streaks)`
   - Average consecutive: `mean(streaks)`

## Usage Example

```python
from run_Strategy import calculate_trade_statistics

# After running a backtest
results = strategy.run_backtest(df)

# Calculate detailed statistics
trade_stats = calculate_trade_statistics(results)

print(f"Max consecutive wins: {trade_stats['max_consecutive_wins']}")
print(f"Max consecutive losses: {trade_stats['max_consecutive_losses']}")
print(f"Avg consecutive wins: {trade_stats['avg_consecutive_wins']:.1f}")
print(f"Avg consecutive losses: {trade_stats['avg_consecutive_losses']:.1f}")
```

## Monte Carlo Integration

In `run_single_monte_carlo()` (run_Strategy.py):
- Calls `calculate_trade_statistics()` for each iteration
- Aggregates consecutive win/loss stats across all iterations
- Stores in results DataFrame attributes for reporting

## Files That Display These Metrics

1. **run_Strategy.py**: Main runner displays consecutive stats in output
2. **validation scripts**: Use metrics for robustness testing
3. **analysis scripts**: Include consecutive metrics in reports

## Notes

- The metrics calculation handles edge cases (empty trades, all wins, all losses)
- Compatible with both forex and crypto strategies
- Provides insight into strategy consistency and risk profile
- Critical for understanding drawdown periods and winning streaks