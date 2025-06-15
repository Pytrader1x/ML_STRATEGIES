"""
Debug exits_in_candle processing
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')

# Load data
df_full = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
df_full.set_index('DateTime', inplace=True)
df = df_full.iloc[-5000:].copy()

# Add indicators
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df)
df = TIC.add_intelligent_chop(df)

# Create strategy with debug enabled
strategy_config = OptimizedStrategyConfig(
    initial_capital=1_000_000, risk_per_trade=0.002, sl_max_pips=10.0,
    sl_atr_multiplier=1.0, tp_atr_multipliers=(0.2, 0.3, 0.5),
    max_tp_percent=0.003, tsl_activation_pips=15, tsl_min_profit_pips=1,
    tsl_initial_buffer_multiplier=1.0, trailing_atr_multiplier=1.2,
    tp_range_market_multiplier=0.5, tp_trend_market_multiplier=0.7,
    tp_chop_market_multiplier=0.3, sl_range_market_multiplier=0.7,
    exit_on_signal_flip=False, partial_profit_before_sl=False,
    debug_decisions=True, use_daily_sharpe=True  # Enable debug
)

# Patch the strategy to add debug logging for exits_in_candle
class DebugStrategy(OptimizedProdStrategy):
    def run_backtest(self, df: pd.DataFrame):
        # Store original check_exit_conditions
        original_check = self.signal_generator.check_exit_conditions
        
        def debug_check_exit_conditions(row, trade, current_time, exits_in_candle=None):
            # Add debug print
            if str(current_time) == '2025-05-02 09:15:00':
                print(f"\n[DEBUG] check_exit_conditions at {current_time}")
                print(f"  exits_in_candle: {exits_in_candle}")
                print(f"  trade.tp_hits: {trade.tp_hits}")
                print(f"  High: {row['High']:.5f}, Low: {row['Low']:.5f}")
                print(f"  TP1: {trade.take_profits[0]:.5f}, TP2: {trade.take_profits[1]:.5f}")
            
            result = original_check(row, trade, current_time, exits_in_candle)
            
            if str(current_time) == '2025-05-02 09:15:00' and result[0]:
                print(f"  -> Exit found: {result[1]}, percent: {result[2]}")
            
            return result
        
        # Replace method
        self.signal_generator.check_exit_conditions = debug_check_exit_conditions
        
        # Run normal backtest
        return super().run_backtest(df)

strategy = DebugStrategy(strategy_config)

# Run only relevant part of data
start_idx = df.index.get_loc(pd.Timestamp('2025-05-02 05:00:00'))
end_idx = df.index.get_loc(pd.Timestamp('2025-05-02 10:00:00'))
subset_df = df.iloc[start_idx:end_idx+1]

print("Running backtest on subset...")
results = strategy.run_backtest(subset_df)