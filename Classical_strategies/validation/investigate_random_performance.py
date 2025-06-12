"""
Investigation of why random entries perform so well
Focus on transaction costs, execution mechanics, and data issues
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import random
from datetime import datetime

warnings.filterwarnings('ignore')

class RandomPerformanceInvestigation:
    
    def test_pure_buy_and_hold(self, df):
        """Test buy and hold strategy as baseline"""
        print("\n" + "="*80)
        print("BUY AND HOLD TEST")
        print("="*80)
        
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]
        
        # Calculate returns
        total_return = (final_price - initial_price) / initial_price * 100
        
        # Calculate daily returns for Sharpe
        daily_returns = df['Close'].pct_change().dropna()
        
        # Annualized Sharpe (15-min bars to yearly)
        periods_per_year = 252 * 96  # 96 fifteen-minute bars per day
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        sharpe = mean_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0
        
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Initial price: {initial_price:.5f}")
        print(f"Final price: {final_price:.5f}")
        print(f"Total return: {total_return:.2f}%")
        print(f"Sharpe ratio: {sharpe:.3f}")
        
        return total_return, sharpe
    
    def test_alternating_trades(self, df):
        """Test alternating long/short trades every N bars"""
        print("\n" + "="*80)
        print("ALTERNATING TRADES TEST")
        print("="*80)
        
        trade_interval = 100  # Trade every 100 bars
        trades = []
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0
        capital = 100000
        
        for i in range(trade_interval, len(df), trade_interval):
            current_price = df['Close'].iloc[i]
            
            if position == 0:
                # Enter long
                position = 1
                entry_price = current_price
            elif position == 1:
                # Exit long, enter short
                pnl = (current_price - entry_price) * 1000000  # 1M position
                trades.append(pnl)
                position = -1
                entry_price = current_price
            else:  # position == -1
                # Exit short, enter long
                pnl = (entry_price - current_price) * 1000000
                trades.append(pnl)
                position = 1
                entry_price = current_price
        
        # Close final position
        if position != 0:
            final_price = df['Close'].iloc[-1]
            if position == 1:
                pnl = (final_price - entry_price) * 1000000
            else:
                pnl = (entry_price - final_price) * 1000000
            trades.append(pnl)
        
        # Calculate metrics
        if trades:
            total_pnl = sum(trades)
            total_return = total_pnl / capital * 100
            win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100
            
            print(f"Total trades: {len(trades)}")
            print(f"Total P&L: ${total_pnl:,.0f}")
            print(f"Total return: {total_return:.1f}%")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Avg trade P&L: ${np.mean(trades):,.0f}")
        
        return trades
    
    def analyze_price_movements(self, df):
        """Analyze price movement characteristics"""
        print("\n" + "="*80)
        print("PRICE MOVEMENT ANALYSIS")
        print("="*80)
        
        # Calculate pip movements
        pip_changes = df['Close'].diff() * 10000
        
        # Statistics
        print(f"Average pip change per bar: {pip_changes.mean():.2f}")
        print(f"Std dev of pip changes: {pip_changes.std():.2f}")
        print(f"Max pip change: {pip_changes.max():.1f}")
        print(f"Min pip change: {pip_changes.min():.1f}")
        
        # Percentage of bars with >10 pip moves
        large_moves = pip_changes[abs(pip_changes) > 10]
        print(f"\nBars with >10 pip moves: {len(large_moves)} ({len(large_moves)/len(pip_changes)*100:.1f}%)")
        
        # Check for trending behavior
        up_moves = pip_changes[pip_changes > 0]
        down_moves = pip_changes[pip_changes < 0]
        
        print(f"\nUp moves: {len(up_moves)} ({len(up_moves)/len(pip_changes)*100:.1f}%)")
        print(f"Down moves: {len(down_moves)} ({len(down_moves)/len(pip_changes)*100:.1f}%)")
        
        # Rolling volatility
        rolling_vol = pip_changes.rolling(window=96).std()  # Daily volatility
        print(f"\nAverage daily pip volatility: {rolling_vol.mean():.1f}")
        
        return pip_changes
    
    def test_transaction_costs(self, df, spread_pips=1.0):
        """Test impact of realistic transaction costs"""
        print("\n" + "="*80)
        print(f"TRANSACTION COST TEST (Spread = {spread_pips} pips)")
        print("="*80)
        
        # Simple random trades with spread
        n_trades = 100
        trades = []
        
        for _ in range(n_trades):
            # Random entry and exit points
            entry_idx = random.randint(0, len(df) - 200)
            exit_idx = entry_idx + random.randint(10, 100)
            
            if exit_idx >= len(df):
                continue
            
            entry_price = df['Close'].iloc[entry_idx]
            exit_price = df['Close'].iloc[exit_idx]
            direction = random.choice([1, -1])
            
            if direction == 1:  # Long
                # Pay spread on entry
                entry_price += spread_pips * 0.0001
                pip_move = (exit_price - entry_price) * 10000
            else:  # Short
                # Pay spread on entry
                entry_price -= spread_pips * 0.0001
                pip_move = (entry_price - exit_price) * 10000
            
            pnl = pip_move * 100  # $100 per pip for 1M position
            trades.append(pnl)
        
        # Calculate results
        total_pnl = sum(trades)
        avg_pnl = np.mean(trades)
        win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100
        
        print(f"Total trades: {len(trades)}")
        print(f"Total P&L: ${total_pnl:,.0f}")
        print(f"Average P&L: ${avg_pnl:.0f}")
        print(f"Win rate: {win_rate:.1f}%")
        
        # Expected cost from spread
        expected_cost = n_trades * spread_pips * 100
        print(f"\nExpected spread cost: ${expected_cost:,.0f}")
        print(f"Actual total P&L: ${total_pnl:,.0f}")
        print(f"Net after spread: ${total_pnl - expected_cost:,.0f}")
        
        return trades
    
    def check_data_snooping(self, df):
        """Check for potential data snooping issues"""
        print("\n" + "="*80)
        print("DATA SNOOPING CHECK")
        print("="*80)
        
        # Check if Close prices are used in indicators before they should be known
        # In FX, typically use Bid/Ask or at least acknowledge the spread
        
        # Check for perfect fills
        print("Checking for unrealistic fills...")
        
        # Simulate some trades and check fill prices
        test_trades = 20
        perfect_fills = 0
        
        for _ in range(test_trades):
            idx = random.randint(100, len(df) - 100)
            bar = df.iloc[idx]
            
            # Check if we could get filled at exact Close price
            # In reality, retail traders can't trade at the exact close
            close_price = bar['Close']
            high_price = bar['High']
            low_price = bar['Low']
            
            # For a market order, we'd get filled at ask (long) or bid (short)
            # Close price is typically mid-price
            if close_price == high_price or close_price == low_price:
                perfect_fills += 1
        
        print(f"Bars where close = high or low: {perfect_fills}/{test_trades}")
        
        # Check for after-hours or weekend data
        weekend_bars = df[df.index.dayofweek >= 5]
        print(f"\nWeekend bars in data: {len(weekend_bars)}")
        
        return perfect_fills
    
    def test_random_with_realistic_constraints(self, df):
        """Test random strategy with realistic constraints"""
        print("\n" + "="*80)
        print("REALISTIC RANDOM STRATEGY TEST")
        print("="*80)
        
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=False
        )
        
        # Add realistic constraints
        class RealisticRandomStrategy(OptimizedProdStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.min_bars_between_trades = 20
                self.last_trade_bar = -100
                self.spread_pips = 1.0  # 1 pip spread
                self.slippage_pips = 0.5  # 0.5 pip slippage
                
            def calculate_position_size(self, *args, **kwargs):
                # Fixed 1M position
                return 1_000_000
            
            def execute_trade(self, signal, current_bar, current_idx):
                """Override to add realistic execution costs"""
                # Check minimum bars between trades
                if current_idx - self.last_trade_bar < self.min_bars_between_trades:
                    return None
                
                # Get base price
                price = current_bar['Close']
                
                # Add spread and slippage
                if signal > 0:  # Long
                    entry_price = price + (self.spread_pips + self.slippage_pips) * 0.0001
                else:  # Short
                    entry_price = price - (self.spread_pips + self.slippage_pips) * 0.0001
                
                self.last_trade_bar = current_idx
                
                # Create trade with adjusted entry price
                # Note: This is simplified - need to properly override trade execution
                return entry_price
            
            def generate_signal(self, row, prev_row=None):
                """Random signals with constraints"""
                if random.random() < 0.01:  # 1% chance per bar
                    return 1 if random.random() < 0.5 else -1
                return 0
        
        # Test the realistic random strategy
        strategy = RealisticRandomStrategy(config)
        
        # Add indicators
        test_df = df.iloc[-10000:].copy()  # Last 10k bars
        test_df = TIC.add_neuro_trend_intelligent(test_df)
        test_df = TIC.add_market_bias(test_df)
        test_df = TIC.add_intelligent_chop(test_df)
        
        results = strategy.run_backtest(test_df)
        
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Total Return: {results['total_return']:.1f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
        
        return results
    
    def run_investigation(self):
        """Run complete investigation"""
        print("="*80)
        print("RANDOM PERFORMANCE INVESTIGATION")
        print("="*80)
        print(f"Started: {datetime.now()}")
        
        # Load data
        df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Use recent data
        test_df = df['2022-01-01':'2023-12-31'].copy()
        print(f"\nTest period: {test_df.index[0]} to {test_df.index[-1]}")
        print(f"Total bars: {len(test_df):,}")
        
        # 1. Buy and hold baseline
        bh_return, bh_sharpe = self.test_pure_buy_and_hold(test_df)
        
        # 2. Alternating trades
        alt_trades = self.test_alternating_trades(test_df)
        
        # 3. Price movement analysis
        pip_changes = self.analyze_price_movements(test_df)
        
        # 4. Transaction cost test
        self.test_transaction_costs(test_df, spread_pips=0.0)  # No spread
        self.test_transaction_costs(test_df, spread_pips=1.0)  # 1 pip spread
        self.test_transaction_costs(test_df, spread_pips=2.0)  # 2 pip spread
        
        # 5. Data snooping check
        self.check_data_snooping(test_df)
        
        # 6. Realistic random strategy
        self.test_random_with_realistic_constraints(test_df)
        
        print(f"\nCompleted: {datetime.now()}")


def main():
    investigator = RandomPerformanceInvestigation()
    investigator.run_investigation()


if __name__ == "__main__":
    main()