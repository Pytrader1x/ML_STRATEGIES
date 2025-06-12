"""
Deep Individual Trade Analysis
Examines 20 random trades in extreme detail to understand entry/exit mechanics
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
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class TradeForensics:
    """Forensic analysis of individual trades"""
    
    def __init__(self):
        self.trades_analyzed = []
        
    def create_strategy_with_trade_capture(self):
        """Create strategy that captures detailed trade information"""
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            tsl_activation_pips=3,
            tsl_min_profit_pips=1,
            verbose=True
        )
        
        class TradeCapturingStrategy(OptimizedProdStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.trade_details = []
                self.bar_data = []
                
            def run_backtest(self, df):
                """Override to capture bar-by-bar data"""
                self.df = df.copy()
                self.bar_data = []
                
                # Run normal backtest
                results = super().run_backtest(df)
                
                # Store detailed trade info
                if hasattr(self, 'trades') and self.trades:
                    for trade in self.trades:
                        if hasattr(trade, 'entry_time'):
                            # Find entry bar index
                            entry_idx = df.index.get_loc(trade.entry_time)
                            exit_idx = df.index.get_loc(trade.exit_time) if hasattr(trade, 'exit_time') else None
                            
                            # Capture surrounding bars
                            context_start = max(0, entry_idx - 10)
                            context_end = min(len(df), (exit_idx if exit_idx else entry_idx) + 10)
                            
                            trade_detail = {
                                'trade': trade,
                                'entry_idx': entry_idx,
                                'exit_idx': exit_idx,
                                'context_bars': df.iloc[context_start:context_end].copy(),
                                'entry_bar': df.iloc[entry_idx].copy(),
                                'exit_bar': df.iloc[exit_idx].copy() if exit_idx else None
                            }
                            
                            self.trade_details.append(trade_detail)
                
                return results
        
        return TradeCapturingStrategy(config)
    
    def analyze_single_trade(self, trade_detail, trade_num):
        """Deep analysis of a single trade"""
        print(f"\n{'='*80}")
        print(f"TRADE #{trade_num} FORENSIC ANALYSIS")
        print(f"{'='*80}")
        
        trade = trade_detail['trade']
        entry_bar = trade_detail['entry_bar']
        exit_bar = trade_detail['exit_bar']
        context_bars = trade_detail['context_bars']
        
        # Basic trade info
        print(f"\n1. TRADE SUMMARY")
        print(f"   Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
        if hasattr(trade, 'exit_time'):
            print(f"   Exit:  {trade.exit_time} @ {trade.exit_price:.5f}")
            print(f"   Duration: {trade.exit_time - trade.entry_time}")
        print(f"   Direction: {'LONG' if trade.direction == 1 else 'SHORT'}")
        print(f"   P&L: ${trade.pnl:.2f}")
        
        # Entry analysis
        print(f"\n2. ENTRY ANALYSIS")
        print(f"   Entry Bar OHLC: O={entry_bar['Open']:.5f}, H={entry_bar['High']:.5f}, L={entry_bar['Low']:.5f}, C={entry_bar['Close']:.5f}")
        print(f"   Indicators at entry:")
        print(f"     - NTI_Direction: {entry_bar.get('NTI_Direction', 'N/A')}")
        print(f"     - MB_Bias: {entry_bar.get('MB_Bias', 'N/A')}")
        print(f"     - IC_Signal: {entry_bar.get('IC_Signal', 'N/A')}")
        
        # Check entry logic
        entry_valid = self.validate_entry_logic(trade, entry_bar)
        print(f"   Entry logic valid: {'✅ YES' if entry_valid else '❌ NO'}")
        
        # Pre-entry context
        print(f"\n3. PRE-ENTRY CONTEXT (5 bars before)")
        entry_idx_in_context = context_bars.index.get_loc(trade.entry_time)
        for i in range(max(0, entry_idx_in_context-5), entry_idx_in_context):
            bar = context_bars.iloc[i]
            print(f"   {bar.name}: C={bar['Close']:.5f}, NTI={bar.get('NTI_Direction', 'N/A'):>2}, MB={bar.get('MB_Bias', 'N/A'):>2}")
        
        # Exit analysis
        if exit_bar is not None:
            print(f"\n4. EXIT ANALYSIS")
            print(f"   Exit Bar OHLC: O={exit_bar['Open']:.5f}, H={exit_bar['High']:.5f}, L={exit_bar['Low']:.5f}, C={exit_bar['Close']:.5f}")
            print(f"   Exit reason: {getattr(trade, 'exit_reason', 'Unknown')}")
            
            # Calculate actual vs expected exit
            if trade.direction == 1:  # Long
                pip_move = (trade.exit_price - trade.entry_price) * 10000
                max_favorable = (exit_bar['High'] - trade.entry_price) * 10000
                max_adverse = (trade.entry_price - exit_bar['Low']) * 10000
            else:  # Short
                pip_move = (trade.entry_price - trade.exit_price) * 10000
                max_favorable = (trade.entry_price - exit_bar['Low']) * 10000
                max_adverse = (exit_bar['High'] - trade.entry_price) * 10000
            
            print(f"   Pip movement: {pip_move:.1f} pips")
            print(f"   Max favorable excursion: {max_favorable:.1f} pips")
            print(f"   Max adverse excursion: {max_adverse:.1f} pips")
            
            # Check if exit makes sense
            exit_valid = self.validate_exit_logic(trade, exit_bar, context_bars)
            print(f"   Exit logic valid: {'✅ YES' if exit_valid else '❌ NO'}")
        
        # Price action analysis
        print(f"\n5. PRICE ACTION ANALYSIS")
        self.analyze_price_action(trade, context_bars)
        
        # Risk/Reward analysis
        print(f"\n6. RISK/REWARD ANALYSIS")
        if hasattr(trade, 'stop_loss') and hasattr(trade, 'take_profits'):
            sl_pips = abs(trade.entry_price - trade.stop_loss) * 10000
            tp1_pips = abs(trade.take_profits[0] - trade.entry_price) * 10000 if len(trade.take_profits) > 0 else 0
            
            print(f"   Stop Loss: {sl_pips:.1f} pips")
            print(f"   Take Profit 1: {tp1_pips:.1f} pips")
            print(f"   Risk/Reward: 1:{tp1_pips/sl_pips:.2f}" if sl_pips > 0 else "   Risk/Reward: N/A")
        
        # Anomaly detection
        anomalies = self.detect_anomalies(trade, context_bars)
        if anomalies:
            print(f"\n⚠️  ANOMALIES DETECTED:")
            for anomaly in anomalies:
                print(f"   - {anomaly}")
        
        return entry_valid, exit_valid, anomalies
    
    def validate_entry_logic(self, trade, entry_bar):
        """Validate if entry follows strategy rules"""
        nti = entry_bar.get('NTI_Direction', 0)
        mb = entry_bar.get('MB_Bias', 0)
        ic = entry_bar.get('IC_Signal', 0)
        
        # For long trades
        if trade.direction == 1:
            return nti > 0 and mb > 0
        # For short trades
        else:
            return nti < 0 and mb < 0
    
    def validate_exit_logic(self, trade, exit_bar, context_bars):
        """Validate if exit follows strategy rules"""
        exit_reason = getattr(trade, 'exit_reason', 'Unknown')
        
        if 'STOP_LOSS' in str(exit_reason):
            # Check if stop loss was actually hit
            if trade.direction == 1:
                return exit_bar['Low'] <= trade.stop_loss
            else:
                return exit_bar['High'] >= trade.stop_loss
                
        elif 'TAKE_PROFIT' in str(exit_reason):
            # Check if take profit was hit
            if hasattr(trade, 'take_profits') and len(trade.take_profits) > 0:
                if trade.direction == 1:
                    return exit_bar['High'] >= trade.take_profits[0]
                else:
                    return exit_bar['Low'] <= trade.take_profits[0]
        
        return True  # Other exit reasons assumed valid
    
    def analyze_price_action(self, trade, context_bars):
        """Analyze price action around the trade"""
        entry_idx = context_bars.index.get_loc(trade.entry_time)
        
        # Calculate volatility
        returns = context_bars['Close'].pct_change()
        volatility = returns.std() * np.sqrt(96 * 252)  # Annualized
        print(f"   Context volatility: {volatility*100:.1f}% annualized")
        
        # Check for gaps
        gaps = []
        for i in range(1, len(context_bars)):
            gap_size = abs(context_bars.iloc[i]['Open'] - context_bars.iloc[i-1]['Close'])
            if gap_size > 0.0010:  # 10 pip gap
                gaps.append((context_bars.index[i], gap_size * 10000))
        
        if gaps:
            print(f"   Gaps detected: {len(gaps)}")
            for time, size in gaps[:3]:  # Show first 3
                print(f"     - {time}: {size:.1f} pips")
        
        # Trend analysis
        if entry_idx >= 5:
            pre_entry_trend = context_bars['Close'].iloc[entry_idx-5:entry_idx].pct_change().sum()
            print(f"   5-bar pre-entry trend: {pre_entry_trend*100:.2f}%")
    
    def detect_anomalies(self, trade, context_bars):
        """Detect any anomalies in the trade"""
        anomalies = []
        
        # Check for instant profit
        if hasattr(trade, 'pnl') and hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
            trade_duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
            if trade.pnl > 500 and trade_duration < 30:  # $500+ profit in < 30 min
                anomalies.append(f"Large profit (${trade.pnl:.0f}) in short time ({trade_duration:.0f} min)")
        
        # Check for perfect entries/exits
        entry_bar = context_bars.loc[trade.entry_time]
        if trade.direction == 1 and abs(trade.entry_price - entry_bar['Low']) < 0.0002:
            anomalies.append("Entry near perfect low of bar")
        elif trade.direction == -1 and abs(trade.entry_price - entry_bar['High']) < 0.0002:
            anomalies.append("Entry near perfect high of bar")
        
        # Check for unrealistic fills
        if hasattr(trade, 'partial_exits'):
            for partial in trade.partial_exits:
                if hasattr(partial, 'price'):
                    bar = context_bars.loc[partial.time] if partial.time in context_bars.index else None
                    if bar is not None:
                        if partial.price > bar['High'] or partial.price < bar['Low']:
                            anomalies.append(f"Partial exit at {partial.price:.5f} outside bar range")
        
        return anomalies
    
    def analyze_random_trades(self, n_trades=20):
        """Analyze n random trades in detail"""
        print("="*80)
        print(f"ANALYZING {n_trades} RANDOM TRADES")
        print("="*80)
        
        # Load data
        df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Use a specific period for consistency
        test_df = df['2022-01-01':'2023-12-31'].copy()
        
        print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
        print(f"Total bars: {len(test_df):,}")
        
        # Add indicators
        test_df = TIC.add_neuro_trend_intelligent(test_df)
        test_df = TIC.add_market_bias(test_df)
        test_df = TIC.add_intelligent_chop(test_df)
        
        # Run strategy with trade capture
        strategy = self.create_strategy_with_trade_capture()
        results = strategy.run_backtest(test_df)
        
        print(f"\nBacktest complete: {results['total_trades']} trades")
        print(f"Win rate: {results['win_rate']:.1f}%")
        print(f"Sharpe: {results['sharpe_ratio']:.3f}")
        
        # Select random trades
        if strategy.trade_details:
            selected_trades = random.sample(
                strategy.trade_details, 
                min(n_trades, len(strategy.trade_details))
            )
            
            # Analyze each trade
            summary = {
                'valid_entries': 0,
                'valid_exits': 0,
                'total_anomalies': 0,
                'anomaly_types': []
            }
            
            for i, trade_detail in enumerate(selected_trades, 1):
                entry_valid, exit_valid, anomalies = self.analyze_single_trade(trade_detail, i)
                
                if entry_valid:
                    summary['valid_entries'] += 1
                if exit_valid:
                    summary['valid_exits'] += 1
                summary['total_anomalies'] += len(anomalies)
                summary['anomaly_types'].extend(anomalies)
            
            # Summary report
            print(f"\n{'='*80}")
            print("TRADE ANALYSIS SUMMARY")
            print(f"{'='*80}")
            print(f"Valid entries: {summary['valid_entries']}/{n_trades} ({summary['valid_entries']/n_trades*100:.0f}%)")
            print(f"Valid exits: {summary['valid_exits']}/{n_trades} ({summary['valid_exits']/n_trades*100:.0f}%)")
            print(f"Total anomalies: {summary['total_anomalies']}")
            
            if summary['anomaly_types']:
                print("\nMost common anomalies:")
                from collections import Counter
                anomaly_counts = Counter(summary['anomaly_types'])
                for anomaly, count in anomaly_counts.most_common(5):
                    print(f"  - {anomaly}: {count} times")
            
            return summary
        else:
            print("No trades found to analyze!")
            return None
    
    def compare_random_vs_strategy_trades(self):
        """Compare random entry trades with strategy trades"""
        print("\n" + "="*80)
        print("RANDOM VS STRATEGY TRADE COMPARISON")
        print("="*80)
        
        # Load data
        df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        test_df = df['2023-01-01':'2023-06-30'].copy()
        
        # Add indicators
        test_df = TIC.add_neuro_trend_intelligent(test_df)
        test_df = TIC.add_market_bias(test_df)
        test_df = TIC.add_intelligent_chop(test_df)
        
        # Run normal strategy
        print("\n1. Running normal strategy...")
        normal_strategy = self.create_strategy_with_trade_capture()
        normal_results = normal_strategy.run_backtest(test_df)
        
        # Run random strategy
        print("\n2. Running random entry strategy...")
        random_strategy = self.create_random_strategy_with_capture()
        random_results = random_strategy.run_backtest(test_df)
        
        # Compare results
        print("\n3. COMPARISON RESULTS")
        print(f"{'Metric':<20} {'Normal':>15} {'Random':>15}")
        print("-" * 50)
        print(f"{'Sharpe Ratio':<20} {normal_results['sharpe_ratio']:>15.3f} {random_results['sharpe_ratio']:>15.3f}")
        print(f"{'Total Return %':<20} {normal_results['total_return']:>15.1f} {random_results['total_return']:>15.1f}")
        print(f"{'Win Rate %':<20} {normal_results['win_rate']:>15.1f} {random_results['win_rate']:>15.1f}")
        print(f"{'Total Trades':<20} {normal_results['total_trades']:>15} {random_results['total_trades']:>15}")
        print(f"{'Avg Win $':<20} {normal_results['avg_win']:>15.0f} {random_results['avg_win']:>15.0f}")
        print(f"{'Avg Loss $':<20} {normal_results['avg_loss']:>15.0f} {random_results['avg_loss']:>15.0f}")
        
        # Analyze trade patterns
        if normal_strategy.trade_details and random_strategy.trade_details:
            print("\n4. TRADE PATTERN ANALYSIS")
            
            # Entry time distribution
            normal_hours = [t['trade'].entry_time.hour for t in normal_strategy.trade_details]
            random_hours = [t['trade'].entry_time.hour for t in random_strategy.trade_details]
            
            print(f"\nMost active hours:")
            print(f"  Normal: {Counter(normal_hours).most_common(3)}")
            print(f"  Random: {Counter(random_hours).most_common(3)}")
            
            # Trade duration
            normal_durations = []
            random_durations = []
            
            for t in normal_strategy.trade_details:
                if hasattr(t['trade'], 'exit_time'):
                    duration = (t['trade'].exit_time - t['trade'].entry_time).total_seconds() / 60
                    normal_durations.append(duration)
            
            for t in random_strategy.trade_details:
                if hasattr(t['trade'], 'exit_time'):
                    duration = (t['trade'].exit_time - t['trade'].entry_time).total_seconds() / 60
                    random_durations.append(duration)
            
            if normal_durations and random_durations:
                print(f"\nAverage trade duration (minutes):")
                print(f"  Normal: {np.mean(normal_durations):.1f}")
                print(f"  Random: {np.mean(random_durations):.1f}")
    
    def create_random_strategy_with_capture(self):
        """Create random strategy with trade capture"""
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            verbose=True
        )
        
        class RandomTradeCapturingStrategy(self.create_strategy_with_trade_capture().__class__):
            def generate_signal(self, row, prev_row=None):
                """Random entry signals"""
                if random.random() < 0.02:  # 2% chance
                    return 1 if random.random() < 0.5 else -1
                return 0
        
        return RandomTradeCapturingStrategy(config)


def main():
    """Run trade forensics analysis"""
    analyzer = TradeForensics()
    
    # Analyze random trades
    trade_summary = analyzer.analyze_random_trades(n_trades=20)
    
    # Compare random vs strategy
    analyzer.compare_random_vs_strategy_trades()
    
    print("\n" + "="*80)
    print("FORENSIC ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()