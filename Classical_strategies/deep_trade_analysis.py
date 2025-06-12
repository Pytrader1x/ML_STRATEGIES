"""
Deep Trade Analysis - Validate Individual Trade Logic and Realism
Analyzes 20 random trades from each strategy configuration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection, ExitReason
import sys
sys.path.append('..')
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


class DeepTradeAnalyzer:
    def __init__(self, currency='AUDUSD'):
        self.currency = currency
        self.trade_analyses = []
        
    def load_data(self):
        """Load and prepare data"""
        print(f"Loading {self.currency} data...")
        df = pd.read_csv(f'../data/{self.currency}_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Add indicators
        df = TIC.add_neuro_trend_intelligent(df)
        df = TIC.add_market_bias(df)
        df = TIC.add_intelligent_chop(df)
        
        print(f"Data loaded: {len(df):,} rows")
        return df
    
    def analyze_trade_entry(self, df, trade, entry_idx):
        """Analyze trade entry logic"""
        entry_bar = df.iloc[entry_idx]
        prev_bars = df.iloc[max(0, entry_idx-5):entry_idx]  # 5 bars before entry
        
        analysis = {
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'direction': 'LONG' if trade.direction == TradeDirection.LONG else 'SHORT',
            'position_size': trade.position_size,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profits[0] if trade.take_profits else None,  # Primary TP
        }
        
        # Check entry conditions
        analysis['entry_signals'] = {
            'NTI_Direction': entry_bar['NTI_Direction'],
            'MB_Bias': entry_bar['MB_Bias'],
            'IC_Signal': entry_bar['IC_Signal'],
            'IC_Regime': entry_bar['IC_Regime'],
            'IC_ATR_Normalized': entry_bar['IC_ATR_Normalized']
        }
        
        # Validate entry logic
        nti_mb_aligned = entry_bar['NTI_Direction'] == entry_bar['MB_Bias']
        signal_valid = entry_bar['IC_Signal'] != 0
        # NTI_Direction is 1 for long, -1 for short
        trade_dir_value = 1 if trade.direction == TradeDirection.LONG else -1
        direction_match = entry_bar['NTI_Direction'] == trade_dir_value
        
        analysis['entry_validation'] = {
            'nti_mb_aligned': nti_mb_aligned,
            'ic_signal_present': signal_valid,
            'direction_correct': direction_match,
            'all_conditions_met': nti_mb_aligned and signal_valid and direction_match
        }
        
        # Check if entry price is realistic
        analysis['price_validation'] = {
            'bar_open': entry_bar['Open'],
            'bar_high': entry_bar['High'],
            'bar_low': entry_bar['Low'],
            'bar_close': entry_bar['Close'],
            'entry_within_bar': entry_bar['Low'] <= trade.entry_price <= entry_bar['High'],
            'entry_at_close': abs(trade.entry_price - entry_bar['Close']) < 0.00001
        }
        
        # Calculate stop loss distance
        if self.currency == 'USDJPY' or 'JPY' in self.currency:
            pip_size = 0.01
        else:
            pip_size = 0.0001
            
        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / pip_size
        tp_distance_pips = abs(analysis['take_profit'] - trade.entry_price) / pip_size if analysis['take_profit'] else 0
        
        analysis['risk_reward'] = {
            'sl_distance_pips': round(sl_distance_pips, 1),
            'tp_distance_pips': round(tp_distance_pips, 1),
            'risk_reward_ratio': round(tp_distance_pips / sl_distance_pips, 2) if sl_distance_pips > 0 else 0
        }
        
        # Check market conditions
        analysis['market_context'] = {
            'atr_value': entry_bar['IC_ATR_Normalized'],
            'market_regime': ['Choppy', 'Ranging', 'Trending'][int(entry_bar['IC_Regime'])],
            'recent_volatility': prev_bars['IC_ATR_Normalized'].mean() if len(prev_bars) > 0 else 0
        }
        
        return analysis
    
    def analyze_trade_exit(self, df, trade, entry_idx):
        """Analyze trade exit logic"""
        exit_idx = None
        exit_analysis = {
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason.value if trade.exit_reason else 'Unknown',
            'pnl_pips': getattr(trade, 'pnl_pips', 0),
            'pnl_dollars': trade.pnl
        }
        
        # Find exit bar
        for i in range(entry_idx + 1, len(df)):
            if df.index[i] >= trade.exit_time:
                exit_idx = i
                break
        
        if exit_idx is None:
            exit_analysis['validation_error'] = 'Exit time not found in data'
            return exit_analysis
        
        exit_bar = df.iloc[exit_idx]
        bars_held = exit_idx - entry_idx
        
        exit_analysis['trade_duration'] = {
            'bars_held': bars_held,
            'time_held': str(trade.exit_time - trade.entry_time)
        }
        
        # Validate exit price
        exit_analysis['price_validation'] = {
            'bar_open': exit_bar['Open'],
            'bar_high': exit_bar['High'],
            'bar_low': exit_bar['Low'],
            'bar_close': exit_bar['Close'],
            'exit_within_bar': exit_bar['Low'] <= trade.exit_price <= exit_bar['High']
        }
        
        # Check exit reason validity
        if trade.exit_reason == ExitReason.STOP_LOSS:
            if trade.direction == TradeDirection.LONG:  # Long
                sl_hit = exit_bar['Low'] <= trade.stop_loss
                exit_at_sl = abs(trade.exit_price - trade.stop_loss) < 0.00001
            else:  # Short
                sl_hit = exit_bar['High'] >= trade.stop_loss
                exit_at_sl = abs(trade.exit_price - trade.stop_loss) < 0.00001
                
            exit_analysis['sl_validation'] = {
                'sl_could_be_hit': sl_hit,
                'exit_exactly_at_sl': exit_at_sl,
                'realistic': sl_hit and not exit_at_sl  # Should have slippage
            }
            
        elif trade.exit_reason in [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3]:
            tp1 = trade.take_profits[0] if trade.take_profits else None
            if tp1 and trade.direction == TradeDirection.LONG:  # Long
                tp_hit = exit_bar['High'] >= tp1
                exit_at_tp = abs(trade.exit_price - tp1) < 0.00001
            elif tp1:  # Short
                tp_hit = exit_bar['Low'] <= tp1
                exit_at_tp = abs(trade.exit_price - tp1) < 0.00001
            else:
                tp_hit = False
                exit_at_tp = False
                
            exit_analysis['tp_validation'] = {
                'tp_could_be_hit': tp_hit,
                'exit_exactly_at_tp': exit_at_tp,
                'realistic': tp_hit and not exit_at_tp  # Should have slippage
            }
            
        elif trade.exit_reason == ExitReason.SIGNAL_FLIP:
            # Check if signals actually flipped
            entry_bar = df.iloc[entry_idx]
            signal_flipped = exit_bar['NTI_Direction'] * entry_bar['NTI_Direction'] < 0
            
            exit_analysis['signal_flip_validation'] = {
                'signal_actually_flipped': signal_flipped,
                'exit_at_close': abs(trade.exit_price - exit_bar['Close']) < 0.00001
            }
        
        # Check for unrealistic scenarios
        exit_analysis['realism_checks'] = {
            'both_sl_tp_in_bar': False,
            'impossible_fill': False
        }
        
        # Check if both SL and TP could be hit in same bar
        tp1 = trade.take_profits[0] if trade.take_profits else None
        if tp1 and trade.direction == TradeDirection.LONG:  # Long
            both_possible = (exit_bar['Low'] <= trade.stop_loss) and (exit_bar['High'] >= tp1)
        elif tp1:  # Short
            both_possible = (exit_bar['High'] >= trade.stop_loss) and (exit_bar['Low'] <= tp1)
        else:
            both_possible = False
            
        if both_possible:
            exit_analysis['realism_checks']['both_sl_tp_in_bar'] = True
            exit_analysis['realism_checks']['which_hit_first'] = 'UNKNOWN - Requires tick data'
        
        return exit_analysis
    
    def run_deep_analysis(self, df, config_name, strategy, num_trades=20):
        """Run deep analysis on random trades"""
        print(f"\n{'='*80}")
        print(f"Deep Trade Analysis - {config_name}")
        print(f"{'='*80}")
        
        # Get a sample period with good trading activity
        sample_start = len(df) // 2  # Start from middle of data
        sample_df = df.iloc[sample_start:sample_start + 20000].copy()
        
        # Run backtest to get trades
        print("Running backtest to generate trades...")
        results = strategy.run_backtest(sample_df)
        
        if not results['trades'] or len(results['trades']) == 0:
            print("No trades generated!")
            return []
        
        # Select random trades
        all_trades = results['trades']
        num_to_analyze = min(num_trades, len(all_trades))
        selected_indices = np.random.choice(len(all_trades), num_to_analyze, replace=False)
        selected_trades = [all_trades[i] for i in sorted(selected_indices)]
        
        print(f"Analyzing {num_to_analyze} trades out of {len(all_trades)} total trades")
        
        analyses = []
        for i, trade in enumerate(selected_trades):
            print(f"\n--- Trade {i+1}/{num_to_analyze} ---")
            
            # Find entry bar index
            entry_idx = None
            for idx in range(len(sample_df)):
                if sample_df.index[idx] == trade.entry_time:
                    entry_idx = idx
                    break
            
            if entry_idx is None:
                print(f"WARNING: Could not find entry time {trade.entry_time}")
                continue
            
            # Analyze entry
            entry_analysis = self.analyze_trade_entry(sample_df, trade, entry_idx)
            
            # Analyze exit
            exit_analysis = self.analyze_trade_exit(sample_df, trade, entry_idx)
            
            # Combine analyses
            trade_analysis = {
                'trade_number': i + 1,
                'entry': entry_analysis,
                'exit': exit_analysis,
                'overall_realism': self.assess_overall_realism(entry_analysis, exit_analysis)
            }
            
            analyses.append(trade_analysis)
            
            # Print summary
            self.print_trade_summary(trade_analysis)
        
        return analyses
    
    def assess_overall_realism(self, entry_analysis, exit_analysis):
        """Assess overall trade realism"""
        issues = []
        
        # Entry issues
        if not entry_analysis['entry_validation']['all_conditions_met']:
            issues.append('Entry conditions not properly met')
        
        if entry_analysis['price_validation']['entry_at_close']:
            issues.append('Entry exactly at close price (unrealistic)')
        
        if not entry_analysis['price_validation']['entry_within_bar']:
            issues.append('Entry price outside bar range (impossible)')
        
        # Exit issues
        if 'sl_validation' in exit_analysis:
            if exit_analysis['sl_validation'].get('exit_exactly_at_sl'):
                issues.append('Exit exactly at SL without slippage (unrealistic)')
        
        if 'tp_validation' in exit_analysis:
            if exit_analysis['tp_validation'].get('exit_exactly_at_tp'):
                issues.append('Exit exactly at TP without slippage (unrealistic)')
        
        if not exit_analysis['price_validation']['exit_within_bar']:
            issues.append('Exit price outside bar range (impossible)')
        
        if exit_analysis['realism_checks']['both_sl_tp_in_bar']:
            issues.append('Both SL and TP within same bar (outcome uncertain)')
        
        return {
            'realistic': len(issues) == 0,
            'issues': issues,
            'severity': 'OK' if len(issues) == 0 else ('MODERATE' if len(issues) <= 2 else 'SEVERE')
        }
    
    def print_trade_summary(self, analysis):
        """Print trade analysis summary"""
        entry = analysis['entry']
        exit = analysis['exit']
        realism = analysis['overall_realism']
        
        print(f"\nEntry: {entry['entry_time']} @ {entry['entry_price']:.5f} ({entry['direction']})")
        print(f"  Signals: NTI={entry['entry_signals']['NTI_Direction']}, "
              f"MB={entry['entry_signals']['MB_Bias']}, "
              f"IC={entry['entry_signals']['IC_Signal']}")
        print(f"  Valid Entry: {entry['entry_validation']['all_conditions_met']}")
        print(f"  Risk/Reward: {entry['risk_reward']['sl_distance_pips']:.1f} pips SL, "
              f"{entry['risk_reward']['tp_distance_pips']:.1f} pips TP "
              f"(RR: {entry['risk_reward']['risk_reward_ratio']})")
        
        print(f"\nExit: {exit['exit_time']} @ {exit['exit_price']:.5f} ({exit['exit_reason']})")
        print(f"  Duration: {exit['trade_duration']['bars_held']} bars "
              f"({exit['trade_duration']['time_held']})")
        print(f"  P&L: {exit['pnl_pips']:.1f} pips (${exit['pnl_dollars']:.2f})")
        
        print(f"\nRealism: {realism['severity']}")
        if realism['issues']:
            for issue in realism['issues']:
                print(f"  ⚠️  {issue}")
    
    def generate_detailed_report(self, all_analyses):
        """Generate detailed validation report"""
        print(f"\n{'='*80}")
        print("DEEP TRADE VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        total_trades = sum(len(analyses) for analyses in all_analyses.values())
        realistic_trades = 0
        all_issues = []
        
        for config_name, analyses in all_analyses.items():
            print(f"\n{config_name}:")
            config_realistic = sum(1 for a in analyses if a['overall_realism']['realistic'])
            realistic_trades += config_realistic
            
            print(f"  Realistic trades: {config_realistic}/{len(analyses)} "
                  f"({config_realistic/len(analyses)*100:.1f}%)")
            
            # Collect issues
            issue_counts = {}
            for analysis in analyses:
                for issue in analysis['overall_realism']['issues']:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    all_issues.append(issue)
            
            print("  Common issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {issue}: {count} trades")
        
        # Overall summary
        print(f"\n{'='*60}")
        print("OVERALL FINDINGS:")
        print(f"{'='*60}")
        print(f"Total trades analyzed: {total_trades}")
        print(f"Realistic trades: {realistic_trades} ({realistic_trades/total_trades*100:.1f}%)")
        
        # Most common issues
        from collections import Counter
        issue_counter = Counter(all_issues)
        print("\nMost common realism issues:")
        for issue, count in issue_counter.most_common(5):
            print(f"  - {issue}: {count} occurrences ({count/total_trades*100:.1f}%)")
        
        # Save detailed report
        report = {
            'analysis_date': datetime.now().isoformat(),
            'currency': self.currency,
            'total_trades_analyzed': total_trades,
            'realistic_trades': realistic_trades,
            'realism_percentage': realistic_trades/total_trades*100,
            'all_analyses': all_analyses,
            'common_issues': dict(issue_counter)
        }
        
        with open('deep_trade_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to deep_trade_analysis_report.json")
        
        return report


def main():
    """Run deep trade analysis"""
    analyzer = DeepTradeAnalyzer('AUDUSD')
    
    # Load data
    df = analyzer.load_data()
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk", OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=False
        )),
        ("Config 2: Scalping", OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            verbose=False
        ))
    ]
    
    all_analyses = {}
    
    for config_name, config in configs:
        strategy = OptimizedProdStrategy(config)
        analyses = analyzer.run_deep_analysis(df, config_name, strategy, num_trades=20)
        all_analyses[config_name] = analyses
    
    # Generate summary report
    analyzer.generate_detailed_report(all_analyses)


if __name__ == "__main__":
    main()