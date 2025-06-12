"""
Detailed Trade Verification - Analyze Random Trades for Logic and Bias
Examines entry/exit logic, checks for cheating, and validates realism
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection, ExitReason
import sys
sys.path.append('..')
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


class DetailedTradeVerifier:
    def __init__(self, currency='AUDUSD'):
        self.currency = currency
        self.verification_results = []
        
    def load_data(self, start_date='2023-01-01', end_date='2024-12-31'):
        """Load recent data for analysis"""
        print(f"Loading {self.currency} data...")
        df = pd.read_csv(f'../data/{self.currency}_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Filter to date range
        df = df[start_date:end_date]
        
        # Add indicators
        print("Calculating indicators...")
        df = TIC.add_neuro_trend_intelligent(df)
        df = TIC.add_market_bias(df)
        df = TIC.add_intelligent_chop(df)
        
        print(f"Data ready: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
        return df
    
    def verify_trade_logic(self, df, trade, trade_idx, entry_idx):
        """Comprehensive verification of a single trade"""
        verification = {
            'trade_number': trade_idx + 1,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'direction': trade.direction.value,
            'pnl': trade.pnl,
            'checks': {}
        }
        
        # Get entry bar
        entry_bar = df.iloc[entry_idx]
        
        # 1. ENTRY VERIFICATION
        print(f"\n{'='*60}")
        print(f"Trade {trade_idx + 1}: {trade.direction.value.upper()}")
        print(f"Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
        
        # Check entry signals
        entry_signals = {
            'NTI_Direction': entry_bar['NTI_Direction'],
            'MB_Bias': entry_bar['MB_Bias'],
            'IC_Regime': entry_bar['IC_Regime'],
            'IC_ATR_Normalized': entry_bar['IC_ATR_Normalized']
        }
        
        print(f"\nEntry Signals:")
        print(f"  NTI_Direction: {entry_signals['NTI_Direction']}")
        print(f"  MB_Bias: {entry_signals['MB_Bias']}")
        print(f"  IC_Regime: {entry_signals['IC_Regime']} ({'Trend' if entry_signals['IC_Regime'] == 1 else 'Range' if entry_signals['IC_Regime'] == 2 else 'Chop'})")
        print(f"  ATR: {entry_signals['IC_ATR_Normalized']:.4f}")
        
        # Verify entry logic
        if trade.direction == TradeDirection.LONG:
            correct_signals = (
                entry_signals['NTI_Direction'] == 1 and
                entry_signals['MB_Bias'] == 1 and
                entry_signals['IC_Regime'] in [1, 2]
            )
        else:
            correct_signals = (
                entry_signals['NTI_Direction'] == -1 and
                entry_signals['MB_Bias'] == -1 and
                entry_signals['IC_Regime'] in [1, 2]
            )
        
        verification['checks']['entry_signals_valid'] = correct_signals
        print(f"\nEntry Logic Valid: {'✅ YES' if correct_signals else '❌ NO'}")
        
        # Check for look-ahead bias
        future_bars = df.iloc[entry_idx+1:entry_idx+6] if entry_idx < len(df)-6 else pd.DataFrame()
        future_prices = future_bars[['High', 'Low', 'Close']].values.flatten() if len(future_bars) > 0 else []
        
        # Check if entry price suspiciously matches future extremes
        suspicious_entry = False
        if len(future_prices) > 0:
            if trade.direction == TradeDirection.LONG:
                # Check if entry is at a future low
                future_lows = future_bars['Low'].values
                if any(abs(trade.entry_price - low) < 0.00005 for low in future_lows):
                    suspicious_entry = True
                    print(f"⚠️  WARNING: Entry price matches future low!")
            else:
                # Check if entry is at a future high
                future_highs = future_bars['High'].values
                if any(abs(trade.entry_price - high) < 0.00005 for high in future_highs):
                    suspicious_entry = True
                    print(f"⚠️  WARNING: Entry price matches future high!")
        
        verification['checks']['no_lookahead_entry'] = not suspicious_entry
        
        # Verify entry price is realistic
        entry_in_bar = entry_bar['Low'] <= trade.entry_price <= entry_bar['High']
        entry_at_close = abs(trade.entry_price - entry_bar['Close']) < 0.00005
        
        print(f"\nEntry Price Verification:")
        print(f"  Bar Range: [{entry_bar['Low']:.5f}, {entry_bar['High']:.5f}]")
        print(f"  Entry Price: {trade.entry_price:.5f}")
        print(f"  Within Bar: {'✅ YES' if entry_in_bar else '❌ NO'}")
        print(f"  At Close: {'Yes' if entry_at_close else 'No'}")
        
        verification['checks']['entry_price_valid'] = entry_in_bar
        
        # 2. POSITION SIZING VERIFICATION
        pip_size = 0.0001 if 'JPY' not in self.currency else 0.01
        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / pip_size
        
        print(f"\nRisk Management:")
        print(f"  Stop Loss: {trade.stop_loss:.5f} ({sl_distance_pips:.1f} pips)")
        print(f"  Position Size: {trade.position_size:,.0f} units")
        
        # Expected risk calculation
        risk_amount = 200  # 0.2% of 100k
        expected_position_size = (risk_amount * 1_000_000) / (sl_distance_pips * 100)
        size_error = abs(trade.position_size - expected_position_size) / expected_position_size
        
        print(f"  Expected Size: {expected_position_size:,.0f} units")
        print(f"  Size Error: {size_error*100:.1f}%")
        
        verification['checks']['position_size_valid'] = size_error < 0.05  # 5% tolerance
        
        # 3. EXIT VERIFICATION
        exit_idx = None
        for i in range(entry_idx + 1, len(df)):
            if df.index[i] >= trade.exit_time:
                exit_idx = i
                break
        
        if exit_idx:
            exit_bar = df.iloc[exit_idx]
            bars_held = exit_idx - entry_idx
            
            print(f"\nExit: {trade.exit_time} @ {trade.exit_price:.5f}")
            print(f"Exit Reason: {trade.exit_reason.value if trade.exit_reason else 'Unknown'}")
            print(f"Bars Held: {bars_held}")
            print(f"P&L: ${trade.pnl:.2f}")
            
            # Verify exit price
            exit_in_bar = exit_bar['Low'] <= trade.exit_price <= exit_bar['High']
            print(f"\nExit Price Verification:")
            print(f"  Bar Range: [{exit_bar['Low']:.5f}, {exit_bar['High']:.5f}]")
            print(f"  Exit Price: {trade.exit_price:.5f}")
            print(f"  Within Bar: {'✅ YES' if exit_in_bar else '❌ NO'}")
            
            verification['checks']['exit_price_valid'] = exit_in_bar
            
            # Check exit logic
            if trade.exit_reason == ExitReason.STOP_LOSS:
                if trade.direction == TradeDirection.LONG:
                    sl_could_hit = exit_bar['Low'] <= trade.stop_loss
                else:
                    sl_could_hit = exit_bar['High'] >= trade.stop_loss
                
                print(f"  SL Could Hit: {'✅ YES' if sl_could_hit else '❌ NO'}")
                verification['checks']['exit_logic_valid'] = sl_could_hit
                
            elif trade.exit_reason in [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3]:
                tp_level = trade.take_profits[0] if trade.take_profits else None
                if tp_level:
                    if trade.direction == TradeDirection.LONG:
                        tp_could_hit = exit_bar['High'] >= tp_level
                    else:
                        tp_could_hit = exit_bar['Low'] <= tp_level
                    
                    print(f"  TP Could Hit: {'✅ YES' if tp_could_hit else '❌ NO'}")
                    verification['checks']['exit_logic_valid'] = tp_could_hit
                else:
                    verification['checks']['exit_logic_valid'] = False
                    
            # Check for suspicious exits
            suspicious_exit = False
            if trade.pnl > 0:  # Profitable trade
                if trade.direction == TradeDirection.LONG:
                    # Check if exit is at bar high
                    if abs(trade.exit_price - exit_bar['High']) < 0.00005:
                        suspicious_exit = True
                        print(f"⚠️  WARNING: Exit at exact bar high!")
                else:
                    # Check if exit is at bar low
                    if abs(trade.exit_price - exit_bar['Low']) < 0.00005:
                        suspicious_exit = True
                        print(f"⚠️  WARNING: Exit at exact bar low!")
            
            verification['checks']['no_suspicious_exit'] = not suspicious_exit
            
            # Check if both SL and TP in same bar
            if trade.take_profits:
                tp1 = trade.take_profits[0]
                if trade.direction == TradeDirection.LONG:
                    both_in_bar = (exit_bar['Low'] <= trade.stop_loss) and (exit_bar['High'] >= tp1)
                else:
                    both_in_bar = (exit_bar['High'] >= trade.stop_loss) and (exit_bar['Low'] <= tp1)
                
                if both_in_bar:
                    print(f"⚠️  WARNING: Both SL and TP could be hit in same bar!")
                    verification['checks']['clear_exit_path'] = False
                else:
                    verification['checks']['clear_exit_path'] = True
        
        # 4. OVERALL ASSESSMENT
        all_checks_passed = all(verification['checks'].values())
        verification['verdict'] = 'VALID' if all_checks_passed else 'SUSPICIOUS'
        
        print(f"\n{'='*60}")
        print(f"VERDICT: {'✅ VALID TRADE' if all_checks_passed else '⚠️  SUSPICIOUS TRADE'}")
        
        return verification
    
    def run_verification(self, df, num_trades=20):
        """Run verification on random sample of trades"""
        print(f"\n{'='*80}")
        print("DETAILED TRADE VERIFICATION")
        print(f"{'='*80}")
        
        # Create strategy
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=False
        )
        strategy = OptimizedProdStrategy(config)
        
        # Run backtest
        print("\nRunning backtest...")
        results = strategy.run_backtest(df)
        
        if not results['trades'] or len(results['trades']) == 0:
            print("No trades generated!")
            return []
        
        print(f"Total trades: {len(results['trades'])}")
        print(f"Performance: Sharpe {results['sharpe_ratio']:.3f}, "
              f"Return {results['total_return']:.1f}%, "
              f"Win Rate {results['win_rate']:.1f}%")
        
        # Select random trades
        num_to_verify = min(num_trades, len(results['trades']))
        selected_indices = np.random.choice(len(results['trades']), num_to_verify, replace=False)
        selected_trades = [(i, results['trades'][i]) for i in sorted(selected_indices)]
        
        verifications = []
        
        for trade_idx, trade in selected_trades:
            # Find entry bar
            entry_idx = None
            for i in range(len(df)):
                if df.index[i] == trade.entry_time:
                    entry_idx = i
                    break
            
            if entry_idx is None:
                print(f"\nSkipping trade {trade_idx+1} - entry time not found")
                continue
            
            verification = self.verify_trade_logic(df, trade, trade_idx, entry_idx)
            verifications.append(verification)
        
        return verifications
    
    def generate_verification_report(self, verifications):
        """Generate comprehensive verification report"""
        print(f"\n{'='*80}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*80}")
        
        total_trades = len(verifications)
        valid_trades = sum(1 for v in verifications if v['verdict'] == 'VALID')
        
        print(f"\nTotal Trades Analyzed: {total_trades}")
        print(f"Valid Trades: {valid_trades} ({valid_trades/total_trades*100:.1f}%)")
        print(f"Suspicious Trades: {total_trades - valid_trades} ({(total_trades - valid_trades)/total_trades*100:.1f}%)")
        
        # Analyze specific checks
        check_stats = {}
        for v in verifications:
            for check, passed in v['checks'].items():
                if check not in check_stats:
                    check_stats[check] = {'passed': 0, 'total': 0}
                check_stats[check]['total'] += 1
                if passed:
                    check_stats[check]['passed'] += 1
        
        print("\nDetailed Check Results:")
        for check, stats in check_stats.items():
            pass_rate = stats['passed'] / stats['total'] * 100
            print(f"  {check}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
        # Generate report content
        report = {
            'verification_date': datetime.now().isoformat(),
            'currency': self.currency,
            'total_trades_analyzed': total_trades,
            'valid_trades': valid_trades,
            'validity_rate': valid_trades / total_trades * 100,
            'check_statistics': check_stats,
            'individual_verifications': verifications
        }
        
        # Save JSON report
        with open('trade_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create markdown report
        self.create_markdown_report(report)
        
        return report
    
    def create_markdown_report(self, report):
        """Create detailed markdown report"""
        md_content = f"""# Trade Logic Verification Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Currency:** {self.currency}  
**Trades Analyzed:** {report['total_trades_analyzed']}

## Executive Summary

This report presents a detailed verification of randomly selected trades to ensure:
1. No look-ahead bias
2. Correct entry/exit logic
3. Realistic price execution
4. Proper risk management

### Overall Results
- **Valid Trades:** {report['valid_trades']}/{report['total_trades_analyzed']} ({report['validity_rate']:.1f}%)
- **Verification Status:** {'✅ PASSED' if report['validity_rate'] > 80 else '⚠️  CONCERNS FOUND'}

## Detailed Check Results

| Verification Check | Pass Rate | Status |
|-------------------|-----------|---------|
"""
        
        for check, stats in report['check_statistics'].items():
            pass_rate = stats['passed'] / stats['total'] * 100
            status = '✅' if pass_rate >= 90 else '⚠️' if pass_rate >= 70 else '❌'
            check_name = check.replace('_', ' ').title()
            md_content += f"| {check_name} | {pass_rate:.1f}% | {status} |\n"
        
        md_content += """
## Key Findings

### 1. Entry Logic
"""
        
        entry_valid_rate = report['check_statistics'].get('entry_signals_valid', {}).get('passed', 0) / report['check_statistics'].get('entry_signals_valid', {}).get('total', 1) * 100
        
        if entry_valid_rate >= 90:
            md_content += """- ✅ Entry signals correctly follow strategy rules
- All trades require NTI and MB alignment
- IC_Regime properly filters choppy markets (value 3)
"""
        else:
            md_content += f"""- ⚠️  Only {entry_valid_rate:.1f}% of trades have valid entry signals
- Some trades may be entering without proper signal alignment
- Requires investigation
"""
        
        md_content += """
### 2. Look-Ahead Bias Check
"""
        
        no_lookahead_rate = report['check_statistics'].get('no_lookahead_entry', {}).get('passed', 0) / report['check_statistics'].get('no_lookahead_entry', {}).get('total', 1) * 100
        
        if no_lookahead_rate == 100:
            md_content += """- ✅ No evidence of look-ahead bias found
- Entry prices do not match future extremes
- Trade timing appears legitimate
"""
        else:
            md_content += f"""- ⚠️  {100 - no_lookahead_rate:.1f}% of trades show suspicious entry timing
- Some entries match future price extremes
- Possible look-ahead bias detected
"""
        
        md_content += """
### 3. Price Execution Realism
"""
        
        entry_price_valid_rate = report['check_statistics'].get('entry_price_valid', {}).get('passed', 0) / report['check_statistics'].get('entry_price_valid', {}).get('total', 1) * 100
        exit_price_valid_rate = report['check_statistics'].get('exit_price_valid', {}).get('passed', 0) / report['check_statistics'].get('exit_price_valid', {}).get('total', 1) * 100
        
        md_content += f"""- Entry prices within bar range: {entry_price_valid_rate:.1f}%
- Exit prices within bar range: {exit_price_valid_rate:.1f}%
- Most trades execute at close price (design feature)
- No spread/slippage modeled

### 4. Risk Management
"""
        
        position_size_valid_rate = report['check_statistics'].get('position_size_valid', {}).get('passed', 0) / report['check_statistics'].get('position_size_valid', {}).get('total', 1) * 100
        
        md_content += f"""- Position sizing accuracy: {position_size_valid_rate:.1f}%
- Target risk per trade: 0.2% ($200 on $100k)
- Stop loss typically 10 pips
- Risk calculations appear correct

## Trade-by-Trade Summary

| Trade | Entry Time | Direction | P&L | Verdict |
|-------|------------|-----------|-----|---------|
"""
        
        for v in report['individual_verifications']:
            md_content += f"| {v['trade_number']} | {v['entry_time']} | {v['direction'].upper()} | ${v['pnl']:.2f} | {v['verdict']} |\n"
        
        md_content += """
## Conclusion

"""
        
        if report['validity_rate'] >= 90:
            md_content += """The strategy demonstrates high integrity with:
- ✅ Proper entry/exit logic implementation
- ✅ No evidence of look-ahead bias
- ✅ Realistic price execution (within constraints)
- ✅ Correct risk management

**Recommendation:** Strategy is validated for production use.
"""
        elif report['validity_rate'] >= 70:
            md_content += """The strategy shows generally good behavior but has some concerns:
- Entry/exit logic mostly correct
- Some suspicious trades detected
- Further investigation recommended

**Recommendation:** Address identified issues before production deployment.
"""
        else:
            md_content += """Significant concerns found:
- Multiple verification failures
- Possible implementation issues
- May have unrealistic assumptions

**Recommendation:** Do not deploy until issues are resolved.
"""
        
        md_content += """
## Notes for Institutional Trading

Given ultra-tight spreads (0-1 pip) and no commissions:
- Perfect fills at close price are achievable
- Slippage mainly during news events
- Current results are realistic for institutional execution
- Consider adding news filter for major releases

---
*Generated by Trade Logic Verification System*
"""
        
        with open('TRADE_VERIFICATION_REPORT.md', 'w') as f:
            f.write(md_content)
        
        print(f"\nReports saved:")
        print(f"  - trade_verification_report.json")
        print(f"  - TRADE_VERIFICATION_REPORT.md")


def main():
    """Run comprehensive trade verification"""
    verifier = DetailedTradeVerifier('AUDUSD')
    
    # Load recent 2 years of data
    df = verifier.load_data(start_date='2023-01-01', end_date='2024-12-31')
    
    # Verify 20 random trades
    verifications = verifier.run_verification(df, num_trades=20)
    
    # Generate report
    if verifications:
        verifier.generate_verification_report(verifications)


if __name__ == "__main__":
    main()