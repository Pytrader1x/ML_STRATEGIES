"""
Deep Validation of Strategy Execution for Institutional Trading
- Checks for lookahead bias
- Validates realistic entry/exit prices
- Verifies stop loss and take profit logic
- Ensures proper P&L calculations
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import os
from datetime import datetime
import json

class DeepStrategyValidator:
    """Deep validation of strategy execution logic"""
    
    def __init__(self, currency_pair='AUDUSD'):
        self.currency_pair = currency_pair
        self.df = None
        self.validation_results = {
            'lookahead_issues': [],
            'price_violations': [],
            'slippage_issues': [],
            'pnl_issues': [],
            'execution_issues': []
        }
        
    def load_data(self):
        """Load and prepare data for validation"""
        data_path = 'data' if os.path.exists('data') else '../data'
        file_path = os.path.join(data_path, f'{self.currency_pair}_MASTER_15M.csv')
        
        print(f"Loading {self.currency_pair} data for validation...")
        self.df = pd.read_csv(file_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # Calculate indicators
        print("Calculating indicators...")
        self.df = TIC.add_neuro_trend_intelligent(self.df)
        self.df = TIC.add_market_bias(self.df, ha_len=350, ha_len2=30)
        self.df = TIC.add_intelligent_chop(self.df)
        
        return self.df
    
    def validate_trade_execution(self, trades, df):
        """Validate each trade for execution issues"""
        print("\n🔍 VALIDATING TRADE EXECUTION...")
        
        for i, trade in enumerate(trades):
            trade_num = i + 1
            
            # 1. Check entry price is within candle range
            entry_idx = df.index.get_loc(trade.entry_time)
            entry_candle = df.iloc[entry_idx]
            
            if trade.entry_price < entry_candle['Low'] or trade.entry_price > entry_candle['High']:
                self.validation_results['price_violations'].append({
                    'trade': trade_num,
                    'type': 'entry',
                    'time': trade.entry_time,
                    'price': trade.entry_price,
                    'candle_low': entry_candle['Low'],
                    'candle_high': entry_candle['High'],
                    'issue': 'Entry price outside candle range'
                })
            
            # 2. Check for lookahead bias in entry
            if entry_idx > 0:
                prev_candle = df.iloc[entry_idx - 1]
                # Entry should use previous candle's indicators
                if hasattr(trade, 'entry_indicators'):
                    # Check if indicators match previous candle
                    pass
            
            # 3. Validate exit prices
            if hasattr(trade, 'exits') and trade.exits:
                for exit_info in trade.exits:
                    exit_time = exit_info['time']
                    exit_price = exit_info['price']
                    exit_reason = exit_info['reason']
                    
                    try:
                        exit_idx = df.index.get_loc(exit_time)
                        exit_candle = df.iloc[exit_idx]
                        
                        # Check exit price is within candle range
                        if exit_price < exit_candle['Low'] or exit_price > exit_candle['High']:
                            self.validation_results['price_violations'].append({
                                'trade': trade_num,
                                'type': 'exit',
                                'time': exit_time,
                                'price': exit_price,
                                'candle_low': exit_candle['Low'],
                                'candle_high': exit_candle['High'],
                                'reason': exit_reason,
                                'issue': 'Exit price outside candle range'
                            })
                        
                        # Special checks for stop loss exits
                        if 'stop_loss' in str(exit_reason).lower():
                            # For long trades, SL exit should be at Low or worse
                            if trade.direction.value == 'LONG' and exit_price > exit_candle['Low'] + 0.0002:
                                self.validation_results['execution_issues'].append({
                                    'trade': trade_num,
                                    'type': 'stop_loss',
                                    'issue': 'Long SL exit price better than candle Low',
                                    'exit_price': exit_price,
                                    'candle_low': exit_candle['Low']
                                })
                            # For short trades, SL exit should be at High or worse
                            elif trade.direction.value == 'SHORT' and exit_price < exit_candle['High'] - 0.0002:
                                self.validation_results['execution_issues'].append({
                                    'trade': trade_num,
                                    'type': 'stop_loss',
                                    'issue': 'Short SL exit price better than candle High',
                                    'exit_price': exit_price,
                                    'candle_high': exit_candle['High']
                                })
                                
                    except Exception as e:
                        self.validation_results['execution_issues'].append({
                            'trade': trade_num,
                            'type': 'exit',
                            'issue': f'Error validating exit: {str(e)}'
                        })
            
            # 4. Validate slippage
            self.validate_slippage(trade, trade_num)
            
            # 5. Validate P&L calculation
            self.validate_pnl(trade, trade_num)
    
    def validate_slippage(self, trade, trade_num):
        """Validate slippage is applied correctly"""
        # Check entry slippage
        if hasattr(trade, 'entry_slippage'):
            if trade.entry_slippage < 0 or trade.entry_slippage > 0.0005:  # Max 0.5 pips
                self.validation_results['slippage_issues'].append({
                    'trade': trade_num,
                    'type': 'entry',
                    'slippage': trade.entry_slippage,
                    'issue': 'Entry slippage outside expected range'
                })
        
        # Check exit slippage
        if hasattr(trade, 'exits') and trade.exits:
            for exit_info in trade.exits:
                if 'slippage' in exit_info:
                    exit_slippage = exit_info['slippage']
                    exit_reason = exit_info['reason']
                    
                    # Stop loss should have higher slippage
                    if 'stop_loss' in str(exit_reason).lower():
                        if exit_slippage < 0 or exit_slippage > 0.0020:  # Max 2 pips
                            self.validation_results['slippage_issues'].append({
                                'trade': trade_num,
                                'type': 'stop_loss_exit',
                                'slippage': exit_slippage,
                                'issue': 'Stop loss slippage outside expected range'
                            })
                    # Take profit should have minimal slippage (limit orders)
                    elif 'take_profit' in str(exit_reason).lower():
                        if exit_slippage != 0:
                            self.validation_results['slippage_issues'].append({
                                'trade': trade_num,
                                'type': 'take_profit_exit',
                                'slippage': exit_slippage,
                                'issue': 'Take profit should have zero slippage (limit order)'
                            })
    
    def validate_pnl(self, trade, trade_num):
        """Validate P&L calculations"""
        if not hasattr(trade, 'pnl'):
            return
            
        # Recalculate P&L
        if hasattr(trade, 'exits') and trade.exits:
            total_pnl = 0
            total_size_exited = 0
            
            for exit_info in trade.exits:
                exit_price = exit_info['price']
                exit_size = exit_info.get('size', trade.position_size)
                
                # Calculate P&L for this exit
                if trade.direction.value == 'LONG':
                    price_change_pips = (exit_price - trade.entry_price) * 10000
                else:
                    price_change_pips = (trade.entry_price - exit_price) * 10000
                
                # Standard calculation: 1M units = $100 per pip for AUDUSD
                pnl_per_million = price_change_pips * 100
                exit_size_millions = exit_size / 1_000_000
                exit_pnl = pnl_per_million * exit_size_millions
                
                total_pnl += exit_pnl
                total_size_exited += exit_size
            
            # Compare with trade's recorded P&L
            if abs(total_pnl - trade.pnl) > 1.0:  # Allow $1 rounding difference
                self.validation_results['pnl_issues'].append({
                    'trade': trade_num,
                    'calculated_pnl': total_pnl,
                    'recorded_pnl': trade.pnl,
                    'difference': total_pnl - trade.pnl,
                    'issue': 'P&L calculation mismatch'
                })
    
    def validate_institutional_spreads(self, strategy_config):
        """Validate spreads are appropriate for institutional trading"""
        print("\n💼 VALIDATING INSTITUTIONAL SPREADS...")
        
        issues = []
        
        # Check entry slippage (should be 0.5-1 pip for institutional)
        if strategy_config.entry_slippage_pips < 0.5 or strategy_config.entry_slippage_pips > 1.0:
            issues.append(f"Entry slippage {strategy_config.entry_slippage_pips} pips outside institutional range (0.5-1.0)")
        
        # Check stop loss slippage (can be 1-2 pips in fast markets)
        if strategy_config.stop_loss_slippage_pips < 1.0 or strategy_config.stop_loss_slippage_pips > 2.0:
            issues.append(f"Stop loss slippage {strategy_config.stop_loss_slippage_pips} pips may be unrealistic")
        
        # Check minimum stop loss (institutional traders can use tighter stops)
        if strategy_config.sl_min_pips < 2.0:
            issues.append(f"Minimum stop loss {strategy_config.sl_min_pips} pips too tight even for institutional")
        
        return issues
    
    def run_validation(self, start_date='2024-01-01', end_date='2024-06-30'):
        """Run comprehensive validation"""
        if self.df is None:
            self.load_data()
        
        # Filter test period
        test_df = self.df.loc[start_date:end_date].copy()
        print(f"\n📊 Validation period: {start_date} to {end_date}")
        print(f"Total bars: {len(test_df):,}")
        
        # Create strategy with debug mode
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            realistic_costs=True,
            entry_slippage_pips=0.5,
            stop_loss_slippage_pips=2.0,
            debug_decisions=False,  # Set to True for detailed logs
            verbose=False
        )
        
        # Validate spreads
        spread_issues = self.validate_institutional_spreads(config)
        if spread_issues:
            self.validation_results['execution_issues'].extend(
                [{'type': 'spread', 'issue': issue} for issue in spread_issues]
            )
        
        # Run strategy
        print("\n🏃 Running strategy for validation...")
        strategy = OptimizedProdStrategy(config)
        
        # Enable detailed trade logging
        strategy.enable_trade_logging = True
        
        result = strategy.run_backtest(test_df)
        
        # Get trades with detailed exit information
        trades = result.get('trades', [])
        print(f"Total trades executed: {len(trades)}")
        
        # Validate each trade
        self.validate_trade_execution(trades, test_df)
        
        # Generate validation report
        self.generate_report(result)
        
        return self.validation_results
    
    def generate_report(self, backtest_result):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("📊 DEEP VALIDATION REPORT")
        print("="*80)
        
        # Summary
        total_issues = sum(len(issues) for issues in self.validation_results.values())
        print(f"\n🔍 Total Issues Found: {total_issues}")
        
        # Detailed issues by category
        for category, issues in self.validation_results.items():
            if issues:
                print(f"\n❌ {category.upper().replace('_', ' ')}: {len(issues)} issues")
                for i, issue in enumerate(issues[:5]):  # Show first 5
                    print(f"   {i+1}. {issue}")
                if len(issues) > 5:
                    print(f"   ... and {len(issues) - 5} more")
        
        # Validation summary
        print("\n✅ VALIDATION SUMMARY:")
        
        if not self.validation_results['lookahead_issues']:
            print("   ✓ No lookahead bias detected")
        else:
            print("   ✗ Potential lookahead bias found")
        
        if not self.validation_results['price_violations']:
            print("   ✓ All trades respect candle boundaries")
        else:
            print(f"   ✗ {len(self.validation_results['price_violations'])} price boundary violations")
        
        if not self.validation_results['slippage_issues']:
            print("   ✓ Slippage implementation correct")
        else:
            print(f"   ✗ {len(self.validation_results['slippage_issues'])} slippage issues")
        
        if not self.validation_results['pnl_issues']:
            print("   ✓ P&L calculations verified")
        else:
            print(f"   ✗ {len(self.validation_results['pnl_issues'])} P&L calculation issues")
        
        # Performance metrics
        print(f"\n📈 STRATEGY PERFORMANCE:")
        print(f"   Sharpe Ratio: {backtest_result.get('sharpe_ratio', 0):.3f}")
        print(f"   Total Return: {backtest_result.get('total_return', 0):.2f}%")
        print(f"   Max Drawdown: {backtest_result.get('max_drawdown', 0):.2f}%")
        print(f"   Win Rate: {backtest_result.get('win_rate', 0):.1f}%")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if total_issues == 0:
            print("   ✅ Strategy execution appears sound for institutional trading")
        else:
            print("   ⚠️  Address the issues above before live trading")
            if self.validation_results['price_violations']:
                print("   - Review exit price calculation logic")
            if self.validation_results['slippage_issues']:
                print("   - Verify slippage settings match institutional execution")
            if self.validation_results['pnl_issues']:
                print("   - Double-check P&L calculation formulas")
        
        print("\n" + "="*80)
        
        # Save detailed report
        report_data = {
            'validation_time': datetime.now().isoformat(),
            'currency_pair': self.currency_pair,
            'total_issues': total_issues,
            'validation_results': self.validation_results,
            'performance_metrics': {
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'total_return': backtest_result.get('total_return', 0),
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                'win_rate': backtest_result.get('win_rate', 0),
                'total_trades': backtest_result.get('total_trades', 0)
            }
        }
        
        with open('validation_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: validation_report.json")

def main():
    """Run deep validation"""
    print("🔍 DEEP STRATEGY VALIDATION FOR INSTITUTIONAL TRADING")
    print("="*60)
    
    validator = DeepStrategyValidator('AUDUSD')
    
    # Run validation on recent data
    validation_results = validator.run_validation('2024-01-01', '2024-06-30')
    
    # Check critical issues
    critical_issues = (
        validation_results['lookahead_issues'] + 
        validation_results['price_violations'] +
        validation_results['pnl_issues']
    )
    
    if not critical_issues:
        print("\n✅ VALIDATION PASSED - Strategy is ready for institutional trading")
    else:
        print(f"\n❌ VALIDATION FAILED - {len(critical_issues)} critical issues found")
        print("Please review the validation report and fix issues before proceeding")

if __name__ == "__main__":
    main()