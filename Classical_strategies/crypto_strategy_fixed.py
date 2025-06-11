"""
Fixed Crypto Trading Strategy - Direct Percentage-Based Implementation
Works directly with crypto prices without pip conversion
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')


@dataclass
class CryptoTrade:
    """Represents a single crypto trade"""
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # In base currency (e.g., ETH)
    direction: int  # 1 for long, -1 for short
    stop_loss: float
    take_profits: List[float]
    tp_sizes: List[float]  # Partial TP sizes
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None


class CryptoStrategy:
    """
    Crypto-specific trading strategy using percentage-based calculations
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        self.current_capital = config['initial_capital']
        
    def calculate_position_size(self, price, stop_loss_price):
        """
        Calculate position size based on risk per trade
        """
        risk_amount = self.current_capital * self.config['risk_per_trade']
        price_risk = abs(price - stop_loss_price) / price
        
        if price_risk == 0:
            return 0
        
        # Position value in dollars
        position_value = risk_amount / price_risk
        
        # Position size in crypto units
        position_size = position_value / price
        
        # Apply max position size limit
        max_position_value = self.current_capital * self.config['max_position_size']
        max_position_size = max_position_value / price
        
        return min(position_size, max_position_size)
    
    def run_backtest(self, df):
        """
        Run backtest on crypto data
        """
        # Ensure we have required indicators
        required_cols = ['NTI_Direction', 'MB_Bias', 'IC_Signal']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required indicator: {col}")
        
        # Initialize tracking variables
        self.trades = []
        self.equity_curve = [self.config['initial_capital']]
        self.current_capital = self.config['initial_capital']
        open_trade = None
        
        # Main backtest loop
        for i in range(50, len(df)):  # Start after warm-up period
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['Close']
            
            # Check if we have an open trade
            if open_trade is not None:
                # Check exit conditions
                exit_signal = self._check_exit_conditions(
                    open_trade, current_bar, df.iloc[i-1], current_time
                )
                
                if exit_signal:
                    # Close trade
                    open_trade = self._close_trade(
                        open_trade, exit_signal['price'], 
                        current_time, exit_signal['reason']
                    )
                    self.trades.append(open_trade)
                    open_trade = None
            
            # Check for new entry signal
            if open_trade is None and self._check_entry_signal(df, i):
                # Calculate stops and targets
                atr = self._calculate_atr(df, i)
                direction = 1 if current_bar['NTI_Direction'] > 0 else -1
                
                # Stop loss calculation
                sl_pct = min(
                    self.config['sl_max_pct'],
                    self.config['sl_atr_multiplier'] * atr / current_price
                )
                sl_price = current_price * (1 - sl_pct) if direction > 0 else current_price * (1 + sl_pct)
                
                # Take profit calculations
                tp_prices = []
                for tp_mult in self.config['tp_atr_multipliers']:
                    tp_pct = min(
                        self.config['max_tp_pct'],
                        tp_mult * atr / current_price
                    )
                    tp_price = current_price * (1 + tp_pct) if direction > 0 else current_price * (1 - tp_pct)
                    tp_prices.append(tp_price)
                
                # Calculate position size
                position_size = self.calculate_position_size(current_price, sl_price)
                
                if position_size > 0:
                    # Create new trade
                    open_trade = CryptoTrade(
                        entry_time=current_time,
                        entry_price=current_price,
                        position_size=position_size,
                        direction=direction,
                        stop_loss=sl_price,
                        take_profits=tp_prices,
                        tp_sizes=[0.3, 0.3, 0.4]  # Partial TP sizes
                    )
            
            # Update equity curve
            current_equity = self._calculate_current_equity(open_trade, current_price)
            self.equity_curve.append(current_equity)
        
        # Close any remaining open trade
        if open_trade is not None:
            open_trade = self._close_trade(
                open_trade, df.iloc[-1]['Close'], 
                df.index[-1], 'End of Data'
            )
            self.trades.append(open_trade)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(df)
    
    def _check_entry_signal(self, df, i):
        """Check if entry conditions are met"""
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Basic entry: NTI signal change
        nti_changed = (current['NTI_Direction'] != prev['NTI_Direction'] and 
                      current['NTI_Direction'] != 0)
        
        # Additional filters
        mb_aligned = (current['MB_Bias'] == current['NTI_Direction'])
        not_choppy = (current['IC_Signal'] != 0)  # Not in chop zone
        
        return nti_changed and mb_aligned and not_choppy
    
    def _check_exit_conditions(self, trade, current_bar, prev_bar, current_time):
        """Check various exit conditions"""
        current_price = current_bar['Close']
        
        # Check stop loss
        if trade.direction > 0:  # Long
            if current_bar['Low'] <= trade.stop_loss:
                return {'price': trade.stop_loss, 'reason': 'Stop Loss'}
        else:  # Short
            if current_bar['High'] >= trade.stop_loss:
                return {'price': trade.stop_loss, 'reason': 'Stop Loss'}
        
        # Check take profits
        for i, tp_price in enumerate(trade.take_profits):
            if trade.tp_sizes[i] > 0:  # TP not yet hit
                if trade.direction > 0 and current_bar['High'] >= tp_price:
                    return {'price': tp_price, 'reason': f'Take Profit {i+1}'}
                elif trade.direction < 0 and current_bar['Low'] <= tp_price:
                    return {'price': tp_price, 'reason': f'Take Profit {i+1}'}
        
        # Check signal flip
        if self.config['exit_on_signal_flip']:
            signal_flipped = (current_bar['NTI_Direction'] * trade.direction < 0)
            if signal_flipped:
                # Check minimum profit and time conditions
                time_held = (current_time - trade.entry_time).total_seconds() / 3600
                profit_pct = (current_price - trade.entry_price) / trade.entry_price * trade.direction
                
                if (time_held >= self.config['signal_flip_min_hours'] and
                    profit_pct >= self.config['signal_flip_min_profit_pct']):
                    return {'price': current_price, 'reason': 'Signal Flip'}
        
        # Trailing stop logic
        if self.config['use_trailing_stop']:
            profit_pct = (current_price - trade.entry_price) / trade.entry_price * trade.direction
            
            if profit_pct >= self.config['tsl_activation_pct']:
                # Calculate trailing stop level
                if trade.direction > 0:
                    tsl_price = current_price * (1 - self.config['tsl_buffer_pct'])
                    if current_bar['Low'] <= tsl_price:
                        return {'price': tsl_price, 'reason': 'Trailing Stop'}
                else:
                    tsl_price = current_price * (1 + self.config['tsl_buffer_pct'])
                    if current_bar['High'] >= tsl_price:
                        return {'price': tsl_price, 'reason': 'Trailing Stop'}
        
        return None
    
    def _close_trade(self, trade, exit_price, exit_time, exit_reason):
        """Close a trade and calculate P&L"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
        trade.pnl_pct = price_change_pct * trade.direction * 100
        trade.pnl_dollars = trade.position_size * (exit_price - trade.entry_price) * trade.direction
        
        # Update capital
        self.current_capital += trade.pnl_dollars
        
        return trade
    
    def _calculate_current_equity(self, open_trade, current_price):
        """Calculate current equity including open P&L"""
        equity = self.current_capital
        
        if open_trade is not None:
            unrealized_pnl = open_trade.position_size * (current_price - open_trade.entry_price) * open_trade.direction
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_atr(self, df, i, period=14):
        """Calculate ATR for position"""
        if i < period:
            return 0
        
        high_low = df.iloc[i-period:i]['High'] - df.iloc[i-period:i]['Low']
        high_close = abs(df.iloc[i-period:i]['High'] - df.iloc[i-period+1:i+1]['Close'].shift(1))
        low_close = abs(df.iloc[i-period:i]['Low'] - df.iloc[i-period+1:i+1]['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.mean()
    
    def _calculate_performance_metrics(self, df):
        """Calculate strategy performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return_pct': 0,
                'profit_factor': 0
            }
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl_pct > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        # Returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Sharpe ratio (annualized for crypto 24/7 trading)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 96)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # Total return
        total_return_pct = (equity_array[-1] - equity_array[0]) / equity_array[0] * 100
        
        # Profit factor
        gross_profits = sum(t.pnl_dollars for t in self.trades if t.pnl_dollars > 0)
        gross_losses = abs(sum(t.pnl_dollars for t in self.trades if t.pnl_dollars < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'final_capital': self.current_capital,
            'total_pnl': self.current_capital - self.config['initial_capital']
        }


def create_conservative_crypto_config():
    """Conservative configuration for crypto trading"""
    return {
        'initial_capital': 100000,
        'risk_per_trade': 0.001,  # 0.1% risk per trade
        'max_position_size': 0.05,  # Max 5% of capital in one position
        
        # Stop loss and take profit (as percentages)
        'sl_max_pct': 0.03,  # 3% max stop loss
        'sl_atr_multiplier': 2.0,
        'tp_atr_multipliers': [1.0, 2.0, 3.0],  # ATR multiples for TPs
        'max_tp_pct': 0.06,  # 6% max take profit
        
        # Trailing stop
        'use_trailing_stop': True,
        'tsl_activation_pct': 0.01,  # Activate at 1% profit
        'tsl_buffer_pct': 0.005,  # 0.5% buffer
        
        # Exit conditions
        'exit_on_signal_flip': True,
        'signal_flip_min_profit_pct': 0.005,  # 0.5% min profit
        'signal_flip_min_hours': 2.0,
    }


def create_aggressive_crypto_config():
    """Aggressive configuration for crypto trading"""
    return {
        'initial_capital': 100000,
        'risk_per_trade': 0.002,  # 0.2% risk per trade
        'max_position_size': 0.1,  # Max 10% of capital in one position
        
        # Stop loss and take profit (as percentages)
        'sl_max_pct': 0.015,  # 1.5% max stop loss
        'sl_atr_multiplier': 1.0,
        'tp_atr_multipliers': [0.5, 1.0, 1.5],  # Tighter targets
        'max_tp_pct': 0.03,  # 3% max take profit
        
        # Trailing stop
        'use_trailing_stop': True,
        'tsl_activation_pct': 0.005,  # Activate at 0.5% profit
        'tsl_buffer_pct': 0.003,  # 0.3% buffer
        
        # Exit conditions
        'exit_on_signal_flip': True,
        'signal_flip_min_profit_pct': 0.0,  # Exit immediately
        'signal_flip_min_hours': 0.0,
    }


def run_crypto_validation(data_path='../crypto_data/ETHUSD_MASTER_15M.csv'):
    """Run validation on crypto strategies"""
    
    print("="*80)
    print("CRYPTO STRATEGY VALIDATION - FIXED VERSION")
    print("="*80)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"\nData loaded: {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Test configurations
    configs = [
        ("Conservative", create_conservative_crypto_config()),
        ("Aggressive", create_aggressive_crypto_config())
    ]
    
    results = {}
    
    for config_name, config in configs:
        print(f"\n\n{'='*60}")
        print(f"Testing {config_name} Configuration")
        print(f"{'='*60}")
        
        # Run multiple tests
        test_results = []
        n_tests = 20
        sample_size = 8000
        
        for i in range(n_tests):
            # Random sample
            max_start = len(df) - sample_size
            if max_start <= 0:
                print("Insufficient data for testing")
                break
            
            start_idx = np.random.randint(0, max_start)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            
            # Add indicators
            sample_df = TIC.add_neuro_trend_intelligent(sample_df)
            sample_df = TIC.add_market_bias(sample_df)
            sample_df = TIC.add_intelligent_chop(sample_df)
            
            # Run strategy
            strategy = CryptoStrategy(config)
            
            try:
                metrics = strategy.run_backtest(sample_df)
                test_results.append(metrics)
                
                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{n_tests} tests...")
                    
            except Exception as e:
                print(f"  Error in test {i + 1}: {e}")
                continue
        
        if test_results:
            # Calculate averages
            avg_metrics = {
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in test_results]),
                'avg_return': np.mean([r['total_return_pct'] for r in test_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in test_results]),
                'avg_drawdown': np.mean([r['max_drawdown'] for r in test_results]),
                'avg_trades': np.mean([r['total_trades'] for r in test_results]),
                'sharpe_above_1': sum(1 for r in test_results if r['sharpe_ratio'] > 1.0) / len(test_results) * 100
            }
            
            results[config_name] = avg_metrics
            
            print(f"\nResults Summary:")
            print(f"  Average Sharpe: {avg_metrics['avg_sharpe']:.3f}")
            print(f"  Average Return: {avg_metrics['avg_return']:.1f}%")
            print(f"  Win Rate: {avg_metrics['avg_win_rate']:.1f}%")
            print(f"  Max Drawdown: {avg_metrics['avg_drawdown']:.1f}%")
            print(f"  Avg Trades: {avg_metrics['avg_trades']:.0f}")
            print(f"  % Tests with Sharpe > 1.0: {avg_metrics['sharpe_above_1']:.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_crypto_validation()
    
    if results:
        # Save results
        import json
        with open('results/crypto_strategy_performance.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Results saved to results/crypto_strategy_performance.json")