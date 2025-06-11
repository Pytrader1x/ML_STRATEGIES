"""
Final Optimized Crypto Trading Strategy
Focus on trend following with proper risk management for crypto volatility
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
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
    position_size: float
    direction: int
    stop_loss: float
    take_profit: float  # Single TP for simplicity
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None


class FinalCryptoStrategy:
    """
    Simplified but effective crypto strategy
    Key principles:
    1. Follow strong trends only
    2. Wide stops to avoid noise
    3. Let winners run with trailing stops
    4. Reduce frequency - quality over quantity
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        self.current_capital = config['initial_capital']
        
    def calculate_position_size(self, price, stop_loss_price):
        """Simple fixed fractional position sizing"""
        risk_amount = self.current_capital * self.config['risk_per_trade']
        price_risk_pct = abs(price - stop_loss_price) / price
        
        if price_risk_pct == 0:
            return 0
        
        position_value = risk_amount / price_risk_pct
        position_size = position_value / price
        
        # Max position size
        max_value = self.current_capital * self.config['max_position_pct']
        max_size = max_value / price
        
        return min(position_size, max_size)
    
    def calculate_trend_strength(self, df, i):
        """Calculate trend strength using multiple timeframes"""
        if i < 200:
            return 0
        
        # Multiple moving averages
        ma_20 = df['Close'].iloc[i-20:i].mean()
        ma_50 = df['Close'].iloc[i-50:i].mean()
        ma_100 = df['Close'].iloc[i-100:i].mean()
        ma_200 = df['Close'].iloc[i-200:i].mean()
        
        current_price = df['Close'].iloc[i]
        
        # Bull trend scoring
        bull_score = 0
        if current_price > ma_20:
            bull_score += 1
        if ma_20 > ma_50:
            bull_score += 1
        if ma_50 > ma_100:
            bull_score += 1
        if ma_100 > ma_200:
            bull_score += 1
        if current_price > ma_200 * 1.1:  # 10% above 200MA
            bull_score += 1
        
        # Bear trend scoring
        bear_score = 0
        if current_price < ma_20:
            bear_score += 1
        if ma_20 < ma_50:
            bear_score += 1
        if ma_50 < ma_100:
            bear_score += 1
        if ma_100 < ma_200:
            bear_score += 1
        if current_price < ma_200 * 0.9:  # 10% below 200MA
            bear_score += 1
        
        # Return net score
        return bull_score - bear_score
    
    def check_volatility_filter(self, df, i):
        """Check if volatility is in acceptable range"""
        if i < 100:
            return True
        
        # Calculate recent volatility
        returns = df['Close'].iloc[i-96:i].pct_change().dropna()
        daily_vol = returns.std() * np.sqrt(96)
        
        # Filter out extreme volatility periods
        return 0.02 < daily_vol < 0.15  # 2% to 15% daily volatility
    
    def run_backtest(self, df):
        """Run simplified but effective backtest"""
        # Add required indicators
        required_cols = ['NTI_Direction', 'MB_Bias', 'IC_Signal']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required indicator: {col}")
        
        # Initialize
        self.trades = []
        self.equity_curve = [self.config['initial_capital']]
        self.current_capital = self.config['initial_capital']
        open_trade = None
        
        # Track last signal to avoid repeated entries
        last_signal_direction = 0
        bars_since_exit = 0
        
        # Main loop
        for i in range(200, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['Close']
            
            # Update bars since exit
            if open_trade is None:
                bars_since_exit += 1
            
            # Check open trade
            if open_trade is not None:
                # Simple exit checks
                exit_price = None
                exit_reason = None
                
                # Stop loss
                if open_trade.direction > 0:
                    if current_bar['Low'] <= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = 'Stop Loss'
                else:
                    if current_bar['High'] >= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = 'Stop Loss'
                
                # Take profit
                if exit_price is None:
                    if open_trade.direction > 0:
                        if current_bar['High'] >= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = 'Take Profit'
                    else:
                        if current_bar['Low'] <= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = 'Take Profit'
                
                # Trailing stop (simplified)
                if exit_price is None and self.config['use_trailing_stop']:
                    profit_pct = (current_price - open_trade.entry_price) / open_trade.entry_price * open_trade.direction
                    
                    if profit_pct > self.config['trailing_activation_pct']:
                        # Move stop to breakeven + small profit
                        if open_trade.direction > 0:
                            new_stop = open_trade.entry_price * (1 + self.config['trailing_lock_profit_pct'])
                            open_trade.stop_loss = max(open_trade.stop_loss, new_stop)
                            
                            # Dynamic trailing
                            trail_stop = current_price * (1 - self.config['trailing_distance_pct'])
                            open_trade.stop_loss = max(open_trade.stop_loss, trail_stop)
                        else:
                            new_stop = open_trade.entry_price * (1 - self.config['trailing_lock_profit_pct'])
                            open_trade.stop_loss = min(open_trade.stop_loss, new_stop)
                            
                            trail_stop = current_price * (1 + self.config['trailing_distance_pct'])
                            open_trade.stop_loss = min(open_trade.stop_loss, trail_stop)
                
                # Exit on strong reversal signal
                if exit_price is None and current_bar['NTI_Direction'] * open_trade.direction < 0:
                    trend_strength = self.calculate_trend_strength(df, i)
                    if abs(trend_strength) >= 3 and trend_strength * open_trade.direction < 0:
                        exit_price = current_price
                        exit_reason = 'Trend Reversal'
                
                # Close trade if exit triggered
                if exit_price is not None:
                    open_trade = self._close_trade(open_trade, exit_price, current_time, exit_reason)
                    self.trades.append(open_trade)
                    open_trade = None
                    last_signal_direction = 0
                    bars_since_exit = 0
            
            # Check for new entry
            if open_trade is None and bars_since_exit >= self.config['min_bars_between_trades']:
                # Get signals
                nti_dir = current_bar['NTI_Direction']
                mb_bias = current_bar['MB_Bias']
                
                # Check if new signal
                if nti_dir != 0 and nti_dir != last_signal_direction:
                    # Check trend strength
                    trend_score = self.calculate_trend_strength(df, i)
                    
                    # Strong trend filter
                    if abs(trend_score) >= self.config['min_trend_score']:
                        # Direction must match trend
                        if (trend_score > 0 and nti_dir > 0) or (trend_score < 0 and nti_dir < 0):
                            # Check volatility
                            if self.check_volatility_filter(df, i):
                                # Market bias confirmation
                                if mb_bias == nti_dir:
                                    # Not choppy
                                    if current_bar['IC_Signal'] != 0:
                                        # Calculate ATR for stops
                                        atr = self._calculate_atr(df, i, period=20)
                                        
                                        # Entry setup
                                        direction = nti_dir
                                        
                                        # Wider stops for crypto
                                        atr_in_pct = atr / current_price
                                        sl_distance = max(
                                            self.config['min_stop_pct'],
                                            atr_in_pct * self.config['atr_multiplier_sl']
                                        )
                                        
                                        # Asymmetric risk-reward
                                        tp_distance = sl_distance * self.config['risk_reward_ratio']
                                        
                                        # Calculate prices
                                        if direction > 0:
                                            sl_price = current_price * (1 - sl_distance)
                                            tp_price = current_price * (1 + tp_distance)
                                        else:
                                            sl_price = current_price * (1 + sl_distance)
                                            tp_price = current_price * (1 - tp_distance)
                                        
                                        # Position size
                                        position_size = self.calculate_position_size(current_price, sl_price)
                                        
                                        if position_size > 0:
                                            open_trade = CryptoTrade(
                                                entry_time=current_time,
                                                entry_price=current_price,
                                                position_size=position_size,
                                                direction=direction,
                                                stop_loss=sl_price,
                                                take_profit=tp_price
                                            )
                                            last_signal_direction = nti_dir
            
            # Update equity
            current_equity = self._calculate_current_equity(open_trade, current_price)
            self.equity_curve.append(current_equity)
        
        # Close final trade
        if open_trade is not None:
            open_trade = self._close_trade(
                open_trade, df.iloc[-1]['Close'], df.index[-1], 'End of Data'
            )
            self.trades.append(open_trade)
        
        return self._calculate_performance_metrics()
    
    def _close_trade(self, trade, exit_price, exit_time, exit_reason):
        """Close trade and calculate P&L"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # P&L calculation
        price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
        trade.pnl_pct = price_change_pct * trade.direction * 100
        trade.pnl_dollars = trade.position_size * (exit_price - trade.entry_price) * trade.direction
        
        # Update capital
        self.current_capital += trade.pnl_dollars
        
        return trade
    
    def _calculate_current_equity(self, open_trade, current_price):
        """Calculate equity including open position"""
        equity = self.current_capital
        
        if open_trade is not None:
            unrealized_pnl = open_trade.position_size * (current_price - open_trade.entry_price) * open_trade.direction
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_atr(self, df, i, period=14):
        """Calculate Average True Range"""
        if i < period:
            return df['High'].iloc[:i].mean() - df['Low'].iloc[:i].mean()
        
        high_low = df['High'].iloc[i-period:i] - df['Low'].iloc[i-period:i]
        return high_low.mean()
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return_pct': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in self.trades if t.pnl_pct > 0]
        losing_trades = [t for t in self.trades if t.pnl_pct < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        # Returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Remove zeros
        returns = returns[returns != 0]
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
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
        if losing_trades:
            gross_profits = sum(t.pnl_dollars for t in winning_trades)
            gross_losses = abs(sum(t.pnl_dollars for t in losing_trades))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        else:
            profit_factor = np.inf if winning_trades else 0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'avg_win': np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0,
            'win_count': len(winning_trades),
            'loss_count': len(losing_trades)
        }


def create_final_conservative_config():
    """Final conservative configuration - focus on high probability"""
    return {
        'initial_capital': 100000,
        'risk_per_trade': 0.002,  # 0.2% risk
        'max_position_pct': 0.10,  # 10% max position
        
        # Stops and targets
        'min_stop_pct': 0.04,  # 4% minimum stop
        'atr_multiplier_sl': 3.0,  # 3x ATR for stop
        'risk_reward_ratio': 2.0,  # 2:1 RR minimum
        
        # Trailing stop
        'use_trailing_stop': True,
        'trailing_activation_pct': 0.03,  # Activate at 3%
        'trailing_lock_profit_pct': 0.01,  # Lock 1% profit
        'trailing_distance_pct': 0.02,  # Trail by 2%
        
        # Entry filters
        'min_trend_score': 3,  # Strong trend required (3/5)
        'min_bars_between_trades': 20,  # Space out trades
    }


def create_final_moderate_config():
    """Final moderate configuration - balanced approach"""
    return {
        'initial_capital': 100000,
        'risk_per_trade': 0.0025,  # 0.25% risk
        'max_position_pct': 0.15,  # 15% max position
        
        # Stops and targets
        'min_stop_pct': 0.03,  # 3% minimum stop
        'atr_multiplier_sl': 2.5,  # 2.5x ATR
        'risk_reward_ratio': 1.5,  # 1.5:1 RR
        
        # Trailing stop
        'use_trailing_stop': True,
        'trailing_activation_pct': 0.02,  # Activate at 2%
        'trailing_lock_profit_pct': 0.005,  # Lock 0.5%
        'trailing_distance_pct': 0.015,  # Trail by 1.5%
        
        # Entry filters
        'min_trend_score': 2,  # Moderate trend
        'min_bars_between_trades': 10,
    }


def run_final_validation():
    """Run final validation with focus on what works"""
    
    print("="*80)
    print("FINAL CRYPTO STRATEGY VALIDATION")
    print("Focus: Trend Following with Proper Risk Management")
    print("="*80)
    
    # Load data
    data_path = '../crypto_data/ETHUSD_MASTER_15M.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Add indicators once
    print("\nPreparing data with indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    print(f"Data ready: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
    
    # Test configurations
    configs = [
        ("Conservative Trend", create_final_conservative_config()),
        ("Moderate Trend", create_final_moderate_config())
    ]
    
    # Test on specific market regimes
    test_periods = [
        ("2017-2018 First Bull", "2017-01-01", "2018-12-31"),
        ("2019-2020 Accumulation", "2019-01-01", "2020-12-31"),
        ("2021 Major Bull", "2021-01-01", "2021-12-31"),
        ("2022 Bear Market", "2022-01-01", "2022-12-31"),
        ("2023-2024 Recovery", "2023-01-01", "2024-12-31"),
        ("Last 12 Months", "2024-04-01", "2025-03-31")
    ]
    
    all_results = {}
    
    for config_name, config in configs:
        print(f"\n\n{'='*60}")
        print(f"{config_name} Configuration")
        print(f"{'='*60}")
        
        config_results = {}
        
        for period_name, start_date, end_date in test_periods:
            # Get period data
            period_df = df[start_date:end_date].copy()
            
            if len(period_df) < 500:
                print(f"\n{period_name}: Insufficient data")
                continue
            
            # Run strategy
            strategy = FinalCryptoStrategy(config)
            
            try:
                metrics = strategy.run_backtest(period_df)
                config_results[period_name] = metrics
                
                print(f"\n{period_name}:")
                print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
                print(f"  Return: {metrics['total_return_pct']:.1f}%")
                print(f"  Win Rate: {metrics['win_rate']:.1f}% ({metrics['win_count']}W/{metrics['loss_count']}L)")
                print(f"  Max DD: {metrics['max_drawdown']:.1f}%")
                print(f"  Avg Win: {metrics['avg_win']:.1f}%, Avg Loss: {metrics['avg_loss']:.1f}%")
                
            except Exception as e:
                print(f"\n{period_name}: Error - {e}")
                continue
        
        # Calculate overall metrics
        all_sharpes = [m['sharpe_ratio'] for m in config_results.values()]
        all_returns = [m['total_return_pct'] for m in config_results.values()]
        positive_periods = sum(1 for r in all_returns if r > 0)
        
        print(f"\n{config_name} Summary:")
        print(f"  Average Sharpe: {np.mean(all_sharpes):.3f}")
        print(f"  Average Return: {np.mean(all_returns):.1f}%")
        print(f"  Positive Periods: {positive_periods}/{len(all_returns)}")
        
        all_results[config_name] = config_results
    
    # Save results
    import json
    with open('results/crypto_final_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n✅ Results saved to results/crypto_final_results.json")
    
    return all_results


if __name__ == "__main__":
    results = run_final_validation()
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if results:
        # Find best configuration
        best_config = None
        best_sharpe = -999
        
        for config_name, periods in results.items():
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in periods.values()])
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_config = config_name
        
        print(f"\nBest Configuration: {best_config}")
        print(f"Average Sharpe Ratio: {best_sharpe:.3f}")
        
        if best_sharpe > 0.5:
            print("\n✅ Strategy shows promise for crypto trading")
            print("Recommendation: Further optimization and live testing warranted")
        else:
            print("\n⚠️ Strategy needs more work for consistent profitability")
            print("Recommendation: Consider alternative approaches or hybrid strategies")