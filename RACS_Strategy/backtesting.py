"""
Comprehensive backtesting framework for trading strategies
This is a custom implementation, NOT using backtrader library

Why not backtrader?
1. Full control over execution logic and metrics
2. Lighter weight for our specific needs
3. Easier to add custom metrics (holding periods, entry timing)
4. Better integration with our existing codebase
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Trade information container"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position: str  # 'Long' or 'Short'
    size: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    holding_hours: float
    entry_signal: float
    exit_signal: float
    

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for the strategy"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """Return strategy parameters"""
        pass


class MomentumStrategy(Strategy):
    """Momentum mean reversion strategy"""
    
    def __init__(self, lookback: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based trading signals"""
        df = data.copy()
        
        # Calculate momentum
        df['Momentum'] = df['Close'].pct_change(self.lookback)
        df['Mom_Mean'] = df['Momentum'].rolling(50).mean()
        df['Mom_Std'] = df['Momentum'].rolling(50).std()
        df['Mom_Z'] = (df['Momentum'] - df['Mom_Mean']) / df['Mom_Std']
        
        # Generate signals (mean reversion)
        df['Signal'] = 0
        df.loc[df['Mom_Z'] < -self.entry_z, 'Signal'] = 1  # Buy on extreme negative
        df.loc[df['Mom_Z'] > self.entry_z, 'Signal'] = -1  # Sell on extreme positive
        
        # Exit when momentum normalizes
        df.loc[abs(df['Mom_Z']) < self.exit_z, 'Signal'] = 0
        
        return df
    
    def get_parameters(self) -> Dict:
        return {
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z
        }


class MACrossoverStrategy(Strategy):
    """Moving average crossover strategy"""
    
    def __init__(self, fast: int = 10, slow: int = 50, 
                 vol_filter: bool = True, trend_filter: bool = True):
        self.fast = fast
        self.slow = slow
        self.vol_filter = vol_filter
        self.trend_filter = trend_filter
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals"""
        df = data.copy()
        
        # Moving averages
        df['MA_Fast'] = df['Close'].rolling(self.fast).mean()
        df['MA_Slow'] = df['Close'].rolling(self.slow).mean()
        
        # Basic signals
        df['Signal'] = 0
        df.loc[df['MA_Fast'] > df['MA_Slow'], 'Signal'] = 1
        df.loc[df['MA_Fast'] < df['MA_Slow'], 'Signal'] = -1
        
        # Volatility filter
        if self.vol_filter:
            df['Volatility'] = df['Close'].pct_change().rolling(20).std()
            vol_threshold = df['Volatility'].quantile(0.75)
            df.loc[df['Volatility'] > vol_threshold, 'Signal'] = 0
        
        # Trend strength filter
        if self.trend_filter:
            df['Trend_Strength'] = abs(df['MA_Fast'] - df['MA_Slow']) / df['Close']
            min_strength = 0.001  # 0.1%
            df.loc[df['Trend_Strength'] < min_strength, 'Signal'] = 0
        
        return df
    
    def get_parameters(self) -> Dict:
        return {
            'fast': self.fast,
            'slow': self.slow,
            'vol_filter': self.vol_filter,
            'trend_filter': self.trend_filter
        }


class BreakoutStrategy(Strategy):
    """Breakout strategy"""
    
    def __init__(self, lookback: int = 20, breakout_multi: float = 1.5,
                 volume_confirm: bool = True):
        self.lookback = lookback
        self.breakout_multi = breakout_multi
        self.volume_confirm = volume_confirm
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals"""
        df = data.copy()
        
        # Calculate ATR
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Calculate ranges
        df['High_Roll'] = df['High'].rolling(self.lookback).max()
        df['Low_Roll'] = df['Low'].rolling(self.lookback).min()
        df['Range'] = df['High_Roll'] - df['Low_Roll']
        
        # Breakout levels
        df['Upper_Break'] = df['High_Roll'] + (df['ATR'] * self.breakout_multi)
        df['Lower_Break'] = df['Low_Roll'] - (df['ATR'] * self.breakout_multi)
        
        # Signals
        df['Signal'] = 0
        df.loc[df['Close'] > df['Upper_Break'].shift(1), 'Signal'] = 1
        df.loc[df['Close'] < df['Lower_Break'].shift(1), 'Signal'] = -1
        
        # Volume confirmation
        if self.volume_confirm and 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df.loc[(df['Signal'] != 0) & (df['Volume_Ratio'] < 1.2), 'Signal'] = 0
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_parameters(self) -> Dict:
        return {
            'lookback': self.lookback,
            'breakout_multi': self.breakout_multi,
            'volume_confirm': self.volume_confirm
        }


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.prepare_data()
        self.trades: List[Trade] = []
        self.signals_df = None  # Store signals for visualization
        
    def prepare_data(self):
        """Prepare data with basic calculations"""
        # Basic returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Additional features
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Close_Open_Ratio'] = self.data['Close'] / self.data['Open']
        
        # Volatility
        self.data['Volatility'] = self.data['Returns'].rolling(20).std()
        
        # Volume patterns (if available)
        if 'Volume' in self.data.columns:
            self.data['Volume_MA'] = self.data['Volume'].rolling(20).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
    
    def run_backtest(self, strategy: Strategy, track_trades: bool = False) -> Dict:
        """Run backtest with given strategy"""
        # Generate signals
        df = strategy.generate_signals(self.data)
        
        # Store signals DataFrame for visualization
        self.signals_df = df.copy()
        
        # Track trades if requested
        if track_trades:
            self._track_trades(df)
        
        # Calculate metrics
        return self._calculate_metrics(df)
    
    def _track_trades(self, df: pd.DataFrame):
        """Track individual trades for detailed analysis"""
        self.trades = []
        position = 0
        entry_idx = None
        entry_price = None
        entry_time = None
        
        for idx in range(len(df)):
            current_signal = df.iloc[idx]['Signal']
            
            # Entry
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_idx = idx
                entry_price = df.iloc[idx]['Close']
                entry_time = df.index[idx]
                
            # Exit
            elif position != 0 and (current_signal == 0 or current_signal == -position):
                exit_idx = idx
                exit_price = df.iloc[idx]['Close']
                exit_time = df.index[idx]
                
                # Calculate trade metrics
                if position > 0:  # Long trade
                    pnl = exit_price - entry_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # Short trade
                    pnl = entry_price - exit_price
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                
                holding_bars = exit_idx - entry_idx
                holding_hours = holding_bars * 0.25  # 15-minute bars
                
                # Create trade record
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position='Long' if position > 0 else 'Short',
                    size=1.0,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_bars=holding_bars,
                    holding_hours=holding_hours,
                    entry_signal=position,
                    exit_signal=current_signal
                )
                
                self.trades.append(trade)
                
                # Reset position
                position = current_signal if current_signal != 0 else 0
                if position != 0:
                    entry_idx = idx
                    entry_price = df.iloc[idx]['Close']
                    entry_time = df.index[idx]
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        df = df.dropna()
        
        if len(df) < 100:
            return {
                'sharpe': -999,
                'returns': -100,
                'win_rate': 0,
                'max_dd': 100,
                'trades': 0
            }
        
        # Position and returns
        df['Position'] = df['Signal'].shift(1).fillna(0)
        df['Strat_Returns'] = df['Position'] * df['Returns']
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Core metrics
        total_return = (1 + df['Strat_Returns']).prod() - 1
        
        # Sharpe ratio (annualized for 15-minute bars)
        if df['Strat_Returns'].std() > 0:
            sharpe = np.sqrt(252 * 24 * 4) * df['Strat_Returns'].mean() / df['Strat_Returns'].std()
        else:
            sharpe = 0
        
        # Win rate
        winning_trades = (df['Strat_Returns'] > 0).sum()
        total_trades = (df['Strat_Returns'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Max drawdown
        cumulative = (1 + df['Strat_Returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Prepare base metrics
        metrics = {
            'sharpe': float(sharpe),
            'returns': float(total_return * 100),
            'win_rate': float(win_rate * 100),
            'max_dd': float(abs(max_dd) * 100),
            'trades': int(total_trades)
        }
        
        # Add trade-based metrics if available
        if self.trades:
            metrics.update(self._calculate_trade_metrics())
        
        return metrics
    
    def _calculate_trade_metrics(self) -> Dict:
        """Calculate metrics from individual trades"""
        if not self.trades:
            return {}
        
        # Convert trades to DataFrame for easier analysis
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'position': trade.position,
                'pnl_pct': trade.pnl_pct,
                'holding_bars': trade.holding_bars,
                'holding_hours': trade.holding_hours,
                'entry_hour': trade.entry_time.hour,
                'entry_minute': trade.entry_time.minute
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Holding period metrics
        metrics = {
            'avg_holding_bars': trades_df['holding_bars'].mean(),
            'avg_holding_hours': trades_df['holding_hours'].mean(),
            'median_holding_hours': trades_df['holding_hours'].median(),
            'min_holding_hours': trades_df['holding_hours'].min(),
            'max_holding_hours': trades_df['holding_hours'].max()
        }
        
        # Entry timing analysis
        entry_hours = trades_df['entry_hour'].value_counts()
        metrics['entry_hour_distribution'] = entry_hours.to_dict()
        metrics['top_entry_hours'] = entry_hours.nlargest(3).to_dict()
        
        # Long vs Short analysis
        long_trades = trades_df[trades_df['position'] == 'Long']
        short_trades = trades_df[trades_df['position'] == 'Short']
        
        if len(long_trades) > 0:
            metrics['avg_long_holding_hours'] = long_trades['holding_hours'].mean()
            metrics['long_win_rate'] = (long_trades['pnl_pct'] > 0).sum() / len(long_trades) * 100
        else:
            metrics['avg_long_holding_hours'] = 0
            metrics['long_win_rate'] = 0
            
        if len(short_trades) > 0:
            metrics['avg_short_holding_hours'] = short_trades['holding_hours'].mean()
            metrics['short_win_rate'] = (short_trades['pnl_pct'] > 0).sum() / len(short_trades) * 100
        else:
            metrics['avg_short_holding_hours'] = 0
            metrics['short_win_rate'] = 0
        
        return metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame for analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position': trade.position,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'holding_bars': trade.holding_bars,
                'holding_hours': trade.holding_hours
            })
        
        return pd.DataFrame(trades_data)