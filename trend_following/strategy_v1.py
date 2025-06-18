import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import clone_indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clone_indicators.tic import TIC
from clone_indicators.indicators import *

import ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TrendFollowingStrategy:
    """
    Advanced trend-following strategy using multiple indicators with confluence.
    
    Key concepts:
    1. Multiple timeframe analysis
    2. Indicator confluence scoring
    3. Risk management with dynamic position sizing
    4. Regime detection to avoid choppy markets
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 1,
                 stop_loss_atr_mult: float = 2.0,
                 take_profit_atr_mult: float = 3.0,
                 min_confluence_score: float = 0.6):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.min_confluence_score = min_confluence_score
        
        # Trading state
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe."""
        
        # Use TIC to add advanced indicators
        df = TIC.add_super_trend(df, atr_period=10, multiplier=2.5, inplace=False)
        df = TIC.add_market_bias(df, ha_len=50, ha_len2=10, inplace=True)
        df = TIC.add_neuro_trend_intelligent(df, base_fast=10, base_slow=50, 
                                           confirm_bars=3, dynamic_thresholds=True, 
                                           inplace=True)
        df = TIC.add_intelligent_chop(df, inplace=True)
        df = TIC.add_andean_oscillator(df, length=250, sig_length=25, inplace=True)
        
        # Add traditional indicators from ta library
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
        df['MACD_signal'] = ta.trend.MACD(close=df['Close']).macd_signal()
        df['MACD_diff'] = ta.trend.MACD(close=df['Close']).macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        
        # ATR for risk management
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], 
                                                   close=df['Close'], window=14).average_true_range()
        
        # Volume indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], 
                                                       volume=df.get('Volume', pd.Series(np.ones(len(df))))).on_balance_volume()
        
        # Moving averages for trend
        df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], 
                                                close=df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        return df
    
    def calculate_confluence_score(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calculate a confluence score based on multiple indicators.
        Higher score means stronger trend signal.
        """
        if idx < 200:  # Need enough data for all indicators
            return 0.0
            
        score = 0.0
        max_score = 0.0
        
        # 1. SuperTrend alignment (weight: 2.0)
        if pd.notna(df.loc[idx, 'SuperTrend_Direction']):
            if df.loc[idx, 'SuperTrend_Direction'] == 1:
                score += 2.0
            elif df.loc[idx, 'SuperTrend_Direction'] == -1:
                score -= 2.0
        max_score += 2.0
        
        # 2. Market Bias (weight: 1.5)
        if pd.notna(df.loc[idx, 'MB_Bias']):
            if df.loc[idx, 'MB_Bias'] == 1:
                score += 1.5
            elif df.loc[idx, 'MB_Bias'] == -1:
                score -= 1.5
        max_score += 1.5
        
        # 3. NeuroTrend Intelligent Direction (weight: 2.0)
        if pd.notna(df.loc[idx, 'NTI_Direction']):
            if df.loc[idx, 'NTI_Direction'] == 1:
                score += 2.0
            elif df.loc[idx, 'NTI_Direction'] == -1:
                score -= 2.0
        max_score += 2.0
        
        # 4. Market Regime - avoid choppy markets (weight: 1.5)
        if pd.notna(df.loc[idx, 'IC_Regime']):
            regime = df.loc[idx, 'IC_Regime']
            if regime == 2:  # Strong trend
                score += 1.5 * np.sign(score)
            elif regime == 0:  # Choppy market
                score *= 0.5  # Reduce score in choppy markets
        max_score += 1.5
        
        # 5. Moving Average alignment (weight: 1.0)
        ema20 = df.loc[idx, 'EMA_20']
        ema50 = df.loc[idx, 'EMA_50']
        ema200 = df.loc[idx, 'EMA_200']
        close = df.loc[idx, 'Close']
        
        if close > ema20 > ema50 > ema200:
            score += 1.0
        elif close < ema20 < ema50 < ema200:
            score -= 1.0
        max_score += 1.0
        
        # 6. ADX trend strength (weight: 1.0)
        if pd.notna(df.loc[idx, 'ADX']) and df.loc[idx, 'ADX'] > 25:
            if df.loc[idx, 'DI_plus'] > df.loc[idx, 'DI_minus']:
                score += 1.0
            else:
                score -= 1.0
        max_score += 1.0
        
        # 7. RSI momentum (weight: 0.5)
        if pd.notna(df.loc[idx, 'RSI']):
            rsi = df.loc[idx, 'RSI']
            if 40 < rsi < 60:  # Neutral zone
                pass
            elif rsi > 60:
                score += 0.5
            elif rsi < 40:
                score -= 0.5
        max_score += 0.5
        
        # 8. MACD momentum (weight: 1.0)
        if pd.notna(df.loc[idx, 'MACD_diff']):
            if df.loc[idx, 'MACD_diff'] > 0:
                score += 1.0
            else:
                score -= 1.0
        max_score += 1.0
        
        # 9. Andean Oscillator signals (weight: 1.5)
        if pd.notna(df.loc[idx, 'AO_Bull']) and pd.notna(df.loc[idx, 'AO_Bear']):
            if df.loc[idx, 'AO_Bull'] > df.loc[idx, 'AO_Bear']:
                score += 1.5
            else:
                score -= 1.5
        max_score += 1.5
        
        # Normalize score to [-1, 1]
        normalized_score = score / max_score if max_score > 0 else 0
        
        return normalized_score
    
    def calculate_position_size(self, df: pd.DataFrame, idx: int, 
                              confluence_score: float, current_capital: float) -> float:
        """
        Calculate position size based on risk management rules.
        Uses ATR-based position sizing with confluence adjustment.
        """
        if pd.isna(df.loc[idx, 'ATR']) or df.loc[idx, 'ATR'] == 0:
            return 0
        
        # Base position size from risk per trade
        atr = df.loc[idx, 'ATR']
        close = df.loc[idx, 'Close']
        
        # Risk amount in base currency
        risk_amount = current_capital * self.risk_per_trade
        
        # Stop loss distance in price
        stop_distance = atr * self.stop_loss_atr_mult
        
        # Base position size (in units)
        base_position_size = risk_amount / stop_distance
        
        # Adjust by confluence score (scale from 0.5 to 1.5)
        confidence_multiplier = 0.5 + abs(confluence_score)
        
        # Adjust by market regime
        if pd.notna(df.loc[idx, 'IC_Confidence']):
            regime_confidence = df.loc[idx, 'IC_Confidence'] / 100.0
            confidence_multiplier *= (0.5 + 0.5 * regime_confidence)
        
        # Final position size
        position_size = base_position_size * confidence_multiplier
        
        # Cap at maximum position size (e.g., 10% of capital)
        max_position_value = current_capital * 0.1
        max_position_size = max_position_value / close
        
        return min(position_size, max_position_size)
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on the data with the strategy.
        """
        # Add indicators
        df = self.add_indicators(df)
        
        # Initialize tracking variables
        current_capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [current_capital]
        
        # Skip initial rows until all indicators are ready
        start_idx = 200
        
        for idx in range(start_idx, len(df)):
            current_date = df.index[idx]
            
            # Calculate confluence score
            confluence_score = self.calculate_confluence_score(df, idx)
            
            # Current price info
            close = df.loc[idx, 'Close']
            atr = df.loc[idx, 'ATR']
            
            # Check if we have an open position
            if position is not None:
                # Check exit conditions
                exit_signal = False
                exit_reason = ""
                
                # Stop loss
                if position['type'] == 'long' and close <= position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif position['type'] == 'short' and close >= position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                
                # Take profit
                elif position['type'] == 'long' and close >= position['take_profit']:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif position['type'] == 'short' and close <= position['take_profit']:
                    exit_signal = True
                    exit_reason = "Take Profit"
                
                # Trend reversal
                elif (position['type'] == 'long' and confluence_score < -0.3) or \
                     (position['type'] == 'short' and confluence_score > 0.3):
                    exit_signal = True
                    exit_reason = "Trend Reversal"
                
                if exit_signal:
                    # Close position
                    if position['type'] == 'long':
                        pnl = (close - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - close) * position['size']
                    
                    current_capital += pnl
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': close,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (position['entry_price'] * position['size']),
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Check entry conditions if no position
            elif abs(confluence_score) >= self.min_confluence_score:
                # Determine position type
                position_type = 'long' if confluence_score > 0 else 'short'
                
                # Calculate position size
                position_size = self.calculate_position_size(df, idx, confluence_score, current_capital)
                
                if position_size > 0:
                    # Calculate stop loss and take profit
                    if position_type == 'long':
                        stop_loss = close - (atr * self.stop_loss_atr_mult)
                        take_profit = close + (atr * self.take_profit_atr_mult)
                    else:
                        stop_loss = close + (atr * self.stop_loss_atr_mult)
                        take_profit = close - (atr * self.take_profit_atr_mult)
                    
                    # Enter position
                    position = {
                        'type': position_type,
                        'entry_date': current_date,
                        'entry_price': close,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confluence_score': confluence_score
                    }
                    
                    # Deduct position value from capital (for margin)
                    current_capital -= close * position_size
            
            # Track equity
            if position is not None:
                # Mark to market
                if position['type'] == 'long':
                    unrealized_pnl = (close - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - close) * position['size']
                
                equity_curve.append(current_capital + position['entry_price'] * position['size'] + unrealized_pnl)
            else:
                equity_curve.append(current_capital)
        
        # Close any remaining position
        if position is not None:
            close = df.iloc[-1]['Close']
            if position['type'] == 'long':
                pnl = (close - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - close) * position['size']
            
            current_capital += pnl
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': close,
                'size': position['size'],
                'pnl': pnl,
                'pnl_pct': pnl / (position['entry_price'] * position['size']),
                'exit_reason': 'End of Data'
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity_curve, df.index[start_idx:])
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': equity_curve[-1]
        }
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float], 
                         dates: pd.DatetimeIndex) -> Dict:
        """Calculate performance metrics."""
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'annual_return': 0
            }
        
        # Trade statistics
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        # Calculate returns for Sharpe ratio
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Assume 15-minute bars, calculate daily returns
        # 96 bars per day (24 hours * 4 bars/hour)
        bars_per_day = 96
        daily_returns = []
        
        for i in range(0, len(returns), bars_per_day):
            daily_return = np.prod(1 + returns[i:i+bars_per_day]) - 1
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Calculate Sharpe ratio (annualized)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        equity_peaks = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - equity_peaks) / equity_peaks
        max_drawdown = np.min(drawdowns)
        
        # Calculate returns
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (dates[-1] - dates[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': sum(pnls),
            'avg_trade': np.mean(pnls) if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }
        
        return metrics


def run_strategy_v1():
    """Run the first version of the strategy."""
    
    # Load data
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Initialize strategy
    strategy = TrendFollowingStrategy(
        initial_capital=1000000,
        risk_per_trade=0.02,
        max_positions=1,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        min_confluence_score=0.6
    )
    
    # Run backtest
    print("Running Strategy V1...")
    results = strategy.backtest(df)
    
    # Save results
    with open('results_v1.json', 'w') as f:
        json.dump({
            'version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'risk_per_trade': strategy.risk_per_trade,
                'stop_loss_atr_mult': strategy.stop_loss_atr_mult,
                'take_profit_atr_mult': strategy.take_profit_atr_mult,
                'min_confluence_score': strategy.min_confluence_score
            },
            'metrics': results['metrics'],
            'total_trades': len(results['trades'])
        }, f, indent=2)
    
    # Print metrics
    print("\nStrategy V1 Results:")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    
    return results


if __name__ == "__main__":
    run_strategy_v1()