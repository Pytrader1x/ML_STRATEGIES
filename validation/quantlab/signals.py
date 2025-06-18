"""
Signal Generation Module
Clean, testable signal functions with zero look-ahead bias
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def momentum(prices: pd.Series, 
             lookback: int = 40,
             entry_z: float = 1.5,
             exit_z: float = 0.5,
             rolling_window: int = 50) -> pd.DataFrame:
    """
    Momentum Z-Score Mean Reversion Signal
    
    Parameters:
    -----------
    prices : pd.Series
        Close prices with DateTime index
    lookback : int
        Momentum lookback period
    entry_z : float
        Z-score threshold for entry
    exit_z : float
        Z-score threshold for exit
    rolling_window : int
        Window for calculating mean/std
        
    Returns:
    --------
    pd.DataFrame with columns:
        - momentum: Raw momentum values
        - z_score: Normalized momentum
        - signal: Trading signal (-1, 0, 1)
        - entry: Entry points
        - exit: Exit points
    """
    
    # Calculate momentum
    momentum = prices.pct_change(lookback)
    
    # Calculate rolling statistics
    mom_mean = momentum.rolling(rolling_window, min_periods=rolling_window//2).mean()
    mom_std = momentum.rolling(rolling_window, min_periods=rolling_window//2).std()
    
    # Calculate z-score
    z_score = (momentum - mom_mean) / mom_std
    
    # Generate signals
    signal = pd.Series(0, index=prices.index)
    
    # Entry signals
    signal[z_score < -entry_z] = 1   # Long when oversold
    signal[z_score > entry_z] = -1    # Short when overbought
    
    # Exit signals - need to handle position tracking
    position = 0
    signals_clean = []
    entries = []
    exits = []
    
    for i in range(len(signal)):
        if position == 0:  # No position
            if signal.iloc[i] != 0:
                position = signal.iloc[i]
                signals_clean.append(position)
                entries.append(True)
                exits.append(False)
            else:
                signals_clean.append(0)
                entries.append(False)
                exits.append(False)
        else:  # Have position
            z = z_score.iloc[i] if not pd.isna(z_score.iloc[i]) else float('inf')
            
            # Check exit conditions
            if (position == 1 and z > -exit_z) or (position == -1 and z < exit_z):
                signals_clean.append(0)
                entries.append(False)
                exits.append(True)
                position = 0
            # Check reversal
            elif signal.iloc[i] == -position:
                position = signal.iloc[i]
                signals_clean.append(position)
                entries.append(True)
                exits.append(True)
            else:
                signals_clean.append(position)
                entries.append(False)
                exits.append(False)
    
    return pd.DataFrame({
        'momentum': momentum,
        'z_score': z_score,
        'signal': signals_clean,
        'entry': entries,
        'exit': exits
    }, index=prices.index)


def ma_crossover(prices: pd.Series,
                fast_period: int = 20,
                slow_period: int = 50,
                confirmation_bars: int = 1) -> pd.DataFrame:
    """
    Moving Average Crossover Signal
    
    Parameters:
    -----------
    prices : pd.Series
        Close prices
    fast_period : int
        Fast MA period
    slow_period : int
        Slow MA period  
    confirmation_bars : int
        Bars to confirm crossover
        
    Returns:
    --------
    pd.DataFrame with signal information
    """
    
    # Calculate MAs
    fast_ma = prices.rolling(fast_period).mean()
    slow_ma = prices.rolling(slow_period).mean()
    
    # Raw crossover
    raw_signal = np.where(fast_ma > slow_ma, 1, -1)
    
    # Add confirmation requirement
    signal = pd.Series(0, index=prices.index)
    
    for i in range(confirmation_bars, len(signal)):
        # Check if all confirmation bars have same signal
        if all(raw_signal[i-j] == raw_signal[i] for j in range(confirmation_bars)):
            signal.iloc[i] = raw_signal[i]
    
    return pd.DataFrame({
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'signal': signal,
        'entry': signal.diff() != 0,
        'exit': signal.diff() != 0
    }, index=prices.index)


def validate_signals(signals: pd.DataFrame, prices: pd.Series) -> Dict:
    """
    Validate signals for look-ahead bias and other issues
    
    Returns dict with validation results
    """
    
    results = {
        'has_lookahead': False,
        'signal_count': (signals['signal'] != 0).sum(),
        'long_count': (signals['signal'] > 0).sum(),
        'short_count': (signals['signal'] < 0).sum(),
        'issues': []
    }
    
    # Check for look-ahead bias
    # Signal at time t should only use data up to time t-1
    for col in signals.columns:
        if col == 'signal':
            continue
            
        # Check if any signal component uses future data
        shifted = signals[col].shift(1)
        if not signals[col].equals(shifted):
            # This is expected, but check correlation with future prices
            future_corr = signals[col].corr(prices.shift(-1))
            if abs(future_corr) > 0.1:  # Suspicious correlation with future
                results['has_lookahead'] = True
                results['issues'].append(f"{col} may have look-ahead bias (future corr: {future_corr:.3f})")
    
    return results