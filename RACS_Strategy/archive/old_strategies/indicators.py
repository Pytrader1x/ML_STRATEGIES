import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Try to import additional optimization libraries
try:
    import numba as nb
    HAS_NUMBA = True
    # Set optimal thread count
    import os
    nb.set_num_threads(os.cpu_count())
except ImportError:
    HAS_NUMBA = False
    nb = None

# Try to import numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorator if numba is not available
    def jit(*args, **kwargs):  # noqa: ARG001
        def decorator(func):
            return func
        return decorator
    # Define dummy prange
    prange = range


def is_numba_available():
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


def set_numba_enabled(enabled):
    """Enable or disable Numba usage."""
    global NUMBA_AVAILABLE
    NUMBA_AVAILABLE = enabled


@jit(nopython=True, cache=True, parallel=False)
def _supertrend_core_numba(high, low, close, src, atr_period, multiplier):
    """
    Numba-accelerated core computation for SuperTrend indicator.
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        src: Source prices array
        atr_period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        Tuple of (up, down, trend, atr_values)
    """
    n = len(high)
    
    # Calculate True Range components
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    
    # Fix first value
    hc[0] = hl[0]
    lc[0] = hl[0]
    
    # True Range
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Fast EMA for ATR
    alpha = 2.0 / (atr_period + 1)
    atr_values = np.empty(n, dtype=np.float64)
    atr_values[0] = tr[0]
    
    for i in range(1, n):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    # Calculate basic bands
    basic_up = src - multiplier * atr_values
    basic_down = src + multiplier * atr_values
    
    # Pre-allocate result arrays
    up = np.empty(n, dtype=np.float64)
    down = np.empty(n, dtype=np.float64)
    trend = np.empty(n, dtype=np.int32)
    
    # Initialize
    up[0] = basic_up[0]
    down[0] = basic_down[0]
    trend[0] = 1
    
    # Main calculation loop
    for i in range(1, n):
        prev_close = src[i-1]
        prev_up = up[i-1]
        prev_down = down[i-1]
        prev_trend = trend[i-1]
        curr_close = src[i]
        
        # Update bands
        if prev_close > prev_up:
            up[i] = max(basic_up[i], prev_up)
        else:
            up[i] = basic_up[i]
            
        if prev_close < prev_down:
            down[i] = min(basic_down[i], prev_down)
        else:
            down[i] = basic_down[i]
        
        # Update trend
        if prev_trend == 1:
            if curr_close <= up[i-1]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if curr_close >= down[i-1]:
                trend[i] = 1
            else:
                trend[i] = -1
    
    return up, down, trend, atr_values


@jit(nopython=True, cache=True)
def _fast_ema_numba(values, period):
    """Numba-accelerated EMA calculation."""
    alpha = 2.0 / (period + 1)
    n = len(values)
    ema = np.empty(n, dtype=np.float64)
    ema[0] = values[0]
    for i in range(1, n):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema


@jit(nopython=True, cache=True)
def _market_bias_core_numba(open_vals, high_vals, low_vals, close_vals, ha_len, ha_len2):
    """
    Numba-accelerated core computation for Market Bias indicator.
    
    Args:
        open_vals: Open prices array
        high_vals: High prices array
        low_vals: Low prices array
        close_vals: Close prices array
        ha_len: First EMA period
        ha_len2: Second EMA period
    
    Returns:
        Tuple of (mb_h2, mb_l2, mb_ha_avg, mb_bias, mb_o2, mb_c2)
    """
    n = len(open_vals)
    
    # 1. Initial Data Smoothing
    ha_ema_open = _fast_ema_numba(open_vals, ha_len)
    ha_ema_close = _fast_ema_numba(close_vals, ha_len)
    ha_ema_high = _fast_ema_numba(high_vals, ha_len)
    ha_ema_low = _fast_ema_numba(low_vals, ha_len)
    
    # 2. Heikin-Ashi Style Candle Construction (Smoothed)
    ha_close_val = (ha_ema_open + ha_ema_high + ha_ema_low + ha_ema_close) / 4
    
    # Vectorized HA open calculation
    ha_open_val = np.empty(n, dtype=np.float64)
    ha_open_val[0] = (ha_ema_open[0] + ha_ema_close[0]) / 2
    
    for i in range(1, n):
        ha_open_val[i] = (ha_open_val[i-1] + ha_close_val[i-1]) / 2
    
    ha_high_val = np.maximum(ha_ema_high, np.maximum(ha_open_val, ha_close_val))
    ha_low_val = np.minimum(ha_ema_low, np.minimum(ha_open_val, ha_close_val))
    
    # 3. Secondary Smoothing (Signal Lines)
    mb_o2 = _fast_ema_numba(ha_open_val, ha_len2)
    mb_c2 = _fast_ema_numba(ha_close_val, ha_len2)
    mb_h2 = _fast_ema_numba(ha_high_val, ha_len2)
    mb_l2 = _fast_ema_numba(ha_low_val, ha_len2)
    
    # 4. Bias Determination
    mb_ha_avg = (mb_h2 + mb_l2) / 2
    mb_bias = np.where(mb_c2 > mb_o2, 1, -1)  # 1 for Bullish, -1 for Bearish
    
    return mb_h2, mb_l2, mb_ha_avg, mb_bias, mb_o2, mb_c2

# SuperTrend Indicator with Numba acceleration support
def supertrend_indicator(df, atr_period=10, multiplier=3.0, source_col='Close', use_numba=True):
    """
    Ultra-optimized SuperTrend indicator with Numba acceleration support.
    
    This version uses Numba JIT compilation when available for extreme performance,
    falling back to optimized NumPy operations if Numba is not installed.
    
    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', source_col columns.
        atr_period (int): The period for ATR calculation.
        multiplier (float): The ATR multiplier.
        source_col (str): The column name for the source price.
        use_numba (bool): Whether to use Numba acceleration if available (default: True).
    
    Returns:
        pd.DataFrame: DataFrame with SuperTrend components.
    """
    # Extract arrays - working with raw numpy is faster
    n = len(df)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    src = df[source_col].values.astype(np.float64)
    
    # Try to use Numba-accelerated version first
    if NUMBA_AVAILABLE and use_numba:
        try:
            up, down, trend, atr_values = _supertrend_core_numba(
                high, low, close, src, atr_period, multiplier
            )
            supertrend = np.where(trend == 1, up, down)
            
            return pd.DataFrame({
                'SuperTrend_Up': up,
                'SuperTrend_Down': down,
                'SuperTrend_Direction': trend,
                'ATR': atr_values,
                'SuperTrend_Line': supertrend
            }, index=df.index)
        except Exception:
            # Fall back to non-Numba version if there's any issue
            pass
    
    # Fast ATR calculation using exponential smoothing
    # Calculate True Range components
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    
    # Fix first value
    hc[0] = hl[0]
    lc[0] = hl[0]
    
    # True Range
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Fast EMA for ATR using direct calculation
    alpha = 2.0 / (atr_period + 1)
    atr_values = np.empty(n, dtype=np.float64)
    atr_values[0] = tr[0]
    
    # Vectorized EMA calculation
    for i in range(1, n):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    # Calculate basic bands
    basic_up = src - multiplier * atr_values
    basic_down = src + multiplier * atr_values
    
    # Pre-allocate result arrays
    up = np.empty(n, dtype=np.float64)
    down = np.empty(n, dtype=np.float64)
    trend = np.empty(n, dtype=np.int32)
    
    # Initialize
    up[0] = basic_up[0]
    down[0] = basic_down[0]
    trend[0] = 1
    
    # Main calculation - optimized with minimal branching
    # This is the fastest possible implementation in pure Python/NumPy
    prev_close = src[0]
    prev_up = up[0]
    prev_down = down[0]
    prev_trend = 1
    
    for i in range(1, n):
        curr_close = src[i]
        
        # Update bands
        if prev_close > prev_up:
            new_up = max(basic_up[i], prev_up)
        else:
            new_up = basic_up[i]
            
        if prev_close < prev_down:
            new_down = min(basic_down[i], prev_down)
        else:
            new_down = basic_down[i]
        
        up[i] = new_up
        down[i] = new_down
        
        # Update trend
        if prev_trend == 1:
            if curr_close <= prev_up:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if curr_close >= prev_down:
                trend[i] = 1
            else:
                trend[i] = -1
        
        # Update previous values
        prev_close = curr_close
        prev_up = new_up
        prev_down = new_down
        prev_trend = trend[i]
    
    # Create final line
    supertrend = np.where(trend == 1, up, down)
    
    # Return as DataFrame
    return pd.DataFrame({
        'SuperTrend_Up': up,
        'SuperTrend_Down': down,
        'SuperTrend_Direction': trend,
        'ATR': atr_values,
        'SuperTrend_Line': supertrend
    }, index=df.index)

# Market Bias Indicator using Heikin-Ashi Style Candles 
def market_bias_indicator(df, ha_len=300, ha_len2=30, use_numba=True):
    """
    Optimized Market Bias indicator with Numba acceleration support.
    
    This version uses Numba JIT compilation when available for extreme performance,
    falling back to optimized NumPy operations if Numba is not installed.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        ha_len (int): Period for the first layer of EMA smoothing.
        ha_len2 (int): Period for the second layer of EMA smoothing (signal lines).
        use_numba (bool): Whether to use Numba acceleration if available (default: True).
    
    Returns:
        pd.DataFrame: DataFrame with Market Bias components.
    """
    # Work with numpy arrays for better performance
    n = len(df)
    open_vals = df['Open'].values.astype(np.float64)
    high_vals = df['High'].values.astype(np.float64)
    low_vals = df['Low'].values.astype(np.float64)
    close_vals = df['Close'].values.astype(np.float64)
    
    # Try to use Numba-accelerated version first
    if NUMBA_AVAILABLE and use_numba:
        try:
            mb_h2, mb_l2, mb_ha_avg, mb_bias, mb_o2, mb_c2 = _market_bias_core_numba(
                open_vals, high_vals, low_vals, close_vals, ha_len, ha_len2
            )
            
            return pd.DataFrame({
                'MB_h2': mb_h2,
                'MB_l2': mb_l2,
                'MB_ha_avg': mb_ha_avg,
                'MB_Bias': mb_bias,
                'MB_o2': mb_o2,
                'MB_c2': mb_c2
            }, index=df.index)
        except Exception:
            # Fall back to non-Numba version if there's any issue
            pass
    
    # Fast EMA calculation function
    def fast_ema(values, period):
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        return ema
    
    # 1. Initial Data Smoothing
    ha_ema_open = fast_ema(open_vals, ha_len)
    ha_ema_close = fast_ema(close_vals, ha_len)
    ha_ema_high = fast_ema(high_vals, ha_len)
    ha_ema_low = fast_ema(low_vals, ha_len)
    
    # 2. Heikin-Ashi Style Candle Construction (Smoothed)
    ha_close_val = (ha_ema_open + ha_ema_high + ha_ema_low + ha_ema_close) / 4
    
    # Vectorized HA open calculation
    ha_open_val = np.empty(n, dtype=np.float64)
    ha_open_val[0] = (ha_ema_open[0] + ha_ema_close[0]) / 2
    
    for i in range(1, n):
        ha_open_val[i] = (ha_open_val[i-1] + ha_close_val[i-1]) / 2
    
    ha_high_val = np.maximum(ha_ema_high, np.maximum(ha_open_val, ha_close_val))
    ha_low_val = np.minimum(ha_ema_low, np.minimum(ha_open_val, ha_close_val))
    
    # 3. Secondary Smoothing (Signal Lines)
    mb_o2 = fast_ema(ha_open_val, ha_len2)
    mb_c2 = fast_ema(ha_close_val, ha_len2)
    mb_h2 = fast_ema(ha_high_val, ha_len2)
    mb_l2 = fast_ema(ha_low_val, ha_len2)
    
    # 4. Bias Determination
    mb_ha_avg = (mb_h2 + mb_l2) / 2
    mb_bias = np.where(mb_c2 > mb_o2, 1, -1)  # 1 for Bullish, -1 for Bearish
    
    return pd.DataFrame({
        'MB_h2': mb_h2,
        'MB_l2': mb_l2,
        'MB_ha_avg': mb_ha_avg,
        'MB_Bias': mb_bias,
        'MB_o2': mb_o2,
        'MB_c2': mb_c2
    }, index=df.index)



# Support/Resistance Indicator using Fractal Patterns
@jit(nopython=True, cache=True)
def _fractal_sr_core_numba(high, low):
    """
    Numba-accelerated fractal pattern detection for Support/Resistance.
    
    Args:
        high: High prices array
        low: Low prices array
    
    Returns:
        Tuple of (fractal_highs, fractal_lows)
    """
    n = len(high)
    fractal_highs = np.full(n, np.nan)
    fractal_lows = np.full(n, np.nan)
    
    # Process bars from index 2 to n-3 (need 2 bars on each side)
    for i in range(2, n - 2):
        # Check for support level (fractal low)
        if (low[i] < low[i-1] and low[i] < low[i+1] and 
            low[i+1] < low[i+2] and low[i-1] < low[i-2]):
            fractal_lows[i] = low[i]
        
        # Check for resistance level (fractal high)
        if (high[i] > high[i-1] and high[i] > high[i+1] and 
            high[i+1] > high[i+2] and high[i-1] > high[i-2]):
            fractal_highs[i] = high[i]
    
    return fractal_highs, fractal_lows


def support_resistance_indicator_fractal(df, noise_filter=True, use_numba=True):
    """
    Fractal-based Support and Resistance Identification using showcase logic.
    
    This indicator identifies support and resistance levels by finding fractal patterns
    based on the logic from showcase_finding_support_and_resistance_levels_using_python.py.
    
    Algorithm:
    1. Support: Low[i] < Low[i-1] and Low[i] < Low[i+1] and Low[i+1] < Low[i+2] and Low[i-1] < Low[i-2]
    2. Resistance: High[i] > High[i-1] and High[i] > High[i+1] and High[i+1] > High[i+2] and High[i-1] > High[i-2]
    3. Optional noise filtering to remove levels that are too close to each other
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        noise_filter (bool): Whether to apply noise filtering (default: True)
        use_numba (bool): Whether to use Numba acceleration if available (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with support/resistance columns:
            - SR_FractalHighs: Fractal resistance levels (price at resistance points)
            - SR_FractalLows: Fractal support levels (price at support points)
            - SR_Levels: All significant levels (both support and resistance)
            - SR_LevelTypes: Type of each level ('Support' or 'Resistance')
            - SR_LevelStrengths: How many times each level has been tested
    """
    # Input validation
    if len(df) < 5:
        raise ValueError("DataFrame must have at least 5 rows for fractal analysis")
    
    # Extract price arrays
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    n = len(df)
    
    # Initialize result arrays
    fractal_highs = np.full(n, np.nan)
    fractal_lows = np.full(n, np.nan)
    levels = np.full(n, np.nan)
    level_types = np.full(n, '', dtype=object)
    level_strengths = np.full(n, np.nan)
    
    # Lists to store all detected levels
    all_levels = []  # (index, price, type)
    
    # Try to use Numba-accelerated version first
    if NUMBA_AVAILABLE and use_numba:
        try:
            fractal_highs, fractal_lows = _fractal_sr_core_numba(high, low)
        except Exception:
            # Fall back to non-Numba version if there's any issue
            use_numba = False
    
    if not use_numba:
        # Pure Python implementation
        # Process bars from index 2 to n-3 (need 2 bars on each side)
        for i in range(2, n - 2):
            # Check for support level (fractal low)
            if (low[i] < low[i-1] and low[i] < low[i+1] and 
                low[i+1] < low[i+2] and low[i-1] < low[i-2]):
                fractal_lows[i] = low[i]
                all_levels.append((i, low[i], 'Support'))
            
            # Check for resistance level (fractal high)
            if (high[i] > high[i-1] and high[i] > high[i+1] and 
                high[i+1] > high[i+2] and high[i-1] > high[i-2]):
                fractal_highs[i] = high[i]
                all_levels.append((i, high[i], 'Resistance'))
    else:
        # Extract levels from Numba results
        for i in range(n):
            if not np.isnan(fractal_lows[i]):
                all_levels.append((i, fractal_lows[i], 'Support'))
            if not np.isnan(fractal_highs[i]):
                all_levels.append((i, fractal_highs[i], 'Resistance'))
    
    # Apply noise filtering if requested
    if noise_filter and len(all_levels) > 0:
        # Calculate mean candle size for noise threshold
        candle_sizes = high - low
        mean_size = np.mean(candle_sizes[candle_sizes > 0])
        
        # Filter out levels that are too close to existing ones
        filtered_levels = []
        for idx, price, level_type in all_levels:
            # Check if this level is too close to any existing filtered level
            too_close = False
            for _, existing_price, _ in filtered_levels:
                if abs(price - existing_price) < mean_size:
                    too_close = True
                    break
            
            if not too_close:
                filtered_levels.append((idx, price, level_type))
        
        all_levels = filtered_levels
    
    # Count level strengths (how many times each level appears)
    level_counts = {}
    for _, price, _ in all_levels:
        rounded_price = round(price, 5)
        level_counts[rounded_price] = level_counts.get(rounded_price, 0) + 1
    
    # Fill the result arrays
    for idx, price, level_type in all_levels:
        levels[idx] = price
        level_types[idx] = level_type
        level_strengths[idx] = level_counts.get(round(price, 5), 1)
    
    return pd.DataFrame({
        'SR_FractalHighs': fractal_highs,
        'SR_FractalLows': fractal_lows,
        'SR_Levels': levels,
        'SR_LevelTypes': level_types,
        'SR_LevelStrengths': level_strengths
    }, index=df.index)


# NeuroTrend Indicator with Numba acceleration support
@jit(nopython=True, cache=True)
def _neurotrend_core_numba(high, low, close, volume,
                           atr_period=14, rsi_period=14, dmi_period=14,
                           base_fast_len=10, base_slow_len=50,
                           volatility_factor=2.0, momentum_factor=0.5,
                           slope_smooth=3, confidence_smooth=5):
    """
    Numba-accelerated core computation for NeuroTrend indicator.
    
    Args:
        high: High prices array
        low: Low prices array  
        close: Close prices array
        volume: Volume array
        atr_period: ATR period for volatility
        rsi_period: RSI period for momentum
        dmi_period: DMI period for trend strength
        base_fast_len: Base fast EMA length
        base_slow_len: Base slow EMA length
        volatility_factor: Factor for volatility adaptation
        momentum_factor: Factor for momentum adaptation
        slope_smooth: Smoothing period for slope calculation
        confidence_smooth: Smoothing period for confidence score
    
    Returns:
        Tuple of (fast_ema, slow_ema, slope_power, trend_phase, trend_direction,
                  confidence, reversal_risk, stall_detected, slope_forecast)
    """
    n = len(close)
    
    # Initialize result arrays
    fast_ema = np.empty(n, dtype=np.float64)
    slow_ema = np.empty(n, dtype=np.float64)
    slope_power = np.empty(n, dtype=np.float64)
    trend_phase = np.empty(n, dtype=np.int32)  # 0=Neutral, 1=Impulse, 2=Cooling, 3=Reversal
    trend_direction = np.empty(n, dtype=np.int32)  # 1=Bull, -1=Bear
    confidence = np.empty(n, dtype=np.float64)
    reversal_risk = np.empty(n, dtype=np.bool_)
    stall_detected = np.empty(n, dtype=np.bool_)
    slope_forecast = np.empty(n, dtype=np.float64)
    
    # Calculate ATR for volatility
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc[0] = hl[0]
    lc[0] = hl[0]
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Fast EMA for ATR
    alpha_atr = 2.0 / (atr_period + 1)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha_atr * tr[i] + (1 - alpha_atr) * atr[i-1]
    
    # Calculate RSI for momentum
    gains = np.maximum(0.0, np.diff(close))
    losses = np.maximum(0.0, -np.diff(close))
    gains = np.concatenate((np.array([0.0]), gains))
    losses = np.concatenate((np.array([0.0]), losses))
    
    # Average gains and losses using EMA
    alpha_rsi = 2.0 / (rsi_period + 1)
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gains[0]
    avg_loss[0] = losses[0]
    
    for i in range(1, n):
        avg_gain[i] = alpha_rsi * gains[i] + (1 - alpha_rsi) * avg_gain[i-1]
        avg_loss[i] = alpha_rsi * losses[i] + (1 - alpha_rsi) * avg_loss[i-1]
    
    # Calculate RSI
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
    
    # Calculate DMI components for confidence
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0.0
    low_diff[0] = 0.0
    
    # Directional Movement
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smooth DM
    alpha_dmi = 2.0 / (dmi_period + 1)
    smooth_plus_dm = np.empty(n, dtype=np.float64)
    smooth_minus_dm = np.empty(n, dtype=np.float64)
    smooth_plus_dm[0] = plus_dm[0]
    smooth_minus_dm[0] = minus_dm[0]
    
    for i in range(1, n):
        smooth_plus_dm[i] = alpha_dmi * plus_dm[i] + (1 - alpha_dmi) * smooth_plus_dm[i-1]
        smooth_minus_dm[i] = alpha_dmi * minus_dm[i] + (1 - alpha_dmi) * smooth_minus_dm[i-1]
    
    # Calculate DI+ and DI-
    di_plus = np.empty(n, dtype=np.float64)
    di_minus = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        if atr[i] > 0:
            di_plus[i] = (smooth_plus_dm[i] / atr[i]) * 100
            di_minus[i] = (smooth_minus_dm[i] / atr[i]) * 100
        else:
            di_plus[i] = 0.0
            di_minus[i] = 0.0
    
    # Calculate ADX
    di_diff = np.abs(di_plus - di_minus)
    di_sum = di_plus + di_minus
    dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0.0)
    
    # Smooth ADX
    adx = np.empty(n, dtype=np.float64)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = alpha_dmi * dx[i] + (1 - alpha_dmi) * adx[i-1]
    
    # Main calculation loop
    for i in range(n):
        # Adaptive EMA lengths based on market conditions
        # Normalize ATR and RSI
        atr_norm = atr[i] / close[i] if close[i] > 0 else 0.0
        rsi_norm = (rsi[i] - 50.0) / 50.0  # -1 to 1
        
        # Adjust EMA lengths
        fast_len = max(5, min(20, base_fast_len + int(atr_norm * volatility_factor * 10)))
        slow_len = max(20, min(100, base_slow_len + int(abs(rsi_norm) * momentum_factor * 20)))
        
        # Calculate adaptive EMAs
        if i == 0:
            fast_ema[i] = close[i]
            slow_ema[i] = close[i]
        else:
            alpha_fast = 2.0 / (fast_len + 1)
            alpha_slow = 2.0 / (slow_len + 1)
            fast_ema[i] = alpha_fast * close[i] + (1 - alpha_fast) * fast_ema[i-1]
            slow_ema[i] = alpha_slow * close[i] + (1 - alpha_slow) * slow_ema[i-1]
        
        # Calculate slope and power
        if i >= slope_smooth:
            # Slope calculation
            fast_slope = (fast_ema[i] - fast_ema[i-slope_smooth]) / slope_smooth
            slow_slope = (slow_ema[i] - slow_ema[i-slope_smooth]) / slope_smooth
            
            # Normalize slopes
            price_range = max(0.0001, np.max(high[max(0,i-20):i+1]) - np.min(low[max(0,i-20):i+1]))
            fast_slope_norm = fast_slope / price_range
            slow_slope_norm = slow_slope / price_range
            
            # Slope power metric
            slope_power[i] = (fast_slope_norm - slow_slope_norm) * 100
            
            # Trend direction
            trend_direction[i] = 1 if fast_ema[i] > slow_ema[i] else -1
            
            # Trend phase classification
            slope_threshold = 0.5
            cooling_threshold = 0.2
            
            if abs(slope_power[i]) > slope_threshold:
                if slope_power[i] > 0 and trend_direction[i] == 1:
                    trend_phase[i] = 1  # Bullish Impulse
                elif slope_power[i] < 0 and trend_direction[i] == -1:
                    trend_phase[i] = 1  # Bearish Impulse
                else:
                    trend_phase[i] = 3  # Reversal
            elif abs(slope_power[i]) > cooling_threshold:
                trend_phase[i] = 2  # Cooling
            else:
                trend_phase[i] = 0  # Neutral
            
            # Confidence score based on ADX and trend alignment
            trend_alignment = 1.0 if (di_plus[i] > di_minus[i] and trend_direction[i] == 1) or \
                                    (di_minus[i] > di_plus[i] and trend_direction[i] == -1) else 0.5
            confidence[i] = min(100.0, adx[i] * trend_alignment)
            
            # Reversal risk detection
            momentum_divergence = (rsi[i] > 70 and trend_direction[i] == 1 and slope_power[i] < 0) or \
                                 (rsi[i] < 30 and trend_direction[i] == -1 and slope_power[i] > 0)
            reversal_risk[i] = momentum_divergence or trend_phase[i] == 3
            
            # Stall detection
            stall_detected[i] = abs(slope_power[i]) < 0.1 and atr_norm < 0.01
            
            # Slope forecast (simple linear projection)
            if i >= slope_smooth * 2:
                slope_change = slope_power[i] - slope_power[i-slope_smooth]
                slope_forecast[i] = slope_power[i] + slope_change
            else:
                slope_forecast[i] = slope_power[i]
            
        else:
            # Initialize early values
            slope_power[i] = 0.0
            trend_direction[i] = 1 if close[i] > close[0] else -1
            trend_phase[i] = 0
            confidence[i] = 50.0
            reversal_risk[i] = False
            stall_detected[i] = False
            slope_forecast[i] = 0.0
    
    return (fast_ema, slow_ema, slope_power, trend_phase, trend_direction,
            confidence, reversal_risk, stall_detected, slope_forecast)


def neurotrend_indicator(df, atr_period=14, rsi_period=14, dmi_period=14,
                        base_fast_len=10, base_slow_len=50,
                        volatility_factor=2.0, momentum_factor=0.5,
                        slope_smooth=3, confidence_smooth=5,
                        use_numba=True):
    """
    NeuroTrend - Advanced Adaptive Trend Analysis Indicator.
    
    This indicator uses neural network-inspired adaptive techniques to analyze
    market trends with dynamic EMAs that adjust based on volatility (ATR) and
    momentum (RSI). It provides trend phase classification, confidence scoring,
    and reversal/stall detection without look-ahead bias.
    
    Features:
    - Adaptive EMAs with dynamic lengths based on market conditions
    - Slope calculation and power metrics for trend strength
    - Trend phase classification (Impulse, Cooling, Neutral, Reversal)
    - Confidence scoring using DMI/ADX
    - Reversal and stall detection
    - Slope forecasting for trend projection
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data (Volume optional)
        atr_period (int): ATR period for volatility measurement (default: 14)
        rsi_period (int): RSI period for momentum measurement (default: 14)
        dmi_period (int): DMI/ADX period for trend strength (default: 14)
        base_fast_len (int): Base length for fast adaptive EMA (default: 10)
        base_slow_len (int): Base length for slow adaptive EMA (default: 50)
        volatility_factor (float): Factor for volatility-based adaptation (default: 2.0)
        momentum_factor (float): Factor for momentum-based adaptation (default: 0.5)
        slope_smooth (int): Smoothing period for slope calculation (default: 3)
        confidence_smooth (int): Smoothing period for confidence score (default: 5)
        use_numba (bool): Whether to use Numba acceleration if available (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with NeuroTrend components:
            - NT_FastEMA: Adaptive fast EMA
            - NT_SlowEMA: Adaptive slow EMA
            - NT_SlopePower: Slope power metric (-100 to 100)
            - NT_TrendPhase: Trend phase classification (text)
            - NT_TrendDirection: Trend direction (1 for bullish, -1 for bearish)
            - NT_Confidence: Confidence score (0-100)
            - NT_ReversalRisk: Reversal risk flag (boolean)
            - NT_StallDetected: Stall detection flag (boolean)
            - NT_SlopeForecast: Projected slope for next period
    """
    # Input validation - Volume is now optional
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    if len(df) < max(atr_period, rsi_period, dmi_period, base_slow_len):
        raise ValueError("Insufficient data for NeuroTrend calculation")
    
    # Extract price arrays
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    
    # Handle missing volume - use a synthetic volume based on price range
    if 'Volume' in df.columns:
        volume = df['Volume'].values.astype(np.float64)
    else:
        # Create synthetic volume based on price range and volatility
        # This is a simple proxy that increases with price movement
        price_range = high - low
        price_change = np.abs(np.diff(close, prepend=close[0]))
        volume = price_range * price_change * 1000000  # Scale factor for reasonable values
    
    # Try to use Numba-accelerated version first
    if NUMBA_AVAILABLE and use_numba:
        try:
            (fast_ema, slow_ema, slope_power, trend_phase_num, trend_direction,
             confidence, reversal_risk, stall_detected, slope_forecast) = _neurotrend_core_numba(
                high, low, close, volume,
                atr_period, rsi_period, dmi_period,
                base_fast_len, base_slow_len,
                volatility_factor, momentum_factor,
                slope_smooth, confidence_smooth
            )
            
            # Convert numeric trend phase to text
            phase_map = {0: 'Neutral', 1: 'Impulse', 2: 'Cooling', 3: 'Reversal'}
            trend_phase = np.array([phase_map[p] for p in trend_phase_num], dtype=object)
            
            return pd.DataFrame({
                'NT_FastEMA': fast_ema,
                'NT_SlowEMA': slow_ema,
                'NT_SlopePower': slope_power,
                'NT_TrendPhase': trend_phase,
                'NT_TrendDirection': trend_direction,
                'NT_Confidence': confidence,
                'NT_ReversalRisk': reversal_risk,
                'NT_StallDetected': stall_detected,
                'NT_SlopeForecast': slope_forecast
            }, index=df.index)
            
        except Exception:
            # Fall back to non-Numba version if there's any issue
            pass
    
    # Pure Python/NumPy implementation (fallback)
    n = len(df)
    
    # Initialize result arrays
    fast_ema = np.empty(n, dtype=np.float64)
    slow_ema = np.empty(n, dtype=np.float64)
    slope_power = np.empty(n, dtype=np.float64)
    trend_phase = np.empty(n, dtype=object)
    trend_direction = np.empty(n, dtype=np.int32)
    confidence = np.empty(n, dtype=np.float64)
    reversal_risk = np.empty(n, dtype=bool)
    stall_detected = np.empty(n, dtype=bool)
    slope_forecast = np.empty(n, dtype=np.float64)
    
    # Calculate ATR for volatility
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc[0] = hl[0]
    lc[0] = hl[0]
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Fast EMA for ATR
    alpha_atr = 2.0 / (atr_period + 1)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha_atr * tr[i] + (1 - alpha_atr) * atr[i-1]
    
    # Calculate RSI for momentum
    gains = np.maximum(0.0, np.diff(close))
    losses = np.maximum(0.0, -np.diff(close))
    gains = np.concatenate(([0.0], gains))
    losses = np.concatenate(([0.0], losses))
    
    # Average gains and losses
    avg_gain = pd.Series(gains).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=rsi_period, adjust=False).mean().values
    
    # Calculate RSI
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100 if avg_gain.any() else 50, rsi)
    
    # Calculate DMI/ADX for confidence
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Smooth DM
    smooth_plus_dm = pd.Series(plus_dm).ewm(span=dmi_period, adjust=False).mean().values
    smooth_minus_dm = pd.Series(minus_dm).ewm(span=dmi_period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr > 0, (smooth_plus_dm / atr) * 100, 0)
    di_minus = np.where(atr > 0, (smooth_minus_dm / atr) * 100, 0)
    
    # Calculate ADX
    di_diff = np.abs(di_plus - di_minus)
    di_sum = di_plus + di_minus
    dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0)
    adx = pd.Series(dx).ewm(span=dmi_period, adjust=False).mean().values
    
    # Main calculation loop
    for i in range(n):
        # Adaptive EMA lengths
        atr_norm = atr[i] / close[i] if close[i] > 0 else 0.0
        rsi_norm = (rsi[i] - 50.0) / 50.0
        
        fast_len = max(5, min(20, base_fast_len + int(atr_norm * volatility_factor * 10)))
        slow_len = max(20, min(100, base_slow_len + int(abs(rsi_norm) * momentum_factor * 20)))
        
        # Calculate adaptive EMAs
        if i == 0:
            fast_ema[i] = close[i]
            slow_ema[i] = close[i]
        else:
            alpha_fast = 2.0 / (fast_len + 1)
            alpha_slow = 2.0 / (slow_len + 1)
            fast_ema[i] = alpha_fast * close[i] + (1 - alpha_fast) * fast_ema[i-1]
            slow_ema[i] = alpha_slow * close[i] + (1 - alpha_slow) * slow_ema[i-1]
        
        # Calculate slope and power
        if i >= slope_smooth:
            # Slope calculation
            fast_slope = (fast_ema[i] - fast_ema[i-slope_smooth]) / slope_smooth
            slow_slope = (slow_ema[i] - slow_ema[i-slope_smooth]) / slope_smooth
            
            # Normalize slopes
            price_range = max(0.0001, np.max(high[max(0,i-20):i+1]) - np.min(low[max(0,i-20):i+1]))
            fast_slope_norm = fast_slope / price_range
            slow_slope_norm = slow_slope / price_range
            
            # Slope power metric
            slope_power[i] = (fast_slope_norm - slow_slope_norm) * 100
            
            # Trend direction
            trend_direction[i] = 1 if fast_ema[i] > slow_ema[i] else -1
            
            # Trend phase classification
            slope_threshold = 0.5
            cooling_threshold = 0.2
            
            if abs(slope_power[i]) > slope_threshold:
                if slope_power[i] > 0 and trend_direction[i] == 1:
                    trend_phase[i] = 'Impulse'
                elif slope_power[i] < 0 and trend_direction[i] == -1:
                    trend_phase[i] = 'Impulse'
                else:
                    trend_phase[i] = 'Reversal'
            elif abs(slope_power[i]) > cooling_threshold:
                trend_phase[i] = 'Cooling'
            else:
                trend_phase[i] = 'Neutral'
            
            # Confidence score
            trend_alignment = 1.0 if (di_plus[i] > di_minus[i] and trend_direction[i] == 1) or \
                                    (di_minus[i] > di_plus[i] and trend_direction[i] == -1) else 0.5
            confidence[i] = min(100.0, adx[i] * trend_alignment)
            
            # Reversal risk detection
            momentum_divergence = (rsi[i] > 70 and trend_direction[i] == 1 and slope_power[i] < 0) or \
                                 (rsi[i] < 30 and trend_direction[i] == -1 and slope_power[i] > 0)
            reversal_risk[i] = momentum_divergence or trend_phase[i] == 'Reversal'
            
            # Stall detection
            stall_detected[i] = abs(slope_power[i]) < 0.1 and atr_norm < 0.01
            
            # Slope forecast
            if i >= slope_smooth * 2:
                slope_change = slope_power[i] - slope_power[i-slope_smooth]
                slope_forecast[i] = slope_power[i] + slope_change
            else:
                slope_forecast[i] = slope_power[i]
            
        else:
            # Initialize early values
            slope_power[i] = 0.0
            trend_direction[i] = 1 if close[i] > close[0] else -1
            trend_phase[i] = 'Neutral'
            confidence[i] = 50.0
            reversal_risk[i] = False
            stall_detected[i] = False
            slope_forecast[i] = 0.0
    
    return pd.DataFrame({
        'NT_FastEMA': fast_ema,
        'NT_SlowEMA': slow_ema,
        'NT_SlopePower': slope_power,
        'NT_TrendPhase': trend_phase,
        'NT_TrendDirection': trend_direction,
        'NT_Confidence': confidence,
        'NT_ReversalRisk': reversal_risk,
        'NT_StallDetected': stall_detected,
        'NT_SlopeForecast': slope_forecast
    }, index=df.index)


# NeuroTrend Intelligent - Anti-Whipsaw Implementation
@jit(nopython=True, cache=True)
def _apply_hysteresis_numba(raw_dir: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Numba-accelerated hysteresis filter for direction changes.
    
    Args:
        raw_dir: Raw direction signals (+1/-1)
        k: Number of consecutive bars required for confirmation
    
    Returns:
        Confirmed direction signals
    """
    n = len(raw_dir)
    confirmed = np.empty(n, dtype=np.float64)
    confirmed[:] = np.nan
    
    # Track consecutive count
    count = 1
    current_dir = raw_dir[0]
    confirmed[0] = current_dir
    
    for i in range(1, n):
        if raw_dir[i] == current_dir:
            count += 1
        else:
            # Direction changed
            if count >= k:
                # Previous direction was confirmed, start counting new direction
                count = 1
                current_dir = raw_dir[i]
            else:
                # Not enough confirmation, maintain previous direction
                confirmed[i] = confirmed[i-1] if i > 0 else current_dir
                continue
        
        # Check if we have enough confirmation
        if count >= k:
            # Backfill the confirmed direction
            for j in range(max(0, i - k + 1), i + 1):
                if np.isnan(confirmed[j]):
                    confirmed[j] = current_dir
        else:
            confirmed[i] = confirmed[i-1] if i > 0 else current_dir
    
    return confirmed


@jit(nopython=True, cache=True)
def _calculate_dynamic_thresholds_numba(slope_power: np.ndarray, window: int = 500,
                                       hi_percentile: float = 0.8, lo_percentile: float = 0.6) -> tuple:
    """
    Calculate dynamic thresholds based on rolling percentiles.
    
    Args:
        slope_power: Slope power values
        window: Rolling window size
        hi_percentile: High threshold percentile (0-1)
        lo_percentile: Low threshold percentile (0-1)
    
    Returns:
        Tuple of (hi_threshold, lo_threshold) arrays
    """
    n = len(slope_power)
    hi_thresh = np.empty(n, dtype=np.float64)
    lo_thresh = np.empty(n, dtype=np.float64)
    
    abs_slope = np.abs(slope_power)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_data = abs_slope[start_idx:i+1]
        
        if len(window_data) >= 20:  # Minimum samples for percentile
            sorted_data = np.sort(window_data)
            hi_idx = int(hi_percentile * (len(sorted_data) - 1))
            lo_idx = int(lo_percentile * (len(sorted_data) - 1))
            hi_thresh[i] = sorted_data[hi_idx]
            lo_thresh[i] = sorted_data[lo_idx]
        else:
            # Use default thresholds for early bars
            hi_thresh[i] = 0.5
            lo_thresh[i] = 0.2
    
    return hi_thresh, lo_thresh


@jit(nopython=True, cache=True)
def _neurotrend_intelligent_core_numba(high, low, close, volume,
                                     atr_period=14, rsi_period=14, dmi_period=14,
                                     base_fast_len=10, base_slow_len=50,
                                     volatility_factor=2.0, momentum_factor=0.5,
                                     slope_smooth=3, confidence_smooth=5,
                                     confirm_bars=3, dynamic_thresholds=True,
                                     threshold_window=500, hi_percentile=0.8, lo_percentile=0.6,
                                     atr_multiplier=1.25, atr_z_threshold=1.0):
    """
    Numba-accelerated core computation for NeuroTrend Intelligent indicator.
    
    This version includes anti-whipsaw features:
    - Hysteresis confirmation
    - Dynamic thresholds
    - Volatility regime adaptation
    """
    n = len(close)
    
    # Initialize result arrays
    fast_ema = np.empty(n, dtype=np.float64)
    slow_ema = np.empty(n, dtype=np.float64)
    slope_power = np.empty(n, dtype=np.float64)
    trend_phase = np.empty(n, dtype=np.int32)  # 0=Neutral, 1=Impulse, 2=Cooling, 3=Reversal
    trend_direction_raw = np.empty(n, dtype=np.int32)  # 1=Bull, -1=Bear
    trend_direction_confirmed = np.empty(n, dtype=np.int32)
    confidence = np.empty(n, dtype=np.float64)
    reversal_risk = np.empty(n, dtype=np.bool_)
    stall_detected = np.empty(n, dtype=np.bool_)
    slope_forecast = np.empty(n, dtype=np.float64)
    
    # Calculate ATR for volatility
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc[0] = hl[0]
    lc[0] = hl[0]
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Fast EMA for ATR
    alpha_atr = 2.0 / (atr_period + 1)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha_atr * tr[i] + (1 - alpha_atr) * atr[i-1]
    
    # Calculate ATR z-score for volatility regime detection
    atr_mean = np.empty(n, dtype=np.float64)
    atr_std = np.empty(n, dtype=np.float64)
    atr_z = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        window_start = max(0, i - 99)
        window_atr = atr[window_start:i+1]
        if len(window_atr) >= 20:
            atr_mean[i] = np.mean(window_atr)
            atr_std[i] = np.std(window_atr)
            if atr_std[i] > 0:
                atr_z[i] = (atr[i] - atr_mean[i]) / atr_std[i]
            else:
                atr_z[i] = 0.0
        else:
            atr_mean[i] = atr[i]
            atr_std[i] = 0.0
            atr_z[i] = 0.0
    
    # Calculate RSI for momentum
    gains = np.maximum(0.0, np.diff(close))
    losses = np.maximum(0.0, -np.diff(close))
    gains = np.concatenate((np.array([0.0]), gains))
    losses = np.concatenate((np.array([0.0]), losses))
    
    # Average gains and losses using EMA
    alpha_rsi = 2.0 / (rsi_period + 1)
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gains[0]
    avg_loss[0] = losses[0]
    
    for i in range(1, n):
        avg_gain[i] = alpha_rsi * gains[i] + (1 - alpha_rsi) * avg_gain[i-1]
        avg_loss[i] = alpha_rsi * losses[i] + (1 - alpha_rsi) * avg_loss[i-1]
    
    # Calculate RSI
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
    
    # Calculate DMI components for confidence
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0.0
    low_diff[0] = 0.0
    
    # Directional Movement
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smooth DM
    alpha_dmi = 2.0 / (dmi_period + 1)
    smooth_plus_dm = np.empty(n, dtype=np.float64)
    smooth_minus_dm = np.empty(n, dtype=np.float64)
    smooth_plus_dm[0] = plus_dm[0]
    smooth_minus_dm[0] = minus_dm[0]
    
    for i in range(1, n):
        smooth_plus_dm[i] = alpha_dmi * plus_dm[i] + (1 - alpha_dmi) * smooth_plus_dm[i-1]
        smooth_minus_dm[i] = alpha_dmi * minus_dm[i] + (1 - alpha_dmi) * smooth_minus_dm[i-1]
    
    # Calculate DI+ and DI-
    di_plus = np.empty(n, dtype=np.float64)
    di_minus = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        if atr[i] > 0:
            di_plus[i] = (smooth_plus_dm[i] / atr[i]) * 100
            di_minus[i] = (smooth_minus_dm[i] / atr[i]) * 100
        else:
            di_plus[i] = 0.0
            di_minus[i] = 0.0
    
    # Calculate ADX
    di_diff = np.abs(di_plus - di_minus)
    di_sum = di_plus + di_minus
    dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0.0)
    
    # Smooth ADX
    adx = np.empty(n, dtype=np.float64)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = alpha_dmi * dx[i] + (1 - alpha_dmi) * adx[i-1]
    
    # Main calculation loop
    for i in range(n):
        # Volatility regime adaptation
        volatility_mult = 1.0
        if atr_z[i] > atr_z_threshold:
            volatility_mult = atr_multiplier
        elif atr_z[i] < -atr_z_threshold:
            volatility_mult = 1.0 / atr_multiplier
        
        # Adaptive EMA lengths based on market conditions
        atr_norm = atr[i] / close[i] if close[i] > 0 else 0.0
        rsi_norm = (rsi[i] - 50.0) / 50.0  # -1 to 1
        
        # Adjust EMA lengths with volatility regime consideration
        fast_len_base = max(5, min(20, base_fast_len + int(atr_norm * volatility_factor * 10)))
        slow_len_base = max(20, min(100, base_slow_len + int(abs(rsi_norm) * momentum_factor * 20)))
        
        fast_len = int(fast_len_base * volatility_mult)
        slow_len = int(slow_len_base * volatility_mult)
        
        # Calculate adaptive EMAs
        if i == 0:
            fast_ema[i] = close[i]
            slow_ema[i] = close[i]
        else:
            alpha_fast = 2.0 / (fast_len + 1)
            alpha_slow = 2.0 / (slow_len + 1)
            fast_ema[i] = alpha_fast * close[i] + (1 - alpha_fast) * fast_ema[i-1]
            slow_ema[i] = alpha_slow * close[i] + (1 - alpha_slow) * slow_ema[i-1]
        
        # Calculate slope and power
        if i >= slope_smooth:
            # Slope calculation
            fast_slope = (fast_ema[i] - fast_ema[i-slope_smooth]) / slope_smooth
            slow_slope = (slow_ema[i] - slow_ema[i-slope_smooth]) / slope_smooth
            
            # Normalize slopes
            price_range = max(0.0001, np.max(high[max(0,i-20):i+1]) - np.min(low[max(0,i-20):i+1]))
            fast_slope_norm = fast_slope / price_range
            slow_slope_norm = slow_slope / price_range
            
            # Slope power metric
            slope_power[i] = (fast_slope_norm - slow_slope_norm) * 100
            
            # Raw trend direction
            trend_direction_raw[i] = 1 if fast_ema[i] > slow_ema[i] else -1
            
            # Dynamic thresholds
            if dynamic_thresholds and i >= threshold_window:
                window_start = max(0, i - threshold_window + 1)
                window_slope = np.abs(slope_power[window_start:i+1])
                sorted_slope = np.sort(window_slope)
                hi_idx = int(hi_percentile * (len(sorted_slope) - 1))
                lo_idx = int(lo_percentile * (len(sorted_slope) - 1))
                slope_threshold = sorted_slope[hi_idx] * volatility_mult
                cooling_threshold = sorted_slope[lo_idx] * volatility_mult
            else:
                slope_threshold = 0.5 * volatility_mult
                cooling_threshold = 0.2 * volatility_mult
            
            # Trend phase classification with dynamic thresholds
            if abs(slope_power[i]) > slope_threshold:
                if slope_power[i] > 0 and trend_direction_raw[i] == 1:
                    trend_phase[i] = 1  # Bullish Impulse
                elif slope_power[i] < 0 and trend_direction_raw[i] == -1:
                    trend_phase[i] = 1  # Bearish Impulse
                else:
                    trend_phase[i] = 3  # Reversal
            elif abs(slope_power[i]) > cooling_threshold:
                trend_phase[i] = 2  # Cooling
            else:
                trend_phase[i] = 0  # Neutral
            
            # Confidence score based on ADX and trend alignment
            trend_alignment = 1.0 if (di_plus[i] > di_minus[i] and trend_direction_raw[i] == 1) or \
                                    (di_minus[i] > di_plus[i] and trend_direction_raw[i] == -1) else 0.5
            confidence[i] = min(100.0, adx[i] * trend_alignment)
            
            # Reversal risk detection
            momentum_divergence = (rsi[i] > 70 and trend_direction_raw[i] == 1 and slope_power[i] < 0) or \
                                 (rsi[i] < 30 and trend_direction_raw[i] == -1 and slope_power[i] > 0)
            reversal_risk[i] = momentum_divergence or trend_phase[i] == 3
            
            # Stall detection
            stall_detected[i] = abs(slope_power[i]) < 0.1 and atr_norm < 0.01
            
            # Slope forecast (simple linear projection)
            if i >= slope_smooth * 2:
                slope_change = slope_power[i] - slope_power[i-slope_smooth]
                slope_forecast[i] = slope_power[i] + slope_change
            else:
                slope_forecast[i] = slope_power[i]
            
        else:
            # Initialize early values
            slope_power[i] = 0.0
            trend_direction_raw[i] = 1 if close[i] > close[0] else -1
            trend_phase[i] = 0
            confidence[i] = 50.0
            reversal_risk[i] = False
            stall_detected[i] = False
            slope_forecast[i] = 0.0
    
    # Apply hysteresis to trend direction
    trend_direction_confirmed = _apply_hysteresis_numba(trend_direction_raw, confirm_bars)
    
    # Additional arrays for diagnostics
    hi_thresh, lo_thresh = _calculate_dynamic_thresholds_numba(slope_power, threshold_window, hi_percentile, lo_percentile)
    
    return (fast_ema, slow_ema, slope_power, trend_phase, trend_direction_raw, trend_direction_confirmed,
            confidence, reversal_risk, stall_detected, slope_forecast, hi_thresh, lo_thresh, atr_z)


def neurotrend_intelligent(df, atr_period=14, rsi_period=14, dmi_period=14,
                          base_fast_len=10, base_slow_len=50,
                          volatility_factor=2.0, momentum_factor=0.5,
                          slope_smooth=3, confidence_smooth=5,
                          confirm_bars=3, dynamic_thresholds=True,
                          threshold_window=500, hi_percentile=0.8, lo_percentile=0.6,
                          atr_multiplier=1.25, atr_z_threshold=1.0,
                          enable_diagnostics=False, use_numba=True):
    """
    NeuroTrend Intelligent - Advanced Anti-Whipsaw Trend Analysis.
    
    This is an improved version of NeuroTrend with multiple anti-whipsaw features:
    1. Hysteresis confirmation filter
    2. Dynamic threshold scaling
    3. Volatility regime adaptation
    4. Diagnostic capabilities
    
    Args:
        df: DataFrame with OHLC data
        
        Core parameters:
        - atr_period: ATR period for volatility (default: 14)
        - rsi_period: RSI period for momentum (default: 14)
        - dmi_period: DMI/ADX period for trend strength (default: 14)
        - base_fast_len: Base fast EMA length (default: 10)
        - base_slow_len: Base slow EMA length (default: 50)
        - volatility_factor: Volatility adaptation factor (default: 2.0)
        - momentum_factor: Momentum adaptation factor (default: 0.5)
        - slope_smooth: Slope calculation smoothing (default: 3)
        - confidence_smooth: Confidence score smoothing (default: 5)
        
        Anti-whipsaw parameters:
        - confirm_bars: Bars required for direction confirmation (default: 3)
        - dynamic_thresholds: Use dynamic thresholds (default: True)
        - threshold_window: Window for threshold calculation (default: 500)
        - hi_percentile: High threshold percentile (default: 0.8)
        - lo_percentile: Low threshold percentile (default: 0.6)
        - atr_multiplier: Multiplier for high volatility regime (default: 1.25)
        - atr_z_threshold: Z-score threshold for regime detection (default: 1.0)
        
        Diagnostic parameters:
        - enable_diagnostics: Enable diagnostic outputs (default: False)
        - use_numba: Use Numba acceleration if available (default: True)
    
    Returns:
        DataFrame with NeuroTrend Intelligent components
    """
    # Input validation
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    if len(df) < max(atr_period, rsi_period, dmi_period, base_slow_len, threshold_window):
        raise ValueError("Insufficient data for NeuroTrend Intelligent calculation")
    
    # Extract price arrays
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    
    # Handle missing volume
    if 'Volume' in df.columns:
        volume = df['Volume'].values.astype(np.float64)
    else:
        price_range = high - low
        price_change = np.abs(np.diff(close, prepend=close[0]))
        volume = price_range * price_change * 1000000
    
    # Try to use Numba-accelerated version
    if NUMBA_AVAILABLE and use_numba:
        try:
            (fast_ema, slow_ema, slope_power, trend_phase_num, trend_direction_raw, trend_direction_confirmed,
             confidence, reversal_risk, stall_detected, slope_forecast, hi_thresh, lo_thresh, atr_z) = \
                _neurotrend_intelligent_core_numba(
                    high, low, close, volume,
                    atr_period, rsi_period, dmi_period,
                    base_fast_len, base_slow_len,
                    volatility_factor, momentum_factor,
                    slope_smooth, confidence_smooth,
                    confirm_bars, dynamic_thresholds,
                    threshold_window, hi_percentile, lo_percentile,
                    atr_multiplier, atr_z_threshold
                )
            
            # Convert numeric trend phase to text
            phase_map = {0: 'Neutral', 1: 'Impulse', 2: 'Cooling', 3: 'Reversal'}
            trend_phase = np.array([phase_map[p] for p in trend_phase_num], dtype=object)
            
        except Exception:
            # Fall back to non-Numba version if there's any issue
            raise NotImplementedError("Pure Python fallback not implemented in this version")
    else:
        raise NotImplementedError("Pure Python fallback not implemented in this version")
    
    # Create base result DataFrame
    result_df = pd.DataFrame({
        'NTI_FastEMA': fast_ema,
        'NTI_SlowEMA': slow_ema,
        'NTI_SlopePower': slope_power,
        'NTI_TrendPhase': trend_phase,
        'NTI_DirectionRaw': trend_direction_raw,
        'NTI_Direction': trend_direction_confirmed,
        'NTI_Confidence': confidence,
        'NTI_ReversalRisk': reversal_risk,
        'NTI_StallDetected': stall_detected,
        'NTI_SlopeForecast': slope_forecast
    }, index=df.index)
    
    # Add diagnostic columns if enabled
    if enable_diagnostics:
        result_df['NTI_HiThreshold'] = hi_thresh
        result_df['NTI_LoThreshold'] = lo_thresh
        result_df['NTI_ATR_Z'] = atr_z
        result_df['NTI_DirectionChanged'] = result_df['NTI_Direction'].diff() != 0
        
        # Track flip events
        flip_mask = result_df['NTI_DirectionChanged']
        result_df['NTI_FlipTimestamp'] = np.where(flip_mask, df.index, pd.NaT)
        result_df['NTI_FlipFromDir'] = np.where(flip_mask, result_df['NTI_Direction'].shift(1), np.nan)
        result_df['NTI_FlipToDir'] = np.where(flip_mask, result_df['NTI_Direction'], np.nan)
    
    return result_df


# Fast implementation helper functions (Numba-optimized)
if HAS_NUMBA:
    @nb.njit(nopython=True, cache=True)
    def apply_hysteresis(raw_dir: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Numba-accelerated hysteresis filter for direction changes.
        
        Args:
            raw_dir: Raw direction signals (+1/-1)
            k: Number of consecutive bars required for confirmation
        
        Returns:
            Confirmed direction signals
        """
        n = len(raw_dir)
        confirmed = np.empty(n, dtype=np.float64)
        confirmed[:] = np.nan
        
        # Track consecutive count
        count = 1
        current_dir = raw_dir[0]
        confirmed[0] = current_dir
        
        for i in range(1, n):
            if raw_dir[i] == current_dir:
                count += 1
            else:
                # Direction changed
                if count >= k:
                    # Previous direction was confirmed, start counting new direction
                    count = 1
                    current_dir = raw_dir[i]
                else:
                    # Not enough confirmation, maintain previous direction
                    confirmed[i] = confirmed[i-1] if i > 0 else current_dir
                    continue
            
            # Check if we have enough confirmation
            if count >= k:
                # Backfill the confirmed direction
                for j in range(max(0, i - k + 1), i + 1):
                    if np.isnan(confirmed[j]):
                        confirmed[j] = current_dir
            else:
                confirmed[i] = confirmed[i-1] if i > 0 else current_dir
        
        return confirmed

    @nb.njit(fastmath=True, cache=True)
    def calculate_dynamic_thresholds_rolling(abs_slope: np.ndarray, window: int = 500,
                                            hi_percentile: float = 0.8, lo_percentile: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dynamic thresholds based on rolling percentiles.
        Matches the original implementation exactly.
        """
        n = len(abs_slope)
        hi_thresh = np.empty(n)
        lo_thresh = np.empty(n)
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            window_data = abs_slope[start_idx:i+1]
            
            if len(window_data) >= 20:  # Minimum samples for percentile
                sorted_data = np.sort(window_data)
                hi_idx = int(hi_percentile * (len(sorted_data) - 1))
                lo_idx = int(lo_percentile * (len(sorted_data) - 1))
                hi_thresh[i] = sorted_data[hi_idx]
                lo_thresh[i] = sorted_data[lo_idx]
            else:
                # Use default thresholds for early bars
                hi_thresh[i] = 0.5
                lo_thresh[i] = 0.2
        
        return hi_thresh, lo_thresh

    @nb.njit(fastmath=True, cache=True)
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      period: int) -> np.ndarray:
        """Calculate Average True Range."""
        n = len(close)
        tr = np.empty(n)
        atr = np.empty(n)
        
        # First TR
        tr[0] = high[0] - low[0]
        atr[0] = tr[0]
        
        # Calculate TR
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # EMA smoothing
        alpha = 2.0 / (period + 1)
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return atr

    @nb.njit(fastmath=True, cache=True)
    def calculate_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using EMA smoothing to match original."""
        n = len(close)
        rsi = np.empty(n)
        
        # Calculate price changes
        gains = np.zeros(n)
        losses = np.zeros(n)
        
        for i in range(1, n):
            change = close[i] - close[i-1]
            if change > 0:
                gains[i] = change
            else:
                losses[i] = -change
        
        # EMA of gains and losses - matching original implementation
        alpha = 2.0 / (period + 1)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[0] = gains[0]
        avg_loss[0] = losses[0]
        
        for i in range(1, n):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i-1]
        
        # Calculate RSI
        for i in range(n):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
        return rsi

    @nb.njit(fastmath=True, cache=True)
    def calculate_dmi_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          period: int, atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate DMI and ADX using EMA smoothing to match original."""
        n = len(high)
        
        # Initialize arrays
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        # Calculate directional movements - match original logic exactly
        for i in range(1, n):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            
            # Match original: plus_dm when up > down AND up > 0
            if up > down and up > 0:
                plus_dm[i] = up
            else:
                plus_dm[i] = 0.0
                
            # Match original: minus_dm when down > up AND down > 0    
            if down > up and down > 0:
                minus_dm[i] = down
            else:
                minus_dm[i] = 0.0
        
        # No need to calculate TR since we use ATR passed in
        
        # Smooth using EMA to match original implementation
        alpha_dmi = 2.0 / (period + 1)
        smooth_plus_dm = np.zeros(n)
        smooth_minus_dm = np.zeros(n)
        smooth_plus_dm[0] = plus_dm[0]
        smooth_minus_dm[0] = minus_dm[0]
        
        for i in range(1, n):
            smooth_plus_dm[i] = alpha_dmi * plus_dm[i] + (1 - alpha_dmi) * smooth_plus_dm[i-1]
            smooth_minus_dm[i] = alpha_dmi * minus_dm[i] + (1 - alpha_dmi) * smooth_minus_dm[i-1]
        
        # Calculate DI+ and DI- using ATR (not smooth_tr)
        di_plus = np.zeros(n)
        di_minus = np.zeros(n)
        
        # Use the ATR that was calculated earlier
        for i in range(n):
            if atr[i] > 0:
                di_plus[i] = (smooth_plus_dm[i] / atr[i]) * 100
                di_minus[i] = (smooth_minus_dm[i] / atr[i]) * 100
        
        # Calculate DX and ADX using EMA smoothing
        di_diff = np.abs(di_plus - di_minus)
        di_sum = di_plus + di_minus
        dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0.0)
        
        # Smooth ADX using EMA
        adx = np.zeros(n)
        adx[0] = dx[0]
        for i in range(1, n):
            adx[i] = alpha_dmi * dx[i] + (1 - alpha_dmi) * adx[i-1]
        
        return di_plus, di_minus, adx

    @nb.njit(fastmath=True, cache=True)
    def incremental_stats(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate rolling mean and std matching original implementation."""
        n = len(values)
        mean = np.empty(n)
        std = np.empty(n)
        
        for i in range(n):
            window_start = max(0, i - window + 1)
            window_vals = values[window_start:i+1]
            
            if len(window_vals) >= 20:
                mean[i] = np.mean(window_vals)
                std[i] = np.std(window_vals)
            else:
                mean[i] = values[i]
                std[i] = 0.0
        
        return mean, std

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def nti_fast_kernel_phase2(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                               fast_ema: np.ndarray, slow_ema: np.ndarray,
                               atr: np.ndarray, rsi: np.ndarray,
                               di_plus: np.ndarray, di_minus: np.ndarray, adx: np.ndarray,
                               hi_thresh: np.ndarray, lo_thresh: np.ndarray,
                               confirm_bars: int) -> Tuple[np.ndarray, ...]:
        """
        Phase 2: Parallel calculation of slope, phase, confidence, etc.
        This runs after EMAs have been calculated sequentially.
        """
        n = len(close)
        
        # Pre-allocate arrays
        slope_power = np.empty(n)
        trend_phase = np.empty(n, dtype=np.int32)
        direction_raw = np.zeros(n, dtype=np.int32)
        confidence = np.empty(n)
        reversal_risk = np.zeros(n, dtype=np.bool_)
        stall_detected = np.zeros(n, dtype=np.bool_)
        slope_forecast = np.empty(n)
        
        # Parallel loop for independent calculations
        for i in nb.prange(n):
            # Slope power calculation
            if i >= 3:  # slope_smooth = 3
                price_range = np.max(high[max(0, i-20):i+1]) - np.min(low[max(0, i-20):i+1])
                if price_range > 0.0001:
                    fast_slope = (fast_ema[i] - fast_ema[i-3]) / 3.0
                    slow_slope = (slow_ema[i] - slow_ema[i-3]) / 3.0
                    fast_slope_norm = fast_slope / price_range
                    slow_slope_norm = slow_slope / price_range
                    slope_val = (fast_slope_norm - slow_slope_norm) * 100
                    
                    # Clamp to [-100, 100]
                    if slope_val < -100:
                        slope_power[i] = -100
                    elif slope_val > 100:
                        slope_power[i] = 100
                    else:
                        slope_power[i] = slope_val
                else:
                    slope_power[i] = 0
            else:
                # Initialize early values
                slope_power[i] = 0
                trend_phase[i] = 0  # Neutral
                direction_raw[i] = 1 if close[i] > close[0] else -1
                confidence[i] = 50.0
                reversal_risk[i] = False
                stall_detected[i] = False
                slope_forecast[i] = 0.0
                continue  # Skip the rest of the loop for early values
            
            # Raw direction
            direction_raw[i] = 1 if fast_ema[i] > slow_ema[i] else -1
            
            # Phase detection with direction alignment check
            abs_slope = abs(slope_power[i])
            if abs_slope > hi_thresh[i]:
                # Check for direction alignment for Impulse
                if (slope_power[i] > 0 and direction_raw[i] == 1) or \
                   (slope_power[i] < 0 and direction_raw[i] == -1):
                    trend_phase[i] = 1  # Impulse
                else:
                    trend_phase[i] = 3  # Reversal
            elif abs_slope > lo_thresh[i]:
                trend_phase[i] = 2  # Cooling
            else:
                trend_phase[i] = 0  # Neutral
            
            # Confidence calculation - match original implementation
            # Confidence score based on ADX and trend alignment
            trend_alignment = 1.0 if (di_plus[i] > di_minus[i] and direction_raw[i] == 1) or \
                                    (di_minus[i] > di_plus[i] and direction_raw[i] == -1) else 0.5
            confidence[i] = min(100.0, adx[i] * trend_alignment)
            
            # Risk flags
            if i >= 10:
                slope_variance = np.var(slope_power[max(0, i-10):i+1])
                reversal_risk[i] = (confidence[i] < 30 and slope_variance > 100) or trend_phase[i] == 3
                
                recent_slopes = slope_power[max(0, i-5):i+1]
                stall_detected[i] = np.std(recent_slopes) < 2.0 and abs(np.mean(recent_slopes)) < 5
            
            # Slope forecast
            if i >= 3:
                weights = np.array([0.5, 0.3, 0.2])
                recent = slope_power[i-2:i+1]
                slope_forecast[i] = np.sum(recent * weights)
            else:
                slope_forecast[i] = slope_power[i]
        
        return (slope_power, trend_phase, direction_raw, 
                confidence, reversal_risk, stall_detected, slope_forecast)


def neurotrend_intelligent_fast(df: pd.DataFrame,
                               base_fast_len: int = 10,
                               base_slow_len: int = 50,
                               atr_period: int = 14,
                               rsi_period: int = 14,
                               dmi_period: int = 14,
                               volatility_factor: float = 1.0,
                               momentum_factor: float = 0.5,
                               slope_smooth: int = 3,
                               confidence_smooth: int = 5,
                               confirm_bars: int = 3,
                               dynamic_thresholds: bool = True,
                               hi_percentile: float = 0.8,
                               lo_percentile: float = 0.6,
                               enable_diagnostics: bool = False,
                               use_numba: bool = True,
                               num_threads: Optional[int] = None) -> pd.DataFrame:
    """
    Fast implementation of NeuroTrend Intelligent using P² algorithm and parallel processing.
    
    This version achieves ~10x speedup over the original by:
    1. Using P² algorithm for O(1) rolling quantiles instead of O(n log n) sorting
    2. Separating sequential EMA calculation from parallel slope/phase calculations
    3. Incremental statistics calculation
    4. Optimized memory access patterns
    
    The outputs are identical to the original implementation up to IEEE floating-point rounding.
    
    All parameters and outputs are identical to the original neurotrend_intelligent function.
    """
    
    if not HAS_NUMBA or not use_numba:
        # Fall back to original implementation
        return neurotrend_intelligent(df, atr_period=atr_period, rsi_period=rsi_period, 
                                     dmi_period=dmi_period, base_fast_len=base_fast_len,
                                     base_slow_len=base_slow_len, volatility_factor=volatility_factor,
                                     momentum_factor=momentum_factor, slope_smooth=slope_smooth,
                                     confidence_smooth=confidence_smooth, confirm_bars=confirm_bars,
                                     dynamic_thresholds=dynamic_thresholds, hi_percentile=hi_percentile,
                                     lo_percentile=lo_percentile, enable_diagnostics=enable_diagnostics,
                                     use_numba=False)
    
    # Set thread count if specified
    if num_threads is not None:
        nb.set_num_threads(num_threads)
    
    # Extract numpy arrays
    high = df['High'].to_numpy(dtype=np.float64)
    low = df['Low'].to_numpy(dtype=np.float64) 
    close = df['Close'].to_numpy(dtype=np.float64)
    n = len(close)
    
    # Phase 1: Calculate indicators (vectorized)
    atr = calculate_atr(high, low, close, atr_period)
    rsi = calculate_rsi(close, rsi_period)
    di_plus, di_minus, adx = calculate_dmi_adx(high, low, close, dmi_period, atr)
    
    # Calculate ATR statistics
    atr_mean, atr_std = incremental_stats(atr, 100)
    
    # Phase 1.5: Sequential EMA calculation with adaptive periods
    fast_ema = np.empty(n)
    slow_ema = np.empty(n)
    
    for i in range(n):
        # ATR z-score calculation matching original
        if i >= 19:  # Need at least 20 values
            z_score = (atr[i] - atr_mean[i]) / atr_std[i] if atr_std[i] > 0 else 0.0
        else:
            z_score = 0.0
        
        # Volatility regime adaptation
        volatility_mult = 1.0
        if z_score > 1.0:  # atr_z_threshold default
            volatility_mult = 1.25  # atr_multiplier default
        elif z_score < -1.0:
            volatility_mult = 1.0 / 1.25
        
        # Adaptive EMA lengths based on market conditions
        # Normalize ATR and RSI
        atr_norm = atr[i] / close[i] if close[i] > 0 else 0.0
        rsi_norm = (rsi[i] - 50.0) / 50.0  # -1 to 1
        
        # Handle NaN values
        if np.isnan(atr_norm):
            atr_norm = 0.0
        if np.isnan(rsi_norm):
            rsi_norm = 0.0
        
        # Adjust EMA lengths with volatility regime consideration
        fast_len_base = max(5, min(20, base_fast_len + int(atr_norm * volatility_factor * 10)))
        slow_len_base = max(20, min(100, base_slow_len + int(abs(rsi_norm) * momentum_factor * 20)))
        
        fast_len = int(fast_len_base * volatility_mult)
        slow_len = int(slow_len_base * volatility_mult)
        
        # Calculate adaptive EMAs
        if i == 0:
            fast_ema[i] = close[i]
            slow_ema[i] = close[i]
        else:
            fast_alpha = 2.0 / (fast_len + 1)
            slow_alpha = 2.0 / (slow_len + 1)
            fast_ema[i] = fast_alpha * close[i] + (1 - fast_alpha) * fast_ema[i-1]
            slow_ema[i] = slow_alpha * close[i] + (1 - slow_alpha) * slow_ema[i-1]
    
    # Calculate initial slope for threshold calculation
    initial_slope = np.zeros(n)
    for i in range(3, n):  # slope_smooth = 3
        price_range = np.max(high[max(0, i-20):i+1]) - np.min(low[max(0, i-20):i+1])
        if price_range > 0.0001:
            fast_slope = (fast_ema[i] - fast_ema[i-3]) / 3.0
            slow_slope = (slow_ema[i] - slow_ema[i-3]) / 3.0
            initial_slope[i] = abs((fast_slope - slow_slope) / price_range * 100)
    
    # Calculate dynamic thresholds using rolling window (matching original)
    if dynamic_thresholds:
        hi_thresh, lo_thresh = calculate_dynamic_thresholds_rolling(initial_slope, 500, hi_percentile, lo_percentile)
    else:
        hi_thresh = np.full(n, 20.0)
        lo_thresh = np.full(n, 10.0)
    
    # Phase 2: Parallel calculation of derived metrics
    (slope_power, trend_phase, direction_raw,
     confidence, reversal_risk, stall_detected, slope_forecast) = nti_fast_kernel_phase2(
        high, low, close, fast_ema, slow_ema,
        atr, rsi, di_plus, di_minus, adx,
        hi_thresh, lo_thresh, confirm_bars
    )
    
    # Apply hysteresis to direction (must be done sequentially)
    direction = apply_hysteresis(direction_raw, confirm_bars)
    
    # Convert trend phase to strings
    phase_map = {0: 'Neutral', 1: 'Impulse', 2: 'Cooling', 3: 'Reversal'}
    trend_phase_str = pd.Series(trend_phase).map(phase_map)
    
    # Build output DataFrame
    output_df = pd.DataFrame(index=df.index)
    
    # Core outputs
    output_df['NTI_FastEMA'] = fast_ema
    output_df['NTI_SlowEMA'] = slow_ema
    output_df['NTI_SlopePower'] = slope_power
    output_df['NTI_TrendPhase'] = trend_phase_str.values
    output_df['NTI_DirectionRaw'] = direction_raw
    output_df['NTI_Direction'] = direction
    output_df['NTI_Confidence'] = confidence
    output_df['NTI_ReversalRisk'] = reversal_risk
    output_df['NTI_StallDetected'] = stall_detected
    output_df['NTI_SlopeForecast'] = slope_forecast
    
    # Diagnostic outputs
    if enable_diagnostics:
        output_df['NTI_HiThreshold'] = hi_thresh
        output_df['NTI_LoThreshold'] = lo_thresh
        
        # Calculate ATR z-scores for diagnostics
        atr_z = np.zeros(n)
        for i in range(n):
            if atr_std[i] > 0:
                atr_z[i] = (atr[i] - atr_mean[i]) / atr_std[i]
            else:
                atr_z[i] = 0.0
        output_df['NTI_ATR_Z'] = atr_z
        
        # Direction change tracking
        direction_changed = np.zeros(n, dtype=bool)
        flip_timestamps = []
        flip_from = []
        flip_to = []
        
        for i in range(1, n):
            if direction[i] != direction[i-1]:
                direction_changed[i] = True
                flip_timestamps.append(df.index[i])
                flip_from.append(direction[i-1])
                flip_to.append(direction[i])
        
        output_df['NTI_DirectionChanged'] = direction_changed
        
        if flip_timestamps:
            output_df['NTI_FlipTimestamp'] = pd.NaT
            output_df['NTI_FlipFromDir'] = np.nan
            output_df['NTI_FlipToDir'] = np.nan
            
            for i, ts in enumerate(flip_timestamps):
                idx = output_df.index.get_loc(ts)
                output_df.iloc[idx, output_df.columns.get_loc('NTI_FlipTimestamp')] = ts
                output_df.iloc[idx, output_df.columns.get_loc('NTI_FlipFromDir')] = flip_from[i]
                output_df.iloc[idx, output_df.columns.get_loc('NTI_FlipToDir')] = flip_to[i]
    
    return output_df


def neurotrend_3state(df, 
                     # Core NeuroTrend parameters
                     atr_period=14, rsi_period=14, dmi_period=14,
                     base_fast_len=10, base_slow_len=50,
                     volatility_factor=2.0, momentum_factor=0.5,
                     # Ranging detection parameters
                     slope_threshold=15.0, confidence_threshold=30.0,
                     adx_threshold=25.0, consolidation_bars=10,
                     range_atr_ratio=0.5, ranging_persistence=5,
                     # General parameters
                     use_numba=True):
    """
    3-State NeuroTrend indicator with ranging market detection.
    
    Combines NeuroTrend Intelligent with ranging detection for three states:
    - 1: Uptrend (Bullish)
    - 0: Ranging/Choppy (Neutral)
    - -1: Downtrend (Bearish)
    
    Args:
        df: DataFrame with OHLC data
        
        Core parameters (same as neurotrend_intelligent):
        - atr_period, rsi_period, dmi_period: Indicator periods
        - base_fast_len, base_slow_len: EMA lengths
        - volatility_factor, momentum_factor: Adaptation factors
        
        Ranging detection parameters:
        - slope_threshold: Max slope power for ranging (default: 15.0)
        - confidence_threshold: Max confidence for ranging (default: 30.0)
        - adx_threshold: Max ADX for ranging (default: 25.0)
        - consolidation_bars: Lookback for range calculation (default: 10)
        - range_atr_ratio: Price range/ATR threshold (default: 0.5)
        - ranging_persistence: Bars to confirm ranging (default: 5)
        
        use_numba: Use Numba acceleration if available
    
    Returns:
        DataFrame with columns:
        - NT3_Direction: 3-state direction (1, 0, -1)
        - NT3_State: State name ('Uptrend', 'Ranging', 'Downtrend')
        - NT3_IsRanging: Boolean ranging flag
        - NT3_Confidence: Adjusted confidence
        - NT3_Support/Resistance: Dynamic levels based on state
        - Plus all standard NeuroTrend Intelligent columns
    """
    # Step 1: Calculate NeuroTrend Intelligent
    nt_result = neurotrend_intelligent(
        df, 
        atr_period=atr_period, rsi_period=rsi_period, dmi_period=dmi_period,
        base_fast_len=base_fast_len, base_slow_len=base_slow_len,
        volatility_factor=volatility_factor, momentum_factor=momentum_factor,
        use_numba=use_numba
    )
    
    # Step 2: Merge with original data
    result_df = df.copy()
    for col in nt_result.columns:
        result_df[col] = nt_result[col]
    
    # Step 3: Add ranging detection
    result_df = add_ranging_detection(
        result_df,
        nti_columns={
            'slope_power': 'NTI_SlopePower',
            'confidence': 'NTI_Confidence',
            'direction': 'NTI_Direction',
            'trend_phase': 'NTI_TrendPhase',
            'stall': 'NTI_StallDetected'
        },
        slope_threshold=slope_threshold,
        confidence_threshold=confidence_threshold,
        adx_threshold=adx_threshold,
        consolidation_bars=consolidation_bars,
        range_atr_ratio=range_atr_ratio,
        ranging_persistence=ranging_persistence
    )
    
    # Step 4: Create simplified 3-state outputs
    result_df['NT3_Direction'] = result_df['NTI_Direction_3State']
    
    # State names
    state_map = {1: 'Uptrend', 0: 'Ranging', -1: 'Downtrend'}
    result_df['NT3_State'] = result_df['NT3_Direction'].map(state_map)
    
    # Adjusted confidence (reduced when ranging)
    result_df['NT3_Confidence'] = result_df['NTI_Confidence'].copy()
    result_df.loc[result_df['NTI_IsRanging'], 'NT3_Confidence'] *= 0.5
    
    # Copy ranging flag
    result_df['NT3_IsRanging'] = result_df['NTI_IsRanging']
    
    # Dynamic support/resistance based on state
    result_df['NT3_Support'] = np.nan
    result_df['NT3_Resistance'] = np.nan
    
    # In uptrend: support = recent lows
    uptrend_mask = result_df['NT3_Direction'] == 1
    if uptrend_mask.any():
        result_df.loc[uptrend_mask, 'NT3_Support'] = (
            result_df.loc[uptrend_mask, 'Low'].rolling(10, min_periods=1).min()
        )
    
    # In downtrend: resistance = recent highs
    downtrend_mask = result_df['NT3_Direction'] == -1
    if downtrend_mask.any():
        result_df.loc[downtrend_mask, 'NT3_Resistance'] = (
            result_df.loc[downtrend_mask, 'High'].rolling(10, min_periods=1).max()
        )
    
    # In ranging: both support and resistance
    ranging_mask = result_df['NT3_Direction'] == 0
    if ranging_mask.any():
        result_df.loc[ranging_mask, 'NT3_Support'] = (
            result_df.loc[ranging_mask, 'Low'].rolling(20, min_periods=1).min()
        )
        result_df.loc[ranging_mask, 'NT3_Resistance'] = (
            result_df.loc[ranging_mask, 'High'].rolling(20, min_periods=1).max()
        )
    
    return result_df


# Ranging detection functions (moved from neurotrend_enhanced.py)
@jit(nopython=True, cache=True)
def _detect_ranging_market_numba(
    slope_power: np.ndarray,
    confidence: np.ndarray,
    atr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    adx: np.ndarray,
    trend_phase_num: np.ndarray,
    stall_detected: np.ndarray,
    slope_threshold: float = 15.0,
    confidence_threshold: float = 30.0,
    adx_threshold: float = 25.0,
    consolidation_bars: int = 10,
    range_atr_ratio: float = 0.5
) -> np.ndarray:
    """
    Numba-optimized ranging market detection.
    
    Returns:
    - is_ranging: Boolean array indicating ranging market conditions
    """
    n = len(slope_power)
    is_ranging = np.zeros(n, dtype=np.bool_)
    
    for i in range(consolidation_bars, n):
        # Calculate price range over consolidation period
        price_range = np.max(high[i-consolidation_bars:i+1]) - np.min(low[i-consolidation_bars:i+1])
        range_to_atr = price_range / atr[i] if atr[i] > 0 else 0
        
        # Check multiple ranging conditions
        low_slope = abs(slope_power[i]) < slope_threshold
        low_confidence = confidence[i] < confidence_threshold
        low_adx = adx[i] < adx_threshold if not np.isnan(adx[i]) else False
        tight_range = range_to_atr < range_atr_ratio
        is_stalled = stall_detected[i]
        is_neutral = trend_phase_num[i] == 0
        
        # Ranging if any strong indicator or multiple weak indicators
        strong_ranging = low_adx or tight_range or is_stalled
        weak_ranging = (low_slope and low_confidence) or (low_slope and is_neutral)
        
        is_ranging[i] = strong_ranging or weak_ranging
    
    return is_ranging


@jit(nopython=True, cache=True)
def _smooth_direction_transitions_numba(
    direction: np.ndarray,
    is_ranging: np.ndarray,
    smooth_window: int = 3,
    ranging_persistence: int = 5
) -> np.ndarray:
    """
    Smooth transitions between trending and ranging states.
    Prevents rapid switching.
    """
    n = len(direction)
    smoothed = direction.copy()
    ranging_count = np.zeros(n, dtype=np.int32)
    
    # Count consecutive ranging signals
    for i in range(1, n):
        if is_ranging[i]:
            ranging_count[i] = ranging_count[i-1] + 1
        else:
            ranging_count[i] = 0
    
    # Apply ranging state with persistence
    for i in range(ranging_persistence, n):
        # Only switch to ranging after persistent signals
        if ranging_count[i] >= ranging_persistence:
            smoothed[i] = 0
        # Keep ranging until strong trend emerges
        elif smoothed[i-1] == 0 and is_ranging[i]:
            smoothed[i] = 0
    
    return smoothed


def add_ranging_detection(
    df: pd.DataFrame,
    nti_columns: dict = None,
    slope_threshold: float = 15.0,
    confidence_threshold: float = 30.0,
    adx_threshold: float = 25.0,
    consolidation_bars: int = 10,
    range_atr_ratio: float = 0.5,
    ranging_persistence: int = 5,
    calculate_adx: bool = True,
    adx_period: int = 14
) -> pd.DataFrame:
    """
    Add ranging/choppy market detection to existing NeuroTrend Intelligent data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with NTI indicators already calculated
    nti_columns : dict, optional
        Mapping of NTI column names if different from defaults
    slope_threshold : float
        Maximum absolute slope power for ranging (default: 15.0)
    confidence_threshold : float
        Maximum confidence for ranging (default: 30.0)
    adx_threshold : float
        Maximum ADX for ranging market (default: 25.0)
    consolidation_bars : int
        Lookback period for price range calculation (default: 10)
    range_atr_ratio : float
        Price range to ATR ratio threshold (default: 0.5)
    ranging_persistence : int
        Bars required to confirm ranging state (default: 5)
    calculate_adx : bool
        Whether to calculate ADX if not present (default: True)
    adx_period : int
        Period for ADX calculation (default: 14)
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - NTI_IsRanging: Boolean ranging market flag
        - NTI_Direction_3State: Direction with ranging (1, 0, -1)
        - NTI_RangingMetrics: Dict with detailed ranging metrics
    """
    result_df = df.copy()
    
    # Default column mapping
    default_columns = {
        'slope_power': 'NTI_SlopePower',
        'confidence': 'NTI_Confidence',
        'direction': 'NTI_Direction',
        'trend_phase': 'NTI_TrendPhase',
        'stall': 'NTI_StallDetected',
        'atr': 'ATR'
    }
    
    columns = default_columns.copy()
    if nti_columns:
        columns.update(nti_columns)
    
    # Ensure required columns exist
    for key, col in columns.items():
        if col not in result_df.columns and key != 'atr':
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Calculate ATR if not present
    if columns['atr'] not in result_df.columns:
        # Simple ATR calculation
        high = result_df['High'].values
        low = result_df['Low'].values
        close = result_df['Close'].values
        
        hl = high - low
        hc = np.abs(high - np.roll(close, 1))
        lc = np.abs(low - np.roll(close, 1))
        tr = np.maximum(hl, np.maximum(hc, lc))
        tr[0] = hl[0]
        
        result_df[columns['atr']] = pd.Series(tr).rolling(14).mean().bfill().values
    
    # Calculate ADX if requested and not present
    if calculate_adx and 'ADX' not in result_df.columns:
        # Simple ADX calculation
        high = result_df['High'].values
        low = result_df['Low'].values
        close = result_df['Close'].values
        
        # True Range
        hl = high - low
        hc = np.abs(high - np.roll(close, 1))
        lc = np.abs(low - np.roll(close, 1))
        tr = np.maximum(hl, np.maximum(hc, lc))
        tr[0] = hl[0]
        
        # Directional Movement
        up = high - np.roll(high, 1)
        down = np.roll(low, 1) - low
        up[0] = 0
        down[0] = 0
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smoothed indicators
        atr = pd.Series(tr).rolling(adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(adx_period).mean() / atr
        
        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(adx_period).mean()
        
        result_df['ADX'] = adx.fillna(0).values
        result_df['PLUS_DI'] = plus_di.fillna(0).values
        result_df['MINUS_DI'] = minus_di.fillna(0).values
    
    # Extract arrays
    slope_power = result_df[columns['slope_power']].values
    confidence = result_df[columns['confidence']].values
    atr = result_df[columns['atr']].values
    high = result_df['High'].values
    low = result_df['Low'].values
    stall_detected = result_df[columns['stall']].astype(bool).values
    
    # Convert trend phase to numeric if needed
    trend_phase = result_df[columns['trend_phase']]
    if trend_phase.dtype == 'object':
        phase_map = {'Neutral': 0, 'Impulse': 1, 'Cooling': 2, 'Reversal': 3}
        trend_phase_num = trend_phase.map(phase_map).fillna(0).values
    else:
        trend_phase_num = trend_phase.values
    
    # Use ADX if available, otherwise use NaN array
    if 'ADX' in result_df.columns:
        adx = result_df['ADX'].values
    else:
        adx = np.full(len(df), np.nan)
    
    # Detect ranging markets
    use_numba = NUMBA_AVAILABLE
    if use_numba:
        is_ranging = _detect_ranging_market_numba(
            slope_power, confidence, atr, high, low, adx,
            trend_phase_num, stall_detected,
            slope_threshold, confidence_threshold, adx_threshold,
            consolidation_bars, range_atr_ratio
        )
    else:
        # Pure Python fallback
        is_ranging = np.zeros(len(df), dtype=bool)
        for i in range(consolidation_bars, len(df)):
            price_range = high[i-consolidation_bars:i+1].max() - low[i-consolidation_bars:i+1].min()
            range_to_atr = price_range / atr[i] if atr[i] > 0 else 0
            
            is_ranging[i] = (
                (abs(slope_power[i]) < slope_threshold) or
                (confidence[i] < confidence_threshold) or
                (not np.isnan(adx[i]) and adx[i] < adx_threshold) or
                (range_to_atr < range_atr_ratio) or
                stall_detected[i] or
                (trend_phase_num[i] == 0)
            )
    
    # Get original direction
    direction = result_df[columns['direction']].values
    
    # Apply smoothing to prevent rapid transitions
    if use_numba:
        direction_3state = _smooth_direction_transitions_numba(
            direction, is_ranging, 3, ranging_persistence
        )
    else:
        # Simple Python smoothing
        direction_3state = direction.copy()
        ranging_count = 0
        for i in range(len(direction)):
            if is_ranging[i]:
                ranging_count += 1
                if ranging_count >= ranging_persistence:
                    direction_3state[i] = 0
            else:
                if ranging_count < ranging_persistence:
                    ranging_count = 0
    
    # Calculate additional ranging metrics
    result_df['NTI_IsRanging'] = is_ranging
    result_df['NTI_Direction_3State'] = direction_3state
    
    # Add detailed metrics for analysis
    ranging_metrics = []
    for i in range(len(df)):
        if i >= consolidation_bars:
            price_range = high[max(0, i-consolidation_bars):i+1].max() - low[max(0, i-consolidation_bars):i+1].min()
            range_to_atr = price_range / atr[i] if atr[i] > 0 else 0
            
            metrics = {
                'slope_abs': abs(slope_power[i]),
                'confidence': confidence[i],
                'adx': adx[i] if not np.isnan(adx[i]) else None,
                'range_atr_ratio': range_to_atr,
                'is_stalled': bool(stall_detected[i]),
                'trend_phase': trend_phase.iloc[i]
            }
        else:
            metrics = {}
        
        ranging_metrics.append(metrics)
    
    result_df['NTI_RangingMetrics'] = ranging_metrics
    
    # Add summary statistics
    total_bars = len(df)
    ranging_bars = is_ranging.sum()
    ranging_pct = (ranging_bars / total_bars * 100) if total_bars > 0 else 0
    
    # Count transitions
    transitions = np.diff(direction_3state)
    trend_to_range = ((transitions == -1) | (transitions == 1)).sum()
    range_to_trend = ((direction_3state[:-1] == 0) & (direction_3state[1:] != 0)).sum()
    
    print(f"\n🎯 Ranging Market Detection Results:")
    print(f"   📊 Total bars analyzed: {total_bars}")
    print(f"   🔍 Ranging market bars: {ranging_bars} ({ranging_pct:.1f}%)")
    print(f"   ↔️  Trending market bars: {total_bars - ranging_bars} ({100-ranging_pct:.1f}%)")
    print(f"   🔄 Trend→Range transitions: {trend_to_range}")
    print(f"   📈 Range→Trend transitions: {range_to_trend}")
    
    return result_df


@jit(nopython=True, cache=True)
def _andean_oscillator_numba(close, open_prices, length, sig_length):
    """
    Numba-optimized Andean Oscillator calculation.
    
    Based on exponential envelopes algorithm by alexgrover.
    """
    n = len(close)
    alpha = 2.0 / (length + 1)
    
    # Initialize arrays
    up1 = np.zeros(n)
    up2 = np.zeros(n)
    dn1 = np.zeros(n)
    dn2 = np.zeros(n)
    bull = np.zeros(n)
    bear = np.zeros(n)
    
    # Initial values
    up1[0] = max(close[0], open_prices[0])
    up2[0] = max(close[0] * close[0], open_prices[0] * open_prices[0])
    dn1[0] = min(close[0], open_prices[0])
    dn2[0] = min(close[0] * close[0], open_prices[0] * open_prices[0])
    
    # Calculate exponential envelopes
    for i in range(1, n):
        C = close[i]
        O = open_prices[i]
        C2 = C * C
        O2 = O * O
        
        # Upper envelope
        up1[i] = max(C, O, up1[i-1] - (up1[i-1] - C) * alpha)
        up2[i] = max(C2, O2, up2[i-1] - (up2[i-1] - C2) * alpha)
        
        # Lower envelope
        dn1[i] = min(C, O, dn1[i-1] + (C - dn1[i-1]) * alpha)
        dn2[i] = min(C2, O2, dn2[i-1] + (C2 - dn2[i-1]) * alpha)
        
        # Components
        bull_val = dn2[i] - dn1[i] * dn1[i]
        bear_val = up2[i] - up1[i] * up1[i]
        
        # Ensure non-negative values for sqrt
        bull[i] = np.sqrt(max(0, bull_val))
        bear[i] = np.sqrt(max(0, bear_val))
    
    # Calculate signal line (EMA of max(bull, bear))
    signal = np.zeros(n)
    signal_alpha = 2.0 / (sig_length + 1)
    
    # Initialize signal
    signal[0] = max(bull[0], bear[0])
    
    # Calculate signal EMA
    for i in range(1, n):
        max_val = max(bull[i], bear[i])
        signal[i] = signal_alpha * max_val + (1 - signal_alpha) * signal[i-1]
    
    return bull, bear, signal, up1, up2, dn1, dn2


def add_andean_oscillator(df, length=250, sig_length=25, inplace=False):
    """
    Add Andean Oscillator to the DataFrame.
    
    The Andean Oscillator uses exponential envelopes to identify trend components.
    It calculates bullish and bearish components based on price variance within
    adaptive envelopes.
    
    Args:
        df: DataFrame with OHLC data
        length: Period for exponential envelopes (default: 250)
        sig_length: Signal line EMA period (default: 25)
        inplace: Modify DataFrame in-place (default: False)
    
    Returns:
        DataFrame with Andean Oscillator columns:
        - AO_Bull: Bullish component
        - AO_Bear: Bearish component
        - AO_Signal: Signal line (EMA of max(bull, bear))
        - AO_BullTrend: Bullish trend start markers (A++)
        - AO_BearTrend: Bearish trend start markers (A--)
        - AO_BullTrendEnd: Bullish trend end markers
        - AO_BearTrendEnd: Bearish trend end markers
        - AO_Dominance: Current dominance (1=Bull, -1=Bear, 0=None)
    """
    result_df = df if inplace else df.copy()
    
    # Extract price data
    close = result_df['Close'].values
    open_prices = result_df['Open'].values
    
    # Calculate Andean Oscillator
    if NUMBA_AVAILABLE:
        bull, bear, signal, up1, up2, dn1, dn2 = _andean_oscillator_numba(
            close, open_prices, length, sig_length
        )
    else:
        # Pure Python implementation
        n = len(close)
        alpha = 2.0 / (length + 1)
        
        up1 = np.zeros(n)
        up2 = np.zeros(n)
        dn1 = np.zeros(n)
        dn2 = np.zeros(n)
        bull = np.zeros(n)
        bear = np.zeros(n)
        
        # Initial values
        up1[0] = max(close[0], open_prices[0])
        up2[0] = max(close[0] ** 2, open_prices[0] ** 2)
        dn1[0] = min(close[0], open_prices[0])
        dn2[0] = min(close[0] ** 2, open_prices[0] ** 2)
        
        # Calculate exponential envelopes
        for i in range(1, n):
            C = close[i]
            O = open_prices[i]
            C2 = C * C
            O2 = O * O
            
            # Upper envelope
            up1[i] = max(C, O, up1[i-1] - (up1[i-1] - C) * alpha)
            up2[i] = max(C2, O2, up2[i-1] - (up2[i-1] - C2) * alpha)
            
            # Lower envelope
            dn1[i] = min(C, O, dn1[i-1] + (C - dn1[i-1]) * alpha)
            dn2[i] = min(C2, O2, dn2[i-1] + (C2 - dn2[i-1]) * alpha)
            
            # Components
            bull_val = dn2[i] - dn1[i] * dn1[i]
            bear_val = up2[i] - up1[i] * up1[i]
            
            # Ensure non-negative values for sqrt
            bull[i] = np.sqrt(max(0, bull_val))
            bear[i] = np.sqrt(max(0, bear_val))
        
        # Calculate signal line
        signal = pd.Series(np.maximum(bull, bear)).ewm(span=sig_length, adjust=False).mean().values
    
    # Add to DataFrame
    result_df['AO_Bull'] = bull
    result_df['AO_Bear'] = bear
    result_df['AO_Signal'] = signal
    
    # Implement Pine Script dominance logic
    # bullDominance = bullVal > bearVal and bullVal > signalVal
    # bearDominance = bearVal > bullVal and bearVal > signalVal
    bull_dominance = (bull > bear) & (bull > signal)
    bear_dominance = (bear > bull) & (bear > signal)
    
    # Current dominance: 1 for bull, -1 for bear, 0 for none
    current_dominance = np.zeros(len(bull), dtype=int)
    current_dominance[bull_dominance] = 1
    current_dominance[bear_dominance] = -1
    
    # Detect dominance changes (trend starts and ends)
    # Initialize arrays
    bull_trend_start = np.zeros(len(bull), dtype=bool)
    bear_trend_start = np.zeros(len(bull), dtype=bool)
    bull_trend_end = np.zeros(len(bull), dtype=bool)
    bear_trend_end = np.zeros(len(bull), dtype=bool)
    
    # Track last dominance
    last_dominance = 0
    for i in range(len(bull)):
        curr_dom = current_dominance[i]
        
        # Check for dominance change
        if curr_dom != last_dominance and i > 0:
            if curr_dom == 1:  # Bull dominance starts
                bull_trend_start[i] = True
            elif curr_dom == -1:  # Bear dominance starts
                bear_trend_start[i] = True
            elif curr_dom == 0:  # No dominance (signal > both)
                if last_dominance == 1:  # Bull dominance ends
                    bull_trend_end[i] = True
                elif last_dominance == -1:  # Bear dominance ends
                    bear_trend_end[i] = True
        
        last_dominance = curr_dom
    
    # Mark first value as False to avoid edge effects
    bull_trend_start[0] = False
    bear_trend_start[0] = False
    bull_trend_end[0] = False
    bear_trend_end[0] = False
    
    result_df['AO_BullTrend'] = bull_trend_start
    result_df['AO_BearTrend'] = bear_trend_start
    result_df['AO_BullTrendEnd'] = bull_trend_end
    result_df['AO_BearTrendEnd'] = bear_trend_end
    result_df['AO_Dominance'] = current_dominance  # 1=Bull, -1=Bear, 0=None
    
    # Add envelope values for debugging/analysis (optional)
    result_df['AO_Up1'] = up1
    result_df['AO_Up2'] = up2
    result_df['AO_Dn1'] = dn1
    result_df['AO_Dn2'] = dn2
    
    return result_df


# ============================================================================
# Intelligent Chop Indicator - Advanced Market Regime Detection
# ============================================================================

# Constants for market regimes
REGIME_STRONG_TREND = "Strong Trend"
REGIME_WEAK_TREND = "Weak Trend"
REGIME_QUIET_RANGE = "Quiet Range"
REGIME_VOLATILE_CHOP = "Volatile Chop"
REGIME_TRANSITIONAL = "Transitional"


@jit(nopython=True, cache=True)
def _ic_calculate_true_range(high, low, close):
    """Calculate True Range values for Intelligent Chop."""
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    
    # First value
    tr[0] = high[0] - low[0]
    
    # Subsequent values
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    
    return tr


@jit(nopython=True, cache=True)
def _ic_calculate_atr(tr, period):
    """Calculate Average True Range using EMA for Intelligent Chop."""
    n = len(tr)
    atr = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    
    # Initialize with SMA
    atr[0] = tr[0]
    if period <= n:
        atr[period-1] = np.mean(tr[:period])
        
    # EMA calculation
    for i in range(1, n):
        if i < period:
            atr[i] = (atr[i-1] * i + tr[i]) / (i + 1)
        else:
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    return atr


@jit(nopython=True, cache=True)
def _ic_calculate_dmi_adx(high, low, close, period):
    """Calculate DMI and ADX values for Intelligent Chop."""
    n = len(high)
    
    # Initialize arrays
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    # Calculate directional movements
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    # Calculate True Range
    tr = _ic_calculate_true_range(high, low, close)
    
    # Smooth using Wilder's method (EMA with alpha = 1/period)
    alpha = 1.0 / period
    smooth_plus_dm = np.zeros(n, dtype=np.float64)
    smooth_minus_dm = np.zeros(n, dtype=np.float64)
    smooth_tr = np.zeros(n, dtype=np.float64)
    
    # Initialize
    smooth_plus_dm[period-1] = np.sum(plus_dm[:period])
    smooth_minus_dm[period-1] = np.sum(minus_dm[:period])
    smooth_tr[period-1] = np.sum(tr[:period])
    
    # Smooth values
    for i in range(period, n):
        smooth_plus_dm[i] = smooth_plus_dm[i-1] - smooth_plus_dm[i-1]/period + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i-1] - smooth_minus_dm[i-1]/period + minus_dm[i]
        smooth_tr[i] = smooth_tr[i-1] - smooth_tr[i-1]/period + tr[i]
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    
    for i in range(period-1, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100 * smooth_plus_dm[i] / smooth_tr[i]
            di_minus[i] = 100 * smooth_minus_dm[i] / smooth_tr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    
    for i in range(period-1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Initial ADX
    if 2*period-1 < n:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        
        # Smooth ADX
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return di_plus, di_minus, adx, tr


@jit(nopython=True, cache=True)
def _ic_calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands for Intelligent Chop."""
    n = len(close)
    middle = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    for i in range(period-1, n):
        # Calculate SMA
        middle[i] = np.mean(close[i-period+1:i+1])
        
        # Calculate standard deviation
        std = np.std(close[i-period+1:i+1])
        
        # Calculate bands
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
        
        # Calculate bandwidth (normalized)
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bandwidth


@jit(nopython=True, cache=True)
def _ic_calculate_choppiness_index(high, low, close, period):
    """Calculate Choppiness Index for Intelligent Chop."""
    n = len(high)
    ci = np.full(n, 50.0, dtype=np.float64)  # Default neutral value
    
    # Calculate True Range
    tr = _ic_calculate_true_range(high, low, close)
    
    for i in range(period, n):
        # Sum of True Ranges
        sum_tr = np.sum(tr[i-period+1:i+1])
        
        # High-Low range over period
        period_high = np.max(high[i-period+1:i+1])
        period_low = np.min(low[i-period+1:i+1])
        period_range = period_high - period_low
        
        if period_range > 0:
            # Choppiness Index formula
            ci[i] = 100 * np.log10(sum_tr / period_range) / np.log10(period)
    
    return ci


@jit(nopython=True, cache=True)
def _ic_calculate_efficiency_ratio(close, period):
    """Calculate Kaufman's Efficiency Ratio for Intelligent Chop."""
    n = len(close)
    er = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        # Net change over period
        net_change = abs(close[i] - close[i-period])
        
        # Sum of absolute changes
        sum_changes = 0.0
        for j in range(i-period+1, i+1):
            sum_changes += abs(close[j] - close[j-1])
        
        if sum_changes > 0:
            er[i] = net_change / sum_changes
    
    return er


@jit(nopython=True, cache=True)
def _ic_calculate_hurst_exponent(series, min_lag=2, max_lag=100):
    """
    Calculate Hurst Exponent using R/S analysis for Intelligent Chop.
    
    H > 0.5: Trending (persistent)
    H < 0.5: Mean-reverting (anti-persistent)
    H ≈ 0.5: Random walk
    """
    n = len(series)
    if n < max_lag:
        return 0.5
    
    # Use log returns for more stable calculation
    log_returns = np.log(series[1:] / series[:-1])
    
    # Pre-allocate arrays for results
    max_lags = min(max_lag, n//2) - min_lag
    lags = np.zeros(max_lags, dtype=np.int64)
    rs_values = np.zeros(max_lags, dtype=np.float64)
    valid_count = 0
    
    for lag in range(min_lag, min(max_lag, n//2)):
        # R/S calculation for this lag
        rs_sum = 0.0
        rs_count = 0
        
        for start in range(0, len(log_returns) - lag, lag):
            subset = log_returns[start:start+lag]
            if len(subset) < lag:
                continue
                
            # Mean and cumulative deviations
            mean = np.mean(subset)
            deviations = subset - mean
            cumsum = np.cumsum(deviations)
            
            # Range and standard deviation
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(subset)
            
            if S > 0:
                rs_sum += R / S
                rs_count += 1
        
        if rs_count > 0:
            lags[valid_count] = lag
            rs_values[valid_count] = rs_sum / rs_count
            valid_count += 1
    
    if valid_count < 2:
        return 0.5
    
    # Trim arrays to valid size
    lags = lags[:valid_count]
    rs_values = rs_values[:valid_count]
    
    # Linear regression on log-log plot
    log_lags = np.log(lags.astype(np.float64))
    log_rs = np.log(rs_values)
    
    # Calculate slope (Hurst exponent)
    n_points = valid_count
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xx = np.sum(log_lags * log_lags)
    sum_xy = np.sum(log_lags * log_rs)
    
    denominator = n_points * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return 0.5
    
    H = (n_points * sum_xy - sum_x * sum_y) / denominator
    
    # Clamp to valid range
    return max(0.0, min(1.0, H))


@jit(nopython=True, cache=True, parallel=True)
def _intelligent_chop_core(high, low, close, 
                          adx_period=14, bb_period=20, bb_std=2.0,
                          atr_period=14, chop_period=14, er_period=10,
                          hurst_lag_min=2, hurst_lag_max=100,
                          # Thresholds
                          adx_trend_threshold=25.0,
                          adx_strong_threshold=35.0,
                          adx_choppy_threshold=20.0,
                          chop_threshold=61.8,
                          chop_low_threshold=38.2,
                          bb_squeeze_threshold=0.02,
                          atr_low_percentile=0.25,
                          atr_high_percentile=0.75,
                          er_high_threshold=0.3,
                          er_low_threshold=0.1,
                          hurst_trend_threshold=0.55,
                          hurst_range_threshold=0.45):
    """
    Core Numba-optimized calculation for Intelligent Chop Indicator.
    """
    n = len(close)
    
    # Calculate all indicators
    di_plus, di_minus, adx, tr = _ic_calculate_dmi_adx(high, low, close, adx_period)
    upper_bb, middle_bb, lower_bb, bandwidth = _ic_calculate_bollinger_bands(close, bb_period, bb_std)
    atr = _ic_calculate_atr(tr, atr_period)
    chop = _ic_calculate_choppiness_index(high, low, close, chop_period)
    er = _ic_calculate_efficiency_ratio(close, er_period)
    
    # Normalize ATR
    atr_norm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if close[i] > 0:
            atr_norm[i] = atr[i] / close[i]
    
    # Calculate ATR percentiles over rolling window
    atr_low_level = np.zeros(n, dtype=np.float64)
    atr_high_level = np.zeros(n, dtype=np.float64)
    
    window_size = 100
    for i in range(window_size, n):
        window_atr = atr_norm[i-window_size:i]
        sorted_atr = np.sort(window_atr)
        low_idx = int(atr_low_percentile * (window_size - 1))
        high_idx = int(atr_high_percentile * (window_size - 1))
        atr_low_level[i] = sorted_atr[low_idx]
        atr_high_level[i] = sorted_atr[high_idx]
    
    # Initialize regime arrays
    regime = np.empty(n, dtype=np.int32)  # 0=Chop, 1=Weak Trend, 2=Strong Trend, 3=Range
    confidence = np.zeros(n, dtype=np.float64)
    
    # Process each bar
    for i in prange(n):
        if i < max(adx_period * 2, bb_period, chop_period, er_period):
            regime[i] = 0  # Default to choppy for insufficient data
            confidence[i] = 0.0
            continue
        
        # Primary filter: ADX for trend strength
        is_trending = adx[i] > adx_trend_threshold
        is_strong_trend = adx[i] > adx_strong_threshold
        is_non_trending = adx[i] < adx_choppy_threshold
        
        # Secondary filters
        is_choppy_ci = chop[i] > chop_threshold
        is_trending_ci = chop[i] < chop_low_threshold
        is_low_volatility = i >= window_size and atr_norm[i] < atr_low_level[i]
        is_high_volatility = i >= window_size and atr_norm[i] > atr_high_level[i]
        is_bb_squeeze = bandwidth[i] < bb_squeeze_threshold
        is_efficient = er[i] > er_high_threshold
        is_inefficient = er[i] < er_low_threshold
        
        # Hurst calculation for recent window (expensive, so only when needed)
        hurst = 0.5  # Default neutral
        if i >= hurst_lag_max:
            hurst = _ic_calculate_hurst_exponent(close[i-hurst_lag_max:i], hurst_lag_min, hurst_lag_max) # updated to remove look ahead t+1
        
        is_persistent = hurst > hurst_trend_threshold
        is_mean_reverting = hurst < hurst_range_threshold
        
        # Regime determination logic
        trend_score = 0.0
        chop_score = 0.0
        range_score = 0.0
        
        # Trend indicators
        if is_trending:
            trend_score += 2.0
        if is_strong_trend:
            trend_score += 1.0
        if is_trending_ci:
            trend_score += 1.0
        if is_efficient:
            trend_score += 1.0
        if is_persistent:
            trend_score += 1.0
        if di_plus[i] > di_minus[i] * 1.5 or di_minus[i] > di_plus[i] * 1.5:
            trend_score += 1.0
        
        # Chop indicators
        if is_non_trending:
            chop_score += 2.0
        if is_choppy_ci:
            chop_score += 1.5
        if is_high_volatility and is_inefficient:
            chop_score += 2.0
        if is_inefficient:
            chop_score += 1.0
        if abs(hurst - 0.5) < 0.05:  # Random walk
            chop_score += 1.0
        if bandwidth[i] > 0.10 and is_inefficient:  # Wide bands + inefficient
            chop_score += 1.5
        
        # Range indicators
        if is_non_trending:
            range_score += 1.0
        if is_low_volatility:
            range_score += 2.0
        if is_bb_squeeze:
            range_score += 2.0
        if is_mean_reverting:
            range_score += 1.5
        if bandwidth[i] < 0.05:  # Tight bands
            range_score += 1.0
        
        # Determine regime
        max_score = max(trend_score, chop_score, range_score)
        
        if max_score == trend_score and trend_score > 3.0:
            if is_strong_trend or trend_score > 5.0:
                regime[i] = 2  # Strong Trend
            else:
                regime[i] = 1  # Weak Trend
        elif max_score == range_score and range_score > 3.0:
            regime[i] = 3  # Quiet Range
        else:
            regime[i] = 0  # Volatile Chop (default for ambiguous)
        
        # Calculate confidence
        total_score = trend_score + chop_score + range_score
        if total_score > 0:
            if regime[i] in [1, 2]:  # Trending
                confidence[i] = trend_score / total_score * 100
            elif regime[i] == 3:  # Ranging
                confidence[i] = range_score / total_score * 100
            else:  # Choppy
                confidence[i] = chop_score / total_score * 100
        
        # Boost confidence for clear signals
        if regime[i] == 2 and is_strong_trend and is_efficient:
            confidence[i] = min(100.0, confidence[i] * 1.2)
        elif regime[i] == 3 and is_bb_squeeze and is_mean_reverting:
            confidence[i] = min(100.0, confidence[i] * 1.2)
        elif regime[i] == 0 and is_high_volatility and is_choppy_ci:
            confidence[i] = min(100.0, confidence[i] * 1.2)
    
    return (regime, confidence, adx, chop, bandwidth, atr_norm, er, 
            di_plus, di_minus, upper_bb, middle_bb, lower_bb)


def intelligent_chop(df, 
                    # Indicator periods
                    adx_period=14, bb_period=20, bb_std=2.0,
                    atr_period=14, chop_period=14, er_period=10,
                    hurst_lag_min=2, hurst_lag_max=100,
                    # Thresholds
                    adx_trend_threshold=25.0,
                    adx_strong_threshold=35.0,
                    adx_choppy_threshold=20.0,
                    chop_threshold=61.8,
                    chop_low_threshold=38.2,
                    bb_squeeze_threshold=0.02,
                    atr_low_percentile=0.25,
                    atr_high_percentile=0.75,
                    er_high_threshold=0.3,
                    er_low_threshold=0.1,
                    hurst_trend_threshold=0.55,
                    hurst_range_threshold=0.45,
                    # Control
                    inplace=False):
    """
    Intelligent Chop Indicator - Advanced Market Regime Detection
    
    This indicator synthesizes multiple technical indicators to identify market regimes:
    - Strong Trend: Clear directional movement with high momentum
    - Weak Trend: Some directional bias but less conviction
    - Quiet Range: Low volatility consolidation, mean-reverting behavior
    - Volatile Chop: High volatility without clear direction, dangerous conditions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    
    Indicator Periods:
    adx_period : int, default=14
        Period for ADX calculation
    bb_period : int, default=20
        Period for Bollinger Bands
    bb_std : float, default=2.0
        Standard deviations for Bollinger Bands
    atr_period : int, default=14
        Period for ATR calculation
    chop_period : int, default=14
        Period for Choppiness Index
    er_period : int, default=10
        Period for Efficiency Ratio
    hurst_lag_min : int, default=2
        Minimum lag for Hurst exponent
    hurst_lag_max : int, default=100
        Maximum lag for Hurst exponent
    
    Thresholds:
    adx_trend_threshold : float, default=25.0
        ADX level above which market is considered trending
    adx_strong_threshold : float, default=35.0
        ADX level for strong trend
    adx_choppy_threshold : float, default=20.0
        ADX level below which market is non-trending
    chop_threshold : float, default=61.8
        Choppiness Index level for choppy market
    chop_low_threshold : float, default=38.2
        Choppiness Index level for trending market
    bb_squeeze_threshold : float, default=0.02
        Bollinger Band width threshold for squeeze
    atr_low_percentile : float, default=0.25
        Percentile for low volatility
    atr_high_percentile : float, default=0.75
        Percentile for high volatility
    er_high_threshold : float, default=0.3
        Efficiency Ratio threshold for efficient market
    er_low_threshold : float, default=0.1
        Efficiency Ratio threshold for inefficient market
    hurst_trend_threshold : float, default=0.55
        Hurst exponent threshold for trending
    hurst_range_threshold : float, default=0.45
        Hurst exponent threshold for ranging
    
    inplace : bool, default=False
        Whether to modify DataFrame in place
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the following columns:
        - IC_Regime: Numeric regime (0=Chop, 1=Weak Trend, 2=Strong Trend, 3=Range)
        - IC_RegimeName: Human-readable regime name
        - IC_Confidence: Confidence score (0-100)
        - IC_ADX: Average Directional Index
        - IC_ChoppinessIndex: Choppiness Index
        - IC_BandWidth: Bollinger Band width (normalized)
        - IC_ATR_Normalized: ATR as percentage of price
        - IC_EfficiencyRatio: Kaufman's Efficiency Ratio
        - IC_Signal: Trading signal (-1=avoid, 0=caution, 1=favorable)
        - IC_RiskLevel: Risk assessment (Low/Medium/High)
    """
    # Work with copy if not inplace
    result_df = df if inplace else df.copy()
    
    # Validate inputs
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Extract arrays
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    
    # Run core calculation
    global NUMBA_AVAILABLE
    use_numba = NUMBA_AVAILABLE
    
    if use_numba:
        try:
            (regime, confidence, adx, chop, bandwidth, atr_norm, er,
             di_plus, di_minus, upper_bb, middle_bb, lower_bb) = _intelligent_chop_core(
                high, low, close,
                adx_period, bb_period, bb_std,
                atr_period, chop_period, er_period,
                hurst_lag_min, hurst_lag_max,
                adx_trend_threshold, adx_strong_threshold, adx_choppy_threshold,
                chop_threshold, chop_low_threshold, bb_squeeze_threshold,
                atr_low_percentile, atr_high_percentile,
                er_high_threshold, er_low_threshold,
                hurst_trend_threshold, hurst_range_threshold
            )
        except Exception as e:
            print(f"Numba acceleration failed: {e}. Falling back to pure Python.")
            use_numba = False
    
    if not use_numba:
        raise NotImplementedError("Pure Python fallback not implemented. Install numba: pip install numba")
    
    # Map regime numbers to names
    regime_map = {
        0: REGIME_VOLATILE_CHOP,
        1: REGIME_WEAK_TREND,
        2: REGIME_STRONG_TREND,
        3: REGIME_QUIET_RANGE
    }
    
    regime_names = [regime_map.get(r, REGIME_TRANSITIONAL) for r in regime]
    
    # Generate trading signals based on regime
    signals = np.zeros(len(regime), dtype=np.int32)
    risk_levels = []
    
    for i in range(len(regime)):
        if regime[i] == 2:  # Strong Trend
            signals[i] = 1  # Favorable for trend following
            risk_levels.append("Low")
        elif regime[i] == 1:  # Weak Trend
            signals[i] = 0  # Caution
            risk_levels.append("Medium")
        elif regime[i] == 3:  # Quiet Range
            signals[i] = 0  # Caution (could trade range-bound strategies)
            risk_levels.append("Low")
        else:  # Volatile Chop
            signals[i] = -1  # Avoid trading
            risk_levels.append("High")
    
    # Add results to DataFrame
    result_df['IC_Regime'] = regime
    result_df['IC_RegimeName'] = regime_names
    result_df['IC_Confidence'] = confidence
    result_df['IC_ADX'] = adx
    result_df['IC_ChoppinessIndex'] = chop
    result_df['IC_BandWidth'] = bandwidth * 100  # Convert to percentage
    result_df['IC_ATR_Normalized'] = atr_norm * 100  # Convert to percentage
    result_df['IC_EfficiencyRatio'] = er
    result_df['IC_DI_Plus'] = di_plus
    result_df['IC_DI_Minus'] = di_minus
    result_df['IC_BB_Upper'] = upper_bb
    result_df['IC_BB_Middle'] = middle_bb
    result_df['IC_BB_Lower'] = lower_bb
    result_df['IC_Signal'] = signals
    result_df['IC_RiskLevel'] = risk_levels
    
    return result_df