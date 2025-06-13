import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from pathlib import Path
from .indicators import supertrend_indicator, market_bias_indicator, support_resistance_indicator_fractal, neurotrend_indicator, neurotrend_intelligent, add_andean_oscillator
from .plotting import IndicatorPlotter


class TIC:
    """Technical Indicators Custom - Main class for adding indicators to dataframes."""
    
    @staticmethod
    def detect_timeframe(df: pd.DataFrame) -> str:
        """
        Detect the timeframe/interval of the DataFrame based on index frequency.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            str: Human-readable timeframe (e.g., '1M', '5M', '15M', '1H', '4H', '1D')
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return 'Unknown'
        
        if len(df) < 2:
            return 'Unknown'
        
        # Calculate time differences between consecutive rows
        time_diffs = df.index[1:] - df.index[:-1]
        
        # Get the most common time difference (mode)
        # Convert to seconds for easier comparison
        time_diffs_seconds = time_diffs.total_seconds()
        
        # Use median instead of mode to be more robust to outliers
        median_diff = np.median(time_diffs_seconds)
        
        # Map seconds to human-readable format
        # Define thresholds with some tolerance
        if median_diff < 90:  # Less than 1.5 minutes
            return '1M'
        elif median_diff < 180:  # 1.5 to 3 minutes
            return '2M'
        elif median_diff < 270:  # 3 to 4.5 minutes
            return '3M'
        elif median_diff < 420:  # 4.5 to 7 minutes
            return '5M'
        elif median_diff < 720:  # 7 to 12 minutes
            return '10M'
        elif median_diff < 1200:  # 12 to 20 minutes
            return '15M'
        elif median_diff < 2100:  # 20 to 35 minutes
            return '30M'
        elif median_diff < 5400:  # 35 to 90 minutes
            return '1H'
        elif median_diff < 10800:  # 1.5 to 3 hours
            return '2H'
        elif median_diff < 21600:  # 3 to 6 hours
            return '4H'
        elif median_diff < 64800:  # 6 to 18 hours
            return '12H'
        elif median_diff < 129600:  # 18 to 36 hours
            return '1D'
        elif median_diff < 432000:  # 1.5 to 5 days
            return '1W'
        elif median_diff < 2592000:  # 5 to 30 days
            return '1M'
        else:
            return '>1M'
    
    @staticmethod
    def add_super_trend(df, atr_period=10, multiplier=3.0, source_col='Close', inplace=False, use_numba=True):
        """
        Add SuperTrend indicator to DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            atr_period (int): ATR period (default: 10)
            multiplier (float): ATR multiplier (default: 3.0)
            source_col (str): Source column for calculations (default: 'Close')
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            use_numba (bool): Whether to use Numba acceleration if available (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with SuperTrend columns added
        """
        result = supertrend_indicator(df, atr_period, multiplier, source_col, use_numba)
        
        if inplace:
            for col in result.columns:
                df[col] = result[col]
            return df
        else:
            return pd.concat([df, result], axis=1)
    
    @staticmethod
    def plot(df: pd.DataFrame,
             show_market_bias: bool = True,
             show_supertrend: bool = True,
             show_fractal_sr: bool = True,
             show_neurotrend: bool = True,
             show_andean: bool = False,
             show_signals: bool = True,
             title: Optional[str] = None,
             figsize: Tuple[int, int] = (16, 8),
             save_path: Optional[Union[str, Path]] = None,
             show: bool = True):
        """
        Create a professional trading chart with indicators.
        
        This is a convenience method that creates an IndicatorPlotter instance
        and plots the data with the specified settings.
        
        Args:
            df: DataFrame with OHLC data and indicators (from add_super_trend/add_market_bias/add_fractal_sr)
            show_market_bias: Whether to show Market Bias overlay
            show_supertrend: Whether to show SuperTrend line
            show_fractal_sr: Whether to show Fractal Support/Resistance levels
            show_signals: Whether to show signal panel
            title: Chart title (auto-generated if None)
            figsize: Figure size tuple (width, height)
            save_path: Path to save the chart
            show: Whether to display the chart
            
        Returns:
            matplotlib Figure object
            
        Example:
            # Add indicators
            df = TIC.add_super_trend(df)
            df = TIC.add_market_bias(df, inplace=True)
            df = TIC.add_fractal_sr(df, inplace=True)
            
            # Plot the chart
            TIC.plot(df, title="AUDUSD Analysis", save_path="chart.png")
        """
        # Detect timeframe if not included in title
        detected_timeframe = TIC.detect_timeframe(df)
        
        # Pass the detected timeframe to the plotter
        plotter = IndicatorPlotter()
        return plotter.plot(
            df=df,
            show_market_bias=show_market_bias,
            show_supertrend=show_supertrend,
            show_fractal_sr=show_fractal_sr,
            show_neurotrend=show_neurotrend,
            show_andean=show_andean,
            show_signals=show_signals,
            title=title,
            figsize=figsize,
            save_path=save_path,
            show=show,
            detected_timeframe=detected_timeframe
        )
    
    @staticmethod
    def add_market_bias(df, ha_len=50, ha_len2=10, inplace=False, use_numba=True):
        """
        Add Market Bias indicator to DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            ha_len (int): First EMA smoothing period (default: 50)
            ha_len2 (int): Second EMA smoothing period for signals (default: 10)
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            use_numba (bool): Whether to use Numba acceleration if available (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with Market Bias columns added
        """
        result = market_bias_indicator(df, ha_len, ha_len2, use_numba)
        
        if inplace:
            for col in result.columns:
                df[col] = result[col]
            return df
        else:
            return pd.concat([df, result], axis=1)
    
    @staticmethod
    def add_fractal_sr(df, noise_filter=True, inplace=False, use_numba=True):
        """
        Add Fractal Support and Resistance identification to DataFrame.
        
        Uses the fractal pattern approach from the showcase file to identify
        support and resistance levels without look-ahead bias.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            noise_filter (bool): Whether to apply noise filtering (default: True)
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            use_numba (bool): Whether to use Numba acceleration if available (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with Fractal Support/Resistance columns added
        """
        result = support_resistance_indicator_fractal(df, noise_filter, use_numba)
        
        if inplace:
            for col in result.columns:
                df[col] = result[col]
            return df
        else:
            return pd.concat([df, result], axis=1)
    
    @staticmethod
    def add_neuro_trend(df, base_fast=10, base_slow=21, enable_reflex=False, inplace=False, use_numba=True):
        """
        Add NeuroTrend indicator to DataFrame.
        
        NeuroTrend is an advanced adaptive trend analysis indicator that uses
        neural network-inspired techniques to analyze market trends. It provides
        adaptive EMAs that adjust based on volatility and momentum, along with
        trend phase classification and confidence scoring.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            base_fast (int): Base fast EMA period (default: 10)
            base_slow (int): Base slow EMA period (default: 21)
            enable_reflex (bool): Enable Reflex mode for more responsive adaptation (default: False)
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            use_numba (bool): Whether to use Numba acceleration if available (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with NeuroTrend columns added
            
        Added columns:
        - NT_FastEMA: Adaptive fast EMA
        - NT_SlowEMA: Adaptive slow EMA
        - NT_SlopePower: Slope power metric (-100 to 100)
        - NT_TrendPhase: Trend phase classification ('Impulse', 'Cooling', 'Neutral', 'Reversal')
        - NT_TrendDirection: Trend direction (1 for bullish, -1 for bearish)
        - NT_Confidence: Confidence score (0-100)
        - NT_ReversalRisk: Reversal risk flag (boolean)
        - NT_StallDetected: Stall detection flag (boolean)
        - NT_SlopeForecast: Projected slope for next period
        """
        # Map parameters to neurotrend_indicator function
        # The function has more parameters, so we'll use defaults for the ones not exposed
        result = neurotrend_indicator(
            df, 
            base_fast_len=base_fast,
            base_slow_len=base_slow,
            atr_period=14,  # Default
            rsi_period=14,  # Default
            dmi_period=14,  # Default
            volatility_factor=2.0 if enable_reflex else 1.0,  # Higher factor for Reflex mode
            momentum_factor=0.8 if enable_reflex else 0.5,    # Higher factor for Reflex mode
            slope_smooth=3,  # Default
            confidence_smooth=5,  # Default
            use_numba=use_numba
        )
        
        if inplace:
            for col in result.columns:
                df[col] = result[col]
            return df
        else:
            return pd.concat([df, result], axis=1)
    
    @staticmethod
    def add_neuro_trend_intelligent(df, base_fast=10, base_slow=50, 
                                   confirm_bars=3, dynamic_thresholds=True,
                                   enable_diagnostics=False, inplace=False, use_numba=True):
        """
        Add NeuroTrend Intelligent indicator to DataFrame.
        
        NeuroTrend Intelligent is an enhanced version of NeuroTrend with advanced
        anti-whipsaw features including hysteresis confirmation, dynamic thresholds,
        and volatility regime adaptation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            base_fast (int): Base fast EMA period (default: 10)
            base_slow (int): Base slow EMA period (default: 50)
            confirm_bars (int): Bars required for direction confirmation (default: 3)
            dynamic_thresholds (bool): Use dynamic thresholds (default: True)
            enable_diagnostics (bool): Enable diagnostic outputs (default: False)
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            use_numba (bool): Whether to use Numba acceleration if available (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with NeuroTrend Intelligent columns added
            
        Added columns:
        - NTI_FastEMA: Adaptive fast EMA
        - NTI_SlowEMA: Adaptive slow EMA
        - NTI_SlopePower: Slope power metric (-100 to 100)
        - NTI_TrendPhase: Trend phase classification
        - NTI_DirectionRaw: Raw trend direction (unconfirmed)
        - NTI_Direction: Confirmed trend direction (1=bull, -1=bear)
        - NTI_Confidence: Confidence score (0-100)
        - NTI_ReversalRisk: Reversal risk flag
        - NTI_StallDetected: Stall detection flag
        - NTI_SlopeForecast: Projected slope
        
        If enable_diagnostics=True, additional columns:
        - NTI_HiThreshold: Dynamic high threshold
        - NTI_LoThreshold: Dynamic low threshold
        - NTI_ATR_Z: ATR z-score
        - NTI_DirectionChanged: Direction change flag
        - NTI_FlipTimestamp: Timestamp of direction changes
        - NTI_FlipFromDir: Previous direction before flip
        - NTI_FlipToDir: New direction after flip
        """
        # Try to use fast implementation if available
        try:
            from .indicators_fast_v2 import neurotrend_intelligent_fast
            result = neurotrend_intelligent_fast(
                df,
                base_fast_len=base_fast,
                base_slow_len=base_slow,
                confirm_bars=confirm_bars,
                dynamic_thresholds=dynamic_thresholds,
                enable_diagnostics=enable_diagnostics,
                use_numba=use_numba
            )
        except ImportError:
            # Fall back to original implementation
            result = neurotrend_intelligent(
                df,
                base_fast_len=base_fast,
                base_slow_len=base_slow,
                confirm_bars=confirm_bars,
                dynamic_thresholds=dynamic_thresholds,
                enable_diagnostics=enable_diagnostics,
                use_numba=use_numba
            )
        
        if inplace:
            for col in result.columns:
                df[col] = result[col]
            return df
        else:
            return pd.concat([df, result], axis=1)
    
    @staticmethod
    def add_neuro_trend_intelligent_3state(df, base_fast=10, base_slow=50,
                                          confirm_bars=3, dynamic_thresholds=True,
                                          slope_threshold=15.0, confidence_threshold=30.0,
                                          adx_threshold=25.0, consolidation_bars=10,
                                          range_atr_ratio=0.5, ranging_persistence=5,
                                          calculate_adx=True, adx_period=14,
                                          enable_diagnostics=False, use_numba=True,
                                          inplace=False):
        """
        Add NeuroTrend Intelligent with 3-state detection (Trend Up/Down/Ranging).
        
        This enhanced version detects ranging/choppy markets in addition to trends:
        - 1: Uptrend (green)
        - 0: Ranging/Choppy (grey)
        - -1: Downtrend (red)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
        base_fast : int
            Base fast EMA period (default: 10)
        base_slow : int
            Base slow EMA period (default: 50)
        confirm_bars : int
            Bars for direction confirmation (default: 3)
        dynamic_thresholds : bool
            Use dynamic thresholds (default: True)
        slope_threshold : float
            Max absolute slope for ranging (default: 15.0)
        confidence_threshold : float
            Max confidence for ranging (default: 30.0)
        adx_threshold : float
            Max ADX for ranging (default: 25.0)
        consolidation_bars : int
            Lookback for price range (default: 10)
        range_atr_ratio : float
            Price range/ATR threshold (default: 0.5)
        ranging_persistence : int
            Bars to confirm ranging (default: 5)
        calculate_adx : bool
            Calculate ADX if not present (default: True)
        adx_period : int
            ADX period (default: 14)
        enable_diagnostics : bool
            Enable diagnostic outputs (default: False)
        use_numba : bool
            Use Numba acceleration (default: True)
        inplace : bool
            Modify DataFrame in place (default: False)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with NTI indicators and 3-state detection
        """
        # Use the consolidated neurotrend_3state function
        from .indicators import neurotrend_3state
        result = neurotrend_3state(
            df,
            base_fast_len=base_fast,
            base_slow_len=base_slow,
            slope_threshold=slope_threshold,
            confidence_threshold=confidence_threshold,
            adx_threshold=adx_threshold,
            consolidation_bars=consolidation_bars,
            range_atr_ratio=range_atr_ratio,
            ranging_persistence=ranging_persistence,
            use_numba=use_numba
        )
        
        if inplace:
            for col in result.columns:
                if col not in df.columns:
                    df[col] = result[col]
            return df
        else:
            return result
    
    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """
        Convert a timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1M', '5M', '1H', '1D', 'W', 'M')
            
        Returns:
            int: Number of minutes in the timeframe
        """
        import re
        
        # Handle special cases and pandas offset aliases
        timeframe_upper = timeframe.upper()
        
        # First, try to parse pandas frequency strings
        # Look for patterns like '15T', '1H', '1D' etc
        match = re.match(r'^(\d*)([A-Z]+)$', timeframe_upper)
        if match:
            num_str, unit = match.groups()
            num = int(num_str) if num_str else 1
            
            # Map pandas offset aliases to standard units
            # Handle both old and new pandas formats
            unit_map = {
                'T': 1,      # T is minute in pandas (old format)
                'MIN': 1,    # MIN is minute (new format)
                'M': 1,      # M could be minute in our format
                'H': 60,     # Hour
                'D': 1440,   # Day (24 * 60)
                'W': 10080,  # Week (7 * 24 * 60)
                'MO': 43200, # Month (approximately 30 * 24 * 60)
                'Y': 525600, # Year (365 * 24 * 60)
            }
            
            if unit in unit_map:
                return num * unit_map[unit]
        
        # Handle lowercase pandas formats (min, h, d, etc.)
        timeframe_lower = timeframe.lower()
        match_lower = re.match(r'^(\d*)([a-z]+)$', timeframe_lower)
        if match_lower:
            num_str, unit = match_lower.groups()
            num = int(num_str) if num_str else 1
            
            unit_map_lower = {
                'min': 1,    # min is minute (new pandas format)
                'h': 60,     # hour (new pandas format)
                'd': 1440,   # day
                'w': 10080,  # week
            }
            
            if unit in unit_map_lower:
                return num * unit_map_lower[unit]
        
        raise ValueError(f"Unknown timeframe format: {timeframe}")
    
    @staticmethod
    def resample_ohlc(df, rule):
        """
        Resample OHLC data to a different timeframe.
        
        This method properly aggregates OHLC data using appropriate functions
        for each column type. It handles the Volume column if present.
        
        Note: Downsampling to smaller timeframes is not allowed. You can only
        resample to equal or larger timeframes than the base interval.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and datetime index
            rule (str): Resampling rule (e.g., '1H', '4H', 'D', 'W', 'M')
                        Uses pandas offset aliases
            
        Returns:
            pd.DataFrame: Resampled OHLC DataFrame
            
        Example:
            # Resample 15-minute data to 1-hour
            df_hourly = TIC.resample_ohlc(df_15min, '1H')
            
            # Resample to daily
            df_daily = TIC.resample_ohlc(df, 'D')
            
            # Resample to weekly
            df_weekly = TIC.resample_ohlc(df, 'W')
        """
        # Ensure the dataframe has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for resampling")
        
        # Detect the base timeframe
        base_timeframe = TIC.detect_timeframe(df)
        
        if base_timeframe == 'Unknown':
            raise ValueError("Unable to detect the base timeframe of the DataFrame")
        
        # Convert both timeframes to minutes for comparison
        try:
            base_minutes = TIC._timeframe_to_minutes(base_timeframe)
            target_minutes = TIC._timeframe_to_minutes(rule)
        except ValueError as e:
            raise ValueError(f"Error parsing timeframes: {e}")
        
        # Check if downsampling is attempted
        if target_minutes < base_minutes:
            # Calculate valid intervals
            valid_intervals = []
            common_intervals = ['5M', '15M', '30M', '1H', '2H', '4H', '12H', '1D', '1W', '1M']
            for interval in common_intervals:
                interval_minutes = TIC._timeframe_to_minutes(interval)
                if interval_minutes >= base_minutes:
                    valid_intervals.append(interval)
            
            raise ValueError(
                f"Cannot resample from {base_timeframe} to {rule}. "
                f"Downsampling to smaller timeframes is not possible. "
                f"Valid intervals for {base_timeframe} data are: {', '.join(valid_intervals)}"
            )
        
        # Define aggregation rules for OHLC columns
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        
        # Add Volume aggregation if Volume column exists
        if 'Volume' in df.columns:
            agg_rules['Volume'] = 'sum'
        
        # Perform resampling
        resampled = df.resample(rule).agg(agg_rules)
        
        # Remove any rows with all NaN values (e.g., weekends for daily forex data)
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    @staticmethod
    def plot_multi_timeframe(timeframes_dict: dict,
                           title: Optional[str] = None,
                           show_market_bias: bool = True,
                           show_supertrend: bool = True,
                           show_fractal_sr: bool = True,
                           show_neurotrend: bool = True,
                           show_signals: bool = True,
                           figsize: Tuple[int, int] = (20, 16),
                           layout: Optional[Tuple[int, int]] = None,
                           save_path: Optional[Union[str, Path]] = None,
                           show: bool = True) -> plt.Figure:
        """
        Create a multi-timeframe chart with multiple subplots.
        
        This method creates a grid of subplots, each showing a different timeframe
        of the same data. It uses the existing IndicatorPlotter to maintain
        consistent styling across all subplots.
        
        Args:
            timeframes_dict: Dictionary with timeframe keys and DataFrame values
                           e.g., {'1M': df_1m, '15M': df_15m, '1H': df_1h, 'D': df_daily}
            title: Overall title for the figure (default: auto-generated)
            show_market_bias: Whether to show Market Bias overlay in all subplots
            show_supertrend: Whether to show SuperTrend line in all subplots
            show_fractal_sr: Whether to show Fractal S/R levels in all subplots
            show_signals: Whether to show signal panel in all subplots
            figsize: Size of the entire figure (default: (20, 16))
            layout: Optional tuple specifying grid layout (rows, cols)
                   Default is (2, 2) for up to 4 timeframes
            save_path: Path to save the chart
            show: Whether to display the chart
            
        Returns:
            matplotlib Figure object
            
        Example:
            # Prepare multiple timeframes
            timeframes = {
                '1M': df_1min,
                '15M': df_15min,
                '1H': df_hourly,
                'D': df_daily
            }
            
            # Create multi-timeframe plot
            TIC.plot_multi_timeframe(
                timeframes,
                title="AUDUSD Multi-Timeframe Analysis",
                save_path="multi_tf_chart.png"
            )
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Validate input
        if not timeframes_dict:
            raise ValueError("timeframes_dict cannot be empty")
        
        # Filter out None values and sort timeframes
        valid_timeframes = {k: v for k, v in timeframes_dict.items() if v is not None}
        
        if not valid_timeframes:
            raise ValueError("No valid DataFrames provided in timeframes_dict")
        
        # Define standard timeframe order
        standard_order = ['1M', '2M', '3M', '5M', '10M', '15M', '30M', 
                         '1H', '2H', '4H', '12H', '1D', 'D', '1W', 'W', '1MO', 'M']
        
        # Sort timeframes by standard order
        sorted_timeframes = []
        for tf in standard_order:
            if tf in valid_timeframes:
                sorted_timeframes.append((tf, valid_timeframes[tf]))
        
        # Add any non-standard timeframes at the end
        for tf, df in valid_timeframes.items():
            if tf not in standard_order:
                sorted_timeframes.append((tf, df))
        
        n_plots = len(sorted_timeframes)
        
        # Determine layout
        if layout is None:
            if n_plots <= 1:
                layout = (1, 1)
            elif n_plots <= 2:
                layout = (1, 2)
            elif n_plots <= 4:
                layout = (2, 2)
            elif n_plots <= 6:
                layout = (2, 3)
            elif n_plots <= 9:
                layout = (3, 3)
            else:
                # For more than 9, create a grid that's as square as possible
                cols = int(np.ceil(np.sqrt(n_plots)))
                rows = int(np.ceil(n_plots / cols))
                layout = (rows, cols)
        
        rows, cols = layout
        
        # Apply dark background style
        plt.style.use('dark_background')
        
        # Create figure with GridSpec for better control
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        fig.patch.set_facecolor('#131722')  # TradingView dark background
        
        # Create GridSpec based on whether we need signal panels
        if show_signals:
            # Each subplot needs 3:1 ratio for price:signal
            gs = GridSpec(rows * 4, cols, figure=fig, hspace=0.15, wspace=0.08)
        else:
            gs = GridSpec(rows, cols, figure=fig, hspace=0.2, wspace=0.08)
        
        # Create plotter instance
        plotter = IndicatorPlotter()
        
        # Store all axes for later formatting
        all_axes = []
        
        # Plot each timeframe
        for idx, (timeframe, df) in enumerate(sorted_timeframes):
            if idx >= rows * cols:
                break  # Skip if we have more data than grid positions
            
            row = idx // cols
            col = idx % cols
            
            # Create subplot(s) for this position
            if show_signals:
                # Create main price axis and signal axis
                ax_main = fig.add_subplot(gs[row * 4:(row * 4) + 3, col])
                ax_signal = fig.add_subplot(gs[(row * 4) + 3, col], sharex=ax_main)
                axes = [ax_main, ax_signal]
            else:
                # Create only main price axis
                ax_main = fig.add_subplot(gs[row, col])
                axes = [ax_main]
            
            all_axes.extend(axes)
            
            # Detect timeframe for this DataFrame
            detected_tf = TIC.detect_timeframe(df)
            
            # Create subplot title
            subplot_title = f"{timeframe} Timeframe"
            if hasattr(df.index, 'name') and df.index.name:
                subplot_title = f"{df.index.name} - {subplot_title}"
            
            # Use the plotter's internal plotting methods directly
            # Set up the axes styling
            for ax in axes:
                ax.set_facecolor('#131722')
                ax.grid(True, alpha=0.3, color='#363c4e')
                ax.tick_params(colors='#d1d4dc')
                for spine in ax.spines.values():
                    spine.set_color('#363c4e')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Plot candlesticks
            plotter._plot_candlesticks(ax_main, df)
            
            # Check for available indicators
            has_supertrend = 'SuperTrend_Line' in df.columns
            has_market_bias = 'MB_Bias' in df.columns
            has_fractal_sr = 'SR_FractalHighs' in df.columns
            has_neurotrend = 'NT_FastEMA' in df.columns
            
            # Plot indicators if requested and available
            if show_market_bias and has_market_bias:
                plotter._plot_market_bias(ax_main, df)
            
            if show_neurotrend and has_neurotrend:
                plotter._plot_neurotrend(ax_main, df, show_phase_annotations=False)
            
            if show_supertrend and has_supertrend:
                plotter._plot_supertrend(ax_main, df)
            
            if show_fractal_sr and has_fractal_sr:
                plotter._plot_fractal_sr(ax_main, df)
            
            # Set title and labels
            ax_main.set_title(subplot_title, fontsize=12, color='#d1d4dc', pad=5)
            ax_main.set_ylabel('Price', fontsize=10, color='#d1d4dc')
            ax_main.set_xlim(-1, len(df))
            
            # Format x-axis
            plotter._current_timeframe = detected_tf
            plotter._format_date_axis(ax_main, df)
            
            # Add legend to main axis
            plotter._add_legend(ax_main, has_supertrend and show_supertrend, 
                              has_market_bias and show_market_bias, 
                              has_fractal_sr and show_fractal_sr,
                              has_neurotrend and show_neurotrend)
            
            # Plot signals if requested
            if show_signals and len(axes) > 1:
                plotter._plot_signals(axes[1], df, has_supertrend, has_market_bias, has_neurotrend)
                # Hide x-axis labels on main plot when signal panel is shown
                ax_main.set_xticklabels([])
        
        # Add overall title
        if title is None:
            # Try to get symbol from first DataFrame
            symbol = None
            for _, df in sorted_timeframes:
                if hasattr(df.index, 'name') and df.index.name:
                    symbol = df.index.name
                    break
            
            if symbol:
                title = f"{symbol} - Multi-Timeframe Analysis"
            else:
                title = "Multi-Timeframe Analysis"
        
        fig.suptitle(title, fontsize=16, color='#d1d4dc', y=0.995)
        
        # Adjust subplot spacing
        plt.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.04)
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='#131722', edgecolor='none')
            print(f"Multi-timeframe chart saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    @staticmethod
    def add_andean_oscillator(df, length=250, sig_length=25, inplace=False):
        """
        Add Andean Oscillator to the DataFrame.
        
        The Andean Oscillator uses exponential envelopes to identify trend components.
        It calculates bullish and bearish components based on price variance within
        adaptive envelopes. Created by alexgrover.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            length (int): Period for exponential envelopes (default: 250)
            sig_length (int): Signal line EMA period (default: 25)
            inplace (bool): Whether to modify the DataFrame in place (default: False)
            
        Returns:
            pd.DataFrame: DataFrame with Andean Oscillator columns added:
            - AO_Bull: Bullish component
            - AO_Bear: Bearish component
            - AO_Signal: Signal line (EMA of max(bull, bear))
            - AO_BullTrend: Bullish trend start markers (A++ signals)
            - AO_BearTrend: Bearish trend start markers
            
        Example:
            # Add Andean Oscillator
            df = TIC.add_andean_oscillator(df, length=250, sig_length=25)
            
            # Check for bullish trend starts
            bull_signals = df[df['AO_BullTrend']]
        """
        return add_andean_oscillator(df, length, sig_length, inplace)
    
    @staticmethod
    def add_intelligent_chop(df, adx_period=14, bb_period=20, bb_std=2.0,
                            atr_period=14, chop_period=14, er_period=10,
                            hurst_lag_min=2, hurst_lag_max=100,
                            adx_trend_threshold=25.0, adx_strong_threshold=35.0,
                            adx_choppy_threshold=20.0, chop_threshold=61.8,
                            chop_low_threshold=38.2, bb_squeeze_threshold=0.02,
                            atr_low_percentile=0.25, atr_high_percentile=0.75,
                            er_high_threshold=0.3, er_low_threshold=0.1,
                            hurst_trend_threshold=0.55, hurst_range_threshold=0.45,
                            inplace=False):
        """
        Add Intelligent Chop Indicator - Advanced Market Regime Detection.
        
        This indicator synthesizes multiple technical indicators to identify market regimes:
        - Strong Trend: Clear directional movement with high momentum
        - Weak Trend: Some directional bias but less conviction
        - Quiet Range: Low volatility consolidation, mean-reverting behavior
        - Volatile Chop: High volatility without clear direction, dangerous conditions
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
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
            DataFrame with the following columns added:
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
            
        Example:
            # Add Intelligent Chop indicator
            df = TIC.add_intelligent_chop(df)
            
            # Check current market regime
            current_regime = df['IC_RegimeName'].iloc[-1]
            confidence = df['IC_Confidence'].iloc[-1]
            print(f"Market is in {current_regime} with {confidence:.1f}% confidence")
        """
        from .indicators import intelligent_chop
        return intelligent_chop(df, adx_period=adx_period, bb_period=bb_period,
                               bb_std=bb_std, atr_period=atr_period,
                               chop_period=chop_period, er_period=er_period,
                               hurst_lag_min=hurst_lag_min, hurst_lag_max=hurst_lag_max,
                               adx_trend_threshold=adx_trend_threshold,
                               adx_strong_threshold=adx_strong_threshold,
                               adx_choppy_threshold=adx_choppy_threshold,
                               chop_threshold=chop_threshold,
                               chop_low_threshold=chop_low_threshold,
                               bb_squeeze_threshold=bb_squeeze_threshold,
                               atr_low_percentile=atr_low_percentile,
                               atr_high_percentile=atr_high_percentile,
                               er_high_threshold=er_high_threshold,
                               er_low_threshold=er_low_threshold,
                               hurst_trend_threshold=hurst_trend_threshold,
                               hurst_range_threshold=hurst_range_threshold,
                               inplace=inplace)