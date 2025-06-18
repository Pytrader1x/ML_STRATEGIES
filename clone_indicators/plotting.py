"""
Professional trading chart plotting for technical indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import Optional, Union, Tuple
from pathlib import Path


class IndicatorPlotter:
    """
    Professional trading chart plotter with dark theme.
    It can generate standard indicator plots or a specialized multi-panel
    plot for NeuroTrend Intelligent and 3-State versions.
    """
    # TradingView-style color scheme
    COLORS = {
        'bg': '#131722',
        'grid': '#363c4e',
        'text': '#d1d4dc',
        'bullish': '#26a69a',  # Green
        'bearish': '#ef5350',  # Red
        'neutral': '#787b86',  # Grey for ranging
        'yellow': '#ffd700',
        'orange': '#ff9800',
        'cyan': '#00bcd4',
        'white': '#ffffff'
    }

    def __init__(self):
        """Initialize the plotter with style settings"""
        plt.style.use('dark_background')

    def plot(self,
             df: pd.DataFrame,
             show_market_bias: bool = True,
             show_supertrend: bool = True,
             show_fractal_sr: bool = True,
             show_neurotrend: bool = True,
             show_andean: bool = False,
             show_signals: bool = True,
             title: Optional[str] = None,
             figsize: Tuple[int, int] = (16, 8),
             save_path: Optional[Union[str, Path]] = None,
             show: bool = True,
             detected_timeframe: Optional[str] = None) -> plt.Figure:
        """
        Create a professional trading chart. Routes to specialized plotters based on detected indicators.
        """
        self._current_timeframe = detected_timeframe

        # --- ROUTING LOGIC ---
        # Check for specific NeuroTrend variants
        has_nti = 'NTI_FastEMA' in df.columns
        has_nt3 = 'NT_3State' in df.columns  # Check for explicit 3-state column
        has_nt_normal = 'NT_FastEMA' in df.columns and not has_nti
        
        # Only use specialized layout if:
        # 1. NeuroTrend is requested to be shown
        # 2. We have NTI or explicit 3-state data
        # 3. We don't have other indicators that need the signal subplot
        should_use_specialized = (
            show_neurotrend and 
            (has_nti or has_nt3) and 
            not (show_market_bias and 'MB_Bias' in df.columns) and
            not (show_supertrend and 'SuperTrend_Line' in df.columns) and
            not (show_fractal_sr and 'SR_FractalHighs' in df.columns)
        )

        if should_use_specialized:
            return self._plot_neurotrend_intelligent_layout(
                df=df,
                title=title,
                figsize=(18, 12),
                save_path=save_path,
                show=show,
                detected_timeframe=detected_timeframe
            )

        # --- STANDARD PLOT LOGIC (for all other cases) ---
        return self._plot_standard_layout(
            df, show_market_bias, show_supertrend, show_fractal_sr, show_neurotrend,
            show_andean, show_signals, title, figsize, save_path, show, detected_timeframe
        )

    def _plot_standard_layout(self, df, show_market_bias, show_supertrend, show_fractal_sr, show_neurotrend,
                              show_andean, show_signals, title, figsize, save_path, show, detected_timeframe):
        """Plots the standard layout for indicators like SuperTrend, Market Bias, etc."""
        has_supertrend = 'SuperTrend_Line' in df.columns
        has_market_bias = 'MB_Bias' in df.columns
        has_fractal_sr = 'SR_FractalHighs' in df.columns
        has_nt_normal = 'NT_FastEMA' in df.columns and 'NTI_FastEMA' not in df.columns
        has_andean = 'AO_Bull' in df.columns

        # Count how many indicators will be shown
        active_indicators = sum([
            has_supertrend and show_supertrend,
            has_market_bias and show_market_bias,
            has_nt_normal and show_neurotrend
        ])
        
        # Only create signal panel if we have multiple indicators or specific combinations
        need_signal_panel = show_signals and active_indicators >= 2
        need_andean_panel = show_andean and has_andean
        
        # Determine subplot layout
        if need_andean_panel and need_signal_panel:
            # 3 panels: price, signals, andean
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]*1.3), 
                                               constrained_layout=True,
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            plt.subplots_adjust(hspace=0.05)
            ax_andean = ax3
        elif need_andean_panel:
            # 2 panels: price, andean
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True,
                                          gridspec_kw={'height_ratios': [3, 1]})
            plt.subplots_adjust(hspace=0.05)
            ax2 = None
            ax_andean = ax3
        elif need_signal_panel:
            # 2 panels: price, signals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
            plt.subplots_adjust(hspace=0.05)
            ax_andean = None
        else:
            # 1 panel: price only
            fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)
            ax2 = None
            ax_andean = None

        fig.patch.set_facecolor(self.COLORS['bg'])
        ax1.set_facecolor(self.COLORS['bg'])

        self._plot_candlesticks(ax1, df)

        if show_market_bias and has_market_bias: self._plot_market_bias(ax1, df)
        if show_supertrend and has_supertrend: self._plot_supertrend(ax1, df)
        if show_fractal_sr and has_fractal_sr: self._plot_fractal_sr(ax1, df)
        if show_neurotrend and has_nt_normal: self._plot_neurotrend(ax1, df)
        
        # Add Andean Oscillator markers to price chart if Andean is shown
        if show_andean and has_andean:
            self._plot_andean_markers_on_price(ax1, df)

        ax1.set_xlim(-1, len(df))
        
        if title is None:
            symbol = df.index.name if df.index.name else "Data"
            indicators = []
            if has_supertrend and show_supertrend: indicators.append("SuperTrend")
            if has_market_bias and show_market_bias: indicators.append("Market Bias")
            if has_fractal_sr and show_fractal_sr: indicators.append("Fractal S/R")
            if has_nt_normal and show_neurotrend: indicators.append("NeuroTrend")
            if has_andean and show_andean: indicators.append("Andean")
            indicator_text = " & ".join(indicators) if indicators else "Price Chart"
            if detected_timeframe and detected_timeframe != 'Unknown':
                title = f"{symbol} {detected_timeframe} - {indicator_text}"
            else:
                title = f"{symbol} - {indicator_text}"
        elif detected_timeframe and detected_timeframe != 'Unknown' and detected_timeframe not in title:
            title = f"{title} ({detected_timeframe})"

        ax1.set_title(title, fontsize=14, color=self.COLORS['text'], pad=10)
        ax1.set_ylabel('Price', fontsize=11, color=self.COLORS['text'])
        ax1.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax1.tick_params(colors=self.COLORS['text'])

        self._format_date_axis(ax1, df)
        self._add_legend(ax1, has_supertrend and show_supertrend, has_market_bias and show_market_bias,
                         has_fractal_sr and show_fractal_sr, has_nt_normal and show_neurotrend)

        if ax2 is not None and need_signal_panel:
            self._plot_signals(ax2, df, has_supertrend and show_supertrend, 
                             has_market_bias and show_market_bias, 
                             has_nt_normal and show_neurotrend)
            ax1.tick_params(labelbottom=False)
            ax2.sharex(ax1)

        if ax_andean is not None and need_andean_panel:
            ax_andean.set_facecolor(self.COLORS['bg'])
            self._plot_andean_oscillator(ax_andean, df)
            ax_andean.sharex(ax1)
            # Hide x-axis labels on panels above if Andean is the bottom panel
            if ax2 is not None:
                ax2.tick_params(labelbottom=False)
            else:
                ax1.tick_params(labelbottom=False)

        # Style all axes
        all_axes = [ax for ax in [ax1, ax2, ax_andean] if ax is not None]
        for ax in all_axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(self.COLORS['grid'])
            ax.spines['left'].set_color(self.COLORS['grid'])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg'])
            print(f"Chart saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_neurotrend_intelligent_layout(self, df, title, figsize, save_path, show, detected_timeframe):
        """Creates the specific multi-panel layout for NeuroTrend Intelligent / 3-State."""
        self._current_timeframe = detected_timeframe
        
        # Determine which type of NeuroTrend we're plotting
        if 'NT_3State' in df.columns:
            # This is a 3-state NeuroTrend created in the notebook
            prefix = "NT_"
            direction_col = "NT_3State"
            is_3state = True
            # EMA columns should exist as NT_FastEMA, NT_SlowEMA
            fast_ema_col = "NT_FastEMA"
            slow_ema_col = "NT_SlowEMA"
        elif 'NT3_Direction' in df.columns:
            # This is a different 3-state format from neurotrend_3state
            prefix = "NT3_"
            direction_col = "NT3_Direction"
            is_3state = True
            # For NT3, we need to check if NTI EMAs exist (since NT3 is built on NTI)
            if 'NTI_FastEMA' in df.columns:
                fast_ema_col = "NTI_FastEMA"
                slow_ema_col = "NTI_SlowEMA"
            else:
                # Fallback - no EMAs available
                fast_ema_col = None
                slow_ema_col = None
        else:
            # This is NeuroTrend Intelligent
            prefix = "NTI_"
            direction_col = "NTI_Direction"
            is_3state = False
            fast_ema_col = "NTI_FastEMA"
            slow_ema_col = "NTI_SlowEMA"

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = fig.add_gridspec(3, 1, height_ratios=[5, 2, 1.5], hspace=0.1)
        ax_price = fig.add_subplot(gs[0])
        ax_sub1 = fig.add_subplot(gs[1], sharex=ax_price)
        ax_sub2 = fig.add_subplot(gs[2], sharex=ax_price)
        fig.patch.set_facecolor(self.COLORS['bg'])

        ax_price.set_facecolor(self.COLORS['bg'])
        self._plot_candlesticks(ax_price, df)
        self._plot_neurotrend_background(ax_price, df, direction_col)
        
        # Plot EMAs if columns exist
        if fast_ema_col and slow_ema_col and fast_ema_col in df.columns and slow_ema_col in df.columns:
            self._plot_neurotrend_emas(ax_price, df, fast_ema_col, slow_ema_col, direction_col)
        
        ax_price.set_xlim(-1, len(df))
        ax_price.set_ylabel('Price', color=self.COLORS['text'])
        ax_price.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax_price.tick_params(colors=self.COLORS['text'], labelbottom=False)

        if is_3state:
            if prefix == "NT_":
                # For 3-state NeuroTrend created in notebook
                self._plot_nt_slope_confidence(ax_sub1, df)
                self._plot_nt_phase_or_direction(ax_sub2, df, direction_col)
            else:
                # For NT3 from neurotrend_3state function
                self._plot_nt3_confidence_metrics(ax_sub1, df)
                self._plot_nt3_state_bars(ax_sub2, df)
        else:
            # For NTI, use the existing methods
            self._plot_nti_slope_confidence(ax_sub1, df)
            self._plot_nti_trend_phase(ax_sub2, df)

        if title is None:
            symbol = df.index.name if df.index.name else "Data"
            if is_3state:
                indicator_name = "NeuroTrend 3-State"
            else:
                indicator_name = "NeuroTrend Intelligent"
            title = f"{symbol} - {indicator_name} Analysis"
            if detected_timeframe and detected_timeframe != 'Unknown':
                title += f" ({detected_timeframe})"
        ax_price.set_title(title, fontsize=14, color=self.COLORS['text'], pad=10)

        self._add_legend_nti(ax_price, df, is_3state, prefix)
        self._format_date_axis(ax_sub2, df)

        for ax in [ax_price, ax_sub1, ax_sub2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(self.COLORS['grid'])
            ax.spines['left'].set_color(self.COLORS['grid'])

        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg'])
            print(f"Chart saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_candlesticks(self, ax, df):
        """Plot candlestick chart"""
        x_pos = np.arange(len(df))
        opens, highs, lows, closes = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        for idx in range(len(df)):
            color = self.COLORS['bullish'] if closes[idx] >= opens[idx] else self.COLORS['bearish']
            ax.plot([x_pos[idx], x_pos[idx]], [lows[idx], highs[idx]], color=color, linewidth=1, alpha=0.8)
            body_height = abs(closes[idx] - opens[idx])
            body_bottom = min(opens[idx], closes[idx])
            if body_height < (df['Close'].mean() * 0.0001): body_height = df['Close'].mean() * 0.0001
            ax.add_patch(Rectangle((x_pos[idx] - 0.3, body_bottom), 0.6, body_height, facecolor=color, edgecolor=color, alpha=0.8))

    def _plot_market_bias(self, ax, df):
        """Plot Market Bias overlay"""
        x_pos = np.arange(len(df))
        valid_mask = ~(df['MB_o2'].isna() | df['MB_c2'].isna())
        mb_bias, mb_o2, mb_c2, mb_h2, mb_l2 = df['MB_Bias'].values, df['MB_o2'].values, df['MB_c2'].values, df['MB_h2'].values, df['MB_l2'].values
        for i in np.where(valid_mask)[0]:
            mb_color = self.COLORS['bullish'] if mb_bias[i] == 1 else self.COLORS['bearish']
            ax.plot([x_pos[i], x_pos[i]], [mb_l2[i], mb_h2[i]], color=mb_color, linewidth=10, alpha=0.3, solid_capstyle='round')
            body_bottom, body_top = min(mb_o2[i], mb_c2[i]), max(mb_o2[i], mb_c2[i])
            body_height = body_top - body_bottom
            if body_height < (df['Close'].mean() * 0.0001): body_height = df['Close'].mean() * 0.0001
            ax.add_patch(Rectangle((x_pos[i] - 0.4, body_bottom), 0.8, body_height, facecolor=mb_color, edgecolor='none', alpha=0.4))

    def _plot_supertrend(self, ax, df):
        """Plot SuperTrend indicator as a continuous line"""
        x_pos = np.arange(len(df))
        st_direction, st_line = df['SuperTrend_Direction'].values, df['SuperTrend_Line'].values
        valid_mask = ~np.isnan(st_line)
        if not np.any(valid_mask): return
        i = 0
        while i < len(df) - 1:
            if valid_mask[i]:
                j = i + 1
                while j < len(df) and valid_mask[j] and st_direction[j] == st_direction[i]: j += 1
                color = self.COLORS['bullish'] if st_direction[i] == 1 else self.COLORS['bearish']
                ax.plot(x_pos[i:j], st_line[i:j], color=color, linewidth=2.5, alpha=0.9)
                if j < len(df) and valid_mask[j]:
                    ax.plot([x_pos[j-1], x_pos[j]], [st_line[j-1], st_line[j]], color=self.COLORS['bearish'] if st_direction[i] == 1 else self.COLORS['bullish'], linewidth=2.5, alpha=0.9)
                i = j
            else:
                i += 1

    def _plot_fractal_sr(self, ax, df):
        """Plot Fractal Support and Resistance levels"""
        x_pos = np.arange(len(df))
        fractal_highs = df['SR_FractalHighs'].dropna()
        fractal_lows = df['SR_FractalLows'].dropna()
        if len(fractal_highs) > 0:
            high_indices = [df.index.get_loc(idx) for idx in fractal_highs.index]
            ax.scatter(high_indices, fractal_highs.values, color=self.COLORS['bearish'], marker='v', s=80, alpha=0.8, label='Resistance', zorder=5)
        if len(fractal_lows) > 0:
            low_indices = [df.index.get_loc(idx) for idx in fractal_lows.index]
            ax.scatter(low_indices, fractal_lows.values, color=self.COLORS['bullish'], marker='^', s=80, alpha=0.8, label='Support', zorder=5)
        levels = df['SR_Levels'].dropna()
        level_types = df['SR_LevelTypes'][df['SR_Levels'].notna()]
        if len(levels) > 0:
            level_detection = {}
            for timestamp, (level, level_type) in zip(levels.index, zip(levels.values, level_types.values)):
                level_key = round(level, 5)
                if level_key not in level_detection:
                    level_detection[level_key] = {'level': level, 'type': level_type, 'first_detected': timestamp, 'detection_index': df.index.get_loc(timestamp)}
            for level_info in level_detection.values():
                color = self.COLORS['bullish'] if level_info['type'] == 'Support' else self.COLORS['bearish']
                start_x, end_x = level_info['detection_index'], len(df) - 1
                ax.plot([start_x, end_x], [level_info['level'], level_info['level']], color=color, linestyle='--', alpha=0.7, linewidth=2)
                ax.text(end_x, level_info['level'], f" {level_info['type']}: {level_info['level']:.5f}", ha='left', va='center', fontweight='bold', fontsize=8, color=color, alpha=0.9)

    def _plot_neurotrend(self, ax, df):
        """Plot standard NeuroTrend indicator."""
        # Check if we have a 3-state column to use for direction
        direction_col = 'NT_3State' if 'NT_3State' in df.columns else 'NT_TrendDirection'
        
        self._plot_neurotrend_emas(ax, df, 'NT_FastEMA', 'NT_SlowEMA', direction_col)
        if 'NT_Confidence' in df.columns:
            valid_mask = ~df['NT_FastEMA'].isna()
            confidence_values = df['NT_Confidence'][valid_mask]
            ax.neurotrend_confidence = confidence_values.iloc[-1] if len(confidence_values) > 0 else None
        else:
            ax.neurotrend_confidence = None

    def _plot_neurotrend_emas(self, ax, df, fast_ema_col, slow_ema_col, direction_col):
        """Generalized function to plot NeuroTrend EMAs with glowing effect."""
        x_pos = np.arange(len(df))
        fast_ema, slow_ema, trend_direction = df[fast_ema_col].values, df[slow_ema_col].values, df[direction_col].values
        valid_mask = ~(np.isnan(fast_ema) | np.isnan(slow_ema))
        if not np.any(valid_mask): return

        first_valid_idx = np.where(valid_mask)[0][0]
        glow_widths, glow_alphas = [8, 6, 4, 2], [0.1, 0.15, 0.2, 0.3]
        i = first_valid_idx
        while i < len(df):
            if valid_mask[i]:
                current_dir = trend_direction[i]
                j = i + 1
                while j < len(df) and valid_mask[j] and trend_direction[j] == current_dir: j += 1
                
                if current_dir == 1: color = self.COLORS['bullish']
                elif current_dir == -1: color = self.COLORS['bearish']
                else: color = self.COLORS['neutral']

                for width, alpha in zip(glow_widths, glow_alphas):
                    ax.plot(x_pos[i:j], fast_ema[i:j], color=color, linewidth=width, alpha=alpha, solid_capstyle='round')
                    ax.plot(x_pos[i:j], slow_ema[i:j], color=color, linewidth=width, alpha=alpha, solid_capstyle='round')
                
                ax.plot(x_pos[i:j], fast_ema[i:j], color=color, linewidth=2, alpha=0.9)
                ax.plot(x_pos[i:j], slow_ema[i:j], color=color, linewidth=2, alpha=0.7, linestyle='--')
                ax.fill_between(x_pos[i:j], fast_ema[i:j], slow_ema[i:j], color=color, alpha=0.2)
                i = j
            else:
                i += 1

    def _plot_neurotrend_background(self, ax, df, direction_col):
        """Colors the background of a plot based on a direction column (1, -1, 0)."""
        x_pos = np.arange(len(df))
        direction = df[direction_col].values
        for i in range(len(df)):
            if pd.isna(direction[i]): continue
            if direction[i] == 1: color, alpha = self.COLORS['bullish'], 0.15
            elif direction[i] == -1: color, alpha = self.COLORS['bearish'], 0.15
            else: color, alpha = self.COLORS['neutral'], 0.15
            ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=color, alpha=alpha, ec='none')

    def _plot_nti_slope_confidence(self, ax, df):
        """Plots the Slope Power and Confidence sub-panel."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        ax.plot(x_pos, df["NTI_Confidence"], color=self.COLORS['cyan'], label='Confidence', linewidth=1.5)
        ax.plot(x_pos, df["NTI_SlopePower"], color=self.COLORS['yellow'], label='Slope Power', linewidth=1.5)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_ylabel('Value', color=self.COLORS['text'])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'], labelbottom=False)

    def _plot_nti_trend_phase(self, ax, df):
        """Plots the Trend Phase sub-panel."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        phases = df["NTI_TrendPhase"]
        phase_map = {'Impulse': 1, 'Cooling': 0.5, 'Neutral': 0, 'Reversal': -1}
        color_map = {'Impulse': self.COLORS['bullish'], 'Cooling': self.COLORS['yellow'], 'Neutral': self.COLORS['neutral'], 'Reversal': self.COLORS['bearish']}
        
        for phase_name, y_level in phase_map.items():
            mask = (phases == phase_name)
            ax.fill_between(x_pos, y_level - 0.45, y_level + 0.45, where=mask, color=color_map[phase_name], step='mid', alpha=0.8)

        ax.set_ylabel('Trend Phase', color=self.COLORS['text'])
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks(list(phase_map.values()))
        ax.set_yticklabels(list(phase_map.keys()))
        ax.grid(False)
        ax.grid(True, axis='y', alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])
    
    def _plot_nt_slope_confidence(self, ax, df):
        """Plots the Slope Power and Confidence sub-panel for regular NeuroTrend."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        ax.plot(x_pos, df["NT_Confidence"], color=self.COLORS['cyan'], label='Confidence', linewidth=1.5)
        ax.plot(x_pos, df["NT_SlopePower"], color=self.COLORS['yellow'], label='Slope Power', linewidth=1.5)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=50, color='white', linestyle=':', alpha=0.2, linewidth=1, label='Neutral Threshold')
        ax.set_ylabel('Value', color=self.COLORS['text'])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'], labelbottom=False)
    
    def _plot_nt_phase_or_direction(self, ax, df, direction_col):
        """Plots the 3-state direction sub-panel."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        direction = df[direction_col].values
        
        # Plot bars for each state
        for i in range(len(df)):
            if pd.isna(direction[i]): continue
            if direction[i] == 1:
                color = self.COLORS['bullish']
            elif direction[i] == -1:
                color = self.COLORS['bearish']
            else:  # 0
                color = self.COLORS['neutral']
            ax.bar(x_pos[i], direction[i], color=color, alpha=0.8, width=0.9)
        
        ax.set_ylabel('Market State', color=self.COLORS['text'])
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Bearish', 'Neutral', 'Bullish'])
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])
    
    def _plot_nt3_confidence_metrics(self, ax, df):
        """Plots confidence and slope metrics for NT3."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        
        # Plot NT3 confidence (which is adjusted for ranging)
        if 'NT3_Confidence' in df.columns:
            ax.plot(x_pos, df['NT3_Confidence'], color=self.COLORS['cyan'], label='3-State Confidence', linewidth=1.5)
        
        # Also plot NTI confidence if available for comparison
        if 'NTI_Confidence' in df.columns:
            ax.plot(x_pos, df['NTI_Confidence'], color=self.COLORS['yellow'], label='Raw Confidence', linewidth=1.5, alpha=0.5)
        
        # Plot slope power if available
        if 'NTI_SlopePower' in df.columns:
            ax.plot(x_pos, df['NTI_SlopePower'], color=self.COLORS['orange'], label='Slope Power', linewidth=1.5)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_ylabel('Value', color=self.COLORS['text'])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'], labelbottom=False)
    
    def _plot_nt3_state_bars(self, ax, df):
        """Plots the 3-state bars for NT3."""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        
        # Use NT3_Direction for the state values
        direction = df['NT3_Direction'].values
        
        # Plot colored regions for each state
        for i in range(len(df)):
            if pd.isna(direction[i]): continue
            if direction[i] == 1:
                color = self.COLORS['bullish']
                height = 1
            elif direction[i] == -1:
                color = self.COLORS['bearish']
                height = -1
            else:  # 0 - Ranging
                color = self.COLORS['neutral']
                height = 0.5  # Make ranging bars shorter for visual distinction
            
            ax.bar(x_pos[i], height, color=color, alpha=0.8, width=0.9, bottom=0 if height > 0 else height)
        
        # Add state labels if NT3_State exists
        if 'NT3_State' in df.columns:
            # Find state transitions
            states = df['NT3_State'].values
            prev_state = None
            for i in range(len(df)):
                if not pd.isna(states[i]) and states[i] != prev_state:
                    ax.text(x_pos[i], 0, states[i], rotation=90, ha='center', va='center',
                           fontsize=8, color=self.COLORS['text'], alpha=0.7)
                    prev_state = states[i]
        
        ax.set_ylabel('Market State', color=self.COLORS['text'])
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 0.5, 1])
        ax.set_yticklabels(['Downtrend', 'Neutral', 'Ranging', 'Uptrend'])
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])

    def _add_legend(self, ax, has_supertrend, has_market_bias, has_fractal_sr=False, has_neurotrend=False):
        """Add legend to the standard chart"""
        handles = [mpatches.Patch(color=self.COLORS['bullish'], label='Bullish'), mpatches.Patch(color=self.COLORS['bearish'], label='Bearish')]
        if has_neurotrend:
            try:
                if ax.neurotrend_confidence is not None:
                    handles.append(mpatches.Patch(color='none', label=f'NeuroTrend Confidence: {ax.neurotrend_confidence:.2f}'))
            except AttributeError: pass
        ax.legend(handles=handles, loc='upper left', framealpha=0.8, fontsize=10)

    def _add_legend_nti(self, ax, df, is_3state, prefix):
        """Adds a specific legend for the NTI/NT3 plot."""
        handles = [mpatches.Patch(color=self.COLORS['bullish'], label='Uptrend'),
                   mpatches.Patch(color=self.COLORS['bearish'], label='Downtrend')]
        if is_3state:
            handles.append(mpatches.Patch(color=self.COLORS['neutral'], label='Neutral'))
        
        conf_col = f"{prefix}Confidence"
        if conf_col in df.columns:
            latest_confidence = df[conf_col].dropna().iloc[-1]
            handles.append(mpatches.Patch(color='none', label=f'Confidence: {latest_confidence:.1f}%'))
        ax.legend(handles=handles, loc='upper left', framealpha=0.8, fontsize=10)

    def _plot_andean_oscillator(self, ax, df):
        """Plot Andean Oscillator in a separate panel"""
        if 'AO_Bull' not in df.columns or 'AO_Bear' not in df.columns:
            return
        
        x_pos = np.arange(len(df))
        
        # Get oscillator values
        bull = df['AO_Bull'].values
        bear = df['AO_Bear'].values
        signal = df['AO_Signal'].values if 'AO_Signal' in df.columns else None
        
        # Plot bull and bear components
        ax.plot(x_pos, bull, color=self.COLORS['bullish'], linewidth=2, alpha=0.9, label='Bullish Component')
        ax.plot(x_pos, bear, color=self.COLORS['bearish'], linewidth=2, alpha=0.9, label='Bearish Component')
        
        # Plot signal line if available
        if signal is not None:
            ax.plot(x_pos, signal, color=self.COLORS['orange'], linewidth=2, alpha=0.8, label='Signal')
        
        # Add trend start markers (A++ signals)
        if 'AO_BullTrend' in df.columns:
            bull_starts = df[df['AO_BullTrend']]
            if len(bull_starts) > 0:
                bull_indices = [df.index.get_loc(idx) for idx in bull_starts.index]
                bull_values = bull[bull_indices]
                
                # Plot markers
                ax.scatter(bull_indices, bull_values, color=self.COLORS['bullish'], 
                          marker='^', s=150, alpha=1.0, zorder=5, edgecolor='white', linewidth=1.5)
                
                # Add A++ text above markers
                for idx, val in zip(bull_indices, bull_values):
                    ax.text(idx, val * 1.1, 'A++', ha='center', va='bottom', 
                           color=self.COLORS['bullish'], fontsize=10, fontweight='bold')
        
        if 'AO_BearTrend' in df.columns:
            bear_starts = df[df['AO_BearTrend']]
            if len(bear_starts) > 0:
                bear_indices = [df.index.get_loc(idx) for idx in bear_starts.index]
                bear_values = bear[bear_indices]
                
                # Plot markers
                ax.scatter(bear_indices, bear_values, color=self.COLORS['bearish'], 
                          marker='v', s=150, alpha=1.0, zorder=5, edgecolor='white', linewidth=1.5)
                
                # Add A-- text below markers
                for idx, val in zip(bear_indices, bear_values):
                    ax.text(idx, val * 0.9, 'A--', ha='center', va='top', 
                           color=self.COLORS['bearish'], fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_ylabel('Andean Oscillator', fontsize=10, color=self.COLORS['text'])
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])
        ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    def _plot_andean_markers_on_price(self, ax, df):
        """Plot A++ and A-- markers on the price chart at trend start points"""
        if 'AO_BullTrend' not in df.columns or 'AO_BearTrend' not in df.columns:
            return
        
        # Calculate price range for better positioning
        price_range = df['High'].max() - df['Low'].min()
        marker_offset = price_range * 0.002  # Very small offset (0.2% of price range)
        
        # Get bull trend starts
        if 'AO_BullTrend' in df.columns:
            bull_starts = df[df['AO_BullTrend']]
            if len(bull_starts) > 0:
                bull_indices = [df.index.get_loc(idx) for idx in bull_starts.index]
                # Place markers just below the low price
                bull_y_positions = [df['Low'].iloc[i] - marker_offset for i in bull_indices]
                
                # Plot green up triangles
                ax.scatter(bull_indices, bull_y_positions, color=self.COLORS['bullish'], 
                          marker='^', s=200, alpha=1.0, zorder=10, edgecolor='white', linewidth=2)
                
                # Add A++ text below markers
                for idx, y_pos in zip(bull_indices, bull_y_positions):
                    ax.text(idx, y_pos - marker_offset, 'A++', ha='center', va='top', 
                           color=self.COLORS['bullish'], fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=self.COLORS['bg'], 
                                   edgecolor=self.COLORS['bullish'], alpha=0.8))
        
        # Get bear trend starts
        if 'AO_BearTrend' in df.columns:
            bear_starts = df[df['AO_BearTrend']]
            if len(bear_starts) > 0:
                bear_indices = [df.index.get_loc(idx) for idx in bear_starts.index]
                # Place markers just above the high price
                bear_y_positions = [df['High'].iloc[i] + marker_offset for i in bear_indices]
                
                # Plot red down triangles
                ax.scatter(bear_indices, bear_y_positions, color=self.COLORS['bearish'], 
                          marker='v', s=200, alpha=1.0, zorder=10, edgecolor='white', linewidth=2)
                
                # Add A-- text above markers
                for idx, y_pos in zip(bear_indices, bear_y_positions):
                    ax.text(idx, y_pos + marker_offset, 'A--', ha='center', va='bottom', 
                           color=self.COLORS['bearish'], fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=self.COLORS['bg'], 
                                   edgecolor=self.COLORS['bearish'], alpha=0.8))

    def _plot_signals(self, ax, df, has_supertrend, has_market_bias, has_neurotrend):
        """Plot signal panel for standard indicators"""
        ax.set_facecolor(self.COLORS['bg'])
        x_pos = np.arange(len(df))
        st_signal = df['SuperTrend_Direction'] if has_supertrend else pd.Series(dtype=float)
        mb_signal = df['MB_Bias'] if has_market_bias else pd.Series(dtype=float)
        nt_signal = df['NT_TrendDirection'] if has_neurotrend else pd.Series(dtype=float)

        for i in range(len(df)):
            bullish_count, bearish_count = 0, 0
            if has_supertrend and not pd.isna(st_signal.iloc[i]):
                if st_signal.iloc[i] == 1: bullish_count += 1
                elif st_signal.iloc[i] == -1: bearish_count += 1
            if has_market_bias and not pd.isna(mb_signal.iloc[i]):
                if mb_signal.iloc[i] == 1: bullish_count += 1
                elif mb_signal.iloc[i] == -1: bearish_count += 1
            if has_neurotrend and not pd.isna(nt_signal.iloc[i]):
                if nt_signal.iloc[i] == 1: bullish_count += 1
                elif nt_signal.iloc[i] == -1: bearish_count += 1
            
            total_signals = sum([has_supertrend, has_market_bias, has_neurotrend])
            if total_signals > 0:
                if bullish_count == total_signals: ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=self.COLORS['bullish'], alpha=0.4)
                elif bearish_count == total_signals: ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=self.COLORS['bearish'], alpha=0.4)
                elif bullish_count > bearish_count: ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=self.COLORS['bullish'], alpha=0.2)
                elif bearish_count > bullish_count: ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=self.COLORS['bearish'], alpha=0.2)
                else: ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, color=self.COLORS['neutral'], alpha=0.1)

        if has_supertrend and not st_signal.isna().all(): ax.plot(x_pos[~st_signal.isna()], st_signal[~st_signal.isna()], color=self.COLORS['white'], linewidth=2, label='SuperTrend', alpha=0.9, drawstyle='steps-post')
        if has_market_bias and not mb_signal.isna().all(): ax.plot(x_pos[~mb_signal.isna()], mb_signal[~mb_signal.isna()], color=self.COLORS['orange'], linewidth=2, label='Market Bias', alpha=0.7, linestyle='--', drawstyle='steps-post')
        if has_neurotrend and not nt_signal.isna().all(): ax.plot(x_pos[~nt_signal.isna()], nt_signal[~nt_signal.isna()], color=self.COLORS['yellow'], linewidth=2, label='NeuroTrend', alpha=0.8, linestyle=':', drawstyle='steps-post')

        ax.set_ylabel('Signal', fontsize=10, color=self.COLORS['text'])
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.2, color=self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])
        if has_supertrend or has_market_bias or has_neurotrend: ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
        
        signal_count = sum([has_supertrend, has_market_bias, has_neurotrend])
        description = "Strong Green/Red = All Signals Agree | Light = Partial | Gray = No Confluence" if signal_count >= 2 else "Green = Bullish | Red = Bearish | Gray = Neutral"
        ax.text(0.5, 0.02, description, transform=ax.transAxes, fontsize=9, color=self.COLORS['text'], ha='center', va='bottom')
        self._format_date_axis(ax, df)

    def _format_date_axis(self, ax, df):
        """Format x-axis to show dates properly based on detected timeframe"""
        n_points = len(df)
        detected_timeframe = getattr(self, '_current_timeframe', None)

        if n_points == 0: return

        if detected_timeframe in ['1M', '5M', '15M', '30M', '1H', '4H']:
            max_labels = 10
        else:
            max_labels = 8
        
        step = max(1, n_points // max_labels)
        tick_positions = list(range(0, n_points, step))
        if n_points - 1 not in tick_positions:
            tick_positions.append(n_points - 1)

        date_labels = []
        prev_date_str = ""
        for pos in tick_positions:
            if pos < len(df):
                date = df.index[pos]
                if detected_timeframe in ['1D', 'D', '1W', 'W', '1M', 'M']:
                    current_date_str = date.strftime('%Y-%m-%d')
                else:
                    current_date_str = date.strftime('%Y-%m-%d %H:%M')
                
                if current_date_str.split(' ')[0] != prev_date_str.split(' ')[0]:
                    # Show date if it's a new day
                    if detected_timeframe and 'H' in detected_timeframe:
                        label = date.strftime('%b %d\n%H:%M')
                    else:
                        label = date.strftime('%Y\n%b %d')
                else:
                    # Otherwise, just show time
                    label = date.strftime('%H:%M')
                
                date_labels.append(label)
                prev_date_str = current_date_str

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(date_labels, rotation=0, ha='center', fontsize=9)
        ax.tick_params(axis='x', pad=3)
        ax.grid(True, which='major', axis='x', alpha=0.2)



# ============================================================================
# Combined NeuroTrend + Market Bias + Intelligent Chop Plot
# ============================================================================

def plot_neurotrend_market_bias_chop(df: pd.DataFrame,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (16, 12),
                                    save_path: Optional[Union[str, Path]] = None,
                                    show: bool = True,
                                    show_indicators: bool = True,
                                    show_chop_subplots: bool = True,
                                    use_chop_background: bool = False,
                                    single_plot_height_ratio: float = 0.5,
                                    main_height_ratio: float = 4,
                                    chop_height_ratio: float = 3,
                                    simplified_regime_colors: bool = False,
                                    trend_color: Union[str, Tuple[float, float, float]] = '#4ECDC4',
                                    range_color: Union[str, Tuple[float, float, float]] = '#95A5A6') -> plt.Figure:
    """
    Create a combined plot with NeuroTrend Intelligent and Market Bias overlay on top,
    and optionally Intelligent Chop indicator below.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and indicator columns
    title : str, optional
        Chart title (auto-generated if None)
    figsize : tuple, default=(16, 12)
        Figure size (width, height). When show_chop_subplots=False, height is 
        automatically adjusted by single_plot_height_ratio
    save_path : str or Path, optional
        Path to save the chart
    show : bool, default=True
        Whether to display the chart
    show_indicators : bool, default=True
        Whether to show the indicators subplot (ADX, Choppiness, Efficiency)
    show_chop_subplots : bool, default=True
        Whether to show the Intelligent Chop subplots at all
    use_chop_background : bool, default=False
        If True and show_chop_subplots is False, use Intelligent Chop regime colors 
        as background in the top plot instead of NeuroTrend colors
    single_plot_height_ratio : float, default=0.5
        Height ratio to apply when show_chop_subplots=False. For example, 0.5 means
        the figure height will be half of the original figsize[1]
    main_height_ratio : float, default=4
        Height ratio for the main (top) plot when showing multiple sections
    chop_height_ratio : float, default=3
        Height ratio for the Intelligent Chop section when showing subplots
    simplified_regime_colors : bool, default=False
        If True, uses only two colors: trend_color for Strong/Weak Trend, 
        range_color for Quiet Range/Volatile Chop
    trend_color : str or tuple, default='#4ECDC4' (turquoise)
        Color for trending regimes when simplified_regime_colors=True.
        Can be matplotlib color string ('red', 'blue') or RGB tuple (0.2, 0.4, 0.6)
    range_color : str or tuple, default='#95A5A6' (grey)
        Color for ranging/choppy regimes when simplified_regime_colors=True.
        Can be matplotlib color string or RGB tuple
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Color scheme
    COLORS = {
        'bg': '#131722',
        'grid': '#363c4e',
        'text': '#d1d4dc',
        'bullish': '#26A69A',
        'bearish': '#EF5350',
        'neutral': '#FFB74D',
        'strong_trend': '#81C784',
        'weak_trend': '#FFF176',
        'quiet_range': '#64B5F6',
        'volatile_chop': '#EF5350',
        'white': '#ffffff'
    }
    
    # Apply dark theme
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    # Adjust figsize if not showing chop subplots
    if not show_chop_subplots:
        # Single plot mode - reduce figure height by the specified ratio
        adjusted_height = figsize[1] * single_plot_height_ratio
        fig = plt.figure(figsize=(figsize[0], adjusted_height), constrained_layout=False)
        ax_price = fig.add_subplot(111)
        axes_list = [ax_price]
        ax_chop_price = None
        ax_regime = None
        ax_indicators = None
    else:
        # Define grid layout (2 main sections) with customizable ratios
        gs = fig.add_gridspec(2, 1, height_ratios=[main_height_ratio, chop_height_ratio], hspace=0.15)
        
        # Top section: NeuroTrend + Market Bias
        ax_price = fig.add_subplot(gs[0])
        
        # Bottom section: Intelligent Chop (create sub-grid)
        if show_indicators:
            gs_chop = gs[1].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
            ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
            ax_indicators = fig.add_subplot(gs_chop[2], sharex=ax_price)
            axes_list = [ax_price, ax_chop_price, ax_regime, ax_indicators]
        else:
            gs_chop = gs[1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
            ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
            ax_indicators = None
            axes_list = [ax_price, ax_chop_price, ax_regime]
    
    # Set background color
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in axes_list:
        ax.set_facecolor(COLORS['bg'])
    
    # === TOP SECTION: NeuroTrend + Market Bias ===
    x_pos = np.arange(len(df))
    
    # Determine NeuroTrend type and columns
    if 'NTI_Direction' in df.columns:
        # NeuroTrend Intelligent
        direction_col = 'NTI_Direction'
        confidence_col = 'NTI_Confidence'
        fast_ema_col = 'NTI_FastEMA'
        slow_ema_col = 'NTI_SlowEMA'
    elif 'NT3_Direction' in df.columns:
        # NeuroTrend 3-State
        direction_col = 'NT3_Direction'
        confidence_col = 'NT3_Confidence'
        fast_ema_col = 'NTI_FastEMA' if 'NTI_FastEMA' in df.columns else None
        slow_ema_col = 'NTI_SlowEMA' if 'NTI_SlowEMA' in df.columns else None
    else:
        raise ValueError("No NeuroTrend Intelligent data found in DataFrame")
    
    # Plot background coloring - either NeuroTrend or Intelligent Chop
    if not show_chop_subplots and use_chop_background and 'IC_RegimeName' in df.columns:
        # Use Intelligent Chop regime colors as background
        if simplified_regime_colors:
            # Simplified binary colors: trend vs range
            REGIME_COLORS = {
                'Strong Trend': trend_color,
                'Weak Trend': trend_color,
                'Quiet Range': range_color,
                'Volatile Chop': range_color,
                'Transitional': range_color
            }
        else:
            # Original 4-color scheme
            REGIME_COLORS = {
                'Strong Trend': '#81C784',   # Pastel Green
                'Weak Trend': '#FFF176',     # Pastel Yellow
                'Quiet Range': '#64B5F6',    # Pastel Blue
                'Volatile Chop': '#FFCDD2',  # Light Pastel Red
                'Transitional': '#E0E0E0'    # Light Grey
            }
        
        for i in range(len(df)):
            regime = df['IC_RegimeName'].iloc[i]
            color = REGIME_COLORS.get(regime, COLORS['neutral'])
            ax_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                            color=color, alpha=0.4, ec='none')
    else:
        # Use NeuroTrend background coloring
        for i in range(len(df)):
            direction = df[direction_col].iloc[i]
            if direction == 1:
                color = COLORS['bullish']
            elif direction == -1:
                color = COLORS['bearish']
            else:
                color = COLORS['neutral']
            ax_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                            color=color, alpha=0.15, ec='none')
    
    # Plot candlesticks
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    for idx in range(len(df)):
        close_price = closes[idx]
        open_price = opens[idx]
        high_price = highs[idx]
        low_price = lows[idx]
        
        color = COLORS['bullish'] if close_price >= open_price else COLORS['bearish']
        
        # Wicks
        ax_price.plot([x_pos[idx], x_pos[idx]], [low_price, high_price], 
                     color=color, linewidth=1, alpha=0.8)
        
        # Body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        if body_height < (df['Close'].mean() * 0.0001):
            body_height = df['Close'].mean() * 0.0001
        
        ax_price.add_patch(Rectangle((x_pos[idx] - 0.3, body_bottom), 0.6, body_height, 
                                    facecolor=color, edgecolor=color, alpha=0.8))
    
    # Plot Market Bias overlay if available
    if 'MB_Bias' in df.columns:
        valid_mask = ~(df['MB_o2'].isna() | df['MB_c2'].isna())
        mb_bias = df['MB_Bias'].values
        mb_o2 = df['MB_o2'].values
        mb_c2 = df['MB_c2'].values
        mb_h2 = df['MB_h2'].values
        mb_l2 = df['MB_l2'].values
        
        for i in np.where(valid_mask)[0]:
            mb_color = COLORS['bullish'] if mb_bias[i] == 1 else COLORS['bearish']
            ax_price.plot([x_pos[i], x_pos[i]], [mb_l2[i], mb_h2[i]], 
                         color=mb_color, linewidth=10, alpha=0.3, solid_capstyle='round')
            body_bottom = min(mb_o2[i], mb_c2[i])
            body_top = max(mb_o2[i], mb_c2[i])
            body_height = body_top - body_bottom
            if body_height < (df['Close'].mean() * 0.0001):
                body_height = df['Close'].mean() * 0.0001
            ax_price.add_patch(Rectangle((x_pos[i] - 0.4, body_bottom), 0.8, body_height, 
                                        facecolor=mb_color, edgecolor='none', alpha=0.4))
    
    # Plot NeuroTrend EMAs if available
    if fast_ema_col and slow_ema_col and fast_ema_col in df.columns and slow_ema_col in df.columns:
        fast_ema = df[fast_ema_col].values
        slow_ema = df[slow_ema_col].values
        valid_mask = ~np.isnan(fast_ema)
        
        if np.any(valid_mask):
            # Plot EMAs with color based on direction
            for i in range(len(df) - 1):
                if valid_mask[i] and valid_mask[i + 1]:
                    direction = df[direction_col].iloc[i]
                    if direction == 1:
                        color = COLORS['bullish']
                    elif direction == -1:
                        color = COLORS['bearish']
                    else:
                        color = COLORS['neutral']
                    
                    ax_price.plot([x_pos[i], x_pos[i + 1]], 
                                 [fast_ema[i], fast_ema[i + 1]], 
                                 color=color, linewidth=2, alpha=0.9)
                    ax_price.plot([x_pos[i], x_pos[i + 1]], 
                                 [slow_ema[i], slow_ema[i + 1]], 
                                 color=color, linewidth=2, alpha=0.7, linestyle='--')
    
    # Add confidence text
    if confidence_col in df.columns:
        latest_confidence = df[confidence_col].iloc[-1]
        ax_price.text(0.02, 0.98, f'Confidence: {latest_confidence:.1%}', 
                     transform=ax_price.transAxes, fontsize=10, 
                     color=COLORS['text'], va='top', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], alpha=0.8))
    
    # Add legend
    legend_elements = []
    if 'NTI_Direction' in df.columns or 'NT3_Direction' in df.columns:
        legend_elements.extend([
            Line2D([0], [0], color=COLORS['bullish'], lw=2, label='Uptrend'),
            Line2D([0], [0], color=COLORS['bearish'], lw=2, label='Downtrend')
        ])
        if 'NT3_Direction' in df.columns:
            legend_elements.append(Line2D([0], [0], color=COLORS['neutral'], lw=2, label='Neutral'))
    if 'MB_Bias' in df.columns:
        legend_elements.extend([
            mpatches.Patch(color=COLORS['bullish'], alpha=0.4, label='Bullish Bias'),
            mpatches.Patch(color=COLORS['bearish'], alpha=0.4, label='Bearish Bias')
        ])
    
    ax_price.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    ax_price.set_xlim(-1, len(df))
    ax_price.set_ylabel('Price', fontsize=11, color=COLORS['text'])
    ax_price.grid(True, alpha=0.3, color=COLORS['grid'])
    # Only hide x-axis labels if we're showing chop subplots
    ax_price.tick_params(colors=COLORS['text'], labelbottom=not show_chop_subplots)
    
    # === BOTTOM SECTION: Intelligent Chop ===
    if show_chop_subplots:
        if 'IC_RegimeName' not in df.columns:
            ax_chop_price.text(0.5, 0.5, 'Intelligent Chop data not available', 
                              transform=ax_chop_price.transAxes, 
                              ha='center', va='center', fontsize=12, color=COLORS['text'])
        else:
            # Regime color mapping
            if simplified_regime_colors:
                # Simplified binary colors: trend vs range
                REGIME_COLORS = {
                    'Strong Trend': trend_color,
                    'Weak Trend': trend_color,
                    'Quiet Range': range_color,
                    'Volatile Chop': range_color,
                    'Transitional': range_color
                }
            else:
                # Original 4-color scheme
                REGIME_COLORS = {
                    'Strong Trend': '#81C784',
                    'Weak Trend': '#FFF176',
                    'Quiet Range': '#64B5F6',
                    'Volatile Chop': '#FFCDD2',
                    'Transitional': '#E0E0E0'
                }
            
            # Plot regime background
            for i in range(len(df)):
                regime = df['IC_RegimeName'].iloc[i]
                color = REGIME_COLORS.get(regime, COLORS['white'])
                ax_chop_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                                     color=color, alpha=0.4, ec='none')
            
            # Plot simple price line
            ax_chop_price.plot(x_pos, closes, color=COLORS['white'], linewidth=1.5, alpha=0.9)
            
            # Regime bars
            regime_numeric = df['IC_Regime'].values
            colors_regime = []
            heights = []
            
            for i in range(len(df)):
                regime_name = df['IC_RegimeName'].iloc[i]
                colors_regime.append(REGIME_COLORS.get(regime_name, COLORS['white']))
                
                if regime_numeric[i] == 2:  # Strong Trend
                    heights.append(1.0)
                elif regime_numeric[i] == 1:  # Weak Trend
                    heights.append(0.7)
                elif regime_numeric[i] == 3:  # Quiet Range
                    heights.append(0.5)
                else:  # Choppy
                    heights.append(-0.8)
            
            ax_regime.bar(x_pos, heights, color=colors_regime, alpha=0.8, width=0.9)
            ax_regime.set_ylabel('Regime', fontsize=10, color=COLORS['text'])
            ax_regime.set_ylim(-1, 1.2)
            ax_regime.axhline(y=0, color=COLORS['white'], linestyle='-', alpha=0.3)
            ax_regime.grid(True, alpha=0.2, color=COLORS['grid'])
            # Only hide x-axis labels if indicators subplot is shown
            ax_regime.tick_params(colors=COLORS['text'], labelbottom=not show_indicators)
            
            # Technical indicators (only if show_indicators is True)
            if show_indicators and all(col in df.columns for col in ['IC_ADX', 'IC_ChoppinessIndex', 'IC_EfficiencyRatio']):
                ax_indicators.plot(x_pos, df['IC_ADX'], color='#FFB74D', linewidth=2, label='ADX', alpha=0.9)
                ax_indicators.plot(x_pos, df['IC_ChoppinessIndex'], color='#64B5F6', linewidth=2, label='Choppiness', alpha=0.9)
                ax_indicators.plot(x_pos, df['IC_EfficiencyRatio'] * 100, color='#81C784', linewidth=2, label='Efficiency', alpha=0.9)
                
                ax_indicators.axhline(y=25, color=COLORS['white'], linestyle='--', alpha=0.3)
                ax_indicators.axhline(y=61.8, color=COLORS['white'], linestyle='--', alpha=0.3)
                ax_indicators.set_ylabel('Indicators', fontsize=10, color=COLORS['text'])
                ax_indicators.set_ylim(0, 100)
                ax_indicators.legend(loc='upper left', fontsize=8, framealpha=0.9)
                ax_indicators.grid(True, alpha=0.2, color=COLORS['grid'])
                ax_indicators.tick_params(colors=COLORS['text'])
            
            # Chop legend
            if simplified_regime_colors:
                # Simplified legend with only two categories
                chop_legend = [
                    mpatches.Patch(color=trend_color, alpha=0.6, label='Trending'),
                    mpatches.Patch(color=range_color, alpha=0.6, label='Ranging/Choppy')
                ]
                ax_chop_price.legend(handles=chop_legend, loc='upper right', fontsize=8, 
                                    framealpha=0.9, ncol=2)
            else:
                # Original 4-category legend
                chop_legend = [
                    mpatches.Patch(color='#81C784', alpha=0.6, label='Strong Trend'),
                    mpatches.Patch(color='#FFF176', alpha=0.6, label='Weak Trend'),
                    mpatches.Patch(color='#64B5F6', alpha=0.6, label='Quiet Range'),
                    mpatches.Patch(color='#FFCDD2', alpha=0.6, label='Volatile Chop')
                ]
                ax_chop_price.legend(handles=chop_legend, loc='upper right', fontsize=8, 
                                    framealpha=0.9, ncol=2)
            
            ax_chop_price.set_xlim(-1, len(df))
            ax_chop_price.set_ylabel('Price', fontsize=10, color=COLORS['text'])
            ax_chop_price.grid(True, alpha=0.3, color=COLORS['grid'])
            ax_chop_price.tick_params(colors=COLORS['text'], labelbottom=False)
    
    # Format x-axis on the bottom-most axis
    if isinstance(df.index, pd.DatetimeIndex):
        # Determine which is the bottom-most axis
        if not show_chop_subplots:
            bottom_ax = ax_price
        elif show_indicators and ax_indicators is not None:
            bottom_ax = ax_indicators
        elif ax_regime is not None:
            bottom_ax = ax_regime
        else:
            bottom_ax = ax_price
        
        # Create x-axis positions for plotting (0 to len-1)
        x_dates = df.index
        n_ticks = min(8, len(df))  # Limit number of ticks for readability
        tick_positions = np.linspace(0, len(df)-1, n_ticks, dtype=int)
        
        # Set ticks and labels
        bottom_ax.set_xticks(tick_positions)
        tick_labels = [x_dates[i].strftime('%Y-%m-%d') for i in tick_positions]
        bottom_ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add title
    if title is None:
        symbol = df.index.name if df.index.name else "Market"
        title = f"{symbol} - NeuroTrend Intelligent with Market Bias & Intelligent Chop Analysis"
    ax_price.set_title(title, fontsize=14, color=COLORS['text'], pad=10)
    
    # Style spines
    for ax in axes_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Chart saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# ============================================================================
# Intelligent Chop Indicator Plotting Function
# ============================================================================

def plot_intelligent_chop_indicator(df: pd.DataFrame,
                                   title: Optional[str] = None,
                                   figsize: Tuple[int, int] = (16, 10),
                                   save_path: Optional[Union[str, Path]] = None,
                                   show: bool = True) -> plt.Figure:
    """
    Create a professional multi-panel chart for Intelligent Chop Indicator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and IC indicator columns
    title : str, optional
        Chart title (auto-generated if None)
    figsize : tuple, default=(16, 10)
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the chart
    show : bool, default=True
        Whether to display the chart
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'IC_RegimeName', 'IC_Confidence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column. Run intelligent_chop indicator first.")
    
    # Color scheme - BRIGHTER PASTEL COLORS
    COLORS = {
        'bg': '#131722',
        'grid': '#363c4e',
        'text': '#d1d4dc',
        'strong_trend': '#81C784',  # Bright Pastel Green
        'weak_trend': '#FFF176',    # Bright Pastel Yellow
        'quiet_range': '#64B5F6',   # Bright Pastel Blue
        'volatile_chop': '#EF5350', # Bright Pastel Red
        'transitional': '#BDBDBD',  # Bright Pastel Grey
        'white': '#ffffff',
        'orange': '#FFB74D',
        'candle_up': '#26A69A',     # Teal for up candles
        'candle_down': '#EF5350'    # Red for down candles
    }
    
    # Regime color mapping for backgrounds
    REGIME_COLORS = {
        'Strong Trend': '#81C784',   # Pastel Green
        'Weak Trend': '#FFF176',     # Pastel Yellow
        'Quiet Range': '#64B5F6',    # Pastel Blue
        'Volatile Chop': '#FFCDD2',  # Light Pastel Red
        'Transitional': '#E0E0E0'    # Light Grey
    }
    
    # Apply dark theme
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    # Define grid layout (4 rows with different heights)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.15)
    
    # Create subplots
    ax_price = fig.add_subplot(gs[0])    # Price chart with regime background
    ax_regime = fig.add_subplot(gs[1], sharex=ax_price)   # Regime bars
    ax_indicators = fig.add_subplot(gs[2], sharex=ax_price)  # ADX, CI, ER
    ax_confidence = fig.add_subplot(gs[3], sharex=ax_price)  # Confidence & signals
    
    # Set background color
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in [ax_price, ax_regime, ax_indicators, ax_confidence]:
        ax.set_facecolor(COLORS['bg'])
    
    # 1. Price Chart with Regime Background
    x_pos = np.arange(len(df))
    
    # Plot regime background - BRIGHT PASTEL COLORS
    for i in range(len(df)):
        regime = df['IC_RegimeName'].iloc[i]
        color = REGIME_COLORS.get(regime, COLORS['transitional'])
        alpha = 0.4  # Increased alpha for better visibility
        ax_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                        color=color, alpha=alpha, ec='none')
    
    # Plot candlesticks ONLY
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    for idx in range(len(df)):
        close_price = closes[idx]
        open_price = opens[idx]
        high_price = highs[idx]
        low_price = lows[idx]
        
        # Use specific candle colors
        color = COLORS['candle_up'] if close_price >= open_price else COLORS['candle_down']
        
        # Wicks
        ax_price.plot([x_pos[idx], x_pos[idx]], [low_price, high_price], 
                     color=color, linewidth=1.5, alpha=0.9)
        
        # Body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Ensure minimum body height for visibility
        if body_height < (df['Close'].mean() * 0.0001):
            body_height = df['Close'].mean() * 0.0001
        
        ax_price.add_patch(Rectangle((x_pos[idx] - 0.3, body_bottom), 0.6, body_height, 
                                    facecolor=color, edgecolor=color, alpha=0.9))
    
    # NO Bollinger Bands on price chart - keep it clean
    
    ax_price.set_xlim(-1, len(df))
    
    # IMPORTANT: Set Y-axis limits based on ONLY price data (no indicators)
    price_min = df['Low'].min()
    price_max = df['High'].max()
    
    # Add 3% margin to the price range for better visibility
    price_range = price_max - price_min
    margin = price_range * 0.03
    ax_price.set_ylim(price_min - margin, price_max + margin)
    
    ax_price.set_ylabel('Price', fontsize=11, color=COLORS['text'])
    ax_price.grid(True, alpha=0.3, color=COLORS['grid'])
    ax_price.tick_params(colors=COLORS['text'], labelbottom=False)
    
    # Add title
    if title is None:
        symbol = df.index.name if df.index.name else "Market"
        title = f"{symbol} - Intelligent Chop Indicator Analysis"
    ax_price.set_title(title, fontsize=14, color=COLORS['text'], pad=10)
    
    # Add compact legend for regime colors above the chart
    legend_elements = [
        mpatches.Patch(color='#81C784', alpha=0.6, label='Strong Trend'),
        mpatches.Patch(color='#FFF176', alpha=0.6, label='Weak Trend'),
        mpatches.Patch(color='#64B5F6', alpha=0.6, label='Quiet Range'),
        mpatches.Patch(color='#FFCDD2', alpha=0.6, label='Volatile Chop')
    ]
    
    # Place legend above the plot area, outside the axes
    ax_price.legend(handles=legend_elements, 
                   bbox_to_anchor=(0.5, 1.15), loc='upper center',
                   fontsize=8, framealpha=0.9, ncol=4, 
                   columnspacing=1.0, handlelength=1.5,
                   borderaxespad=0., frameon=True)
    
    # 2. Regime Bars
    regime_numeric = df['IC_Regime'].values
    colors_regime = []
    heights = []
    
    for i in range(len(df)):
        regime_name = df['IC_RegimeName'].iloc[i]
        colors_regime.append(REGIME_COLORS.get(regime_name, COLORS['transitional']))
        
        # Height based on regime type
        if regime_numeric[i] == 2:  # Strong Trend
            heights.append(1.0)
        elif regime_numeric[i] == 1:  # Weak Trend
            heights.append(0.7)
        elif regime_numeric[i] == 3:  # Quiet Range
            heights.append(0.5)
        else:  # Choppy
            heights.append(-0.8)
    
    bars = ax_regime.bar(x_pos, heights, color=colors_regime, alpha=0.8, width=0.9)
    
    ax_regime.set_ylabel('Regime', fontsize=10, color=COLORS['text'])
    ax_regime.set_ylim(-1, 1.2)
    ax_regime.axhline(y=0, color=COLORS['white'], linestyle='-', alpha=0.3)
    ax_regime.grid(True, alpha=0.2, color=COLORS['grid'])
    ax_regime.tick_params(colors=COLORS['text'], labelbottom=False)
    
    # Add regime labels
    ax_regime.text(0.02, 0.9, 'Strong Trend', transform=ax_regime.transAxes, 
                  fontsize=8, color=COLORS['strong_trend'])
    ax_regime.text(0.02, 0.5, 'Range', transform=ax_regime.transAxes, 
                  fontsize=8, color=COLORS['quiet_range'])
    ax_regime.text(0.02, 0.1, 'Chop', transform=ax_regime.transAxes, 
                  fontsize=8, color=COLORS['volatile_chop'])
    
    # 3. Technical Indicators Panel
    # ADX
    ax_indicators.plot(x_pos, df['IC_ADX'], color=COLORS['white'], 
                      linewidth=2, label='ADX', alpha=0.9)
    ax_indicators.axhline(y=25, color=COLORS['white'], linestyle=':', 
                         alpha=0.3, label='Trend Threshold')
    ax_indicators.axhline(y=20, color=COLORS['volatile_chop'], linestyle=':', 
                         alpha=0.3, label='Choppy Threshold')
    
    # Choppiness Index
    ax_indicators.plot(x_pos, df['IC_ChoppinessIndex'], color=COLORS['orange'], 
                      linewidth=1.5, label='Choppiness Index', alpha=0.8)
    ax_indicators.axhline(y=61.8, color=COLORS['orange'], linestyle=':', alpha=0.3)
    ax_indicators.axhline(y=38.2, color=COLORS['orange'], linestyle=':', alpha=0.3)
    
    # Efficiency Ratio (scaled to 0-100)
    ax_indicators.plot(x_pos, df['IC_EfficiencyRatio'] * 100, color=COLORS['quiet_range'], 
                      linewidth=1.5, label='Efficiency Ratio', alpha=0.8)
    
    # Add Bollinger Band Width here (scaled to 0-100 range)
    if 'IC_BandWidth' in df.columns:
        # Scale BandWidth to fit nicely in 0-100 range
        bw_scaled = df['IC_BandWidth'] * 5  # Adjust scaling factor as needed
        ax_indicators.plot(x_pos, bw_scaled, color='#9C27B0', 
                          linewidth=1.5, label='BB Width (scaled)', alpha=0.7)
    
    ax_indicators.set_ylabel('Indicators', fontsize=10, color=COLORS['text'])
    ax_indicators.set_ylim(0, 100)
    ax_indicators.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax_indicators.grid(True, alpha=0.2, color=COLORS['grid'])
    ax_indicators.tick_params(colors=COLORS['text'], labelbottom=False)
    
    # 4. Confidence & Risk Panel
    # Confidence as area chart
    ax_confidence.fill_between(x_pos, 0, df['IC_Confidence'], 
                             color=COLORS['strong_trend'], alpha=0.3, label='Confidence')
    ax_confidence.plot(x_pos, df['IC_Confidence'], color=COLORS['strong_trend'], 
                      linewidth=2, alpha=0.9)
    
    # Risk level as background
    risk_colors = {
        'Low': COLORS['strong_trend'],
        'Medium': COLORS['weak_trend'],
        'High': COLORS['volatile_chop']
    }
    
    for i in range(len(df)):
        risk = df['IC_RiskLevel'].iloc[i]
        color = risk_colors.get(risk, COLORS['transitional'])
        ax_confidence.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                            ymin=0.85, ymax=0.95, color=color, alpha=0.5)
    
    ax_confidence.set_ylabel('Confidence %', fontsize=10, color=COLORS['text'])
    ax_confidence.set_ylim(0, 105)
    ax_confidence.grid(True, alpha=0.2, color=COLORS['grid'])
    ax_confidence.tick_params(colors=COLORS['text'])
    
    # Add risk legend
    ax_confidence.text(0.02, 0.85, 'Risk:', transform=ax_confidence.transAxes, 
                      fontsize=8, color=COLORS['text'])
    ax_confidence.text(0.08, 0.85, 'Low', transform=ax_confidence.transAxes, 
                      fontsize=8, color=COLORS['strong_trend'])
    ax_confidence.text(0.13, 0.85, 'Med', transform=ax_confidence.transAxes, 
                      fontsize=8, color=COLORS['weak_trend'])
    ax_confidence.text(0.18, 0.85, 'High', transform=ax_confidence.transAxes, 
                      fontsize=8, color=COLORS['volatile_chop'])
    
    # Format x-axis dates
    _format_date_axis(ax_confidence, df)
    
    # Style all axes
    for ax in [ax_price, ax_regime, ax_indicators, ax_confidence]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
    
    # Add summary statistics
    total_bars = len(df)
    regime_counts = df['IC_RegimeName'].value_counts()
    
    summary_text = f"Analysis Period: {total_bars} bars | "
    for regime, count in regime_counts.items():
        pct = count / total_bars * 100
        summary_text += f"{regime}: {pct:.1f}% | "
    summary_text = summary_text.rstrip(' | ')
    
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=9, 
            color=COLORS['text'], transform=fig.transFigure)
    
    # Adjust spacing - more room at top for legend
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=COLORS['bg'], edgecolor='none')
        print(f"Chart saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def _format_date_axis(ax, df):
    """Format x-axis to show dates properly."""
    n_points = len(df)
    
    if n_points == 0:
        return
    
    # Determine number of labels based on data size
    if n_points <= 50:
        max_labels = 10
    elif n_points <= 200:
        max_labels = 8
    else:
        max_labels = 10
    
    step = max(1, n_points // max_labels)
    tick_positions = list(range(0, n_points, step))
    
    # Always include the last position
    if tick_positions[-1] != n_points - 1:
        tick_positions.append(n_points - 1)
    
    # Format dates
    date_labels = []
    for pos in tick_positions:
        if pos < len(df):
            date = df.index[pos]
            if hasattr(date, 'strftime'):
                # Determine format based on data frequency
                if n_points <= 100:
                    date_str = date.strftime('%Y-%m-%d\n%H:%M')
                elif n_points <= 500:
                    date_str = date.strftime('%m/%d\n%H:%M')
                else:
                    date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            date_labels.append(date_str)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='x', pad=3)

