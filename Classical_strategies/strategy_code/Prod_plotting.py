"""
Production Strategy Plotting Module
Enhanced plotting functionality for the production trading strategy

Author: Trading System
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .Prod_strategy import Trade, TradeDirection, ExitReason


# ============================================================================
# Plotting Configuration
# ============================================================================

class PlotConfig:
    """Configuration for plotting styling"""
    
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
    
    # Trade marker colors
    TRADE_COLORS = {
        'long_entry': '#43A047',  # Green
        'short_entry': '#E53935',  # Red
        'take_profit': '#43A047',  # Green
        'tp1_pullback': '#4CAF50',  # Light Green
        'stop_loss': '#E53935',  # Red
        'trailing_stop': '#9C27B0',  # Purple
        'signal_flip': '#FF8F00',  # Orange
        'end_of_data': '#9E9E9E'  # Grey
    }


# ============================================================================
# Data Statistics Calculator
# ============================================================================

class DataStatsCalculator:
    """Calculate data statistics for display"""
    
    @staticmethod
    def calculate_data_stats(df: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
        """Calculate comprehensive data statistics"""
        stats = {}
        
        # Basic counts
        stats['total_rows'] = len(df)
        
        # Time period analysis
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index[0]
            end_date = df.index[-1]
            total_duration = end_date - start_date
            
            # Calculate days and months
            stats['total_days'] = total_duration.days
            stats['total_months'] = round(total_duration.days / 30.44, 1)  # Average days per month
            stats['start_date'] = start_date.strftime('%Y-%m-%d')
            stats['end_date'] = end_date.strftime('%Y-%m-%d')
            
            # Calculate timeframe if regular intervals
            if len(df) > 1:
                time_diff = df.index[1] - df.index[0]
                if time_diff.total_seconds() == 900:  # 15 minutes
                    stats['timeframe'] = '15M'
                elif time_diff.total_seconds() == 3600:  # 1 hour
                    stats['timeframe'] = '1H'
                elif time_diff.total_seconds() == 14400:  # 4 hours
                    stats['timeframe'] = '4H'
                elif time_diff.total_seconds() == 86400:  # 1 day
                    stats['timeframe'] = '1D'
                else:
                    minutes = int(time_diff.total_seconds() / 60)
                    stats['timeframe'] = f'{minutes}M'
        else:
            stats['timeframe'] = 'Unknown'
            stats['total_days'] = 'N/A'
            stats['total_months'] = 'N/A'
        
        return stats


# ============================================================================
# Main Plotting Class
# ============================================================================

class ProductionPlotter:
    """Enhanced plotting for production strategy"""
    
    def __init__(self):
        self.config = PlotConfig()
        self.stats_calculator = DataStatsCalculator()
    
    def plot_strategy_results(self, 
                            df: pd.DataFrame,
                            results: Dict,
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (20, 14),
                            save_path: Optional[Union[str, Path]] = None,
                            show: bool = True,
                            show_chop_subplots: bool = False,
                            use_chop_background: bool = True,
                            single_plot_height_ratio: float = 0.85,
                            simplified_regime_colors: bool = True,
                            trend_color: str = '#2ECC71',
                            range_color: str = '#95A5A6',
                            show_pnl: bool = True,
                            show_position_sizes: bool = False) -> plt.Figure:
        """
        Create comprehensive strategy results plot
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data and indicators
        results : Dict
            Strategy backtest results
        title : str, optional
            Chart title
        figsize : tuple
            Figure size (width, height)
        save_path : str or Path, optional
            Path to save the chart
        show : bool
            Whether to display the chart
        show_chop_subplots : bool
            Whether to show Intelligent Chop subplots
        use_chop_background : bool
            Use Intelligent Chop colors as background
        single_plot_height_ratio : float
            Height ratio when not showing chop subplots
        simplified_regime_colors : bool
            Use simplified regime colors
        trend_color : str
            Color for trending regimes
        range_color : str
            Color for ranging regimes
        show_pnl : bool
            Whether to show P&L subplot
        show_position_sizes : bool
            Whether to show position sizes subplot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Apply dark theme
        plt.style.use('dark_background')
        
        # Calculate data statistics
        data_stats = self.stats_calculator.calculate_data_stats(df)
        
        # Create figure and subplots
        fig, axes = self._create_subplots(
            figsize, show_chop_subplots, show_pnl, 
            show_position_sizes, len(df), results.get('trades', [])
        )
        
        # Set background colors
        fig.patch.set_facecolor(self.config.COLORS['bg'])
        for ax in axes:
            ax.set_facecolor(self.config.COLORS['bg'])
        
        # Plot main price chart
        self._plot_price_chart(
            axes[0], df, use_chop_background, simplified_regime_colors,
            trend_color, range_color, results.get('trades', [])
        )
        
        # Plot position sizes if requested
        if show_position_sizes and len(axes) > 1:
            ax_positions = self._find_positions_axis(axes, show_pnl, show_chop_subplots)
            if ax_positions is not None:
                self._plot_position_sizes(ax_positions, df, results.get('trades', []))
        
        # Plot P&L if requested
        if show_pnl and len(axes) > 1:
            ax_pnl = self._find_pnl_axis(axes, show_position_sizes, show_chop_subplots)
            if ax_pnl is not None:
                self._plot_pnl(ax_pnl, df, results.get('trades', []))
        
        # Plot Intelligent Chop if requested
        if show_chop_subplots:
            self._plot_chop_subplots(axes, df, simplified_regime_colors, trend_color, range_color)
        
        # Add title and performance metrics
        self._add_title_and_metrics(
            axes[0], title, results, data_stats
        )
        
        # Format x-axis
        self._format_x_axis(axes, df, show_chop_subplots, show_pnl, show_position_sizes)
        
        # Style spines
        self._style_spines(axes)
        
        plt.tight_layout()
        
        # Save and show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.config.COLORS['bg'])
            print(f"Chart saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _create_subplots(self, figsize, show_chop_subplots, show_pnl, 
                        show_position_sizes, data_len, trades):
        """Create subplot layout"""
        if not show_chop_subplots:
            # Single plot mode
            subplot_count = 1
            height_ratios = [3]
            
            if show_position_sizes and trades:
                subplot_count += 1
                height_ratios.append(0.8)
            
            if show_pnl and trades:
                subplot_count += 1
                height_ratios.append(1)
            
            if subplot_count == 1:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                return fig, [ax]
            else:
                adjusted_height = figsize[1] * (0.85 if subplot_count > 1 else 1.0)
                fig, axes = plt.subplots(
                    subplot_count, 1, 
                    figsize=(figsize[0], adjusted_height),
                    height_ratios=height_ratios,
                    sharex=True
                )
                return fig, axes if isinstance(axes, np.ndarray) else [axes]
        else:
            # Multiple subplots mode with chop
            # Implementation for chop subplots would go here
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            return fig, [ax]
    
    def _find_positions_axis(self, axes, show_pnl, show_chop_subplots):
        """Find the positions subplot axis"""
        if len(axes) < 2:
            return None
        
        if not show_chop_subplots:
            if show_pnl:
                return axes[1] if len(axes) > 2 else None
            else:
                return axes[1]
        
        return None
    
    def _find_pnl_axis(self, axes, show_position_sizes, show_chop_subplots):
        """Find the P&L subplot axis"""
        if len(axes) < 2:
            return None
        
        if not show_chop_subplots:
            if show_position_sizes:
                return axes[2] if len(axes) > 2 else None
            else:
                return axes[1]
        
        return None
    
    def _plot_price_chart(self, ax, df, use_chop_background, simplified_regime_colors,
                         trend_color, range_color, trades):
        """Plot the main price chart with trades"""
        x_pos = np.arange(len(df))
        
        # Plot background coloring
        self._plot_background_colors(
            ax, df, x_pos, use_chop_background, simplified_regime_colors,
            trend_color, range_color
        )
        
        # Plot candlesticks
        self._plot_candlesticks(ax, df, x_pos)
        
        # Plot Market Bias overlay
        self._plot_market_bias_overlay(ax, df, x_pos)
        
        # Plot NeuroTrend EMAs
        self._plot_neurotrend_emas(ax, df, x_pos)
        
        # Plot trades
        if trades:
            self._plot_trades(ax, df, x_pos, trades)
        
        # Add confidence text
        self._add_confidence_text(ax, df)
        
        # Add legend
        self._add_main_legend(ax, df, trades)
        
        # Format main chart
        ax.set_xlim(-1, len(df))
        ax.set_ylabel('Price', fontsize=11, color=self.config.COLORS['text'])
        ax.grid(True, alpha=0.3, color=self.config.COLORS['grid'])
    
    def _plot_background_colors(self, ax, df, x_pos, use_chop_background, 
                               simplified_regime_colors, trend_color, range_color):
        """Plot background colors based on regime or NeuroTrend"""
        # Determine NeuroTrend columns
        if 'NTI_Direction' in df.columns:
            direction_col = 'NTI_Direction'
        elif 'NT3_Direction' in df.columns:
            direction_col = 'NT3_Direction'
        else:
            return
        
        if use_chop_background and 'IC_RegimeName' in df.columns:
            # Use Intelligent Chop colors
            if simplified_regime_colors:
                regime_colors = {
                    'Strong Trend': trend_color,
                    'Weak Trend': trend_color,
                    'Quiet Range': range_color,
                    'Volatile Chop': range_color,
                    'Transitional': range_color
                }
            else:
                regime_colors = {
                    'Strong Trend': '#81C784',
                    'Weak Trend': '#FFF176',
                    'Quiet Range': '#64B5F6',
                    'Volatile Chop': '#FFCDD2',
                    'Transitional': '#E0E0E0'
                }
            
            for i in range(len(df)):
                regime = df['IC_RegimeName'].iloc[i]
                color = regime_colors.get(regime, self.config.COLORS['neutral'])
                ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                          color=color, alpha=0.4, ec='none')
        else:
            # Use NeuroTrend colors
            for i in range(len(df)):
                direction = df[direction_col].iloc[i]
                if direction == 1:
                    color = self.config.COLORS['bullish']
                elif direction == -1:
                    color = self.config.COLORS['bearish']
                else:
                    color = self.config.COLORS['neutral']
                ax.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                          color=color, alpha=0.15, ec='none')
    
    def _plot_candlesticks(self, ax, df, x_pos):
        """Plot candlestick chart"""
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        for idx in range(len(df)):
            close_price = closes[idx]
            open_price = opens[idx]
            high_price = highs[idx]
            low_price = lows[idx]
            
            color = (self.config.COLORS['bullish'] if close_price >= open_price 
                    else self.config.COLORS['bearish'])
            
            # Wicks
            ax.plot([x_pos[idx], x_pos[idx]], [low_price, high_price], 
                   color=color, linewidth=1, alpha=0.8)
            
            # Body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height < (df['Close'].mean() * 0.0001):
                body_height = df['Close'].mean() * 0.0001
            
            ax.add_patch(Rectangle((x_pos[idx] - 0.3, body_bottom), 0.6, body_height, 
                                  facecolor=color, edgecolor=color, alpha=0.8))
    
    def _plot_market_bias_overlay(self, ax, df, x_pos):
        """Plot Market Bias overlay"""
        if 'MB_Bias' not in df.columns:
            return
        
        valid_mask = ~(df['MB_o2'].isna() | df['MB_c2'].isna())
        mb_bias = df['MB_Bias'].values
        mb_o2 = df['MB_o2'].values
        mb_c2 = df['MB_c2'].values
        mb_h2 = df['MB_h2'].values
        mb_l2 = df['MB_l2'].values
        
        for i in np.where(valid_mask)[0]:
            mb_color = (self.config.COLORS['bullish'] if mb_bias[i] == 1 
                       else self.config.COLORS['bearish'])
            ax.plot([x_pos[i], x_pos[i]], [mb_l2[i], mb_h2[i]], 
                   color=mb_color, linewidth=10, alpha=0.3, solid_capstyle='round')
            
            body_bottom = min(mb_o2[i], mb_c2[i])
            body_top = max(mb_o2[i], mb_c2[i])
            body_height = body_top - body_bottom
            
            if body_height < (df['Close'].mean() * 0.0001):
                body_height = df['Close'].mean() * 0.0001
            
            ax.add_patch(Rectangle((x_pos[i] - 0.4, body_bottom), 0.8, body_height, 
                                  facecolor=mb_color, edgecolor='none', alpha=0.4))
    
    def _plot_neurotrend_emas(self, ax, df, x_pos):
        """Plot NeuroTrend EMAs"""
        # Determine column names
        if 'NTI_FastEMA' in df.columns and 'NTI_SlowEMA' in df.columns:
            fast_ema_col = 'NTI_FastEMA'
            slow_ema_col = 'NTI_SlowEMA'
            direction_col = 'NTI_Direction'
        else:
            return
        
        fast_ema = df[fast_ema_col].values
        slow_ema = df[slow_ema_col].values
        valid_mask = ~np.isnan(fast_ema)
        
        if not np.any(valid_mask):
            return
        
        # Plot EMAs with color based on direction
        for i in range(len(df) - 1):
            if valid_mask[i] and valid_mask[i + 1]:
                direction = df[direction_col].iloc[i]
                if direction == 1:
                    color = self.config.COLORS['bullish']
                elif direction == -1:
                    color = self.config.COLORS['bearish']
                else:
                    color = self.config.COLORS['neutral']
                
                ax.plot([x_pos[i], x_pos[i + 1]], 
                       [fast_ema[i], fast_ema[i + 1]], 
                       color=color, linewidth=2, alpha=0.9)
                ax.plot([x_pos[i], x_pos[i + 1]], 
                       [slow_ema[i], slow_ema[i + 1]], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
    
    def _plot_trades(self, ax, df, x_pos, trades):
        """Plot trade markers and levels"""
        for trade in trades:
            # Convert Trade object to dict if needed
            if hasattr(trade, '__dict__'):
                trade_dict = {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'direction': trade.direction.value if isinstance(trade.direction, TradeDirection) else trade.direction,
                    'exit_reason': trade.exit_reason.value if isinstance(trade.exit_reason, ExitReason) else trade.exit_reason,
                    'take_profits': trade.take_profits,
                    'stop_loss': trade.stop_loss,
                    'partial_exits': trade.partial_exits
                }
            else:
                trade_dict = trade
            
            # Find indices
            entry_idx = self._find_time_index(df, trade_dict['entry_time'])
            exit_idx = self._find_time_index(df, trade_dict['exit_time'])
            
            # Plot entry marker
            if entry_idx is not None:
                self._plot_entry_marker(ax, x_pos, entry_idx, trade_dict)
            
            # Plot exit marker
            if exit_idx is not None:
                self._plot_exit_marker(ax, x_pos, exit_idx, trade_dict)
            
            # Plot TP/SL levels
            if entry_idx is not None:
                self._plot_trade_levels(ax, x_pos, entry_idx, exit_idx, trade_dict, len(df))
            
            # Plot partial exits
            if trade_dict.get('partial_exits'):
                self._plot_partial_exits(ax, df, x_pos, trade_dict)
    
    def _plot_position_sizes(self, ax, df, trades):
        """Plot position sizes over time"""
        x_pos = np.arange(len(df))
        position_sizes = np.zeros(len(df))
        
        # Build position timeline
        position_events = []
        
        for trade in trades:
            entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
            exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time')
            position_size = trade.position_size if hasattr(trade, 'position_size') else trade.get('position_size', 1000000)
            direction = trade.direction if hasattr(trade, 'direction') else trade.get('direction')
            partial_exits = trade.partial_exits if hasattr(trade, 'partial_exits') else trade.get('partial_exits', [])
            
            # Convert direction to string if needed
            if hasattr(direction, 'value'):
                direction = direction.value
            
            entry_idx = self._find_time_index(df, entry_time)
            
            if entry_idx is not None:
                position_events.append({
                    'idx': entry_idx,
                    'size': position_size / 1000000,
                    'direction': direction,
                    'event': 'entry'
                })
                
                # Add partial exits
                remaining_size = position_size
                for partial_exit in partial_exits:
                    partial_time = partial_exit.time if hasattr(partial_exit, 'time') else partial_exit['time']
                    partial_size = partial_exit.size if hasattr(partial_exit, 'size') else partial_exit.get('size', position_size / 3)
                    
                    partial_idx = self._find_time_index(df, partial_time)
                    if partial_idx is not None:
                        remaining_size -= partial_size
                        position_events.append({
                            'idx': partial_idx,
                            'size': remaining_size / 1000000,
                            'direction': direction,
                            'event': 'partial_exit'
                        })
                
                # Add final exit
                if exit_time:
                    exit_idx = self._find_time_index(df, exit_time)
                    if exit_idx is not None:
                        position_events.append({
                            'idx': exit_idx,
                            'size': 0,
                            'direction': direction,
                            'event': 'exit'
                        })
        
        # Sort by index
        position_events.sort(key=lambda x: x['idx'])
        
        # Build position size array
        current_positions = {}
        for event in position_events:
            idx = event['idx']
            
            if event['event'] == 'entry':
                trade_id = f"{idx}_{event['direction']}"
                current_positions[trade_id] = event['size']
            elif event['event'] in ['partial_exit', 'exit']:
                # Find corresponding trade
                for trade_id in list(current_positions.keys()):
                    if event['direction'] in trade_id:
                        if event['event'] == 'exit':
                            del current_positions[trade_id]
                        else:
                            current_positions[trade_id] = event['size']
                        break
            
            # Update position size array
            total_long = sum(size for tid, size in current_positions.items() if 'long' in tid)
            total_short = sum(size for tid, size in current_positions.items() if 'short' in tid)
            net_position = total_long - total_short
            
            position_sizes[idx:] = net_position
        
        # Plot bars
        colors = ['#43A047' if size > 0 else '#E53935' if size < 0 else self.config.COLORS['grid'] 
                 for size in position_sizes]
        ax.bar(x_pos, position_sizes, color=colors, alpha=0.7, width=0.9)
        
        # Format axis
        ax.axhline(y=0, color=self.config.COLORS['white'], linestyle='-', linewidth=1, alpha=0.5)
        ax.set_ylabel('Position Size (M)', fontsize=10, color=self.config.COLORS['text'])
        ax.set_xlim(-1, len(df))
        ax.grid(True, alpha=0.2, color=self.config.COLORS['grid'])
        
        # Add stats
        max_pos = np.max(np.abs(position_sizes))
        avg_pos = np.mean(np.abs(position_sizes[position_sizes != 0])) if np.any(position_sizes != 0) else 0
        
        ax.text(0.02, 0.95, f'Max: {max_pos:.1f}M', 
               transform=ax.transAxes, fontsize=8, 
               color=self.config.COLORS['text'], va='top')
        ax.text(0.02, 0.75, f'Avg: {avg_pos:.1f}M', 
               transform=ax.transAxes, fontsize=8, 
               color=self.config.COLORS['text'], va='top')
    
    def _plot_pnl(self, ax, df, trades):
        """Plot cumulative P&L"""
        x_pos = np.arange(len(df))
        cumulative_pnl = [0]
        pnl_times = [0]
        
        # Collect P&L events
        pnl_events = []
        
        for trade in trades:
            # Add partial exits
            partial_exits = trade.partial_exits if hasattr(trade, 'partial_exits') else trade.get('partial_exits', [])
            for partial_exit in partial_exits:
                partial_time = partial_exit.time if hasattr(partial_exit, 'time') else partial_exit['time']
                partial_pnl = partial_exit.pnl if hasattr(partial_exit, 'pnl') else partial_exit['pnl']
                pnl_events.append({
                    'time': partial_time,
                    'pnl': partial_pnl,
                    'type': 'partial'
                })
            
            # Add final exit
            exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time')
            if exit_time:
                # Calculate remaining P&L
                total_pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
                partial_pnl_sum = sum(pe.pnl if hasattr(pe, 'pnl') else pe['pnl'] for pe in partial_exits)
                remaining_pnl = total_pnl - partial_pnl_sum
                
                if remaining_pnl != 0:
                    pnl_events.append({
                        'time': exit_time,
                        'pnl': remaining_pnl,
                        'type': 'final'
                    })
        
        # Sort events by time
        pnl_events.sort(key=lambda x: x['time'])
        
        # Build cumulative P&L
        for event in pnl_events:
            event_idx = self._find_time_index(df, event['time'])
            if event_idx is not None:
                cumulative_pnl.append(cumulative_pnl[-1] + event['pnl'])
                pnl_times.append(event_idx)
        
        # Extend to end
        if pnl_times[-1] < len(df) - 1:
            pnl_times.append(len(df) - 1)
            cumulative_pnl.append(cumulative_pnl[-1])
        
        # Plot
        ax.step(pnl_times, cumulative_pnl, where='post', color='#FFD700', linewidth=2, alpha=0.9)
        ax.fill_between(pnl_times, 0, cumulative_pnl, step='post', alpha=0.3, 
                       color='#43A047' if cumulative_pnl[-1] >= 0 else '#E53935')
        
        # Zero line
        ax.axhline(y=0, color=self.config.COLORS['white'], linestyle='-', linewidth=1, alpha=0.5)
        
        # Format
        ax.set_ylabel('Cumulative P&L ($)', fontsize=10, color=self.config.COLORS['text'])
        ax.set_xlim(-1, len(df))
        ax.grid(True, alpha=0.2, color=self.config.COLORS['grid'])
        
        # Add stats
        final_pnl = cumulative_pnl[-1]
        max_pnl = max(cumulative_pnl)
        min_pnl = min(cumulative_pnl)
        
        ax.text(0.02, 0.95, f'Final P&L: ${final_pnl:.2f}', 
               transform=ax.transAxes, fontsize=9, 
               color='#43A047' if final_pnl >= 0 else '#E53935', va='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=self.config.COLORS['bg'], alpha=0.8))
        ax.text(0.02, 0.75, f'Max: ${max_pnl:.2f}', 
               transform=ax.transAxes, fontsize=8, 
               color=self.config.COLORS['text'], va='top')
        ax.text(0.02, 0.55, f'Min: ${min_pnl:.2f}', 
               transform=ax.transAxes, fontsize=8, 
               color=self.config.COLORS['text'], va='top')
    
    def _add_title_and_metrics(self, ax, title, results, data_stats):
        """Add title and performance metrics with data statistics"""
        if title is None:
            title = "Production Strategy Results"
        
        ax.set_title(title, fontsize=14, color=self.config.COLORS['text'], pad=30)
        
        if results:
            # Extract key metrics
            metrics_text = []
            
            # First row - Performance metrics
            if 'win_rate' in results:
                metrics_text.append(f"Win Rate: {results['win_rate']:.1f}%")
            if 'sharpe_ratio' in results:
                metrics_text.append(f"Sharpe: {results['sharpe_ratio']:.2f}")
            if 'profit_factor' in results:
                metrics_text.append(f"PF: {results['profit_factor']:.2f}")
            
            # Second row - P&L and returns
            if 'total_pnl' in results:
                metrics_text.append(f"P&L: ${results['total_pnl']:,.0f}")
            if 'total_return' in results:
                metrics_text.append(f"Return: {results['total_return']:.2f}%")
            if 'max_drawdown' in results:
                metrics_text.append(f"DD: {results['max_drawdown']:.2f}%")
            
            # Third row - Data statistics
            data_text = []
            data_text.append(f"Rows: {data_stats['total_rows']:,}")
            
            # Show either days or months based on duration
            if data_stats['total_days'] != 'N/A' and data_stats['total_months'] != 'N/A':
                if data_stats['total_months'] >= 1.0:
                    data_text.append(f"Period: {data_stats['total_months']} months")
                else:
                    data_text.append(f"Period: {data_stats['total_days']} days")
            
            if 'timeframe' in data_stats:
                data_text.append(f"TF: {data_stats['timeframe']}")
            
            # Create three rows
            row1 = "  |  ".join(metrics_text[:3]) if len(metrics_text) >= 3 else "  |  ".join(metrics_text)
            row2 = "  |  ".join(metrics_text[3:6]) if len(metrics_text) > 3 else ""
            row3 = "  |  ".join(data_text)
            
            # Add metrics text
            y_pos = 0.99
            for row in [row1, row2, row3]:
                if row:
                    ax.text(0.5, y_pos, row, 
                           transform=ax.transAxes, 
                           fontsize=10, 
                           color=self.config.COLORS['text'], 
                           ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.4', 
                                   facecolor=self.config.COLORS['bg'], 
                                   edgecolor=self.config.COLORS['grid'],
                                   alpha=0.9))
                    y_pos -= 0.04
    
    def _find_time_index(self, df, timestamp):
        """Find index for a given timestamp"""
        if timestamp is None:
            return None
        
        for i, ts in enumerate(df.index):
            if ts == timestamp:
                return i
        return None
    
    def _plot_entry_marker(self, ax, x_pos, entry_idx, trade_dict):
        """Plot trade entry marker with position size"""
        direction = trade_dict['direction']
        entry_price = trade_dict['entry_price']
        
        # Get position size in millions
        position_size = trade_dict.get('position_size', 1000000) / 1000000
        if hasattr(trade_dict.get('position_size'), 'value'):
            position_size = trade_dict['position_size'].value / 1000000
        
        if direction == 'long':
            ax.scatter(x_pos[entry_idx], entry_price, 
                      marker='^', s=200, color=self.config.TRADE_COLORS['long_entry'], 
                      edgecolor='white', linewidth=2, zorder=5)
        else:
            ax.scatter(x_pos[entry_idx], entry_price, 
                      marker='v', s=200, color=self.config.TRADE_COLORS['short_entry'], 
                      edgecolor='white', linewidth=2, zorder=5)
        
        # Add position size annotation
        ax.text(x_pos[entry_idx] - 0.5, entry_price, 
               f'{position_size:.1f}M', 
               fontsize=7, color='white', 
               va='center', ha='right', 
               bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor=self.config.TRADE_COLORS['long_entry' if direction == 'long' else 'short_entry'], 
                        edgecolor='none', 
                        alpha=0.8))
    
    def _plot_exit_marker(self, ax, x_pos, exit_idx, trade_dict):
        """Plot trade exit marker"""
        exit_reason = trade_dict.get('exit_reason', '')
        exit_price = trade_dict.get('exit_price')
        entry_price = trade_dict.get('entry_price')
        direction = trade_dict.get('direction')
        
        # Determine exit color
        if 'take_profit' in str(exit_reason):
            exit_color = self.config.TRADE_COLORS['take_profit']
        elif exit_reason == 'tp1_pullback':
            exit_color = self.config.TRADE_COLORS['tp1_pullback']
        elif exit_reason == 'stop_loss':
            exit_color = self.config.TRADE_COLORS['stop_loss']
        elif exit_reason == 'trailing_stop':
            exit_color = self.config.TRADE_COLORS['trailing_stop']
        elif exit_reason == 'signal_flip':
            exit_color = self.config.TRADE_COLORS['signal_flip']
        else:
            exit_color = self.config.TRADE_COLORS['end_of_data']
        
        # Plot marker - smaller size to reduce visual clutter
        ax.scatter(x_pos[exit_idx], exit_price, 
                  marker='x', s=120, color=exit_color, 
                  linewidth=2, zorder=5)
        
        # Add pip and P&L annotation
        if direction == 'long':
            exit_pips = (exit_price - entry_price) * 10000
        else:
            exit_pips = (entry_price - exit_price) * 10000
        
        # Get final P&L and position size if available
        final_pnl = trade_dict.get('pnl', None)
        
        # Calculate remaining position size for final exit
        position_size = trade_dict.get('position_size', 1000000) / 1000000
        if hasattr(trade_dict.get('position_size'), 'value'):
            position_size = trade_dict['position_size'].value / 1000000
            
        # Get partial exits to calculate remaining size
        partial_exits = trade_dict.get('partial_exits', [])
        remaining_size = position_size
        partial_pnl_sum = 0
        for pe in partial_exits:
            pe_size = pe.size if hasattr(pe, 'size') else pe.get('size', 0)
            remaining_size -= pe_size / 1000000
            # Sum partial P&Ls
            pe_pnl = pe.pnl if hasattr(pe, 'pnl') else pe.get('pnl', 0)
            partial_pnl_sum += pe_pnl
            
        # For TSL/final exits, calculate P&L for remaining position if not provided
        if final_pnl is None and remaining_size > 0:
            # Calculate P&L for remaining position
            # remaining_size is in millions, so multiply by 100 (pip value per million)
            if direction == 'long':
                pip_change = (exit_price - entry_price) * 10000  # Convert to pips
                remaining_pnl = pip_change * remaining_size * 100  # size in M * $100/pip/M
            else:
                pip_change = (entry_price - exit_price) * 10000  # Convert to pips
                remaining_pnl = pip_change * remaining_size * 100  # size in M * $100/pip/M
            final_pnl = partial_pnl_sum + remaining_pnl
        
        pip_color = '#43A047' if exit_pips > 0 else '#E53935'
        
        # Format text with size, pips and P&L in compact format
        # For final exits, show exit type and pips only (more concise)
        if exit_reason == 'trailing_stop':
            text = f'TSL|{exit_pips:+.1f}p'
        elif exit_reason == 'stop_loss':
            text = f'SL|{exit_pips:+.1f}p'
        elif 'take_profit' in str(exit_reason):
            # Extract TP number
            tp_num = exit_reason.split('_')[-1] if '_' in exit_reason else '3'
            text = f'TP{tp_num}|{exit_pips:+.1f}p'
        else:
            text = f'{exit_pips:+.1f}p'
            
        ax.text(x_pos[exit_idx] + 0.5, exit_price, 
               text, 
               fontsize=6, color=pip_color, 
               va='center', ha='left', 
               bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor=self.config.COLORS['bg'], 
                        edgecolor=pip_color, 
                        alpha=0.8))
    
    def _plot_trade_levels(self, ax, x_pos, entry_idx, exit_idx, trade_dict, data_len):
        """Plot TP and SL levels"""
        tp_levels = trade_dict.get('take_profits', [])
        sl_level = trade_dict.get('stop_loss')
        
        # Determine end point for levels
        level_end = exit_idx if exit_idx is not None else min(entry_idx + 50, data_len - 1)
        
        # Draw TP levels
        if tp_levels:
            tp_colors = ['#90EE90', '#3CB371', '#228B22']
            for i, tp in enumerate(tp_levels[:3]):
                if tp is not None:
                    ax.plot([x_pos[entry_idx], x_pos[level_end]], [tp, tp], 
                           color=tp_colors[i], linestyle=':', alpha=0.6, linewidth=1)
                    
                    ax.text(x_pos[entry_idx] + 1, tp, f'TP{i+1}', 
                           fontsize=7, color=tp_colors[i], 
                           va='center', ha='left', alpha=0.8)
        
        # Draw SL level
        if sl_level is not None:
            ax.plot([x_pos[entry_idx], x_pos[level_end]], [sl_level, sl_level], 
                   color='#FF6B6B', linestyle=':', alpha=0.6, linewidth=1)
            
            ax.text(x_pos[entry_idx] + 1, sl_level, 'SL', 
                   fontsize=7, color='#FF6B6B', 
                   va='center', ha='left', alpha=0.8)
    
    def _plot_partial_exits(self, ax, df, x_pos, trade_dict):
        """Plot partial exit markers - consolidated to reduce visual spam"""
        partial_exits = trade_dict.get('partial_exits', [])
        entry_price = trade_dict.get('entry_price')
        direction = trade_dict.get('direction')
        
        if not partial_exits:
            return
            
        # Only plot TP exits (not partial profit exits) to reduce clutter
        tp_exits = [p for p in partial_exits 
                    if (hasattr(p, 'tp_level') and p.tp_level > 0) or 
                       (isinstance(p, dict) and p.get('tp_level', 0) > 0)]
        
        if not tp_exits:
            return
            
        tp_exit_colors = ['#90EE90', '#3CB371', '#228B22']
        
        # Plot only significant TP exits (TP1 and TP3, skip TP2 for clarity)
        for partial_exit in tp_exits:
            tp_level = partial_exit.tp_level if hasattr(partial_exit, 'tp_level') else partial_exit.get('tp_level', 0)
            
            # Skip TP2 to reduce clutter (only show TP1 and TP3)
            if tp_level == 2:
                continue
                
            partial_time = partial_exit.time if hasattr(partial_exit, 'time') else partial_exit['time']
            partial_price = partial_exit.price if hasattr(partial_exit, 'price') else partial_exit['price']
            partial_pnl = partial_exit.pnl if hasattr(partial_exit, 'pnl') else partial_exit.get('pnl', None)
            
            partial_idx = self._find_time_index(df, partial_time)
            
            if partial_idx is not None:
                exit_color = tp_exit_colors[min(tp_level - 1, 2)]
                
                # Use smaller markers for partial exits
                ax.scatter(x_pos[partial_idx], partial_price, 
                          marker='o', s=60, color=exit_color, 
                          edgecolor='white', linewidth=1, zorder=5)
                
                # Calculate pips
                if direction == 'long':
                    partial_pips = (partial_price - entry_price) * 10000
                else:
                    partial_pips = (entry_price - partial_price) * 10000
                
                # Get partial exit size
                partial_size = partial_exit.size if hasattr(partial_exit, 'size') else partial_exit.get('size', 0)
                partial_size_m = partial_size / 1000000
                
                # Simplified text - just show TP level and pips
                text = f'TP{tp_level}|+{partial_pips:.1f}p'
                
                # Add compact annotation
                ax.text(x_pos[partial_idx] + 0.3, partial_price, 
                       text, 
                       fontsize=5, color=exit_color, 
                       va='center', ha='left', 
                       bbox=dict(boxstyle='round,pad=0.1', 
                                facecolor=self.config.COLORS['bg'], 
                                edgecolor=exit_color, 
                                alpha=0.7))
    
    def _add_confidence_text(self, ax, df):
        """Add confidence text if available"""
        if 'NTI_Confidence' in df.columns:
            latest_confidence = df['NTI_Confidence'].iloc[-1]
            ax.text(0.02, 0.98, f'Confidence: {latest_confidence:.1%}', 
                   transform=ax.transAxes, fontsize=10, 
                   color=self.config.COLORS['text'], va='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=self.config.COLORS['bg'], alpha=0.8))
    
    def _add_main_legend(self, ax, df, trades):
        """Add comprehensive legend to main chart with proper grouping"""
        legend_elements = []
        seen_labels = set()  # Track seen labels to avoid duplicates
        
        # Analyze trades to determine what exit types are actually present
        exit_types_present = set()
        partial_exit_levels = set()
        
        if trades:
            for trade in trades:
                # Get exit reason
                exit_reason = trade.exit_reason if hasattr(trade, 'exit_reason') else trade.get('exit_reason')
                if hasattr(exit_reason, 'value'):
                    exit_reason = exit_reason.value
                
                if exit_reason:
                    exit_types_present.add(str(exit_reason))
                
                # Check for partial exits
                partial_exits = trade.partial_exits if hasattr(trade, 'partial_exits') else trade.get('partial_exits', [])
                for partial_exit in partial_exits:
                    tp_level = partial_exit.tp_level if hasattr(partial_exit, 'tp_level') else partial_exit.get('tp_level', 0)
                    if tp_level > 0:
                        partial_exit_levels.add(tp_level)
        
        # SECTION 1: Entry markers
        if trades:
            legend_elements.append(
                Line2D([0], [0], marker='^', color='w', markerfacecolor=self.config.TRADE_COLORS['long_entry'], 
                       markersize=12, label='▲ Long Entry (Green)', linestyle='None', markeredgecolor='white', markeredgewidth=1)
            )
            legend_elements.append(
                Line2D([0], [0], marker='v', color='w', markerfacecolor=self.config.TRADE_COLORS['short_entry'], 
                       markersize=12, label='▼ Short Entry (Red)', linestyle='None', markeredgecolor='white', markeredgewidth=1)
            )
            
            # SECTION 2: Partial exits (TP levels)
            if partial_exit_levels:
                tp_colors = ['#90EE90', '#3CB371', '#228B22']
                for tp_level in sorted(partial_exit_levels):
                    if tp_level <= 3:
                        legend_elements.append(
                            Line2D([0], [0], marker='o', color='w', markerfacecolor=tp_colors[tp_level-1], 
                                   markersize=10, label=f'● TP{tp_level} Exit', linestyle='None', 
                                   markeredgecolor='white', markeredgewidth=1)
                        )
            
            # SECTION 3: Final exits
            exit_markers = {
                'take_profit_1': ('✗', self.config.TRADE_COLORS['take_profit'], 'TP1 Final'),
                'take_profit_2': ('✗', self.config.TRADE_COLORS['take_profit'], 'TP2 Final'),
                'take_profit_3': ('✗', self.config.TRADE_COLORS['take_profit'], 'TP3 Final'), 
                'stop_loss': ('✗', self.config.TRADE_COLORS['stop_loss'], 'Stop Loss'),
                'trailing_stop': ('✗', self.config.TRADE_COLORS['trailing_stop'], 'Trailing SL'),
                'signal_flip': ('✗', self.config.TRADE_COLORS['signal_flip'], 'Signal Flip'),
                'end_of_data': ('✗', self.config.TRADE_COLORS['end_of_data'], 'End of Data')
            }
            
            for exit_type in exit_types_present:
                if exit_type in exit_markers:
                    symbol, color, label = exit_markers[exit_type]
                    if label not in seen_labels:
                        legend_elements.append(
                            Line2D([0], [0], marker='x', color=color, 
                                   markersize=12, label=f'{symbol} {label}', linestyle='None', markeredgewidth=2)
                        )
                        seen_labels.add(label)
        
        # SECTION 4: Market indicators (if space permits)
        if 'NTI_Direction' in df.columns or 'NT3_Direction' in df.columns:
            legend_elements.extend([
                Line2D([0], [0], color=self.config.COLORS['bullish'], lw=3, label='— Uptrend', alpha=0.8),
                Line2D([0], [0], color=self.config.COLORS['bearish'], lw=3, label='— Downtrend', alpha=0.8)
            ])
        
        if legend_elements:
            # Create legend with better layout
            legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                             framealpha=0.95, ncol=3, columnspacing=1.0,
                             handletextpad=0.3, handlelength=1.5,
                             bbox_to_anchor=(0.01, 0.99))
            legend.get_frame().set_facecolor(self.config.COLORS['bg'])
            legend.get_frame().set_edgecolor(self.config.COLORS['grid'])
            
            # Set text color
            for text in legend.get_texts():
                text.set_color(self.config.COLORS['text'])
    
    def _plot_chop_subplots(self, axes, df, simplified_regime_colors, trend_color, range_color):
        """Plot Intelligent Chop subplots (placeholder)"""
        # This would contain the chop subplot implementation
        pass
    
    def _format_x_axis(self, axes, df, show_chop_subplots, show_pnl, show_position_sizes):
        """Format x-axis for all subplots"""
        if isinstance(df.index, pd.DatetimeIndex):
            # Determine bottom axis
            bottom_ax = axes[-1]
            
            # Create x-axis positions
            n_ticks = min(8, len(df))
            tick_positions = np.linspace(0, len(df)-1, n_ticks, dtype=int)
            
            # Set ticks and labels
            bottom_ax.set_xticks(tick_positions)
            tick_labels = [df.index[i].strftime('%Y-%m-%d') for i in tick_positions]
            bottom_ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            # Hide x-axis labels for non-bottom axes
            for ax in axes[:-1]:
                ax.tick_params(colors=self.config.COLORS['text'], labelbottom=False)
            
            # Format bottom axis
            bottom_ax.tick_params(colors=self.config.COLORS['text'])
    
    def _style_spines(self, axes):
        """Style subplot spines"""
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(self.config.COLORS['grid'])
            ax.spines['left'].set_color(self.config.COLORS['grid'])


# ============================================================================
# Convenience Functions
# ============================================================================

def plot_production_results(df: pd.DataFrame, results: Dict, **kwargs) -> plt.Figure:
    """Convenience function to create production strategy plots"""
    plotter = ProductionPlotter()
    return plotter.plot_strategy_results(df, results, **kwargs)


if __name__ == "__main__":
    print("Production Strategy Plotting Module Loaded")