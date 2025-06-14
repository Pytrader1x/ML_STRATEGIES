"""
Object-Oriented Strategy Runner - Monte Carlo Testing Framework
Enhanced version with detailed trade analytics and insights
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any
import json
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

__version__ = "3.1.0"  # Enhanced OOP version with detailed analytics


class RunMode(Enum):
    """Enumeration for different running modes"""
    SINGLE = "single"
    MULTI = "multi"
    CRYPTO = "crypto"
    CUSTOM = "custom"


class StrategyType(Enum):
    """Enumeration for strategy types"""
    ULTRA_TIGHT_RISK = "ultra_tight_risk"
    SCALPING = "scalping"
    CRYPTO_CONSERVATIVE = "crypto_conservative"
    CRYPTO_MODERATE = "crypto_moderate"


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    n_iterations: int = 50
    sample_size: int = 8000
    realistic_costs: bool = True
    use_daily_sharpe: bool = True
    debug_mode: bool = False
    show_plots: bool = False
    save_plots: bool = False
    export_trades: bool = True
    calendar_analysis: bool = True
    date_range: Optional[Tuple[str, str]] = None


@dataclass
class DetailedTradeStats:
    """Container for detailed trade statistics"""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L stats
    total_pnl: float = 0.0
    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0
    
    # Pip stats
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_win_pips: float = 0.0
    max_loss_pips: float = 0.0
    
    # Position sizing
    avg_position_size: float = 0.0
    avg_position_size_millions: float = 0.0
    
    # Entry logic breakdown
    entry_logic_counts: Dict[str, int] = field(default_factory=dict)
    entry_logic_pct: Dict[str, float] = field(default_factory=dict)
    
    # Exit logic breakdown
    exit_logic_counts: Dict[str, int] = field(default_factory=dict)
    exit_logic_pct: Dict[str, float] = field(default_factory=dict)
    
    # TP hit rates
    tp1_hit_count: int = 0
    tp2_hit_count: int = 0
    tp3_hit_count: int = 0
    tp1_hit_pct: float = 0.0
    tp2_hit_pct: float = 0.0
    tp3_hit_pct: float = 0.0
    
    # TP exit sizes
    tp1_exit_size_avg: float = 0.0
    tp2_exit_size_avg: float = 0.0
    tp3_exit_size_avg: float = 0.0
    
    # Additional insights
    avg_trade_duration_hours: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_risk_reward: float = 0.0
    
    # Market regime performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Container for backtest results"""
    metrics: Dict[str, float]
    trades: Optional[List] = None
    equity_curve: Optional[pd.Series] = None
    monte_carlo_stats: Optional[Dict] = None
    calendar_performance: Optional[pd.DataFrame] = None
    detailed_stats: Optional[DetailedTradeStats] = None
    
    
class StrategyFactory:
    """Factory class for creating strategy configurations"""
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, config: BacktestConfig) -> OptimizedProdStrategy:
        """Create a strategy based on the specified type"""
        if strategy_type == StrategyType.ULTRA_TIGHT_RISK:
            return StrategyFactory._create_ultra_tight_risk(config)
        elif strategy_type == StrategyType.SCALPING:
            return StrategyFactory._create_scalping(config)
        elif strategy_type == StrategyType.CRYPTO_CONSERVATIVE:
            return StrategyFactory._create_crypto_conservative(config)
        elif strategy_type == StrategyType.CRYPTO_MODERATE:
            return StrategyFactory._create_crypto_moderate(config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @staticmethod
    def _create_ultra_tight_risk(config: BacktestConfig) -> OptimizedProdStrategy:
        """Configuration 1: Ultra-Tight Risk Management"""
        strategy_config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,  # 0.2% risk per trade
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            tsl_activation_pips=3,
            tsl_min_profit_pips=1,
            tsl_initial_buffer_multiplier=1.0,
            trailing_atr_multiplier=0.8,
            tp_range_market_multiplier=0.5,
            tp_trend_market_multiplier=0.7,
            tp_chop_market_multiplier=0.3,
            sl_range_market_multiplier=0.7,
            exit_on_signal_flip=False,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.5,
            partial_profit_size_percent=0.5,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            realistic_costs=config.realistic_costs,
            verbose=False,
            debug_decisions=config.debug_mode,
            use_daily_sharpe=config.use_daily_sharpe
        )
        return OptimizedProdStrategy(strategy_config)
    
    @staticmethod
    def _create_scalping(config: BacktestConfig) -> OptimizedProdStrategy:
        """Configuration 2: Scalping Strategy"""
        strategy_config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,  # 0.1% risk per trade
            sl_max_pips=5.0,
            sl_atr_multiplier=0.5,
            tp_atr_multipliers=(0.1, 0.2, 0.3),
            max_tp_percent=0.002,
            tsl_activation_pips=2,
            tsl_min_profit_pips=0.5,
            tsl_initial_buffer_multiplier=0.5,
            trailing_atr_multiplier=0.5,
            tp_range_market_multiplier=0.3,
            tp_trend_market_multiplier=0.5,
            tp_chop_market_multiplier=0.2,
            sl_range_market_multiplier=0.5,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=0.0,
            signal_flip_min_time_hours=0.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.3,
            partial_profit_size_percent=0.7,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            realistic_costs=config.realistic_costs,
            verbose=False,
            debug_decisions=config.debug_mode,
            use_daily_sharpe=config.use_daily_sharpe
        )
        return OptimizedProdStrategy(strategy_config)
    
    @staticmethod
    def _create_crypto_conservative(config: BacktestConfig) -> 'CryptoStrategy':
        """Conservative crypto strategy configuration"""
        # This would need the crypto strategy implementation
        # For now, returning a placeholder
        raise NotImplementedError("Crypto strategies need to be implemented")
    
    @staticmethod
    def _create_crypto_moderate(config: BacktestConfig) -> 'CryptoStrategy':
        """Moderate crypto strategy configuration"""
        # This would need the crypto strategy implementation
        # For now, returning a placeholder
        raise NotImplementedError("Crypto strategies need to be implemented")


class DataManager:
    """Manages data loading and preparation"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = self._resolve_data_path(data_path)
        self._cache = {}
    
    def _resolve_data_path(self, data_path: Optional[str]) -> str:
        """Resolve the data directory path"""
        if data_path:
            return data_path
            
        # Auto-detect data path
        if os.path.exists('data'):
            return 'data'
        elif os.path.exists('../data'):
            return '../data'
        else:
            raise FileNotFoundError("Cannot find data directory. Please run from project root.")
    
    def load_currency_data(self, currency_pair: str, use_cache: bool = True) -> pd.DataFrame:
        """Load and prepare data for a specific currency pair"""
        # Check cache first
        if use_cache and currency_pair in self._cache:
            print(f"Using cached data for {currency_pair}")
            return self._cache[currency_pair].copy()
        
        file_path = os.path.join(self.data_path, f'{currency_pair}_MASTER_15M.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading {currency_pair} data...")
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        print(f"Total data points: {len(df):,}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Cache the processed data
        if use_cache:
            self._cache[currency_pair] = df.copy()
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        print("Calculating indicators...")
        
        # Helper function to format time
        def format_time(seconds):
            if seconds < 0.1:
                return f"{seconds * 1000:.0f}ms"
            elif seconds < 1.0:
                return f"{seconds * 1000:.0f}ms"
            else:
                return f"{seconds:.1f}s"
        
        # Neuro Trend Intelligent
        print("  Calculating Neuro Trend Intelligent...")
        start_time = time.time()
        df = TIC.add_neuro_trend_intelligent(df)
        elapsed_time = time.time() - start_time
        print(f"  ✓ Completed Neuro Trend Intelligent in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
        
        # Market Bias
        print("  Calculating Market Bias...")
        start_time = time.time()
        df = TIC.add_market_bias(df)
        elapsed_time = time.time() - start_time
        print(f"  ✓ Completed Market Bias in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
        
        # Intelligent Chop
        print("  Calculating Intelligent Chop...")
        start_time = time.time()
        df = TIC.add_intelligent_chop(df)
        elapsed_time = time.time() - start_time
        print(f"  ✓ Completed Intelligent Chop in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
        
        return df
    
    def filter_date_range(self, df: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Filter dataframe by date range"""
        start_date, end_date = date_range
        return df.loc[start_date:end_date]


class EnhancedTradeAnalyzer:
    """Enhanced trade analyzer with detailed insights"""
    
    @staticmethod
    def analyze_trades_detailed(trades: List[Any]) -> DetailedTradeStats:
        """Perform detailed analysis of trades"""
        if not trades:
            return DetailedTradeStats()
        
        stats = DetailedTradeStats()
        stats.total_trades = len(trades)
        
        # Collect data for analysis
        pnl_values = []
        win_pips = []
        loss_pips = []
        position_sizes = []
        entry_logics = []
        exit_reasons = []
        trade_durations = []
        
        tp_hit_counts = {1: 0, 2: 0, 3: 0}
        tp_exit_sizes = {1: [], 2: [], 3: []}
        
        for trade in trades:
            # Basic P&L
            pnl = trade.pnl if hasattr(trade, 'pnl') else 0
            pnl_values.append(pnl)
            
            # Position size
            pos_size = trade.initial_position_size if hasattr(trade, 'initial_position_size') else trade.position_size
            position_sizes.append(pos_size)
            
            # Calculate pips
            direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
            if direction == 'long':
                pips = (trade.exit_price - trade.entry_price) / 0.0001
            else:
                pips = (trade.entry_price - trade.exit_price) / 0.0001
            
            if pnl > 0:
                win_pips.append(pips)
            else:
                loss_pips.append(pips)
            
            # Entry logic
            entry_logic = 'Relaxed (NTI only)' if trade.is_relaxed else 'Standard (NTI+MB+IC)'
            entry_logics.append(entry_logic)
            
            # Exit reason
            exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
            exit_reasons.append(exit_reason)
            
            # Trade duration
            if trade.exit_time and trade.entry_time:
                duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                trade_durations.append(duration_hours)
            
            # TP hits - check exit reason
            exit_reason_str = str(exit_reason).lower()
            if 'take_profit_1' in exit_reason_str or 'tp1' in exit_reason_str:
                tp_hit_counts[1] += 1
            elif 'take_profit_2' in exit_reason_str or 'tp2' in exit_reason_str:
                tp_hit_counts[2] += 1
                tp_hit_counts[1] += 1  # If TP2 hit, TP1 was also hit
            elif 'take_profit_3' in exit_reason_str or 'tp3' in exit_reason_str:
                tp_hit_counts[3] += 1
                tp_hit_counts[2] += 1  # If TP3 hit, TP2 was also hit
                tp_hit_counts[1] += 1  # If TP3 hit, TP1 was also hit
            
            # Alternative: check tp_hits attribute
            if hasattr(trade, 'tp_hits') and trade.tp_hits:
                for tp_level in [1, 2, 3]:
                    if tp_level <= trade.tp_hits:
                        tp_hit_counts[tp_level] += 1
            
            # Partial exits for TP size tracking
            if hasattr(trade, 'partial_exits') and trade.partial_exits:
                for pe in trade.partial_exits:
                    if hasattr(pe, 'tp_level'):
                        tp_level = pe.tp_level
                        size = pe.size if hasattr(pe, 'size') else 0
                        if 1 <= tp_level <= 3:
                            tp_exit_sizes[tp_level].append(size)
        
        # Calculate statistics
        winning_trades = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades = [pnl for pnl in pnl_values if pnl < 0]
        
        stats.winning_trades = len(winning_trades)
        stats.losing_trades = len(losing_trades)
        stats.win_rate = stats.winning_trades / stats.total_trades * 100 if stats.total_trades > 0 else 0
        
        # P&L stats
        stats.total_pnl = sum(pnl_values)
        stats.avg_win_pnl = np.mean(winning_trades) if winning_trades else 0
        stats.avg_loss_pnl = np.mean(losing_trades) if losing_trades else 0
        
        # Pip stats
        stats.avg_win_pips = np.mean(win_pips) if win_pips else 0
        stats.avg_loss_pips = np.mean(loss_pips) if loss_pips else 0
        stats.max_win_pips = max(win_pips) if win_pips else 0
        stats.max_loss_pips = min(loss_pips) if loss_pips else 0
        
        # Position sizing
        stats.avg_position_size = np.mean(position_sizes) if position_sizes else 0
        stats.avg_position_size_millions = stats.avg_position_size / 1e6
        
        # Entry logic breakdown
        entry_counts = Counter(entry_logics)
        stats.entry_logic_counts = dict(entry_counts)
        for logic, count in entry_counts.items():
            stats.entry_logic_pct[logic] = count / stats.total_trades * 100
        
        # Exit logic breakdown
        exit_counts = Counter(exit_reasons)
        stats.exit_logic_counts = dict(exit_counts)
        for reason, count in exit_counts.items():
            stats.exit_logic_pct[reason] = count / stats.total_trades * 100
        
        # TP hit rates
        stats.tp1_hit_count = tp_hit_counts[1]
        stats.tp2_hit_count = tp_hit_counts[2]
        stats.tp3_hit_count = tp_hit_counts[3]
        stats.tp1_hit_pct = stats.tp1_hit_count / stats.total_trades * 100 if stats.total_trades > 0 else 0
        stats.tp2_hit_pct = stats.tp2_hit_count / stats.total_trades * 100 if stats.total_trades > 0 else 0
        stats.tp3_hit_pct = stats.tp3_hit_count / stats.total_trades * 100 if stats.total_trades > 0 else 0
        
        # TP exit sizes
        stats.tp1_exit_size_avg = np.mean(tp_exit_sizes[1]) / 1e6 if tp_exit_sizes[1] else 0
        stats.tp2_exit_size_avg = np.mean(tp_exit_sizes[2]) / 1e6 if tp_exit_sizes[2] else 0
        stats.tp3_exit_size_avg = np.mean(tp_exit_sizes[3]) / 1e6 if tp_exit_sizes[3] else 0
        
        # Additional insights
        stats.avg_trade_duration_hours = np.mean(trade_durations) if trade_durations else 0
        
        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 1
        stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Risk/Reward
        if stats.avg_loss_pips != 0:
            stats.avg_risk_reward = abs(stats.avg_win_pips / stats.avg_loss_pips)
        else:
            stats.avg_risk_reward = float('inf') if stats.avg_win_pips > 0 else 0
        
        return stats
    
    @staticmethod
    def calculate_trade_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed trade statistics including consecutive wins/losses"""
        stats = {}
        
        # Extract basic stats if available
        if 'trades' in results and results['trades'] is not None:
            trades = results['trades']
            if len(trades) > 0:
                # Extract P&L values
                pnl_values = []
                for trade in trades:
                    if hasattr(trade, 'pnl'):  # Trade object
                        if trade.pnl is not None:
                            pnl_values.append(trade.pnl)
                    elif isinstance(trade, dict) and 'pnl' in trade:  # Dictionary
                        if trade['pnl'] is not None:
                            pnl_values.append(trade['pnl'])
                
                # Calculate consecutive wins/losses
                stats = EnhancedTradeAnalyzer._calculate_consecutive_stats(pnl_values)
                stats['num_wins'] = sum(1 for pnl in pnl_values if pnl > 0)
                stats['num_losses'] = sum(1 for pnl in pnl_values if pnl < 0)
            else:
                stats = EnhancedTradeAnalyzer._empty_stats()
        else:
            # Estimate from aggregate stats
            stats = EnhancedTradeAnalyzer._estimate_stats_from_aggregates(results)
        
        return stats
    
    @staticmethod
    def _calculate_consecutive_stats(pnl_values: List[float]) -> Dict[str, Any]:
        """Calculate consecutive win/loss statistics"""
        wins = [1 if pnl > 0 else 0 for pnl in pnl_values]
        losses = [1 if pnl < 0 else 0 for pnl in pnl_values]
        
        # Consecutive wins
        win_streaks = EnhancedTradeAnalyzer._get_streaks(wins)
        loss_streaks = EnhancedTradeAnalyzer._get_streaks(losses)
        
        return {
            'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
            'avg_consecutive_wins': int(round(np.mean(win_streaks))) if win_streaks else 0,
            'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
            'avg_consecutive_losses': int(round(np.mean(loss_streaks))) if loss_streaks else 0,
        }
    
    @staticmethod
    def _get_streaks(binary_list: List[int]) -> List[int]:
        """Get list of streak lengths from binary list"""
        streaks = []
        current_streak = 0
        
        for value in binary_list:
            if value == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return empty statistics structure"""
        return {
            'max_consecutive_wins': 0,
            'avg_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_consecutive_losses': 0,
            'num_wins': 0,
            'num_losses': 0
        }
    
    @staticmethod
    def _estimate_stats_from_aggregates(results: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate consecutive statistics from aggregate results"""
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) / 100
        
        num_wins = int(total_trades * win_rate)
        num_losses = total_trades - num_wins
        
        stats = {
            'num_wins': num_wins,
            'num_losses': num_losses
        }
        
        # Conservative estimates for consecutive stats
        if num_wins > 0:
            # Average consecutive wins
            if win_rate >= 0.8:
                stats['avg_consecutive_wins'] = 4
            elif win_rate >= 0.6:
                stats['avg_consecutive_wins'] = 3
            elif win_rate >= 0.4:
                stats['avg_consecutive_wins'] = 2
            else:
                stats['avg_consecutive_wins'] = 1
            
            # Max consecutive wins
            max_bound = min(num_wins, int(total_trades * 0.3))
            if win_rate >= 0.8:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 8)
            elif win_rate >= 0.6:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 5)
            else:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 3)
        else:
            stats['avg_consecutive_wins'] = 0
            stats['max_consecutive_wins'] = 0
        
        if num_losses > 0:
            # Average consecutive losses
            loss_rate = 1.0 - win_rate
            if loss_rate >= 0.6:
                stats['avg_consecutive_losses'] = 3
            elif loss_rate >= 0.4:
                stats['avg_consecutive_losses'] = 2
            else:
                stats['avg_consecutive_losses'] = 1
            
            # Max consecutive losses
            max_bound = min(num_losses, int(total_trades * 0.2))
            if loss_rate >= 0.4:
                stats['max_consecutive_losses'] = min(max_bound, stats['avg_consecutive_losses'] * 4)
            else:
                stats['max_consecutive_losses'] = min(max_bound, stats['avg_consecutive_losses'] * 3)
        else:
            stats['avg_consecutive_losses'] = 0
            stats['max_consecutive_losses'] = 0
        
        return stats


class TradeExporter:
    """Handles trade export functionality"""
    
    @staticmethod
    def export_trades_detailed(trades: List[Any], currency: str, config_name: str) -> str:
        """Export detailed trade data to CSV"""
        trade_records = []
        
        for i, trade in enumerate(trades, 1):
            record = TradeExporter._create_trade_record(i, trade)
            trade_records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(trade_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_trades_detail_{timestamp}.csv'
        df.to_csv(filepath, index=False, float_format='%.6f')
        
        # Print summary
        TradeExporter._print_export_summary(df, config_name, filepath)
        
        return filepath
    
    @staticmethod
    def _create_trade_record(trade_id: int, trade: Any) -> Dict[str, Any]:
        """Create a trade record dictionary from trade object"""
        # Extract basic trade info
        direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
        exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
        
        # Calculate pip distances
        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / 0.0001
        
        record = {
            'trade_id': trade_id,
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'direction': direction,
            'initial_size_millions': trade.initial_position_size / 1e6 if hasattr(trade, 'initial_position_size') else trade.position_size / 1e6,
            'confidence': trade.confidence,
            'is_relaxed': trade.is_relaxed,
            'entry_logic': 'Relaxed (NTI only)' if trade.is_relaxed else 'Standard (NTI+MB+IC)',
            'sl_price': trade.stop_loss,
            'sl_distance_pips': sl_distance_pips,
            'tp1_price': trade.take_profits[0] if len(trade.take_profits) > 0 else None,
            'tp2_price': trade.take_profits[1] if len(trade.take_profits) > 1 else None,
            'tp3_price': trade.take_profits[2] if len(trade.take_profits) > 2 else None,
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'exit_reason': exit_reason,
            'tp_hits': trade.tp_hits,
            'trade_duration_hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else None,
            'final_pnl': trade.pnl,
        }
        
        # Add partial exit details if available
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            partial_pnl_sum = 0
            for j, pe in enumerate(trade.partial_exits[:3], 1):
                pe_type = pe.exit_type if hasattr(pe, 'exit_type') else f'TP{pe.tp_level}' if hasattr(pe, 'tp_level') else 'PARTIAL'
                pe_size = pe.size / 1e6 if hasattr(pe, 'size') else 0
                pe_pnl = pe.pnl if hasattr(pe, 'pnl') else 0
                partial_pnl_sum += pe_pnl
                
                record[f'partial_exit_{j}_type'] = pe_type
                record[f'partial_exit_{j}_size_m'] = pe_size
                record[f'partial_exit_{j}_pnl'] = pe_pnl
            
            record['partial_pnl_total'] = partial_pnl_sum
        
        # Calculate final P&L components
        if direction == 'long':
            final_pips = (trade.exit_price - trade.entry_price) / 0.0001
        else:
            final_pips = (trade.entry_price - trade.exit_price) / 0.0001
        
        record['final_exit_pips'] = final_pips
        record['pnl_per_million'] = trade.pnl / (record['initial_size_millions']) if record['initial_size_millions'] > 0 else 0
        
        return record
    
    @staticmethod
    def _print_export_summary(df: pd.DataFrame, config_name: str, filepath: str):
        """Print summary of exported trades"""
        print(f"\n📊 Trade Export Summary for {config_name}:")
        print(f"  Total Trades: {len(df)}")
        print(f"  Total P&L: ${df['final_pnl'].sum():,.2f}")
        print(f"  Avg P&L per Trade: ${df['final_pnl'].mean():,.2f}")
        print(f"  Win Rate: {(df['final_pnl'] > 0).sum() / len(df) * 100:.1f}%")
        
        if len(df[df['final_pnl'] > 0]) > 0:
            print(f"  Avg Win: ${df[df['final_pnl'] > 0]['final_pnl'].mean():,.2f}")
        else:
            print("  Avg Win: N/A")
            
        if len(df[df['final_pnl'] < 0]) > 0:
            print(f"  Avg Loss: ${df[df['final_pnl'] < 0]['final_pnl'].mean():,.2f}")
        else:
            print("  Avg Loss: N/A")
            
        print(f"  Saved to: {filepath}")


class MonteCarloSimulator:
    """Handles Monte Carlo simulation logic"""
    
    def __init__(self, strategy: OptimizedProdStrategy, trade_analyzer: EnhancedTradeAnalyzer):
        self.strategy = strategy
        self.trade_analyzer = trade_analyzer
    
    def run_simulation(self, df: pd.DataFrame, config: BacktestConfig) -> Tuple[pd.DataFrame, Dict]:
        """Run Monte Carlo simulation"""
        iteration_results = []
        last_sample_df = None
        last_results = None
        integrity_checks = []
        
        # Track aggregate statistics
        all_trade_stats = {
            'max_consecutive_wins': [],
            'avg_consecutive_wins': [],
            'max_consecutive_losses': [],
            'avg_consecutive_losses': [],
            'num_wins': [],
            'num_losses': [],
            'avg_win_pips': [],
            'avg_loss_pips': [],
            'avg_win_pnl': [],
            'avg_loss_pnl': [],
            'total_pnl': []
        }
        
        # Track detailed stats across iterations
        all_detailed_stats = []
        all_date_info = []
        
        for i in range(config.n_iterations):
            # Get random sample
            sample_df = self._get_sample(df, config.sample_size)
            
            # Get date range of sample
            start_date = sample_df.index[0]
            end_date = sample_df.index[-1]
            
            # Store date info
            all_date_info.append({
                'start_date': start_date,
                'end_date': end_date,
                'start_year': start_date.year,
                'end_year': end_date.year,
                'year_range': f"{start_date.year}-{end_date.year}" if start_date.year != end_date.year else str(start_date.year)
            })
            
            # Run backtest
            results = self.strategy.run_backtest(sample_df)
            
            # Extract metrics
            metrics = self._extract_metrics(results, i)
            iteration_results.append(metrics)
            
            # Track trade statistics
            trade_stats = self.trade_analyzer.calculate_trade_statistics(results)
            for key in trade_stats:
                if key in all_trade_stats:
                    all_trade_stats[key].append(trade_stats[key])
            
            # Extract pip and P&L stats from trades
            if 'trades' in results and results['trades']:
                pip_pnl_stats = self._extract_pip_pnl_stats(results['trades'])
                for key, value in pip_pnl_stats.items():
                    if key in all_trade_stats:
                        all_trade_stats[key].append(value)
                
                # Detailed analysis for each iteration
                detailed_stats = self.trade_analyzer.analyze_trades_detailed(results['trades'])
                all_detailed_stats.append(detailed_stats)
                
                # Print mini summary for this iteration
                self._print_iteration_summary(i + 1, results, start_date, end_date, len(sample_df), pip_pnl_stats)
            else:
                # Print even if no trades
                self._print_iteration_summary(i + 1, results, start_date, end_date, len(sample_df))
            
            # Keep last iteration data for return
            if i == config.n_iterations - 1:
                last_sample_df = sample_df
                last_results = results
            
            # Progress update every 10 iterations
            if (i + 1) % 10 == 0 and (i + 1) < config.n_iterations:
                self._print_progress(i + 1, config.n_iterations, iteration_results)
        
        # Create results dataframe
        results_df = pd.DataFrame(iteration_results)
        
        # Add date info to results_df
        for i, info in enumerate(all_date_info):
            results_df.loc[i, 'start_year'] = info['start_year']
            results_df.loc[i, 'year_range'] = info['year_range']
        
        # Add aggregate statistics
        monte_carlo_stats = self._calculate_aggregate_stats(all_trade_stats)
        
        # Calculate aggregated detailed stats
        aggregated_detailed_stats = self._aggregate_detailed_stats(all_detailed_stats)
        
        return results_df, {
            'last_sample_df': last_sample_df,
            'last_results': last_results,
            'monte_carlo_stats': monte_carlo_stats,
            'integrity_checks': integrity_checks,
            'aggregated_detailed_stats': aggregated_detailed_stats,
            'date_info': all_date_info
        }
    
    def _get_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Get a random sample from the dataframe"""
        max_start = len(df) - sample_size
        if max_start < 0:
            # If date range is too small, use entire range
            return df.copy()
        else:
            start_idx = np.random.randint(0, max_start)
            return df.iloc[start_idx:start_idx + sample_size].copy()
    
    def _extract_metrics(self, results: Dict, iteration: int) -> Dict:
        """Extract metrics from backtest results"""
        return {
            'iteration': iteration + 1,
            'total_return': results.get('total_return', 0),
            'total_trades': results.get('total_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'profit_factor': results.get('profit_factor', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'sortino_ratio': results.get('sortino_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'avg_trade': results.get('avg_trade', 0),
            'avg_win': results.get('avg_win', 0),
            'avg_loss': results.get('avg_loss', 0),
            'best_trade': results.get('best_trade', 0),
            'worst_trade': results.get('worst_trade', 0),
            'recovery_factor': results.get('recovery_factor', 0),
            'win_loss_ratio': results.get('win_loss_ratio', 0),
            'expectancy': results.get('expectancy', 0),
            'sqn': results.get('sqn', 0),
            'trades_per_day': results.get('trades_per_day', 0)
        }
    
    def _print_iteration_summary(self, iteration: int, results: Dict, start_date, end_date, num_rows: int, pip_pnl_stats: Optional[Dict] = None):
        """Print a mini summary for each iteration"""
        # Format dates nicely
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Extract key metrics
        sharpe = results.get('sharpe_ratio', 0)
        total_return = results.get('total_return', 0)
        win_rate = results.get('win_rate', 0)
        trades = results.get('total_trades', 0)
        max_dd = results.get('max_drawdown', 0)
        
        # Get win/loss counts
        wins = int(trades * win_rate / 100) if trades > 0 else 0
        losses = trades - wins
        
        # Get pip stats if available
        if pip_pnl_stats:
            avg_win_pips = pip_pnl_stats.get('avg_win_pips', 0)
            avg_loss_pips = pip_pnl_stats.get('avg_loss_pips', 0)
            pip_str = f"| Pips: +{avg_win_pips:.1f}/-{avg_loss_pips:.1f}"
        else:
            pip_str = ""
        
        # Position integrity check
        integrity = "✓" if results.get('position_integrity', True) else "✗"
        
        # Color coding for Sharpe (using simple text indicators)
        if sharpe >= 2.0:
            sharpe_indicator = "🟢"
        elif sharpe >= 1.0:
            sharpe_indicator = "🟡"
        else:
            sharpe_indicator = "🔴"
        
        # Print formatted line
        print(f"  [{iteration:3d}] "
              f"{sharpe_indicator} Sharpe: {sharpe:6.3f} | "
              f"Return: {total_return:6.1f}% | "
              f"WR: {win_rate:5.1f}% | "
              f"Trades: {trades:4d} ({wins}W/{losses}L) "
              f"{pip_str} | "
              f"DD: {max_dd:4.1f}% | "
              f"Rows: {num_rows:,} | "
              f"{start_str} → {end_str} | "
              f"Integrity: {integrity}")
    
    def _extract_pip_pnl_stats(self, trades: List[Any]) -> Dict[str, float]:
        """Extract pip and P&L statistics from trades"""
        win_pips = []
        loss_pips = []
        win_pnls = []
        loss_pnls = []
        
        for trade in trades:
            # Calculate pips
            direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
            if direction == 'long':
                pips = (trade.exit_price - trade.entry_price) / 0.0001
            else:
                pips = (trade.entry_price - trade.exit_price) / 0.0001
            
            # Get P&L
            pnl = trade.pnl if hasattr(trade, 'pnl') else 0
            
            if pnl > 0:
                win_pips.append(pips)
                win_pnls.append(pnl)
            elif pnl < 0:
                loss_pips.append(abs(pips))
                loss_pnls.append(pnl)
        
        return {
            'avg_win_pips': np.mean(win_pips) if win_pips else 0,
            'avg_loss_pips': np.mean(loss_pips) if loss_pips else 0,
            'avg_win_pnl': np.mean(win_pnls) if win_pnls else 0,
            'avg_loss_pnl': np.mean(loss_pnls) if loss_pnls else 0,
            'total_pnl': sum(win_pnls) + sum(loss_pnls) if (win_pnls or loss_pnls) else 0
        }
    
    def _print_progress(self, current: int, total: int, results: List[Dict]):
        """Print progress update"""
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        print(f"Progress: {current}/{total} iterations - Avg Sharpe: {avg_sharpe:.3f}, Avg Return: {avg_return:.1f}%")
    
    def _calculate_aggregate_stats(self, all_trade_stats: Dict) -> Dict:
        """Calculate aggregate statistics from all iterations"""
        aggregate_stats = {}
        for key, values in all_trade_stats.items():
            if values:
                aggregate_stats[f'{key}_mean'] = np.mean(values)
                aggregate_stats[f'{key}_median'] = np.median(values)
                aggregate_stats[f'{key}_std'] = np.std(values)
        return aggregate_stats
    
    def _aggregate_detailed_stats(self, all_stats: List[DetailedTradeStats]) -> DetailedTradeStats:
        """Aggregate detailed stats across all iterations"""
        if not all_stats:
            return DetailedTradeStats()
        
        agg = DetailedTradeStats()
        
        # Simple averages
        agg.total_trades = np.mean([s.total_trades for s in all_stats])
        agg.winning_trades = np.mean([s.winning_trades for s in all_stats])
        agg.losing_trades = np.mean([s.losing_trades for s in all_stats])
        agg.win_rate = np.mean([s.win_rate for s in all_stats])
        
        # P&L stats
        agg.total_pnl = np.mean([s.total_pnl for s in all_stats])
        agg.avg_win_pnl = np.mean([s.avg_win_pnl for s in all_stats])
        agg.avg_loss_pnl = np.mean([s.avg_loss_pnl for s in all_stats])
        
        # Pip stats
        agg.avg_win_pips = np.mean([s.avg_win_pips for s in all_stats])
        agg.avg_loss_pips = np.mean([s.avg_loss_pips for s in all_stats])
        agg.max_win_pips = np.mean([s.max_win_pips for s in all_stats])
        agg.max_loss_pips = np.mean([s.max_loss_pips for s in all_stats])
        
        # Position sizing
        agg.avg_position_size = np.mean([s.avg_position_size for s in all_stats])
        agg.avg_position_size_millions = np.mean([s.avg_position_size_millions for s in all_stats])
        
        # Entry logic - aggregate percentages
        all_entry_logics = defaultdict(list)
        for s in all_stats:
            for logic, pct in s.entry_logic_pct.items():
                all_entry_logics[logic].append(pct)
        
        for logic, pcts in all_entry_logics.items():
            agg.entry_logic_pct[logic] = np.mean(pcts)
        
        # Exit logic - aggregate percentages
        all_exit_logics = defaultdict(list)
        for s in all_stats:
            for reason, pct in s.exit_logic_pct.items():
                all_exit_logics[reason].append(pct)
        
        for reason, pcts in all_exit_logics.items():
            agg.exit_logic_pct[reason] = np.mean(pcts)
        
        # TP hit rates
        agg.tp1_hit_pct = np.mean([s.tp1_hit_pct for s in all_stats])
        agg.tp2_hit_pct = np.mean([s.tp2_hit_pct for s in all_stats])
        agg.tp3_hit_pct = np.mean([s.tp3_hit_pct for s in all_stats])
        
        # TP exit sizes
        agg.tp1_exit_size_avg = np.mean([s.tp1_exit_size_avg for s in all_stats if s.tp1_exit_size_avg > 0])
        agg.tp2_exit_size_avg = np.mean([s.tp2_exit_size_avg for s in all_stats if s.tp2_exit_size_avg > 0])
        agg.tp3_exit_size_avg = np.mean([s.tp3_exit_size_avg for s in all_stats if s.tp3_exit_size_avg > 0])
        
        # Additional insights
        agg.avg_trade_duration_hours = np.mean([s.avg_trade_duration_hours for s in all_stats])
        agg.profit_factor = np.mean([s.profit_factor for s in all_stats if s.profit_factor != float('inf')])
        agg.avg_risk_reward = np.mean([s.avg_risk_reward for s in all_stats if s.avg_risk_reward != float('inf')])
        
        return agg


class ResultsVisualizer:
    """Handles all visualization and plotting"""
    
    @staticmethod
    def generate_calendar_year_analysis(results_df: pd.DataFrame, config_name: str, 
                                      currency: Optional[str] = None, 
                                      show_plots: bool = False, 
                                      save_plots: bool = False) -> pd.DataFrame:
        """Generate calendar year performance analysis"""
        # Implementation would go here
        # For now, returning empty dataframe
        return pd.DataFrame()
    
    @staticmethod
    def generate_comparison_plots(all_results: Dict, currency: str, 
                                show_plots: bool = False, 
                                save_plots: bool = False):
        """Generate comparison plots for multiple strategies"""
        if not show_plots and not save_plots:
            return
            
        # Import plotting if needed
        import matplotlib.pyplot as plt
        from strategy_code.Prod_plotting import plot_production_results
        
        # Plot results for each strategy
        for config_name, result_data in all_results.items():
            if 'extra_data' in result_data and 'last_results' in result_data['extra_data']:
                last_results = result_data['extra_data']['last_results']
                last_sample_df = result_data['extra_data'].get('last_sample_df')
                
                if last_results and last_sample_df is not None:
                    try:
                        # Create the plot
                        fig = plot_production_results(
                            last_sample_df,
                            last_results
                        )
                        if save_plots:
                            config_short = config_name.replace(":", "").replace(" ", "_").lower()
                            filename = f'charts/{currency}_{config_short}_trades.png'
                            fig.savefig(filename, dpi=150, bbox_inches='tight')
                            print(f"📊 Trades plot saved to: {filename}")
                        if show_plots:
                            plt.show()
                        else:
                            plt.close(fig)
                    except Exception as e:
                        print(f"⚠️  Error creating trade plot: {str(e)}")
        
        # Create comparison plots
        if save_plots or show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{currency} Strategy Comparison', fontsize=16)
            
            # Sharpe Ratio comparison
            ax = axes[0, 0]
            sharpe_data = {}
            for config_name, result_data in all_results.items():
                results_df = result_data['results_df']
                config_short = 'Config 1' if 'Ultra-Tight' in config_name else 'Config 2'
                sharpe_data[config_short] = results_df['sharpe_ratio'].values
            
            ax.boxplot(sharpe_data.values(), labels=sharpe_data.keys())
            ax.set_title('Sharpe Ratio Distribution')
            ax.set_ylabel('Sharpe Ratio')
            ax.grid(True, alpha=0.3)
            
            # Win Rate comparison
            ax = axes[0, 1]
            winrate_data = {}
            for config_name, result_data in all_results.items():
                results_df = result_data['results_df']
                config_short = 'Config 1' if 'Ultra-Tight' in config_name else 'Config 2'
                winrate_data[config_short] = results_df['win_rate'].values
            
            ax.boxplot(winrate_data.values(), labels=winrate_data.keys())
            ax.set_title('Win Rate Distribution')
            ax.set_ylabel('Win Rate (%)')
            ax.grid(True, alpha=0.3)
            
            # Total Return comparison
            ax = axes[1, 0]
            return_data = {}
            for config_name, result_data in all_results.items():
                results_df = result_data['results_df']
                config_short = 'Config 1' if 'Ultra-Tight' in config_name else 'Config 2'
                return_data[config_short] = results_df['total_return'].values
            
            ax.boxplot(return_data.values(), labels=return_data.keys())
            ax.set_title('Total Return Distribution')
            ax.set_ylabel('Total Return (%)')
            ax.grid(True, alpha=0.3)
            
            # Max Drawdown comparison
            ax = axes[1, 1]
            dd_data = {}
            for config_name, result_data in all_results.items():
                results_df = result_data['results_df']
                config_short = 'Config 1' if 'Ultra-Tight' in config_name else 'Config 2'
                dd_data[config_short] = results_df['max_drawdown'].values
            
            ax.boxplot(dd_data.values(), labels=dd_data.keys())
            ax.set_title('Max Drawdown Distribution')
            ax.set_ylabel('Max Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f'charts/{currency}_metrics_comparison.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"\n📊 Comparison plot saved to: {filename}")
            
            if show_plots:
                plt.show()
    
    @staticmethod
    def plot_monte_carlo_results(results_df: pd.DataFrame, config_name: str):
        """Plot Monte Carlo simulation results"""
        # Implementation would go here
        pass


class BaseRunner(ABC):
    """Abstract base class for different running modes"""
    
    def __init__(self, data_manager: DataManager, config: BacktestConfig):
        self.data_manager = data_manager
        self.config = config
        self.trade_analyzer = EnhancedTradeAnalyzer()
        self.trade_exporter = TradeExporter()
        self.visualizer = ResultsVisualizer()
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the strategy testing"""
        pass
    
    def _print_high_level_stats_table(self, results_df: pd.DataFrame):
        """Print high-level statistics table"""
        print("\n📊 HIGH-LEVEL STATISTICS TABLE:")
        print("─" * 120)
        print(f"{'Metric':<20} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'25%':>12} {'Median':>12} {'75%':>12} {'Max':>12}")
        print("─" * 120)
        
        metrics = [
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Total Return (%)', 'total_return'),
            ('Win Rate (%)', 'win_rate'),
            ('Profit Factor', 'profit_factor'),
            ('Max Drawdown (%)', 'max_drawdown'),
            ('Total Trades', 'total_trades'),
            ('Avg Trade ($)', 'avg_trade'),
            ('Avg Win ($)', 'avg_win'),
            ('Avg Loss ($)', 'avg_loss'),
            ('Recovery Factor', 'recovery_factor'),
            ('SQN', 'sqn')
        ]
        
        for display_name, col_name in metrics:
            if col_name in results_df.columns:
                mean_val = results_df[col_name].mean()
                std_val = results_df[col_name].std()
                min_val = results_df[col_name].min()
                q25_val = results_df[col_name].quantile(0.25)
                median_val = results_df[col_name].median()
                q75_val = results_df[col_name].quantile(0.75)
                max_val = results_df[col_name].max()
                
                # Format based on metric type
                if 'ratio' in col_name.lower() or col_name == 'profit_factor' or col_name == 'sqn' or col_name == 'recovery_factor':
                    format_str = "{:<20} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}"
                elif col_name == 'total_trades':
                    format_str = "{:<20} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f}"
                else:
                    format_str = "{:<20} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f}"
                
                print(format_str.format(display_name, mean_val, std_val, min_val, q25_val, median_val, q75_val, max_val))
        
        print("─" * 120)
    
    def _print_yearly_metrics(self, results_df: pd.DataFrame):
        """Print metrics grouped by year"""
        if 'start_year' not in results_df.columns:
            return
        
        print("\n📅 YEARLY PERFORMANCE METRICS:")
        print("─" * 90)
        print(f"{'Year':<10} {'Count':>8} {'Avg Sharpe':>12} {'Avg Return%':>12} {'Avg WinRate%':>12} {'Avg MaxDD%':>12} {'Avg Trades':>12}")
        print("─" * 90)
        
        # Group by year
        yearly_groups = results_df.groupby('start_year')
        
        for year, group in yearly_groups:
            count = len(group)
            avg_sharpe = group['sharpe_ratio'].mean()
            avg_return = group['total_return'].mean()
            avg_winrate = group['win_rate'].mean()
            avg_maxdd = group['max_drawdown'].mean()
            avg_trades = group['total_trades'].mean()
            
            print(f"{year:<10} {count:>8} {avg_sharpe:>12.3f} {avg_return:>12.1f} {avg_winrate:>12.1f} {avg_maxdd:>12.1f} {avg_trades:>12.0f}")
        
        print("─" * 90)
        
        # Also show year range grouping if samples span multiple years
        if 'year_range' in results_df.columns:
            year_range_groups = results_df.groupby('year_range')
            if len(year_range_groups) < len(results_df):  # Only show if there are multi-year samples
                print("\n📅 MULTI-YEAR SAMPLES:")
                print("─" * 90)
                print(f"{'Year Range':<15} {'Count':>8} {'Avg Sharpe':>12} {'Avg Return%':>12} {'Avg WinRate%':>12}")
                print("─" * 90)
                
                for year_range, group in year_range_groups:
                    if '-' in str(year_range):  # Only show multi-year ranges
                        count = len(group)
                        avg_sharpe = group['sharpe_ratio'].mean()
                        avg_return = group['total_return'].mean()
                        avg_winrate = group['win_rate'].mean()
                        
                        print(f"{year_range:<15} {count:>8} {avg_sharpe:>12.3f} {avg_return:>12.1f} {avg_winrate:>12.1f}")
                
                print("─" * 90)
    
    def _print_summary(self, results_df: pd.DataFrame, config_name: str, detailed_stats: Optional[DetailedTradeStats] = None, monte_carlo_stats: Optional[Dict] = None):
        """Print enhanced summary statistics"""
        print(f"\n{'='*80}")
        print(f"MONTE CARLO RESULTS - {config_name}")
        print(f"{'='*80}")
        print(f"Iterations: {len(results_df)}")
        
        print(f"\n📈 PERFORMANCE METRICS (Mean ± Std):")
        print(f"  Sharpe Ratio:     {results_df['sharpe_ratio'].mean():.3f} ± {results_df['sharpe_ratio'].std():.3f}")
        print(f"  Total Return:     {results_df['total_return'].mean():.1f}% ± {results_df['total_return'].std():.1f}%")
        print(f"  Win Rate:         {results_df['win_rate'].mean():.1f}% ± {results_df['win_rate'].std():.1f}%")
        print(f"  Profit Factor:    {results_df['profit_factor'].mean():.2f} ± {results_df['profit_factor'].std():.2f}")
        print(f"  Max Drawdown:     {results_df['max_drawdown'].mean():.1f}% ± {results_df['max_drawdown'].std():.1f}%")
        
        print(f"\n💰 TRADE STATISTICS:")
        print(f"  Total Trades:     {results_df['total_trades'].mean():.0f} ± {results_df['total_trades'].std():.0f}")
        
        # Use monte carlo stats if available
        if monte_carlo_stats and 'avg_win_pnl_mean' in monte_carlo_stats:
            avg_win_pnl = monte_carlo_stats.get('avg_win_pnl_mean', 0)
            avg_loss_pnl = monte_carlo_stats.get('avg_loss_pnl_mean', 0)
            win_rate = results_df['win_rate'].mean() / 100
            avg_trade_pnl = avg_win_pnl * win_rate + avg_loss_pnl * (1 - win_rate)
            print(f"  Avg Trade P&L:    ${avg_trade_pnl:.2f}")
            print(f"  Avg Win P&L:      ${avg_win_pnl:.2f}")
            print(f"  Avg Loss P&L:     ${avg_loss_pnl:.2f}")
            
            # Pip statistics
            if 'avg_win_pips_mean' in monte_carlo_stats:
                print(f"\n📊 PIP STATISTICS:")
                print(f"  Avg Win Pips:     {monte_carlo_stats.get('avg_win_pips_mean', 0):.1f}")
                print(f"  Avg Loss Pips:    {monte_carlo_stats.get('avg_loss_pips_mean', 0):.1f}")
        else:
            # Fallback calculation
            avg_trade_pnl = results_df['avg_win'].mean() * results_df['win_rate'].mean()/100 + results_df['avg_loss'].mean() * (100-results_df['win_rate'].mean())/100
            print(f"  Avg Trade P&L:    ${avg_trade_pnl:.2f}")
        
        if detailed_stats:
            print(f"\n📊 DETAILED ANALYTICS (Averaged across all iterations):")
            
            print(f"\n  🎯 Win/Loss Analysis:")
            print(f"    Avg Win:        ${detailed_stats.avg_win_pnl:.2f} ({detailed_stats.avg_win_pips:.1f} pips)")
            print(f"    Avg Loss:       ${detailed_stats.avg_loss_pnl:.2f} ({detailed_stats.avg_loss_pips:.1f} pips)")
            print(f"    Max Win:        {detailed_stats.max_win_pips:.1f} pips")
            print(f"    Max Loss:       {detailed_stats.max_loss_pips:.1f} pips")
            print(f"    Risk/Reward:    {detailed_stats.avg_risk_reward:.2f}:1")
            
            print(f"\n  💵 Position Sizing:")
            print(f"    Avg Size:       {detailed_stats.avg_position_size_millions:.2f}M units")
            
            print(f"\n  🚀 Entry Logic Breakdown:")
            for logic, pct in sorted(detailed_stats.entry_logic_pct.items(), key=lambda x: x[1], reverse=True):
                print(f"    {logic:<30} {pct:>5.1f}%")
            
            print(f"\n  🎯 Take Profit Hit Rates:")
            if detailed_stats.tp1_hit_pct > 0:
                print(f"    TP1 Hit:        {detailed_stats.tp1_hit_pct:.1f}% (avg exit: {detailed_stats.tp1_exit_size_avg:.2f}M)")
            else:
                print(f"    TP1 Hit:        {detailed_stats.tp1_hit_pct:.1f}%")
            
            if detailed_stats.tp2_hit_pct > 0:
                print(f"    TP2 Hit:        {detailed_stats.tp2_hit_pct:.1f}% (avg exit: {detailed_stats.tp2_exit_size_avg:.2f}M)")
            else:
                print(f"    TP2 Hit:        {detailed_stats.tp2_hit_pct:.1f}%")
            
            if detailed_stats.tp3_hit_pct > 0:
                print(f"    TP3 Hit:        {detailed_stats.tp3_hit_pct:.1f}% (avg exit: {detailed_stats.tp3_exit_size_avg:.2f}M)")
            else:
                print(f"    TP3 Hit:        {detailed_stats.tp3_hit_pct:.1f}%")
            
            print(f"\n  🚪 Exit Reason Breakdown:")
            exit_sorted = sorted(detailed_stats.exit_logic_pct.items(), key=lambda x: x[1], reverse=True)
            for reason, pct in exit_sorted[:5]:  # Top 5 exit reasons
                reason_display = reason.replace('_', ' ').title()
                print(f"    {reason_display:<30} {pct:>5.1f}%")
            
            print(f"\n  ⏱️  Trade Duration:")
            print(f"    Average:        {detailed_stats.avg_trade_duration_hours:.1f} hours")
            
            # Add consecutive wins/losses if available in monte_carlo_stats
            if monte_carlo_stats:
                print(f"\n  🔥 Win/Loss Streaks:")
                if 'max_consecutive_wins_mean' in monte_carlo_stats:
                    print(f"    Max Consecutive Wins:  {monte_carlo_stats.get('max_consecutive_wins_mean', 0):.1f}")
                    print(f"    Avg Consecutive Wins:  {monte_carlo_stats.get('avg_consecutive_wins_mean', 0):.1f}")
                if 'max_consecutive_losses_mean' in monte_carlo_stats:
                    print(f"    Max Consecutive Losses: {monte_carlo_stats.get('max_consecutive_losses_mean', 0):.1f}")
                    print(f"    Avg Consecutive Losses: {monte_carlo_stats.get('avg_consecutive_losses_mean', 0):.1f}")
            
            # Consistency metrics
            print(f"\n  📈 Consistency Metrics:")
            positive_sharpe = (results_df['sharpe_ratio'] > 1.0).sum()
            profitable = (results_df['total_return'] > 0).sum()
            print(f"    Sharpe > 1.0:   {positive_sharpe}/{len(results_df)} ({positive_sharpe/len(results_df)*100:.0f}%)")
            print(f"    Profitable:     {profitable}/{len(results_df)} ({profitable/len(results_df)*100:.0f}%)")
            
            # Drawdown calculation note
            print(f"\n  📉 Drawdown Calculation Method:")
            print(f"    • Uses Mark-to-Market (MTM) equity including unrealized P&L")
            print(f"    • Calculated on every 15-minute bar")
            print(f"    • High-water mark method: (Current Equity - Peak Equity) / Peak Equity")
            print(f"    • Reports worst peak-to-trough percentage decline")
        else:
            # Fallback to basic stats
            print(f"  Avg Win:          ${results_df['avg_win'].mean():.2f} ± ${results_df['avg_win'].std():.2f}")
            print(f"  Avg Loss:         ${results_df['avg_loss'].mean():.2f} ± ${results_df['avg_loss'].std():.2f}")


class SingleCurrencyRunner(BaseRunner):
    """Runner for single currency mode"""
    
    def __init__(self, data_manager: DataManager, config: BacktestConfig, 
                 currency: str = 'AUDUSD'):
        super().__init__(data_manager, config)
        self.currency = currency
    
    def run(self) -> Dict[str, Any]:
        """Run single currency backtesting"""
        print(f"\n{'='*80}")
        print(f"RUNNING SINGLE CURRENCY MODE - {self.currency}")
        print(f"{'='*80}")
        
        # Load data
        df = self.data_manager.load_currency_data(self.currency)
        
        # Apply date range filter if specified
        if self.config.date_range:
            df = self.data_manager.filter_date_range(df, self.config.date_range)
        
        all_results = {}
        
        # Test both strategy configurations
        for strategy_type, config_name in [
            (StrategyType.ULTRA_TIGHT_RISK, "Config 1: Ultra-Tight Risk Management"),
            (StrategyType.SCALPING, "Config 2: Scalping Strategy")
        ]:
            print(f"\n{'='*60}")
            print(f"Testing {config_name}")
            print(f"{'='*60}")
            
            # Create strategy
            strategy = StrategyFactory.create_strategy(strategy_type, self.config)
            
            # Run Monte Carlo simulation
            simulator = MonteCarloSimulator(strategy, self.trade_analyzer)
            results_df, extra_data = simulator.run_simulation(df, self.config)
            
            # Get aggregated detailed stats
            detailed_stats = extra_data.get('aggregated_detailed_stats')
            monte_carlo_stats = extra_data.get('monte_carlo_stats')
            
            # Print high-level stats table first
            self._print_high_level_stats_table(results_df)
            
            # Print yearly metrics
            self._print_yearly_metrics(results_df)
            
            # Print enhanced summary
            self._print_summary(results_df, config_name, detailed_stats, monte_carlo_stats)
            
            # Export trades if enabled
            if self.config.export_trades and 'last_results' in extra_data:
                last_results = extra_data['last_results']
                if 'trades' in last_results and last_results['trades']:
                    self.trade_exporter.export_trades_detailed(
                        last_results['trades'], 
                        self.currency, 
                        config_name
                    )
            
            # Store results
            all_results[config_name] = {
                'results_df': results_df,
                'extra_data': extra_data,
                'strategy_type': strategy_type,
                'detailed_stats': detailed_stats
            }
        
        # Generate comparison plots
        if not self.config.save_plots and not self.config.show_plots:
            print("\n⚡ Skipping plot generation (use --show-plots or --save-plots to enable)")
        else:
            self.visualizer.generate_comparison_plots(
                all_results, 
                self.currency, 
                self.config.show_plots, 
                self.config.save_plots
            )
        
        return all_results


class MultiCurrencyRunner(BaseRunner):
    """Runner for multi-currency mode"""
    
    def __init__(self, data_manager: DataManager, config: BacktestConfig, 
                 currencies: List[str] = None):
        super().__init__(data_manager, config)
        self.currencies = currencies or ['GBPUSD', 'EURUSD', 'USDJPY', 'NZDUSD', 'USDCAD']
    
    def run(self) -> Dict[str, Any]:
        """Run multi-currency backtesting"""
        print(f"\n{'='*80}")
        print(f"RUNNING MULTI-CURRENCY MODE")
        print(f"Currencies: {', '.join(self.currencies)}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for currency in self.currencies:
            runner = SingleCurrencyRunner(self.data_manager, self.config, currency)
            currency_results = runner.run()
            all_results[currency] = currency_results
        
        # Generate multi-currency summary
        self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, all_results: Dict[str, Any]):
        """Generate multi-currency summary"""
        print(f"\n{'='*80}")
        print("MULTI-CURRENCY SUMMARY")
        print(f"{'='*80}")
        
        # Summary implementation would go here
        pass


class StrategyRunner:
    """Main strategy runner orchestrator"""
    
    def __init__(self):
        self.config = None
        self.data_manager = None
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description=self._get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        
        # Add arguments
        parser.add_argument('--mode', type=str, default='single', 
                          choices=['single', 'multi', 'crypto', 'custom'],
                          help='Testing mode')
        parser.add_argument('--currency', type=str, default='AUDUSD',
                          help='Currency pair to test in single mode')
        parser.add_argument('--currencies', nargs='+', 
                          default=['GBPUSD', 'EURUSD', 'USDJPY', 'NZDUSD', 'USDCAD'],
                          help='List of currency pairs for multi mode')
        parser.add_argument('--iterations', type=int, default=50,
                          help='Number of Monte Carlo iterations')
        parser.add_argument('--sample-size', type=int, default=8000,
                          help='Sample size for each iteration')
        parser.add_argument('--no-plots', action='store_true',
                          help='Skip all chart generation')
        parser.add_argument('--show-plots', action='store_true',
                          help='Display charts in GUI window')
        parser.add_argument('--save-plots', action='store_true',
                          help='Save charts as PNG files')
        parser.add_argument('--realistic-costs', action='store_true', default=True,
                          help='Enable realistic trading costs')
        parser.add_argument('--no-realistic-costs', dest='realistic_costs', action='store_false',
                          help='Disable realistic trading costs')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')
        parser.add_argument('--no-daily-sharpe', action='store_true',
                          help='Disable daily resampling for Sharpe ratio')
        parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                          help='Specify date range (YYYY-MM-DD YYYY-MM-DD)')
        parser.add_argument('--export-trades', action='store_true', default=True,
                          help='Export detailed trades to CSV')
        parser.add_argument('--no-export-trades', dest='export_trades', action='store_false',
                          help='Disable trade export')
        parser.add_argument('--version', '-v', action='version', 
                          version=f'%(prog)s {__version__}')
        
        return parser.parse_args()
    
    def _get_description(self) -> str:
        """Get program description"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║      MONTE CARLO STRATEGY TESTER - Enhanced with Detailed Analytics         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Test proven trading strategies with comprehensive trade insights:           ║
║                                                                              ║
║  FOREX STRATEGIES:                                                           ║
║  • Config 1: Ultra-Tight Risk Management (0.2% risk, 10 pip stops)         ║
║  • Config 2: Scalping Strategy (0.1% risk, 5 pip stops)                    ║
║                                                                              ║
║  NEW ANALYTICS:                                                              ║
║  • Detailed win/loss pip analysis                                           ║
║  • Entry logic breakdown (Standard vs Relaxed)                              ║
║  • Exit reason statistics                                                    ║
║  • Take profit hit rates with position sizing                               ║
║  • Average trade duration and risk/reward ratios                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    
    def _get_epilog(self) -> str:
        """Get program epilog with examples"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           USAGE EXAMPLES                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 BASIC USAGE:
  python run_strategy_oop_enhanced.py
    → Runs default: AUDUSD, 50 iterations with detailed analytics

📊 SINGLE CURRENCY TESTING:
  python run_strategy_oop_enhanced.py --currency GBPUSD --iterations 100
    → Test British Pound with 100 iterations and full analytics
  
  python run_strategy_oop_enhanced.py --sample-size 100000
    → Large sample test (~1 year of data) with comprehensive insights

🌍 MULTI-CURRENCY TESTING:
  python run_strategy_oop_enhanced.py --mode multi
    → Test all major pairs with detailed breakdowns per currency

📈 VISUALIZATION OPTIONS:
  python run_strategy_oop_enhanced.py --show-plots
    → Display enhanced analytics charts interactively
  
  python run_strategy_oop_enhanced.py --save-plots
    → Save detailed analysis charts to PNG files

💡 ENHANCED FEATURES:
  • See exact pip performance for winners vs losers
  • Understand which entry logic performs better
  • Track TP hit rates and optimal exit strategies
  • Analyze trade duration patterns
        """
    
    def run(self, args: argparse.Namespace):
        """Run the strategy based on parsed arguments"""
        # Create configuration
        self.config = BacktestConfig(
            n_iterations=args.iterations,
            sample_size=args.sample_size,
            realistic_costs=args.realistic_costs,
            use_daily_sharpe=not args.no_daily_sharpe,
            debug_mode=args.debug,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
            export_trades=args.export_trades,
            calendar_analysis=True,  # Could add flag for this
            date_range=tuple(args.date_range) if args.date_range else None
        )
        
        # Handle plot flags
        if args.no_plots:
            self.config.show_plots = False
            self.config.save_plots = False
        
        # Create data manager
        self.data_manager = DataManager()
        
        # Run based on mode
        if args.mode == 'single':
            runner = SingleCurrencyRunner(self.data_manager, self.config, args.currency)
        elif args.mode == 'multi':
            runner = MultiCurrencyRunner(self.data_manager, self.config, args.currencies)
        elif args.mode == 'crypto':
            raise NotImplementedError("Crypto mode not yet implemented in OOP version")
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        return runner.run()


def main():
    """Main entry point"""
    runner = StrategyRunner()
    args = runner.parse_arguments()
    
    try:
        results = runner.run(args)
        print("\n✅ Strategy testing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()