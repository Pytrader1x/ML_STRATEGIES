"""
Object-Oriented Strategy Runner - Monte Carlo Testing Framework
Refactored version with cleaner architecture and better organization
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
from tabulate import tabulate
from collections import defaultdict
import calendar

warnings.filterwarnings('ignore')

__version__ = "3.0.0"  # New OOP version


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
    n_iterations: int = 1
    sample_size: int = 2000
    realistic_costs: bool = True
    use_daily_sharpe: bool = True
    debug_mode: bool = True
    show_plots: bool = False
    save_plots: bool = False
    export_trades: bool = True
    calendar_analysis: bool = True
    date_range: Optional[Tuple[str, str]] = None


@dataclass
class BacktestResults:
    """Container for backtest results"""
    metrics: Dict[str, float]
    trades: Optional[List] = None
    equity_curve: Optional[pd.Series] = None
    monte_carlo_stats: Optional[Dict] = None
    calendar_performance: Optional[pd.DataFrame] = None
    
    
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
            sl_min_pips=5.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            tsl_activation_pips=15,  # Fixed: From 3 → 15 (allow TP1 to be reached first)
            tsl_min_profit_pips=1,
            tsl_initial_buffer_multiplier=1.0,
            trailing_atr_multiplier=1.2,  # Fixed: From 0.8 → 1.2 (wider trail distance)
            tp_range_market_multiplier=0.5,
            tp_trend_market_multiplier=0.7,
            tp_chop_market_multiplier=0.3,
            sl_range_market_multiplier=0.7,
            exit_on_signal_flip=False,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=False,
            partial_profit_sl_distance_ratio=0.5,
            partial_profit_size_percent=0.5,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            relaxed_mode=False,  # Require 3 confluence indicators for entry (NTI + MB + IC)
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
            sl_min_pips=3.0,
            sl_max_pips=5.0,
            sl_atr_multiplier=0.5,
            tp_atr_multipliers=(0.1, 0.2, 0.3),
            max_tp_percent=0.002,
            tsl_activation_pips=8,  # Fixed: From 2 → 8 (allow TP1 to be reached first)
            tsl_min_profit_pips=0.5,
            tsl_initial_buffer_multiplier=0.5,
            trailing_atr_multiplier=0.8,  # Fixed: From 0.5 → 0.8 (wider trail distance)
            tp_range_market_multiplier=0.3,
            tp_trend_market_multiplier=0.5,
            tp_chop_market_multiplier=0.2,
            sl_range_market_multiplier=0.5,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=0.0,
            signal_flip_min_time_hours=0.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=False,
            partial_profit_sl_distance_ratio=0.3,
            partial_profit_size_percent=0.7,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            relaxed_mode=False,  # Require 3 confluence indicators for entry (NTI + MB + IC)
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
        df = TIC.add_market_bias(df,ha_len=350, ha_len2=30)
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


class TradeAnalyzer:
    """Analyzes trade statistics and patterns"""
    
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
                stats = TradeAnalyzer._calculate_consecutive_stats(pnl_values)
                stats['num_wins'] = sum(1 for pnl in pnl_values if pnl > 0)
                stats['num_losses'] = sum(1 for pnl in pnl_values if pnl < 0)
                
                # Calculate exit statistics
                exit_stats = TradeAnalyzer._calculate_exit_statistics(trades)
                stats.update(exit_stats)
            else:
                stats = TradeAnalyzer._empty_stats()
        else:
            # Estimate from aggregate stats
            stats = TradeAnalyzer._estimate_stats_from_aggregates(results)
        
        return stats
    
    @staticmethod
    def _calculate_exit_statistics(trades: List[Any]) -> Dict[str, Any]:
        """Calculate detailed exit statistics"""
        exit_counts = defaultdict(int)
        exit_combinations = defaultdict(int)
        total_trades = len(trades)
        
        for trade in trades:
            # Get exit reason
            if hasattr(trade, 'exit_reason'):
                exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else str(trade.exit_reason)
            else:
                exit_reason = 'unknown'
            
            # Track single exit types
            exit_counts[exit_reason] += 1
            
            # Track TP combinations (e.g., TP1+TP2, TP1+TP2+TP3, etc.)
            if hasattr(trade, 'tp_hits') and trade.tp_hits > 0:
                tp_combo = []
                for i in range(1, trade.tp_hits + 1):
                    tp_combo.append(f'TP{i}')
                
                # Check if TSL or SL was also hit
                if 'tsl' in exit_reason.lower():
                    tp_combo.append('TSL')
                elif 'sl' in exit_reason.lower() or 'stop' in exit_reason.lower():
                    tp_combo.append('SL')
                
                combo_key = '+'.join(tp_combo)
                exit_combinations[combo_key] += 1
            else:
                # No TP hits, just the exit reason
                exit_combinations[exit_reason] += 1
        
        # Convert to percentages
        exit_stats = {}
        
        # Single exit type statistics
        for exit_type, count in exit_counts.items():
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            exit_stats[f'exit_{exit_type}_count'] = count
            exit_stats[f'exit_{exit_type}_pct'] = percentage
        
        # Combination statistics
        for combo, count in exit_combinations.items():
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            combo_key = combo.replace('+', '_').replace(' ', '_').lower()
            exit_stats[f'combo_{combo_key}_count'] = count
            exit_stats[f'combo_{combo_key}_pct'] = percentage
        
        # Summary statistics
        tp1_exits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 1)
        tp2_exits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 2)
        tp3_exits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 3)
        
        exit_stats['tp1_hit_count'] = tp1_exits
        exit_stats['tp1_hit_pct'] = (tp1_exits / total_trades * 100) if total_trades > 0 else 0
        exit_stats['tp2_hit_count'] = tp2_exits
        exit_stats['tp2_hit_pct'] = (tp2_exits / total_trades * 100) if total_trades > 0 else 0
        exit_stats['tp3_hit_count'] = tp3_exits
        exit_stats['tp3_hit_pct'] = (tp3_exits / total_trades * 100) if total_trades > 0 else 0
        
        return exit_stats
    
    @staticmethod
    def _calculate_consecutive_stats(pnl_values: List[float]) -> Dict[str, Any]:
        """Calculate consecutive win/loss statistics"""
        wins = [1 if pnl > 0 else 0 for pnl in pnl_values]
        losses = [1 if pnl < 0 else 0 for pnl in pnl_values]
        
        # Consecutive wins
        win_streaks = TradeAnalyzer._get_streaks(wins)
        loss_streaks = TradeAnalyzer._get_streaks(losses)
        
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
            'num_losses': 0,
            'tp1_hit_count': 0,
            'tp1_hit_pct': 0,
            'tp2_hit_count': 0,
            'tp2_hit_pct': 0,
            'tp3_hit_count': 0,
            'tp3_hit_pct': 0,
            'exit_take_profit_1_count': 0,
            'exit_take_profit_1_pct': 0,
            'exit_take_profit_2_count': 0,
            'exit_take_profit_2_pct': 0,
            'exit_take_profit_3_count': 0,
            'exit_take_profit_3_pct': 0,
            'exit_trailing_stop_count': 0,
            'exit_trailing_stop_pct': 0,
            'exit_stop_loss_count': 0,
            'exit_stop_loss_pct': 0
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
                # Determine partial exit type
                if hasattr(pe, 'exit_type'):
                    pe_type = pe.exit_type
                elif hasattr(pe, 'tp_level'):
                    if pe.tp_level == 0:
                        # tp_level=0 indicates partial profit taking before TP levels
                        pe_type = 'PPT'  # Partial Profit Taking
                    else:
                        # tp_level 1, 2, or 3 are actual TP exits
                        pe_type = f'TP{pe.tp_level}'
                else:
                    pe_type = 'PARTIAL'
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
    
    def __init__(self, strategy: OptimizedProdStrategy, trade_analyzer: TradeAnalyzer):
        self.strategy = strategy
        self.trade_analyzer = trade_analyzer
        self.iteration_details = []  # Store detailed results for table display
        self.yearly_performance = defaultdict(list)  # Store yearly breakdown
    
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
            'avg_loss_pips': []
        }
        
        # Track exit statistics across all iterations
        self.aggregate_exit_stats = defaultdict(list)
        
        # Clear previous iteration details
        self.iteration_details = []
        self.yearly_performance = defaultdict(list)
        
        print("\n🔄 Running Monte Carlo Simulations...")
        print(f"  Config: {config.n_iterations} iterations, {config.sample_size:,} rows per test\n")
        
        for i in range(config.n_iterations):
            # Get random sample
            sample_df = self._get_sample(df, config.sample_size)
            
            # Get date range of sample
            start_date = sample_df.index[0]
            end_date = sample_df.index[-1]
            
            # Run backtest
            results = self.strategy.run_backtest(sample_df)
            
            # Extract metrics
            metrics = self._extract_metrics(results, i)
            metrics['start_date'] = start_date
            metrics['end_date'] = end_date
            iteration_results.append(metrics)
            
            # Store detailed iteration info for table
            self._store_iteration_details(i + 1, results, start_date, end_date)
            
            # Track yearly performance
            self._track_yearly_performance(sample_df, results)
            
            # Track trade statistics
            trade_stats = self.trade_analyzer.calculate_trade_statistics(results)
            for key in all_trade_stats:
                if key in trade_stats:
                    all_trade_stats[key].append(trade_stats[key])
            
            # Track exit statistics
            for key, value in trade_stats.items():
                if key.startswith('exit_') or key.startswith('combo_') or key.startswith('tp'):
                    self.aggregate_exit_stats[key].append(value)
            
            # Keep last iteration data for return
            if i == config.n_iterations - 1:
                last_sample_df = sample_df
                last_results = results
            
            # Progress bar
            self._show_progress_bar(i + 1, config.n_iterations)
        
        # Create results dataframe
        results_df = pd.DataFrame(iteration_results)
        
        # Add aggregate statistics
        monte_carlo_stats = self._calculate_aggregate_stats(all_trade_stats)
        
        # Store the simulator instance for later use
        return results_df, {
            'last_sample_df': last_sample_df,
            'last_results': last_results,
            'monte_carlo_stats': monte_carlo_stats,
            'integrity_checks': integrity_checks,
            'simulator': self  # Store simulator instance for table display
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
    
    def _store_iteration_details(self, iteration: int, results: Dict, start_date, end_date):
        """Store iteration details for table display"""
        sharpe = results.get('sharpe_ratio', 0)
        total_return = results.get('total_return', 0)
        win_rate = results.get('win_rate', 0)
        trades = results.get('total_trades', 0)
        max_dd = results.get('max_drawdown', 0)
        profit_factor = results.get('profit_factor', 0)
        avg_win = results.get('avg_win', 0)
        avg_loss = results.get('avg_loss', 0)
        
        # Get win/loss counts
        wins = int(trades * win_rate / 100) if trades > 0 else 0
        losses = trades - wins
        
        # Sharpe quality indicator
        if sharpe >= 2.0:
            quality = "Excellent"
        elif sharpe >= 1.5:
            quality = "Very Good"
        elif sharpe >= 1.0:
            quality = "Good"
        elif sharpe >= 0.5:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        self.iteration_details.append({
            'Iter': iteration,
            'Sharpe': f"{sharpe:.3f}",
            'Quality': quality,
            'Return%': f"{total_return:.1f}",
            'Win Rate%': f"{win_rate:.1f}",
            'Trades': trades,
            'W/L': f"{wins}/{losses}",
            'PF': f"{profit_factor:.2f}",
            'Avg Win': f"${avg_win:,.0f}",
            'Avg Loss': f"${avg_loss:,.0f}",
            'Max DD%': f"{max_dd:.1f}",
            'Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        })
    
    def _track_yearly_performance(self, sample_df: pd.DataFrame, results: Dict):
        """Track performance by calendar year"""
        if 'trades' not in results or not results['trades']:
            return
            
        # Group trades by year
        trades_by_year = defaultdict(list)
        for trade in results['trades']:
            if hasattr(trade, 'exit_time') and trade.exit_time:
                year = trade.exit_time.year
                trades_by_year[year].append(trade)
        
        # Calculate metrics for each year
        for year, year_trades in trades_by_year.items():
            if not year_trades:
                continue
                
            total_pnl = sum(t.pnl for t in year_trades if hasattr(t, 'pnl') and t.pnl is not None)
            num_trades = len(year_trades)
            wins = sum(1 for t in year_trades if hasattr(t, 'pnl') and t.pnl and t.pnl > 0)
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
            
            self.yearly_performance[year].append({
                'total_pnl': total_pnl,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'wins': wins,
                'losses': num_trades - wins
            })
    
    def _show_progress_bar(self, current: int, total: int):
        """Show a progress bar for simulations"""
        progress = current / total
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r  Progress: [{bar}] {current}/{total} ({progress*100:.0f}%)", end='', flush=True)
        if current == total:
            print()  # New line when complete
    
    def display_results_table(self, config_name: str):
        """Display Monte Carlo results in a pretty table format"""
        if not self.iteration_details:
            return
            
        print(f"\n📊 DETAILED MONTE CARLO RESULTS - {config_name}")
        print("="*150)
        
        # Display main results table
        print(tabulate(self.iteration_details, headers='keys', tablefmt='grid', 
                      numalign='right', stralign='left'))
        
        # Calculate and display summary statistics
        self._display_summary_statistics()
        
        # Display yearly performance breakdown
        self._display_yearly_breakdown()
        
        # Display exit statistics
        self._display_exit_statistics()
    
    def _display_summary_statistics(self):
        """Display summary statistics from all iterations"""
        print("\n📈 SUMMARY STATISTICS")
        print("="*80)
        
        # Extract numeric values for calculations
        sharpe_values = [float(d['Sharpe']) for d in self.iteration_details]
        return_values = [float(d['Return%']) for d in self.iteration_details]
        win_rate_values = [float(d['Win Rate%']) for d in self.iteration_details]
        max_dd_values = [float(d['Max DD%']) for d in self.iteration_details]
        pf_values = [float(d['PF']) for d in self.iteration_details]
        
        # Create summary data
        summary_data = [
            ['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            ['Sharpe Ratio', f"{np.mean(sharpe_values):.3f}", f"{np.median(sharpe_values):.3f}", 
             f"{np.std(sharpe_values):.3f}", f"{np.min(sharpe_values):.3f}", 
             f"{np.max(sharpe_values):.3f}", f"{np.max(sharpe_values) - np.min(sharpe_values):.3f}"],
            ['Return %', f"{np.mean(return_values):.1f}", f"{np.median(return_values):.1f}", 
             f"{np.std(return_values):.1f}", f"{np.min(return_values):.1f}", 
             f"{np.max(return_values):.1f}", f"{np.max(return_values) - np.min(return_values):.1f}"],
            ['Win Rate %', f"{np.mean(win_rate_values):.1f}", f"{np.median(win_rate_values):.1f}", 
             f"{np.std(win_rate_values):.1f}", f"{np.min(win_rate_values):.1f}", 
             f"{np.max(win_rate_values):.1f}", f"{np.max(win_rate_values) - np.min(win_rate_values):.1f}"],
            ['Max Drawdown %', f"{np.mean(max_dd_values):.1f}", f"{np.median(max_dd_values):.1f}", 
             f"{np.std(max_dd_values):.1f}", f"{np.min(max_dd_values):.1f}", 
             f"{np.max(max_dd_values):.1f}", f"{np.max(max_dd_values) - np.min(max_dd_values):.1f}"],
            ['Profit Factor', f"{np.mean(pf_values):.2f}", f"{np.median(pf_values):.2f}", 
             f"{np.std(pf_values):.2f}", f"{np.min(pf_values):.2f}", 
             f"{np.max(pf_values):.2f}", f"{np.max(pf_values) - np.min(pf_values):.2f}"]
        ]
        
        print(tabulate(summary_data[1:], headers=summary_data[0], tablefmt='simple', 
                      numalign='right', stralign='left'))
        
        # Distribution analysis
        print("\n📊 DISTRIBUTION ANALYSIS")
        print(f"  Sharpe > 2.0 (Excellent): {sum(1 for s in sharpe_values if s >= 2.0)} iterations ({sum(1 for s in sharpe_values if s >= 2.0)/len(sharpe_values)*100:.1f}%)")
        print(f"  Sharpe > 1.5 (Very Good): {sum(1 for s in sharpe_values if s >= 1.5)} iterations ({sum(1 for s in sharpe_values if s >= 1.5)/len(sharpe_values)*100:.1f}%)")
        print(f"  Sharpe > 1.0 (Good):      {sum(1 for s in sharpe_values if s >= 1.0)} iterations ({sum(1 for s in sharpe_values if s >= 1.0)/len(sharpe_values)*100:.1f}%)")
        print(f"  Sharpe < 0.5 (Poor):      {sum(1 for s in sharpe_values if s < 0.5)} iterations ({sum(1 for s in sharpe_values if s < 0.5)/len(sharpe_values)*100:.1f}%)")
    
    def _display_yearly_breakdown(self):
        """Display performance breakdown by calendar year"""
        if not self.yearly_performance:
            return
            
        print("\n📅 YEARLY PERFORMANCE BREAKDOWN")
        print("="*120)
        
        # Aggregate yearly data
        yearly_summary = []
        for year in sorted(self.yearly_performance.keys()):
            year_data = self.yearly_performance[year]
            if not year_data:
                continue
                
            # Calculate aggregated metrics for the year
            total_pnl = sum(d['total_pnl'] for d in year_data)
            avg_pnl = total_pnl / len(year_data)
            total_trades = sum(d['num_trades'] for d in year_data)
            avg_trades = total_trades / len(year_data)
            avg_win_rate = np.mean([d['win_rate'] for d in year_data])
            total_wins = sum(d['wins'] for d in year_data)
            total_losses = sum(d['losses'] for d in year_data)
            
            yearly_summary.append({
                'Year': year,
                'Samples': len(year_data),
                'Total P&L': f"${total_pnl:,.0f}",
                'Avg P&L/Sample': f"${avg_pnl:,.0f}",
                'Total Trades': total_trades,
                'Avg Trades/Sample': f"{avg_trades:.0f}",
                'Avg Win Rate%': f"{avg_win_rate:.1f}",
                'Total W/L': f"{total_wins}/{total_losses}"
            })
        
        if yearly_summary:
            print(tabulate(yearly_summary, headers='keys', tablefmt='grid', 
                          numalign='right', stralign='left'))
        else:
            print("  No yearly data available")
    
    def _display_exit_statistics(self):
        """Display detailed exit statistics across all iterations"""
        if not self.aggregate_exit_stats:
            return
            
        print("\n🎯 EXIT STATISTICS ANALYSIS")
        print("="*100)
        
        # Calculate mean percentages for each exit type
        exit_summary = []
        
        # Process exit types (using actual ExitReason enum values)
        exit_types = ['take_profit_1', 'take_profit_2', 'take_profit_3', 'trailing_stop', 'stop_loss', 'tp1_pullback', 'signal_flip', 'end_of_data']
        for exit_type in exit_types:
            pct_key = f"exit_{exit_type}_pct"
            count_key = f"exit_{exit_type}_count"
            
            if pct_key in self.aggregate_exit_stats:
                percentages = self.aggregate_exit_stats[pct_key]
                counts = self.aggregate_exit_stats.get(count_key, [])
                
                if percentages:
                    # Format exit names for display
                    if exit_type.startswith('take_profit_'):
                        exit_name = exit_type.replace('take_profit_', 'TP').upper()
                    elif exit_type == 'trailing_stop':
                        exit_name = 'TSL'
                    elif exit_type == 'stop_loss':
                        exit_name = 'SL'
                    elif exit_type == 'tp1_pullback':
                        exit_name = 'TP1 PULLBACK'
                    elif exit_type == 'signal_flip':
                        exit_name = 'SIGNAL FLIP'
                    elif exit_type == 'end_of_data':
                        exit_name = 'END OF DATA'
                    else:
                        exit_name = exit_type.replace('_', ' ').upper()
                    
                    exit_summary.append({
                        'Exit Type': exit_name,
                        'Avg %': f"{np.mean(percentages):.1f}",
                        'Min %': f"{np.min(percentages):.1f}",
                        'Max %': f"{np.max(percentages):.1f}",
                        'Std Dev': f"{np.std(percentages):.1f}",
                        'Total Count': sum(counts) if counts else 0
                    })
        
        if exit_summary:
            print("\n🔸 Individual Exit Types:")
            print(tabulate(exit_summary, headers='keys', tablefmt='simple', 
                          numalign='right', stralign='left'))
        
        # Process exit combinations
        combo_summary = []
        combo_keys = [k for k in self.aggregate_exit_stats.keys() if k.startswith('combo_') and k.endswith('_pct')]
        
        for combo_key in sorted(combo_keys):
            percentages = self.aggregate_exit_stats[combo_key]
            count_key = combo_key.replace('_pct', '_count')
            counts = self.aggregate_exit_stats.get(count_key, [])
            
            if percentages and np.mean(percentages) > 0.1:  # Only show combinations with >0.1% average
                combo_name = combo_key.replace('combo_', '').replace('_pct', '').replace('_', '+').upper()
                combo_summary.append({
                    'Exit Combination': combo_name,
                    'Avg %': f"{np.mean(percentages):.1f}",
                    'Min %': f"{np.min(percentages):.1f}",
                    'Max %': f"{np.max(percentages):.1f}",
                    'Total Count': sum(counts) if counts else 0
                })
        
        if combo_summary:
            print("\n🔸 Exit Combinations (>0.1% avg):")
            # Sort by average percentage descending
            combo_summary.sort(key=lambda x: float(x['Avg %']), reverse=True)
            print(tabulate(combo_summary, headers='keys', tablefmt='simple', 
                          numalign='right', stralign='left'))
        
        # Summary insights
        print("\n💡 Exit Insights:")
        tp1_pct = np.mean(self.aggregate_exit_stats.get('exit_take_profit_1_pct', [0]))
        tp2_pct = np.mean(self.aggregate_exit_stats.get('exit_take_profit_2_pct', [0]))
        tp3_pct = np.mean(self.aggregate_exit_stats.get('exit_take_profit_3_pct', [0]))
        
        print(f"  • TP1 reached in {tp1_pct:.1f}% of trades on average")
        print(f"  • TP2 reached in {tp2_pct:.1f}% of trades on average")
        print(f"  • TP3 reached in {tp3_pct:.1f}% of trades on average")
        
        tsl_pct = np.mean(self.aggregate_exit_stats.get('exit_trailing_stop_pct', [0]))
        sl_pct = np.mean(self.aggregate_exit_stats.get('exit_stop_loss_pct', [0]))
        
        print(f"  • TSL exits: {tsl_pct:.1f}% of trades")
        print(f"  • SL exits: {sl_pct:.1f}% of trades")
    
    def _calculate_aggregate_stats(self, all_trade_stats: Dict) -> Dict:
        """Calculate aggregate statistics from all iterations"""
        aggregate_stats = {}
        for key, values in all_trade_stats.items():
            if values:
                aggregate_stats[f'{key}_mean'] = np.mean(values)
                aggregate_stats[f'{key}_median'] = np.median(values)
                aggregate_stats[f'{key}_std'] = np.std(values)
        return aggregate_stats


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
        print(f"\n🎯 Generating comparison plots for {currency}...")
        
        for config_name, config_data in all_results.items():
            extra_data = config_data['extra_data']
            
            # Check if we have the required data for plotting
            if 'last_sample_df' not in extra_data or 'last_results' not in extra_data:
                print(f"⚠️  Skipping {config_name} - missing plot data")
                continue
            
            print(f"📊 Generating trading chart for {config_name}...")
            
            try:
                # Generate the actual trading chart using the imported plot_production_results
                fig = plot_production_results(
                    df=extra_data['last_sample_df'],
                    results=extra_data['last_results'],
                    title=f"{config_name} - {currency}\nSharpe={extra_data['last_results']['sharpe_ratio']:.3f}, P&L=${extra_data['last_results']['total_pnl']:,.0f}",
                    show_pnl=True,
                    show=show_plots  # Only show if show_plots is True
                )
                
                # Save trade chart if save_plots is enabled
                if save_plots and fig is not None:
                    # Ensure charts directory exists
                    os.makedirs('charts', exist_ok=True)
                    plot_filename = f'charts/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_trades.png'
                    fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    print(f"  💾 Saved: {plot_filename}")
                    
                # Close figure if not showing to free memory
                if not show_plots and fig is not None:
                    plt.close(fig)
                    
            except Exception as e:
                print(f"❌ Error generating plot for {config_name}: {str(e)}")
                continue
    
    @staticmethod
    def plot_monte_carlo_results(results_df: pd.DataFrame, config_name: str):
        """Plot Monte Carlo simulation results"""
        # Implementation would go here
        pass


class BaseRunner(ABC):
    """Abstract base class for different running modes"""
    
    def _print_table_legend(self):
        """Print legend for table columns"""
        print("\n📖 TABLE LEGEND:")
        print("  Iter: Iteration number")
        print("  Sharpe: Sharpe ratio (risk-adjusted returns)")
        print("  Quality: Performance rating based on Sharpe")
        print("  Return%: Total return percentage")
        print("  Win Rate%: Percentage of winning trades")
        print("  W/L: Wins/Losses count")
        print("  PF: Profit Factor (gross wins/gross losses)")
        print("  Max DD%: Maximum drawdown percentage")
        print("  Period: Date range of the sample")
    
    def __init__(self, data_manager: DataManager, config: BacktestConfig):
        self.data_manager = data_manager
        self.config = config
        self.trade_analyzer = TradeAnalyzer()
        self.trade_exporter = TradeExporter()
        self.visualizer = ResultsVisualizer()
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the strategy testing"""
        pass
    
    def _print_condensed_summary(self, results_df: pd.DataFrame, config_name: str):
        """Print condensed summary statistics"""
        print(f"\n🎯 STRATEGY PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Key metrics in a more compact format
        sharpe_mean = results_df['sharpe_ratio'].mean()
        return_mean = results_df['total_return'].mean()
        win_rate_mean = results_df['win_rate'].mean()
        
        # Performance rating
        if sharpe_mean >= 2.0:
            rating = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif sharpe_mean >= 1.5:
            rating = "⭐⭐⭐⭐ VERY GOOD"
        elif sharpe_mean >= 1.0:
            rating = "⭐⭐⭐ GOOD"
        elif sharpe_mean >= 0.5:
            rating = "⭐⭐ MODERATE"
        else:
            rating = "⭐ NEEDS IMPROVEMENT"
        
        print(f"  Overall Rating: {rating}")
        print(f"  Average Sharpe: {sharpe_mean:.3f} (Target: >1.0)")
        print(f"  Average Return: {return_mean:.1f}% per period")
        print(f"  Average Win Rate: {win_rate_mean:.1f}%")
        print(f"  Consistency Score: {(1 - results_df['sharpe_ratio'].std() / sharpe_mean):.1%}" if sharpe_mean > 0 else "  Consistency Score: N/A")
    
    def _save_monte_carlo_results(self, results_df: pd.DataFrame, config_name: str):
        """Save Monte Carlo results to CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_short = config_name.replace(":", "").replace(" ", "_").lower()
        filename = f'results/{self.currency}_{config_short}_monte_carlo.csv'
        
        # Add timestamp as a column
        results_df['timestamp'] = timestamp
        results_df['config'] = config_name
        results_df['currency'] = self.currency
        
        # Save to CSV
        results_df.to_csv(filename, index=False, float_format='%.6f')
        print(f"\n💾 Monte Carlo results saved to: {filename}")
    
    def _display_last_iteration_exit_stats(self, results: Dict[str, Any], config_name: str):
        """Display exit statistics for the last iteration"""
        if 'trades' not in results or not results['trades']:
            return
            
        trades = results['trades']
        total_trades = len(trades)
        
        print(f"\n🎯 LAST ITERATION EXIT BREAKDOWN - {config_name}")
        print("="*80)
        print(f"Total trades in last iteration: {total_trades}")
        
        # Count exits by type
        exit_counts = defaultdict(int)
        tp_combinations = defaultdict(int)
        
        for trade in trades:
            if hasattr(trade, 'exit_reason'):
                exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else str(trade.exit_reason)
                exit_counts[exit_reason] += 1
                
                # Track TP combinations
                if hasattr(trade, 'tp_hits') and trade.tp_hits > 0:
                    if trade.tp_hits == 1:
                        tp_combinations['TP1 only'] += 1
                    elif trade.tp_hits == 2:
                        tp_combinations['TP1+TP2'] += 1
                    elif trade.tp_hits >= 3:
                        tp_combinations['TP1+TP2+TP3'] += 1
        
        # Display exit breakdown
        print("\n📊 Exit Type Breakdown:")
        exit_summary = []
        for exit_type, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            exit_summary.append({
                'Exit Type': exit_type,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        if exit_summary:
            print(tabulate(exit_summary, headers='keys', tablefmt='simple'))
        
        # Display TP combinations if any
        if tp_combinations:
            print("\n🎯 Take Profit Combinations:")
            tp_summary = []
            for combo, count in sorted(tp_combinations.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_trades * 100) if total_trades > 0 else 0
                tp_summary.append({
                    'TP Combination': combo,
                    'Count': count,
                    'Percentage': f"{percentage:.1f}%"
                })
            print(tabulate(tp_summary, headers='keys', tablefmt='simple'))


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
        
        # Test both strategies
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
            
            # Display detailed results table
            if 'simulator' in extra_data:
                extra_data['simulator'].display_results_table(config_name)
            
            # Print table legend on first iteration
            if strategy_type == StrategyType.ULTRA_TIGHT_RISK:
                self._print_table_legend()
            
            # Print traditional summary (now condensed)
            self._print_condensed_summary(results_df, config_name)
            
            # Display exit statistics for the last iteration
            if 'last_results' in extra_data and 'trades' in extra_data['last_results']:
                self._display_last_iteration_exit_stats(extra_data['last_results'], config_name)
            
            # Export trades if enabled
            if self.config.export_trades and 'last_results' in extra_data:
                last_results = extra_data['last_results']
                if 'trades' in last_results and last_results['trades']:
                    self.trade_exporter.export_trades_detailed(
                        last_results['trades'], 
                        self.currency, 
                        config_name
                    )
            
            # Save Monte Carlo results to CSV
            self._save_monte_carlo_results(results_df, config_name)
            
            # Store results
            all_results[config_name] = {
                'results_df': results_df,
                'extra_data': extra_data,
                'strategy_type': strategy_type
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
║          MONTE CARLO STRATEGY TESTER - High Performance Trading             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Test proven trading strategies across Forex and Crypto markets:             ║
║                                                                              ║
║  FOREX STRATEGIES:                                                           ║
║  • Config 1: Ultra-Tight Risk Management (0.2% risk, 10 pip stops)         ║
║  • Config 2: Scalping Strategy (0.1% risk, 5 pip stops)                    ║
║                                                                              ║
║  CRYPTO STRATEGIES:                                                          ║
║  • Conservative: High probability trend following (0.2% risk, 4% stops)     ║
║  • Moderate: Balanced approach (0.25% risk, 3% stops)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    
    def _get_epilog(self) -> str:
        """Get program epilog with examples"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           USAGE EXAMPLES                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 BASIC USAGE:
  python run_strategy_oop.py
    → Runs default: AUDUSD, 50 iterations, 8,000 rows per test

📊 SINGLE CURRENCY TESTING:
  python run_strategy_oop.py --currency GBPUSD
    → Test British Pound with default settings
  
  python run_strategy_oop.py --iterations 100 --sample-size 10000
    → Run 100 tests with 10,000 rows each (more robust results)

🌍 MULTI-CURRENCY TESTING:
  python run_strategy_oop.py --mode multi
    → Test all major pairs: GBPUSD, EURUSD, USDJPY, NZDUSD, USDCAD

📈 VISUALIZATION OPTIONS:
  python run_strategy_oop.py --show-plots
    → Display plots interactively
  
  python run_strategy_oop.py --save-plots
    → Save plots to PNG files

💡 TIPS:
  • Use more iterations (100+) for production decisions
  • Larger sample sizes test strategy robustness over longer periods
  • Multi-currency mode helps identify which pairs work best
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