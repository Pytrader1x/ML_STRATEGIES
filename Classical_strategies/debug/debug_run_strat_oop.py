"""
Debug version of run_strategy_oop.py
- No Monte Carlo simulations
- Always uses last 4,000 rows of data
- Designed for step-by-step debugging in VS Code
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
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union, Any
from enum import Enum

warnings.filterwarnings('ignore')

__version__ = "1.0.0-debug"


class StrategyType(Enum):
    """Enumeration for strategy types"""
    ULTRA_TIGHT_RISK = "ultra_tight_risk"
    SCALPING = "scalping"


@dataclass
class DebugConfig:
    """Configuration for debug backtesting"""
    sample_size: int = 4000  # Fixed sample size
    realistic_costs: bool = True
    use_daily_sharpe: bool = True
    debug_mode: bool = True  # Always true for debugging
    show_plots: bool = True  # Always show plots
    save_plots: bool = False
    export_trades: bool = True
    currency: str = "AUDUSD"
    strategy_type: StrategyType = StrategyType.ULTRA_TIGHT_RISK


class DebugStrategyRunner:
    """Debug version of strategy runner for step-by-step debugging"""
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.df = None
        self.strategy = None
        self.results = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare currency data"""
        print(f"\n{'='*80}")
        print(f"DEBUG MODE - Loading {self.config.currency} data")
        print(f"{'='*80}")
        
        # Determine data path
        if os.path.exists('../data'):
            data_path = '../data'
        elif os.path.exists('data'):
            data_path = 'data'
        else:
            raise FileNotFoundError("Cannot find data directory")
            
        file_path = os.path.join(data_path, f'{self.config.currency}_MASTER_15M.csv')
        
        print(f"Loading from: {file_path}")
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        print(f"Total data points: {len(df):,}")
        print(f"Full date range: {df.index[0]} to {df.index[-1]}")
        
        # Take last N rows for debugging
        df = df.iloc[-self.config.sample_size:]
        print(f"\nDEBUG: Using last {self.config.sample_size:,} rows")
        print(f"Debug date range: {df.index[0]} to {df.index[-1]}")
        
        # Calculate indicators
        print("\nCalculating indicators...")
        df = self._calculate_indicators(df)
        
        self.df = df
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        print("  - Adding Neuro Trend Intelligent...")
        df = TIC.add_neuro_trend_intelligent(df)
        
        print("  - Adding Market Bias...")
        df = TIC.add_market_bias(df)
        
        print("  - Adding Intelligent Chop...")
        df = TIC.add_intelligent_chop(df)
        
        print("  ✓ Indicators calculated")
        return df
    
    def create_strategy(self) -> OptimizedProdStrategy:
        """Create strategy based on configuration"""
        print(f"\nCreating strategy: {self.config.strategy_type.value}")
        
        if self.config.strategy_type == StrategyType.ULTRA_TIGHT_RISK:
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
                relaxed_mode=False,
                realistic_costs=self.config.realistic_costs,
                verbose=True,  # Enable verbose output for debugging
                debug_decisions=True,  # Enable debug output
                use_daily_sharpe=self.config.use_daily_sharpe
            )
        else:  # SCALPING
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
                relaxed_mode=False,
                realistic_costs=self.config.realistic_costs,
                verbose=True,
                debug_decisions=True,
                use_daily_sharpe=self.config.use_daily_sharpe
            )
            
        self.strategy = OptimizedProdStrategy(strategy_config)
        return self.strategy
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run backtest on the data"""
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST")
        print(f"{'='*80}")
        
        # Set breakpoint here for debugging
        print("\n🔴 SET BREAKPOINT HERE to step through backtest")
        print("   You can inspect self.strategy and self.df")
        
        self.results = self.strategy.run_backtest(self.df)
        
        print(f"\nBacktest complete!")
        print(f"Total trades: {self.results.get('total_trades', 0)}")
        print(f"Win rate: {self.results.get('win_rate', 0):.1f}%")
        print(f"Total P&L: ${self.results.get('total_pnl', 0):,.2f}")
        
        return self.results
    
    def export_trades(self) -> Optional[str]:
        """Export trades to CSV"""
        if not self.config.export_trades or not self.results or 'trades' not in self.results:
            return None
            
        trades = self.results['trades']
        if not trades:
            print("No trades to export")
            return None
            
        print(f"\nExporting {len(trades)} trades...")
        
        # Create trade records
        trade_records = []
        for i, trade in enumerate(trades, 1):
            record = self._create_trade_record(i, trade)
            trade_records.append(record)
        
        # Save to CSV
        df_trades = pd.DataFrame(trade_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = f"debug_{self.config.strategy_type.value}"
        filepath = f'results/{self.config.currency}_{config_name}_trades_{timestamp}.csv'
        
        os.makedirs('results', exist_ok=True)
        df_trades.to_csv(filepath, index=False, float_format='%.6f')
        
        print(f"Trades exported to: {filepath}")
        return filepath
    
    def _create_trade_record(self, trade_id: int, trade: Any) -> Dict[str, Any]:
        """Create a trade record for export (simplified from original)"""
        direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
        exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
        
        record = {
            'trade_id': trade_id,
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'direction': direction,
            'position_size_millions': trade.position_size / 1e6,
            'sl_price': trade.stop_loss,
            'tp1_price': trade.take_profits[0] if len(trade.take_profits) > 0 else None,
            'tp2_price': trade.take_profits[1] if len(trade.take_profits) > 1 else None,
            'tp3_price': trade.take_profits[2] if len(trade.take_profits) > 2 else None,
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'exit_reason': exit_reason,
            'tp_hits': trade.tp_hits,
            'final_pnl': trade.pnl,
        }
        
        # Add partial exit info
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            for j, pe in enumerate(trade.partial_exits[:3], 1):
                if hasattr(pe, 'tp_level'):
                    if pe.tp_level == 0:
                        pe_type = 'PPT'  # Partial Profit Taking
                    else:
                        pe_type = f'TP{pe.tp_level}'
                else:
                    pe_type = 'PARTIAL'
                    
                record[f'partial_{j}_type'] = pe_type
                record[f'partial_{j}_pnl'] = pe.pnl if hasattr(pe, 'pnl') else 0
        
        return record
    
    def plot_results(self) -> plt.Figure:
        """Generate and display the results plot"""
        if not self.results or not self.df:
            print("No results to plot")
            return None
            
        print("\nGenerating plot...")
        
        config_name = f"DEBUG {self.config.strategy_type.value.replace('_', ' ').title()}"
        
        fig = plot_production_results(
            df=self.df,
            results=self.results,
            title=f"{config_name} - {self.config.currency}\n" + 
                  f"Last {len(self.df):,} bars | " +
                  f"P&L: ${self.results.get('total_pnl', 0):,.0f} | " +
                  f"Sharpe: {self.results.get('sharpe_ratio', 0):.3f}",
            show_pnl=True,
            show_position_sizes=True,
            show=self.config.show_plots,
            save_path=f'charts/debug_{self.config.currency}_{self.config.strategy_type.value}.png' if self.config.save_plots else None
        )
        
        return fig
    
    def run_complete_debug_session(self):
        """Run complete debug session with all steps"""
        print(f"\n{'='*80}")
        print(f"DEBUG STRATEGY RUNNER v{__version__}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Currency: {self.config.currency}")
        print(f"  Strategy: {self.config.strategy_type.value}")
        print(f"  Sample Size: {self.config.sample_size:,} bars")
        print(f"  Debug Mode: {self.config.debug_mode}")
        print(f"{'='*80}")
        
        # Step 1: Load data
        print("\nStep 1: Loading data...")
        self.load_data()
        
        # Step 2: Create strategy
        print("\nStep 2: Creating strategy...")
        self.create_strategy()
        
        # Step 3: Run backtest
        print("\nStep 3: Running backtest...")
        print("💡 TIP: Set breakpoints in Prod_strategy.py to debug:")
        print("   - run_backtest() method")
        print("   - _check_entry_conditions()")
        print("   - _check_exit_conditions()")
        print("   - _execute_entry()")
        print("   - _execute_exit()")
        
        self.run_backtest()
        
        # Step 4: Export trades
        print("\nStep 4: Exporting trades...")
        self.export_trades()
        
        # Step 5: Plot results
        print("\nStep 5: Plotting results...")
        self.plot_results()
        
        print(f"\n{'='*80}")
        print("DEBUG SESSION COMPLETE")
        print(f"{'='*80}")
        
        return self.results


def main():
    """Main entry point for debug runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Strategy Runner')
    parser.add_argument('--currency', type=str, default='AUDUSD', 
                       help='Currency pair to test')
    parser.add_argument('--strategy', type=str, default='ultra_tight_risk',
                       choices=['ultra_tight_risk', 'scalping'],
                       help='Strategy type to test')
    parser.add_argument('--sample-size', type=int, default=4000,
                       help='Number of bars to use (from end of data)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to file')
    parser.add_argument('--no-costs', action='store_true',
                       help='Disable realistic trading costs')
    
    args = parser.parse_args()
    
    # Create debug configuration
    config = DebugConfig(
        currency=args.currency,
        strategy_type=StrategyType(args.strategy),
        sample_size=args.sample_size,
        save_plots=args.save_plots,
        realistic_costs=not args.no_costs
    )
    
    # Create and run debug session
    runner = DebugStrategyRunner(config)
    results = runner.run_complete_debug_session()
    
    # Keep plot window open if showing
    if config.show_plots:
        print("\nPlot window is open. Close it to exit.")
        plt.show()


if __name__ == "__main__":
    main()