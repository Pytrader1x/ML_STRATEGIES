"""
Simplified single-run strategy tester
Runs only Config 1 (Ultra-Tight Risk Management) on the last 5k rows of AUDUSD data
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
import time

warnings.filterwarnings('ignore')

def calculate_indicators(df):
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

def create_ultra_tight_risk_strategy():
    """Create Configuration 1: Ultra-Tight Risk Management strategy"""
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
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
        realistic_costs=True,
        verbose=False,
        debug_decisions=True,
        use_daily_sharpe=True
    )
    return OptimizedProdStrategy(strategy_config)

def main():
    """Main execution function"""
    print("="*80)
    print("SINGLE RUN STRATEGY TESTER - CONFIG 1 ONLY")
    print("Testing on last 5,000 rows of AUDUSD data")
    print("="*80)
    
    # Configuration
    currency = 'AUDUSD'
    config_name = "Config 1: Ultra-Tight Risk Management"
    
    # Load data
    print(f"\nLoading {currency} data...")
    
    # Auto-detect data path
    if os.path.exists('data'):
        data_path = 'data'
    elif os.path.exists('../data'):
        data_path = '../data'
    else:
        raise FileNotFoundError("Cannot find data directory. Please run from project root.")
    
    file_path = os.path.join(data_path, f'{currency}_MASTER_15M.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load full dataframe
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df_full):,}")
    print(f"Full date range: {df_full.index[0]} to {df_full.index[-1]}")
    
    # Take only the last 5000 rows
    df = df_full.iloc[-5000:].copy()
    print(f"\nUsing last 5,000 rows for testing")
    print(f"Test date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create strategy
    print(f"\nCreating strategy: {config_name}")
    strategy = create_ultra_tight_risk_strategy()
    
    # Run backtest
    print("\nRunning backtest...")
    start_time = time.time()
    results = strategy.run_backtest(df)
    elapsed_time = time.time() - start_time
    print(f"Backtest completed in {elapsed_time:.2f} seconds")
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Key metrics
    print(f"\n📊 KEY PERFORMANCE METRICS:")
    print(f"  Total Return: {results.get('total_return', 0):.2f}%")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    if 'sortino_ratio' in results:
        print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    if 'recovery_factor' in results:
        print(f"  Recovery Factor: {results['recovery_factor']:.2f}")
    
    print(f"\n📈 TRADING STATISTICS:")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
    if 'win_loss_ratio' in results:
        print(f"  Win/Loss Ratio: {results['win_loss_ratio']:.2f}")
    if 'expectancy' in results:
        print(f"  Expectancy: ${results['expectancy']:,.2f}")
    
    print(f"\n💰 P&L BREAKDOWN:")
    print(f"  Total P&L: ${results.get('total_pnl', 0):,.2f}")
    print(f"  Average Trade: ${results.get('avg_trade', 0):,.2f}")
    if 'avg_win' in results:
        print(f"  Average Win: ${results['avg_win']:,.2f}")
    if 'avg_loss' in results:
        print(f"  Average Loss: ${results['avg_loss']:,.2f}")
    if 'best_trade' in results:
        print(f"  Best Trade: ${results['best_trade']:,.2f}")
    if 'worst_trade' in results:
        print(f"  Worst Trade: ${results['worst_trade']:,.2f}")
    
    # Exit statistics
    if 'trades' in results and results['trades']:
        trades = results['trades']
        total_trades = len(trades)
        
        print(f"\n🎯 EXIT BREAKDOWN:")
        
        # Count exits by type
        exit_counts = {}
        for trade in trades:
            if hasattr(trade, 'exit_reason'):
                exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else str(trade.exit_reason)
                exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
        
        # Display exit breakdown
        for exit_type, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            print(f"  {exit_type}: {count} trades ({percentage:.1f}%)")
        
        # TP hit statistics
        tp1_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 1)
        tp2_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 2)
        tp3_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 3)
        
        print(f"\n📍 TAKE PROFIT STATISTICS:")
        print(f"  TP1 Hit: {tp1_hits} trades ({tp1_hits/total_trades*100:.1f}%)")
        print(f"  TP2 Hit: {tp2_hits} trades ({tp2_hits/total_trades*100:.1f}%)")
        print(f"  TP3 Hit: {tp3_hits} trades ({tp3_hits/total_trades*100:.1f}%)")
        
        # Partial Profit statistics
        if 'pp_stats' in results:
            pp_trades = results['pp_stats']['pp_trades']
            pp_percentage = results['pp_stats']['pp_percentage']
            print(f"\n💰 PARTIAL PROFIT STATISTICS:")
            print(f"  PP Exits: {pp_trades} trades ({pp_percentage:.1f}%)")
    
    # Generate and display plot
    print("\n📊 Generating trading chart...")
    try:
        fig = plot_production_results(
            df=df,
            results=results,
            title=f"{config_name} - {currency}\nSharpe={results['sharpe_ratio']:.3f}, P&L=${results['total_pnl']:,.0f}",
            show_pnl=True,
            show=True  # Always show the plot
        )
        
        # Also save the plot
        os.makedirs('charts', exist_ok=True)
        plot_filename = f'charts/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_single_run.png'
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Plot saved to: {plot_filename}")
        
    except Exception as e:
        print(f"❌ Error generating plot: {str(e)}")
    
    # Export trades to CSV
    if 'trades' in results and results['trades']:
        print("\n📄 Exporting trade details...")
        trade_records = []
        
        for i, trade in enumerate(results['trades'], 1):
            # Extract basic trade info
            direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
            exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
            
            # Calculate pip distances
            sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / 0.0001
            
            record = {
                'trade_id': i,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'direction': direction,
                'initial_size_millions': trade.initial_position_size / 1e6 if hasattr(trade, 'initial_position_size') else trade.position_size / 1e6,
                'confidence': trade.confidence,
                'is_relaxed': trade.is_relaxed,
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
            
            trade_records.append(record)
        
        # Save to CSV
        trades_df = pd.DataFrame(trade_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('results', exist_ok=True)
        csv_filename = f'results/{currency}_config_1_single_run_{timestamp}.csv'
        trades_df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"  💾 Trades exported to: {csv_filename}")
        
        # Print trade summary
        print(f"\n  Total trades exported: {len(trades_df)}")
        print(f"  Total P&L: ${trades_df['final_pnl'].sum():,.2f}")
        print(f"  Win rate: {(trades_df['final_pnl'] > 0).sum() / len(trades_df) * 100:.1f}%")
    
    print("\n✅ Single run completed successfully!")

if __name__ == "__main__":
    main()