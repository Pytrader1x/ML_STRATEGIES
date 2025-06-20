"""
Generate zoomed-in charts from Monte Carlo backtests - Fast Version
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def create_strategy(position_size_millions=1.0, initial_capital=1_000_000):
    """Create the validated strategy configuration"""
    config = OptimizedStrategyConfig(
        initial_capital=initial_capital,
        risk_per_trade=0.005,
        base_position_size_millions=position_size_millions,
        sl_min_pips=3.0,
        sl_max_pips=10.0,
        sl_atr_multiplier=0.8,
        tp_atr_multipliers=(0.15, 0.25, 0.4),
        max_tp_percent=0.005,
        tp_range_market_multiplier=0.4,
        tp_trend_market_multiplier=0.6,
        tp_chop_market_multiplier=0.3,
        tsl_activation_pips=8.0,
        tsl_min_profit_pips=1.0,
        trailing_atr_multiplier=0.8,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        relaxed_mode=True,
        relaxed_position_multiplier=0.5,
        realistic_costs=True,
        entry_slippage_pips=0.5,
        stop_loss_slippage_pips=2.0,
        trailing_stop_slippage_pips=1.0,
        take_profit_slippage_pips=0.0,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False,  # CRITICAL: Suppress config printing
        debug_decisions=False,
        use_daily_sharpe=True
    )
    
    return OptimizedProdStrategy(config)

def generate_all_charts(currency_pair='AUDUSD'):
    """Generate both regular and ultra-zoomed charts"""
    
    print(f"🎨 GENERATING ZOOMED CHARTS")
    print("="*60)
    print(f"Currency: {currency_pair}")
    print(f"Regular zoomed charts: 10 (5-30 days)")
    print(f"Ultra-zoomed charts: 5 (1-2 days)")
    
    # Load data
    data_path = 'data' if os.path.exists('data') else '../data'
    file_path = os.path.join(data_path, f'{currency_pair}_MASTER_15M.csv')
    
    print(f"\nLoading data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Data loaded: {len(df):,} rows")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df, ha_len=350, ha_len2=30)
    df = TIC.add_intelligent_chop(df)
    
    # Ensure charts directory exists
    os.makedirs('charts', exist_ok=True)
    
    # Set random seed
    np.random.seed(42)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate regular zoomed charts (5-30 days)
    print("\n📊 Generating regular zoomed charts...")
    for i in range(10):
        zoom_days = np.random.randint(5, 31)
        bars_per_day = 96
        zoom_bars = zoom_days * bars_per_day
        
        max_start = len(df) - zoom_bars - 500
        start_idx = np.random.randint(500, max_start)
        end_idx = start_idx + zoom_bars
        
        zoom_df = df.iloc[start_idx:end_idx].copy()
        start_date = zoom_df.index[0]
        end_date = zoom_df.index[-1]
        
        print(f"  Chart {i+1}: {zoom_days} days", end='', flush=True)
        
        # Run strategy
        strategy = create_strategy()
        result = strategy.run_backtest(zoom_df)
        
        print(f" - Sharpe={result.get('sharpe_ratio', 0):.2f}")
        
        # Create plot
        title = (f"{currency_pair} Zoomed View #{i+1} ({zoom_days} days)\n"
                f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}\n"
                f"Sharpe: {result.get('sharpe_ratio', 0):.2f} | "
                f"Return: {result.get('total_return', 0):.1f}% | "
                f"Trades: {result.get('total_trades', 0)}")
        
        fig = plot_production_results(zoom_df, result, title=title, show_pnl=True, show=False)
        
        filename = f'charts/zoomed_{currency_pair}_{i+1:02d}_{zoom_days}d_{timestamp}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    # Generate ultra-zoomed charts (1-2 days)
    print("\n🔍 Generating ultra-zoomed charts...")
    for i in range(5):
        zoom_days = np.random.choice([1, 2])
        bars_per_day = 96
        zoom_bars = zoom_days * bars_per_day
        
        max_start = len(df) - zoom_bars - 500
        start_idx = np.random.randint(500, max_start)
        end_idx = start_idx + zoom_bars
        
        zoom_df = df.iloc[start_idx:end_idx].copy()
        start_date = zoom_df.index[0]
        end_date = zoom_df.index[-1]
        
        print(f"  Ultra {i+1}: {zoom_days} day{'s' if zoom_days > 1 else ''}", end='', flush=True)
        
        # Run strategy
        strategy = create_strategy()
        result = strategy.run_backtest(zoom_df)
        
        print(f" - Sharpe={result.get('sharpe_ratio', 0):.2f}")
        
        # Create plot with enhanced detail
        title = (f"{currency_pair} Ultra-Zoom #{i+1} ({zoom_days} day{'s' if zoom_days > 1 else ''})\n"
                f"{start_date.strftime('%d %b %Y %H:%M')} to {end_date.strftime('%d %b %Y %H:%M')}\n"
                f"Sharpe: {result.get('sharpe_ratio', 0):.2f} | "
                f"Trades: {result.get('total_trades', 0)}")
        
        fig = plot_production_results(zoom_df, result, title=title, show_pnl=True, show=False)
        
        filename = f'charts/ultra_zoom_{currency_pair}_{i+1:02d}_{zoom_days}d_{timestamp}.png'
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')  # Higher DPI for ultra-zoom
        plt.close(fig)
    
    print(f"\n✅ All charts generated successfully!")
    print(f"📁 Saved to: charts/")
    return timestamp

if __name__ == "__main__":
    generate_all_charts()