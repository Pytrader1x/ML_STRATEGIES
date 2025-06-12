"""
Detailed Trade Analysis for AUDUSD Strategy Validation
Focus on understanding trade mechanics and checking for issues
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def create_config_1_ultra_tight_risk():
    """Configuration 1: Ultra-Tight Risk Management"""
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
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
        verbose=True  # Enable verbose for detailed logging
    )
    return OptimizedProdStrategy(config)


def analyze_single_sample():
    """Analyze a single sample in detail"""
    print("="*80)
    print("DETAILED SINGLE SAMPLE ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Use a specific date range for reproducibility
    start_date = '2022-01-01'
    end_date = '2022-03-31'
    sample_df = df[start_date:end_date].copy()
    
    print(f"\nAnalyzing period: {sample_df.index[0]} to {sample_df.index[-1]}")
    print(f"Total bars: {len(sample_df)}")
    
    # Create strategy
    strategy = create_config_1_ultra_tight_risk()
    
    # Run backtest
    print("\nRunning backtest with verbose logging...")
    results = strategy.run_backtest(sample_df)
    
    # Analyze results
    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"Total return: {results['total_return']:.1f}%")
    print(f"Max drawdown: {results['max_drawdown']:.1f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    
    # Analyze trades
    if 'trades' in results and results['trades']:
        trades = results['trades']
        print(f"\nAnalyzing {len(trades)} trades...")
        
        # Calculate trade statistics
        trade_pnls = []
        trade_sizes = []
        entry_prices = []
        exit_prices = []
        
        for i, trade in enumerate(trades[:10]):  # First 10 trades
            print(f"\n--- Trade {i+1} ---")
            if hasattr(trade, 'entry_time'):
                print(f"Entry: {trade.entry_time} at {trade.entry_price}")
                print(f"Exit: {trade.exit_time} at {trade.exit_price}")
                print(f"Direction: {trade.direction}")
                print(f"Position size: {trade.position_size:,.0f}")
                print(f"P&L: ${trade.pnl:.2f}")
                
                # Calculate pip movement
                if hasattr(trade, 'direction'):
                    if trade.direction == 1:  # Long
                        pip_move = (trade.exit_price - trade.entry_price) * 10000
                    else:  # Short
                        pip_move = (trade.entry_price - trade.exit_price) * 10000
                    print(f"Pip movement: {pip_move:.1f} pips")
                
                # Check if P&L makes sense
                expected_pnl = trade.position_size * abs(trade.exit_price - trade.entry_price)
                if trade.direction == 1:
                    expected_pnl *= 1 if trade.exit_price > trade.entry_price else -1
                else:
                    expected_pnl *= 1 if trade.exit_price < trade.entry_price else -1
                
                print(f"Expected P&L calculation: ${expected_pnl:.2f}")
                
                # Check for partial exits
                if hasattr(trade, 'partial_exits') and trade.partial_exits:
                    print(f"Partial exits: {len(trade.partial_exits)}")
                    for j, partial in enumerate(trade.partial_exits):
                        print(f"  Partial {j+1}: {partial.size:,.0f} units at {partial.price} = ${partial.pnl:.2f}")
                
                trade_pnls.append(trade.pnl)
                trade_sizes.append(trade.position_size)
                entry_prices.append(trade.entry_price)
                exit_prices.append(trade.exit_price)
        
        # Statistical analysis
        print("\n" + "="*80)
        print("TRADE STATISTICS")
        print("="*80)
        
        print(f"\nPosition sizes:")
        print(f"  All trades same size: {len(set(trade_sizes)) == 1}")
        print(f"  Size value: {trade_sizes[0]:,.0f} units")
        
        # Check if position size is always 1M
        if all(size == 1_000_000 for size in trade_sizes):
            print("  ✓ Confirmed: All trades are 1M units (base currency)")
        
        # Calculate average win/loss in pips
        winning_pnls = [p for p in trade_pnls if p > 0]
        losing_pnls = [p for p in trade_pnls if p < 0]
        
        if winning_pnls:
            avg_win_dollars = np.mean(winning_pnls)
            # Estimate average win in pips (assuming 1M position)
            avg_win_pips = avg_win_dollars / 100  # $100 per pip for 1M position
            print(f"\nAverage win: ${avg_win_dollars:.2f} ({avg_win_pips:.1f} pips)")
        
        if losing_pnls:
            avg_loss_dollars = np.mean(losing_pnls)
            avg_loss_pips = abs(avg_loss_dollars) / 100
            print(f"Average loss: ${avg_loss_dollars:.2f} ({avg_loss_pips:.1f} pips)")
        
        # Check for realistic P&L
        max_pnl = max(trade_pnls) if trade_pnls else 0
        min_pnl = min(trade_pnls) if trade_pnls else 0
        
        print(f"\nP&L range:")
        print(f"  Max win: ${max_pnl:.2f}")
        print(f"  Max loss: ${min_pnl:.2f}")
        
        # Warning checks
        print("\n" + "="*80)
        print("VALIDATION CHECKS")
        print("="*80)
        
        warnings = []
        
        # Check if wins are too large
        if winning_pnls and avg_win_pips > 20:
            warnings.append(f"Average win of {avg_win_pips:.1f} pips seems high for scalping")
        
        # Check if all trades are same size
        if len(set(trade_sizes)) == 1:
            print("✓ PASS: Trade sizes are consistent (no compounding)")
        else:
            warnings.append("Trade sizes vary - possible position sizing issue")
        
        # Check win rate
        if results['win_rate'] > 75:
            warnings.append(f"Win rate of {results['win_rate']:.1f}% is suspiciously high")
        
        # Check Sharpe ratio
        if results['sharpe_ratio'] > 2:
            warnings.append(f"Sharpe ratio of {results['sharpe_ratio']:.2f} is unrealistically high")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n✅ No major warnings detected")
    
    return results


def check_indicator_values(df, sample_size=100):
    """Check indicator values for reasonableness"""
    print("\n" + "="*80)
    print("INDICATOR VALUE ANALYSIS")
    print("="*80)
    
    # Sample random rows
    sample_indices = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample_rows = df.iloc[sample_indices]
    
    # Check NTI_Direction
    nti_values = sample_rows['NTI_Direction'].value_counts()
    print("\nNTI_Direction distribution:")
    print(nti_values)
    
    # Check MB_Bias
    mb_values = sample_rows['MB_Bias'].value_counts()
    print("\nMB_Bias distribution:")
    print(mb_values)
    
    # Check IC_Signal
    ic_values = sample_rows['IC_Signal'].value_counts()
    print("\nIC_Signal distribution:")
    print(ic_values)
    
    # Check for suspicious patterns
    if len(nti_values) < 3:  # Should have -1, 0, 1
        print("⚠️  WARNING: NTI_Direction has limited values")
    
    # Check alignment
    alignment = (sample_rows['NTI_Direction'] == sample_rows['MB_Bias']).sum()
    print(f"\nIndicator alignment: {alignment}/{len(sample_rows)} ({alignment/len(sample_rows)*100:.1f}%)")


def main():
    """Run detailed trade analysis"""
    print("AUDUSD Detailed Trade Analysis")
    print("="*80)
    print(f"Analysis started: {datetime.now()}")
    
    # Run single sample analysis
    results = analyze_single_sample()
    
    # Load full data for indicator analysis
    print("\n\nLoading full dataset for indicator analysis...")
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Calculate indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Check indicator values
    check_indicator_values(df)
    
    print(f"\n\nAnalysis completed: {datetime.now()}")


if __name__ == "__main__":
    main()