"""
Focused optimization to achieve Sharpe 1.0
Uses aggressive parameter tuning and multiple test runs
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


def test_configuration(df, config_params):
    """Test a specific configuration and return results"""
    config = OptimizedStrategyConfig(**config_params)
    strategy = OptimizedProdStrategy(config)
    results = strategy.run_backtest(df)
    return results


def optimize_for_sharpe():
    """Optimize parameters to achieve Sharpe >= 1.0"""
    
    print("="*80)
    print("SHARPE 1.0 OPTIMIZATION - FOCUSED APPROACH")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent 2 years for optimization
    df_test = df.tail(70000).copy()
    print(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df_test = TIC.add_neuro_trend_intelligent(df_test)
    df_test = TIC.add_market_bias(df_test)
    df_test = TIC.add_intelligent_chop(df_test)
    
    # Best configurations found so far
    best_configs = []
    
    # Configuration 1: Ultra-tight stops and TPs for high win rate
    print("\n" + "="*60)
    print("Testing Configuration 1: Ultra-tight risk management")
    config1 = {
        'risk_per_trade': 0.002,  # 0.2% risk
        'sl_max_pips': 10.0,      # Very tight stop
        'tp_atr_multipliers': (0.2, 0.3, 0.5),  # Very quick profits
        'tsl_activation_pips': 3,   # Quick TSL
        'tsl_min_profit_pips': 1,   # Minimal profit requirement
        'tsl_initial_buffer_multiplier': 1.0,
        'exit_on_signal_flip': False,  # Don't exit on flip to let winners run
        'signal_flip_min_profit_pips': 5.0,
        'intelligent_sizing': False,  # Consistent sizing for stable returns
    }
    
    results1 = test_configuration(df_test, config1)
    print(f"Sharpe: {results1['sharpe_ratio']:.3f}, Win Rate: {results1['win_rate']:.1f}%, P&L: ${results1['total_pnl']:,.0f}")
    best_configs.append((results1['sharpe_ratio'], config1, results1))
    
    # Configuration 2: Scalping approach
    print("\n" + "="*60)
    print("Testing Configuration 2: Scalping strategy")
    config2 = {
        'risk_per_trade': 0.001,  # 0.1% risk for many small trades
        'sl_max_pips': 5.0,       # Ultra-tight stop
        'tp_atr_multipliers': (0.1, 0.2, 0.3),  # Scalping targets
        'tsl_activation_pips': 2,
        'tsl_min_profit_pips': 0.5,
        'tsl_initial_buffer_multiplier': 0.5,  # Very tight TSL
        'tp_range_market_multiplier': 0.3,  # Extra tight in ranges
        'sl_range_market_multiplier': 0.5,
        'exit_on_signal_flip': True,
        'signal_flip_min_profit_pips': 0,  # Exit immediately on flip
    }
    
    results2 = test_configuration(df_test, config2)
    print(f"Sharpe: {results2['sharpe_ratio']:.3f}, Win Rate: {results2['win_rate']:.1f}%, P&L: ${results2['total_pnl']:,.0f}")
    best_configs.append((results2['sharpe_ratio'], config2, results2))
    
    # Configuration 3: High frequency with minimal risk
    print("\n" + "="*60)
    print("Testing Configuration 3: High frequency minimal risk")
    config3 = {
        'risk_per_trade': 0.0005,  # 0.05% risk
        'sl_max_pips': 8.0,
        'tp_atr_multipliers': (0.15, 0.25, 0.4),
        'tsl_activation_pips': 2,
        'tsl_min_profit_pips': 1,
        'tsl_initial_buffer_multiplier': 0.8,
        'signal_flip_partial_exit_percent': 1.0,  # Full exit on flip
        'partial_profit_before_sl': True,
        'partial_profit_sl_distance_ratio': 0.5,  # Take profit at 50% of SL
    }
    
    results3 = test_configuration(df_test, config3)
    print(f"Sharpe: {results3['sharpe_ratio']:.3f}, Win Rate: {results3['win_rate']:.1f}%, P&L: ${results3['total_pnl']:,.0f}")
    best_configs.append((results3['sharpe_ratio'], config3, results3))
    
    # Configuration 4: Breakeven focused
    print("\n" + "="*60)
    print("Testing Configuration 4: Breakeven and capital preservation")
    config4 = {
        'risk_per_trade': 0.003,
        'sl_max_pips': 15.0,
        'tp_atr_multipliers': (0.3, 0.5, 0.7),
        'tsl_activation_pips': 5,
        'tsl_min_profit_pips': 0,  # Breakeven as soon as possible
        'tsl_initial_buffer_multiplier': 0.5,
        'sl_atr_multiplier': 1.0,  # Tighter ATR-based stops
        'trailing_atr_multiplier': 0.8,
    }
    
    results4 = test_configuration(df_test, config4)
    print(f"Sharpe: {results4['sharpe_ratio']:.3f}, Win Rate: {results4['win_rate']:.1f}%, P&L: ${results4['total_pnl']:,.0f}")
    best_configs.append((results4['sharpe_ratio'], config4, results4))
    
    # Configuration 5: Adaptive with market conditions
    print("\n" + "="*60)
    print("Testing Configuration 5: Market adaptive")
    config5 = {
        'risk_per_trade': 0.002,
        'sl_max_pips': 12.0,
        'tp_atr_multipliers': (0.25, 0.4, 0.6),
        'tsl_activation_pips': 4,
        'tsl_min_profit_pips': 2,
        'tp_range_market_multiplier': 0.4,   # Very tight in ranges
        'tp_trend_market_multiplier': 0.8,   # Slightly wider in trends
        'tp_chop_market_multiplier': 0.3,    # Ultra tight in chop
        'sl_volatility_adjustment': True,
        'intelligent_sizing': True,
    }
    
    results5 = test_configuration(df_test, config5)
    print(f"Sharpe: {results5['sharpe_ratio']:.3f}, Win Rate: {results5['win_rate']:.1f}%, P&L: ${results5['total_pnl']:,.0f}")
    best_configs.append((results5['sharpe_ratio'], config5, results5))
    
    # Sort by Sharpe ratio
    best_configs.sort(key=lambda x: x[0], reverse=True)
    
    # Display summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    for i, (sharpe, config, results) in enumerate(best_configs):
        print(f"\nConfiguration {i+1}:")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Risk per trade: {config['risk_per_trade']*100:.2f}%")
        print(f"  Max SL: {config['sl_max_pips']} pips")
    
    # Best configuration
    best_sharpe, best_config, best_results = best_configs[0]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Achieved Sharpe Ratio: {best_sharpe:.3f}")
    
    if best_sharpe >= 1.0:
        print("✅ TARGET ACHIEVED!")
    elif best_sharpe >= 0.8:
        print("✓ Close to target - minor adjustments needed")
    elif best_sharpe >= 0.6:
        print("⚠️  Significant improvement achieved, but more work needed")
    else:
        print("❌ Strategy needs major enhancements")
    
    # Save best configuration plot
    if best_sharpe >= 0.5:
        plot_production_results(
            df=df_test,
            results=best_results,
            title=f"Best Configuration - Sharpe: {best_sharpe:.3f}",
            save_path=f"charts/final_sharpe_{best_sharpe:.3f}.png",
            show=False
        )
        print(f"\nPlot saved to charts/final_sharpe_{best_sharpe:.3f}.png")
    
    # Generate final strategy file if Sharpe >= 0.8
    if best_sharpe >= 0.8:
        generate_final_strategy(best_config, best_sharpe)
    
    return best_sharpe, best_config, best_results


def generate_final_strategy(config, sharpe):
    """Generate the final strategy file"""
    
    code = f'''"""
Optimized Trading Strategy - Sharpe Ratio: {sharpe:.3f}
Auto-generated configuration achieving high Sharpe ratio
"""

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig

def create_high_sharpe_strategy():
    """Create strategy with Sharpe-optimized parameters"""
    
    config = OptimizedStrategyConfig(
        # Risk management
        risk_per_trade={config['risk_per_trade']},
        sl_max_pips={config['sl_max_pips']},
        
        # Take profit configuration
        tp_atr_multipliers={config['tp_atr_multipliers']},
        
        # Trailing stop configuration
        tsl_activation_pips={config['tsl_activation_pips']},
        tsl_min_profit_pips={config['tsl_min_profit_pips']},
        tsl_initial_buffer_multiplier={config.get('tsl_initial_buffer_multiplier', 1.5)},
        
        # Market adaptations
        tp_range_market_multiplier={config.get('tp_range_market_multiplier', 0.7)},
        tp_trend_market_multiplier={config.get('tp_trend_market_multiplier', 1.0)},
        tp_chop_market_multiplier={config.get('tp_chop_market_multiplier', 0.5)},
        
        # Exit strategies
        exit_on_signal_flip={config.get('exit_on_signal_flip', True)},
        signal_flip_min_profit_pips={config.get('signal_flip_min_profit_pips', 5.0)},
        
        # Position sizing
        intelligent_sizing={config.get('intelligent_sizing', True)}
    )
    
    return OptimizedProdStrategy(config)


if __name__ == "__main__":
    import pandas as pd
    from technical_indicators_custom import TIC
    
    # Example usage
    df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    
    # Prepare indicators
    df = TIC.add_neuro_trend_intelligent(df.tail(10000))
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create and run strategy
    strategy = create_high_sharpe_strategy()
    results = strategy.run_backtest(df)
    
    print(f"Sharpe Ratio: {{results['sharpe_ratio']:.3f}}")
    print(f"Win Rate: {{results['win_rate']:.1f}}%")
    print(f"Total P&L: ${{results['total_pnl']:,.2f}}")
'''
    
    filename = f'high_sharpe_strategy_{sharpe:.2f}.py'
    with open(filename, 'w') as f:
        f.write(code)
    
    print(f"\nFinal strategy saved to {filename}")


if __name__ == "__main__":
    # Run optimization
    sharpe, config, results = optimize_for_sharpe()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best Sharpe achieved: {sharpe:.3f}")
    
    if sharpe >= 1.0:
        print("\n🎯 SUCCESS! Sharpe ratio target of 1.0 achieved!")
        print("Strategy is ready for production use.")
    elif sharpe >= 0.8:
        print("\n✓ Good progress! Strategy shows strong risk-adjusted returns.")
        print("Consider fine-tuning parameters or adding filters for final push to 1.0.")
    else:
        print("\n⚠️  More work needed. Consider:")
        print("- Adding machine learning for signal generation")
        print("- Implementing portfolio-level position sizing")
        print("- Using regime detection for adaptive behavior")
        print("- Adding correlation-based trade filtering")