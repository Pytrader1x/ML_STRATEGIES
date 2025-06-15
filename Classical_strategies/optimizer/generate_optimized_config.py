#!/usr/bin/env python3
"""
Generate optimized strategy configuration from optimization results
"""

import json
import os
from datetime import datetime

def generate_strategy_config(result_file: str):
    """Generate strategy configuration code from optimization results"""
    
    # Load the results
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    best = data['best_result']
    params = best['params']
    
    print(f"\n🏆 OPTIMIZED STRATEGY CONFIGURATION")
    print(f"="*60)
    print(f"Strategy Type: {data['strategy_type']}")
    print(f"Currency: {data['currency']}")
    print(f"Best Sharpe: {best['sharpe_ratio']:.3f}")
    print(f"Total Return: {best['total_return']:.1f}%")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Total Trades: {best['total_trades']}")
    print(f"="*60)
    
    # Generate the strategy configuration code
    config_code = f'''
# Optimized Strategy Configuration
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Sharpe Ratio: {best['sharpe_ratio']:.3f}
# Currency: {data['currency']}

def create_optimized_strategy_{data['strategy_type']}(config: BacktestConfig) -> OptimizedProdStrategy:
    """Optimized Strategy {'1: Ultra-Tight Risk' if data['strategy_type'] == 1 else '2: Scalping'}"""
    
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade={params['risk_per_trade']:.4f},  # {params['risk_per_trade']*100:.2f}% risk
        
        # Stop Loss Configuration
        sl_min_pips={params['sl_min_pips']},
        sl_max_pips={params['sl_max_pips']},
        sl_atr_multiplier={params['sl_atr_multiplier']:.1f},
        sl_volatility_adjustment=True,
        sl_range_market_multiplier=0.7,  # Fixed
        
        # Take Profit Configuration  
        tp_atr_multipliers=(
            {params['tp1_multiplier']:.3f},  # TP1
            {params['tp2_multiplier']:.3f},  # TP2
            {params['tp3_multiplier']:.3f}   # TP3
        ),
        max_tp_percent=0.003,  # 0.3% max
        tp_range_market_multiplier={params['tp_range_market_multiplier']:.2f},
        tp_trend_market_multiplier={params['tp_trend_market_multiplier']:.2f},
        tp_chop_market_multiplier={params['tp_chop_market_multiplier']:.2f},
        
        # Trailing Stop Configuration
        tsl_activation_pips={params['tsl_activation_pips']},
        tsl_min_profit_pips={params['tsl_min_profit_pips']:.1f},
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier={params['trailing_atr_multiplier']:.1f},
        
        # Partial Profit Configuration
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio={params['partial_profit_sl_distance_ratio']:.2f},
        partial_profit_size_percent={params['partial_profit_size_percent']:.2f},
        
        # Other Settings
        intelligent_sizing={bool(params.get('use_intelligent_sizing', 0))},
        exit_on_signal_flip=False,  # Fixed for Strategy 1
        relaxed_mode=False,
        realistic_costs=config.realistic_costs,
        verbose=False,
        debug_decisions=config.debug_mode,
        use_daily_sharpe=config.use_daily_sharpe
    )
    
    return OptimizedProdStrategy(strategy_config)
'''
    
    print("\n📝 GENERATED CONFIGURATION CODE:")
    print(config_code)
    
    # Save to file
    os.makedirs('optimized_configs', exist_ok=True)
    filename = f'optimized_configs/strategy_{data["strategy_type"]}_{data["currency"]}_sharpe_{best["sharpe_ratio"]:.3f}.py'
    with open(filename, 'w') as f:
        f.write(config_code)
    
    print(f"\n💾 Configuration saved to: {filename}")
    
    # Also save a JSON version for easy loading
    json_config = {
        'strategy_type': data['strategy_type'],
        'currency': data['currency'],
        'sharpe_ratio': best['sharpe_ratio'],
        'parameters': params,
        'metrics': {
            'total_return': best['total_return'],
            'win_rate': best['win_rate'],
            'max_drawdown': best['max_drawdown'],
            'profit_factor': best['profit_factor'],
            'total_trades': best['total_trades']
        },
        'generated_at': datetime.now().isoformat()
    }
    
    json_filename = filename.replace('.py', '.json')
    with open(json_filename, 'w') as f:
        json.dump(json_config, f, indent=2)
    
    print(f"💾 JSON config saved to: {json_filename}")
    
    return params


def main():
    """Find and process the latest optimization results"""
    
    # Find the latest optimization result
    import glob
    result_files = glob.glob('optimizer_results/optimization_results_strategy*.json')
    
    if not result_files:
        print("❌ No optimization results found!")
        return
    
    # Get the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📁 Processing: {latest_file}")
    
    # Generate configuration
    generate_strategy_config(latest_file)


if __name__ == "__main__":
    main()