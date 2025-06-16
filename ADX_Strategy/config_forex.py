"""
Forex-optimized configuration for ADX Trend Strategy.

Based on diagnostic analysis of AUDUSD 1H data.
"""

# Forex-optimized Strategy Parameters
FOREX_STRATEGY_PARAMS = {
    # DMI/ADX Parameters
    'adx_period': 14,           # Period for ADX calculation
    'adx_threshold': 30,        # Lower threshold for forex (was 50)
    
    # Williams %R Parameters
    'williams_period': 14,      # Period for Williams %R
    'williams_oversold': -85,   # Slightly more extreme for fewer false signals
    'williams_overbought': -15, # Slightly more extreme for fewer false signals
    
    # Moving Average Parameters
    'sma_period': 50,           # SMA period for stop-loss
    
    # Exit Parameters
    'tp_lookback': 30,          # Lookback period for take-profit levels
    
    # Risk Management
    'risk_percent': 0.02,       # Risk 2% per trade (reduced for forex)
    
    # Logging
    'printlog': False,          # Disable verbose logging
}

# Alternative configurations based on market conditions
CONSERVATIVE_PARAMS = {
    'adx_period': 14,
    'adx_threshold': 35,        # Higher threshold
    'williams_period': 20,      # Longer period for smoother signals
    'williams_oversold': -90,
    'williams_overbought': -10,
    'sma_period': 60,
    'tp_lookback': 40,
    'risk_percent': 0.015,
    'printlog': False,
}

AGGRESSIVE_PARAMS = {
    'adx_period': 14,
    'adx_threshold': 25,        # Lower threshold for more trades
    'williams_period': 10,      # Shorter period for quicker signals
    'williams_oversold': -80,
    'williams_overbought': -20,
    'sma_period': 40,
    'tp_lookback': 20,
    'risk_percent': 0.025,
    'printlog': False,
}

# Backtest parameters for forex
FOREX_BACKTEST_PARAMS = {
    'initial_cash': 10000,
    'commission': 0.0002,       # 2 pips spread typical for major forex pairs
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'data_source': 'csv',
    'symbol': 'AUDUSD',
    'timeframe': '1h',
}

def get_forex_config(style='balanced'):
    """
    Get forex configuration based on trading style.
    
    Parameters:
    -----------
    style : str
        Trading style - 'balanced', 'conservative', or 'aggressive'
    """
    if style == 'conservative':
        strategy_params = CONSERVATIVE_PARAMS
    elif style == 'aggressive':
        strategy_params = AGGRESSIVE_PARAMS
    else:
        strategy_params = FOREX_STRATEGY_PARAMS
    
    return {
        'strategy': strategy_params,
        'backtest': FOREX_BACKTEST_PARAMS,
    }

if __name__ == '__main__':
    # Display configurations
    print("=== FOREX STRATEGY CONFIGURATIONS ===\n")
    
    for style in ['balanced', 'conservative', 'aggressive']:
        config = get_forex_config(style)
        print(f"{style.upper()} Configuration:")
        print("-" * 30)
        for key, value in config['strategy'].items():
            if key != 'printlog':
                print(f"{key:20}: {value}")
        print()