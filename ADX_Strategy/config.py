"""
Configuration file for ADX Trend Strategy.

This file contains all configurable parameters for the strategy,
making it easy to adjust settings without modifying the core code.
"""

# Strategy Parameters
STRATEGY_PARAMS = {
    # DMI/ADX Parameters
    'adx_period': 14,           # Period for ADX calculation
    'adx_threshold': 50,        # Minimum ADX value to confirm strong trend
    
    # Williams %R Parameters
    'williams_period': 14,      # Period for Williams %R
    'williams_oversold': -80,   # Oversold threshold (long entry)
    'williams_overbought': -20, # Overbought threshold (short entry)
    
    # Moving Average Parameters
    'sma_period': 50,           # SMA period for stop-loss
    
    # Exit Parameters
    'tp_lookback': 30,          # Lookback period for take-profit levels
    
    # Risk Management
    'risk_percent': 0.03,       # Risk 3% per trade
    
    # Logging
    'printlog': True,           # Enable/disable logging
}

# Backtest Parameters
BACKTEST_PARAMS = {
    'initial_cash': 10000,      # Starting capital
    'commission': 0.001,        # Commission rate (0.1%)
    'start_date': '2022-01-01', # Backtest start date
    'end_date': '2023-12-31',   # Backtest end date
    'data_source': 'yahoo',     # Data source ('yahoo' or 'csv')
    'symbol': 'SPY',            # Default symbol for testing
    'timeframe': '1h',          # Timeframe for data
}

# Optimization Parameters
OPTIMIZATION_PARAMS = {
    # Parameter ranges for optimization
    'adx_threshold_range': (40, 60, 5),      # (start, stop, step)
    'williams_period_range': (10, 20, 2),    # (start, stop, step)
    'sma_period_range': (40, 60, 10),        # (start, stop, step)
    'tp_lookback_range': (20, 40, 5),        # (start, stop, step)
}

# Data Download Parameters
DATA_PARAMS = {
    'symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],  # Symbols to download
    'interval': '1h',                           # Data interval
    'period': '2y',                            # Download period
    'save_path': 'ADX_Strategy/data/',        # Path to save data
}

# Reporting Parameters
REPORT_PARAMS = {
    'generate_html': True,      # Generate HTML report
    'generate_pdf': False,      # Generate PDF report (requires additional setup)
    'save_trades': True,        # Save trade history to CSV
    'save_equity_curve': True,  # Save equity curve data
    'plot_results': True,       # Plot backtest results
}

# Alert Parameters (for live trading - not implemented yet)
ALERT_PARAMS = {
    'email_alerts': False,      # Send email alerts
    'sms_alerts': False,        # Send SMS alerts
    'webhook_url': None,        # Webhook URL for alerts
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_sharpe_ratio': 1.0,    # Minimum acceptable Sharpe ratio
    'max_drawdown': 20.0,       # Maximum acceptable drawdown (%)
    'min_win_rate': 40.0,       # Minimum win rate (%)
    'min_profit_factor': 1.5,   # Minimum profit factor
}


def get_strategy_config():
    """Get complete strategy configuration."""
    return {
        'strategy': STRATEGY_PARAMS,
        'backtest': BACKTEST_PARAMS,
        'optimization': OPTIMIZATION_PARAMS,
        'data': DATA_PARAMS,
        'report': REPORT_PARAMS,
        'performance': PERFORMANCE_THRESHOLDS,
    }


def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate strategy parameters
    if STRATEGY_PARAMS['adx_threshold'] < 0 or STRATEGY_PARAMS['adx_threshold'] > 100:
        errors.append("ADX threshold must be between 0 and 100")
    
    if STRATEGY_PARAMS['williams_oversold'] > STRATEGY_PARAMS['williams_overbought']:
        errors.append("Williams oversold must be less than overbought")
    
    if STRATEGY_PARAMS['risk_percent'] < 0 or STRATEGY_PARAMS['risk_percent'] > 1:
        errors.append("Risk percent must be between 0 and 1")
    
    # Validate backtest parameters
    if BACKTEST_PARAMS['initial_cash'] <= 0:
        errors.append("Initial cash must be positive")
    
    if BACKTEST_PARAMS['commission'] < 0:
        errors.append("Commission cannot be negative")
    
    return errors


if __name__ == '__main__':
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
        
    # Display current configuration
    print("\nCurrent Strategy Configuration:")
    print("-" * 40)
    for key, value in STRATEGY_PARAMS.items():
        print(f"{key:20}: {value}")
        
    print("\nBacktest Configuration:")
    print("-" * 40)
    for key, value in BACKTEST_PARAMS.items():
        print(f"{key:20}: {value}")