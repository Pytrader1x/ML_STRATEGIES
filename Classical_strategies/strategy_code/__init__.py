"""
Strategy Code Package
Contains the core production trading strategy implementation
"""

from .Prod_strategy import (
    create_optimized_strategy,
    OptimizedProdStrategy,
    OptimizedStrategyConfig,
    Trade,
    TradeDirection,
    ExitReason,
    ConfidenceLevel,
    PartialExit,
    RiskManager,
    PnLCalculator
)

from .Prod_plotting import (
    plot_production_results,
    ProductionPlotter,
    DataStatsCalculator
)

__all__ = [
    # Strategy functions
    'create_optimized_strategy',
    
    # Main classes
    'OptimizedProdStrategy',
    'OptimizedStrategyConfig',
    
    # Plotting
    'plot_production_results',
    'ProductionPlotter',
    'DataStatsCalculator',
    
    # Enums and data classes
    'Trade',
    'TradeDirection',
    'ExitReason',
    'ConfidenceLevel',
    'PartialExit',
    'RiskManager',
    'PnLCalculator'
]