"""
React Plot Integration for run_Strategy.py
Handles the --plot-react flag to export and display charts in React

Author: Trading System
Date: 2024
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from .react_integration import ReactChartIntegration
from .chart_data_exporter import ChartDataExporter


def add_react_plot_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add React plotting arguments to the argument parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to
    """
    parser.add_argument('--plot-react', action='store_true',
                       help='Display charts in React web interface (interactive)')
    parser.add_argument('--export-react', action='store_true',
                       help='Export chart data for React without launching viewer')
    parser.add_argument('--react-port', type=int, default=5173,
                       help='Port for React development server (default: 5173)')
    parser.add_argument('--react-auto-open', action='store_true', default=True,
                       help='Automatically open browser when React server starts')


def handle_react_plotting(df: pd.DataFrame, results: Dict, args: argparse.Namespace,
                         symbol: Optional[str] = None) -> bool:
    """
    Handle React plotting based on command line arguments
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC and indicator data
    results : Dict
        Results dictionary from strategy backtest
    args : argparse.Namespace
        Command line arguments
    symbol : str, optional
        Trading symbol (e.g., 'AUDUSD')
        
    Returns:
    --------
    bool : True if React plotting was handled, False otherwise
    """
    # Check if React plotting is requested
    if not hasattr(args, 'plot_react') and not hasattr(args, 'export_react'):
        return False
        
    if not args.plot_react and not args.export_react:
        return False
    
    # Add symbol to results if provided
    if symbol and 'symbol' not in results:
        results['symbol'] = symbol
    
    # Create integration instance
    integration = ReactChartIntegration()
    
    if args.export_react:
        # Export only mode
        print("\nExporting data for React visualization...")
        output_path = integration.export_only(df, results)
        print(f"✅ Chart data exported to: {output_path}")
        print("\nTo view in React:")
        print("  1. cd react_chart")
        print("  2. npm run dev")
        print("  3. Open http://localhost:5173 in your browser")
        return True
        
    elif args.plot_react:
        # Launch React viewer
        print("\nLaunching React chart viewer...")
        auto_open = args.react_auto_open if hasattr(args, 'react_auto_open') else True
        port = args.react_port if hasattr(args, 'react_port') else 5173
        
        success = integration.export_and_launch(
            df, results,
            auto_open=auto_open,
            port=port
        )
        
        if not success:
            print("❌ Failed to launch React chart viewer")
            return False
            
        return True
    
    return False


def should_skip_matplotlib_plots(args: argparse.Namespace) -> bool:
    """
    Check if matplotlib plots should be skipped when using React
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    bool : True if matplotlib plots should be skipped
    """
    # Skip matplotlib if only React plotting is requested
    if hasattr(args, 'plot_react') and args.plot_react:
        # Skip matplotlib unless explicitly requested with --show-plots
        return not (hasattr(args, 'show_plots') and args.show_plots)
    
    return False


def export_for_react_if_requested(df: pd.DataFrame, results: Dict, 
                                 args: argparse.Namespace) -> None:
    """
    Export data for React if requested, even when using matplotlib plots
    
    This allows users to export data for later React viewing while still
    using traditional matplotlib plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC and indicator data
    results : Dict
        Results dictionary from strategy backtest
    args : argparse.Namespace
        Command line arguments
    """
    if hasattr(args, 'export_react') and args.export_react:
        integration = ReactChartIntegration()
        output_path = integration.export_only(df, results)
        print(f"\n📊 React data exported to: {output_path}")


# Example integration in run_Strategy.py:
"""
# In the main() function, add React arguments:
from strategy_code.plot_react_integration import add_react_plot_arguments
add_react_plot_arguments(parser)

# After running strategy, before or instead of matplotlib plotting:
from strategy_code.plot_react_integration import handle_react_plotting, should_skip_matplotlib_plots

# Handle React plotting
react_handled = handle_react_plotting(results_df, results_dict, args, symbol='AUDUSD')

# Skip matplotlib if React is handling the plotting
if should_skip_matplotlib_plots(args) and react_handled:
    return

# Or always export for React in addition to matplotlib:
export_for_react_if_requested(results_df, results_dict, args)
"""