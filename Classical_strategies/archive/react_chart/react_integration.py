"""
React Chart Integration Module
Handles exporting data and launching React visualization

Author: Trading System
Date: 2024
"""

import os
import json
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from .chart_data_exporter import ChartDataExporter


class ReactChartIntegration:
    """Manages integration between Python strategy and React charts"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.react_dir = self.base_dir / 'react_chart'
        self.data_dir = self.react_dir / 'public'
        self.data_file = self.data_dir / 'chart_data.json'
        
    def export_and_launch(self, df: pd.DataFrame, results: Dict, 
                         auto_open: bool = True, port: int = 5173) -> bool:
        """
        Export data and launch React chart viewer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading dataframe with OHLC and indicators
        results : Dict
            Results dictionary containing trades and metrics
        auto_open : bool
            Automatically open browser when server starts
        port : int
            Port for React dev server
            
        Returns:
        --------
        bool : Success status
        """
        try:
            # Ensure directories exist
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Export data to JSON
            print("\nExporting chart data for React visualization...")
            ChartDataExporter.export_to_json(df, results, str(self.data_file))
            
            # Check if React app is already running
            if self._is_server_running(port):
                print(f"React server already running on port {port}")
                if auto_open:
                    webbrowser.open(f'http://localhost:{port}')
                return True
            
            # Install dependencies if needed
            if not (self.react_dir / 'node_modules').exists():
                print("Installing React dependencies...")
                subprocess.run(['npm', 'install'], cwd=self.react_dir, check=True)
            
            # Start React dev server
            print(f"\nStarting React chart viewer on port {port}...")
            print("Press Ctrl+C in the terminal to stop the server")
            
            # Launch server in subprocess
            env = os.environ.copy()
            process = subprocess.Popen(
                ['npm', 'run', 'dev', '--', '--port', str(port)],
                cwd=self.react_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Open browser if requested
            if auto_open:
                url = f'http://localhost:{port}'
                print(f"\nOpening browser at {url}")
                webbrowser.open(url)
            
            # Keep server running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping React server...")
                process.terminate()
                process.wait()
            
            return True
            
        except Exception as e:
            print(f"Error launching React chart: {e}")
            return False
    
    def export_only(self, df: pd.DataFrame, results: Dict, 
                   output_path: Optional[str] = None) -> str:
        """
        Export data without launching React server
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading dataframe
        results : Dict
            Results dictionary
        output_path : str, optional
            Custom output path for JSON file
            
        Returns:
        --------
        str : Path to exported JSON file
        """
        if output_path is None:
            output_path = str(self.data_file)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Export data
        ChartDataExporter.export_to_json(df, results, output_path)
        
        return output_path
    
    def _is_server_running(self, port: int) -> bool:
        """Check if server is already running on port"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    @staticmethod
    def add_react_plot_option(results_df: pd.DataFrame, results_dict: Dict,
                            show_react: bool = False, export_only: bool = False,
                            output_path: Optional[str] = None) -> None:
        """
        Helper method to integrate with existing plotting workflow
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from strategy
        results_dict : Dict
            Results dictionary from strategy
        show_react : bool
            Launch React chart viewer
        export_only : bool
            Only export data without launching viewer
        output_path : str, optional
            Custom output path for JSON
        """
        if not (show_react or export_only):
            return
        
        integration = ReactChartIntegration()
        
        if export_only:
            path = integration.export_only(results_df, results_dict, output_path)
            print(f"\nChart data exported to: {path}")
            print("To view in React, run: npm run dev in the react_chart directory")
        elif show_react:
            integration.export_and_launch(results_df, results_dict)