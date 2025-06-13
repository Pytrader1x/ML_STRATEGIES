"""
Chart Data Exporter Module
Exports trading data to JSON format for React visualization

Author: Trading System
Date: 2024
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ChartDataExporter:
    """Export trading data to JSON format for React charts"""
    
    @staticmethod
    def export_to_json(df: pd.DataFrame, results: Dict, output_path: str = None) -> Dict:
        """
        Export trading data to JSON format for React visualization
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading dataframe with OHLC and indicator data
        results : Dict
            Results dictionary containing trades and performance metrics
        output_path : str, optional
            Path to save JSON file. If None, returns dict only
            
        Returns:
        --------
        Dict : Chart data in JSON-serializable format
        """
        
        # Initialize data structure
        chart_data = {
            'metadata': {
                'symbol': results.get('symbol', 'Unknown'),
                'timeframe': ChartDataExporter._detect_timeframe(df),
                'start_date': str(df.index[0]) if isinstance(df.index[0], pd.Timestamp) else str(df.index[0]),
                'end_date': str(df.index[-1]) if isinstance(df.index[-1], pd.Timestamp) else str(df.index[-1]),
                'total_rows': len(df)
            },
            'ohlc': [],
            'indicators': {
                'neurotrend': {
                    'fast_ema': [],
                    'slow_ema': [],
                    'direction': []
                },
                'market_bias': {
                    'bias': [],
                    'o2': [],
                    'h2': [],
                    'l2': [],
                    'c2': []
                },
                'intelligent_chop': {
                    'regime': [],
                    'regime_name': []
                }
            },
            'trades': [],
            'performance': {
                'position_sizes': [],
                'cumulative_pnl': [],
                'returns': []
            }
        }
        
        # Export OHLC data with guaranteed unique sequential timestamps
        # Start from a base timestamp and ensure 15-minute intervals
        if len(df) > 0:
            if isinstance(df.index[0], pd.Timestamp):
                # Use a rounded base timestamp to ensure clean intervals
                base_timestamp = int(df.index[0].timestamp())
                # Round to nearest 15-minute boundary for clean chart display
                base_timestamp = (base_timestamp // 900) * 900
            else:
                base_timestamp = int(df.index[0])
        else:
            base_timestamp = 1609459200  # Default: 2021-01-01 00:00:00 UTC
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Guarantee unique sequential timestamps in MILLISECONDS for React chart
            timestamp = (base_timestamp + (i * 900)) * 1000  # Convert to milliseconds
            
            ohlc_point = {
                'time': timestamp,
                'open': float(row['Open']) if not pd.isna(row['Open']) else None,
                'high': float(row['High']) if not pd.isna(row['High']) else None,
                'low': float(row['Low']) if not pd.isna(row['Low']) else None,
                'close': float(row['Close']) if not pd.isna(row['Close']) else None
            }
            chart_data['ohlc'].append(ohlc_point)
            
            # Export NeuroTrend indicators
            if 'NTI_FastEMA' in df.columns:
                chart_data['indicators']['neurotrend']['fast_ema'].append({
                    'time': timestamp,
                    'value': float(row['NTI_FastEMA']) if not pd.isna(row['NTI_FastEMA']) else None
                })
                chart_data['indicators']['neurotrend']['slow_ema'].append({
                    'time': timestamp,
                    'value': float(row['NTI_SlowEMA']) if not pd.isna(row['NTI_SlowEMA']) else None
                })
                chart_data['indicators']['neurotrend']['direction'].append({
                    'time': timestamp,
                    'value': int(row['NTI_Direction']) if not pd.isna(row['NTI_Direction']) else 0
                })
            
            # Export Market Bias
            if 'MB_Bias' in df.columns:
                chart_data['indicators']['market_bias']['bias'].append({
                    'time': timestamp,
                    'value': int(row['MB_Bias']) if not pd.isna(row['MB_Bias']) else 0
                })
                if 'MB_o2' in df.columns:
                    chart_data['indicators']['market_bias']['o2'].append({
                        'time': timestamp,
                        'value': float(row['MB_o2']) if not pd.isna(row['MB_o2']) else None
                    })
                    chart_data['indicators']['market_bias']['h2'].append({
                        'time': timestamp,
                        'value': float(row['MB_h2']) if not pd.isna(row['MB_h2']) else None
                    })
                    chart_data['indicators']['market_bias']['l2'].append({
                        'time': timestamp,
                        'value': float(row['MB_l2']) if not pd.isna(row['MB_l2']) else None
                    })
                    chart_data['indicators']['market_bias']['c2'].append({
                        'time': timestamp,
                        'value': float(row['MB_c2']) if not pd.isna(row['MB_c2']) else None
                    })
            
            # Export Intelligent Chop
            if 'IC_Regime' in df.columns:
                chart_data['indicators']['intelligent_chop']['regime'].append({
                    'time': timestamp,
                    'value': int(row['IC_Regime']) if not pd.isna(row['IC_Regime']) else 0
                })
                if 'IC_RegimeName' in df.columns:
                    chart_data['indicators']['intelligent_chop']['regime_name'].append({
                        'time': timestamp,
                        'value': str(row['IC_RegimeName']) if not pd.isna(row['IC_RegimeName']) else 'Unknown'
                    })
            
            # Export Confidence
            if 'NTI_Confidence' in df.columns:
                if 'confidence' not in chart_data['indicators']:
                    chart_data['indicators']['confidence'] = []
                chart_data['indicators']['confidence'].append({
                    'time': timestamp,
                    'value': float(row['NTI_Confidence']) if not pd.isna(row['NTI_Confidence']) else None
                })
            
            # Export Position Sizes and P&L
            if 'Position_Size' in df.columns:
                chart_data['performance']['position_sizes'].append({
                    'time': timestamp,
                    'value': float(row['Position_Size']) if not pd.isna(row['Position_Size']) else 0
                })
            
            if 'Cumulative_PnL' in df.columns:
                chart_data['performance']['cumulative_pnl'].append({
                    'time': timestamp,
                    'value': float(row['Cumulative_PnL']) if not pd.isna(row['Cumulative_PnL']) else 0
                })
            
            if 'Returns' in df.columns:
                chart_data['performance']['returns'].append({
                    'time': timestamp,
                    'value': float(row['Returns']) if not pd.isna(row['Returns']) else 0
                })
        
        # Export trades
        trades = results.get('trades', [])
        for trade in trades:
            # Handle Trade object or dict
            if hasattr(trade, '__dict__'):
                trade_data = {
                    'entry_time': int(trade.entry_time.timestamp() * 1000) if isinstance(trade.entry_time, pd.Timestamp) else trade.entry_time,
                    'exit_time': int(trade.exit_time.timestamp() * 1000) if isinstance(trade.exit_time, pd.Timestamp) else trade.exit_time,
                    'entry_price': float(trade.entry_price),
                    'exit_price': float(trade.exit_price),
                    'direction': trade.direction.value if hasattr(trade.direction, 'value') else trade.direction,
                    'exit_reason': trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason,
                    'take_profits': [float(tp) for tp in trade.take_profits] if trade.take_profits else [],
                    'stop_loss': float(trade.stop_loss) if trade.stop_loss else None,
                    'pnl': float(trade.pnl) if hasattr(trade, 'pnl') else None,
                    'pnl_pct': float(trade.pnl_pct) if hasattr(trade, 'pnl_pct') else None,
                    'position_size': float(trade.position_size) if hasattr(trade, 'position_size') else 1000000
                }
                
                # Handle partial exits - serialize properly
                if hasattr(trade, 'partial_exits') and trade.partial_exits:
                    partial_exits = []
                    for pe in trade.partial_exits:
                        if hasattr(pe, '__dict__'):
                            # It's a PartialExit object
                            pe_dict = {
                                'time': int(pe.time.timestamp() * 1000) if isinstance(pe.time, pd.Timestamp) else str(pe.time),
                                'price': float(pe.price) if hasattr(pe, 'price') else 0,
                                'size': float(pe.size) if hasattr(pe, 'size') else 0,
                                'tp_level': int(pe.tp_level) if hasattr(pe, 'tp_level') else 1,
                                'pnl': float(pe.pnl) if hasattr(pe, 'pnl') else 0
                            }
                            partial_exits.append(pe_dict)
                        elif isinstance(pe, dict):
                            # Already a dict
                            partial_exits.append(pe)
                        else:
                            # Convert object to string as fallback
                            partial_exits.append(str(pe))
                    trade_data['partial_exits'] = partial_exits
                else:
                    trade_data['partial_exits'] = []
            else:
                # Handle dict trade data
                trade_data = {}
                for key, value in trade.items():
                    if isinstance(value, pd.Timestamp):
                        trade_data[key] = int(value.timestamp() * 1000)
                    elif isinstance(value, (np.integer, np.floating)):
                        trade_data[key] = float(value)
                    elif isinstance(value, list):
                        # Handle list of partial exits or take profits
                        processed_list = []
                        for v in value:
                            if isinstance(v, dict):
                                # Process partial exit dict
                                processed_dict = {}
                                for k, val in v.items():
                                    if isinstance(val, pd.Timestamp):
                                        processed_dict[k] = int(val.timestamp() * 1000)
                                    elif isinstance(val, (np.integer, np.floating)):
                                        processed_dict[k] = float(val)
                                    else:
                                        processed_dict[k] = val
                                processed_list.append(processed_dict)
                            elif hasattr(v, '__dict__'):
                                # Handle PartialExit objects
                                pe_dict = {
                                    'time': int(v.time.timestamp() * 1000) if isinstance(getattr(v, 'time', None), pd.Timestamp) else str(getattr(v, 'time', '')),
                                    'price': float(getattr(v, 'price', 0)),
                                    'size': float(getattr(v, 'size', 0)),
                                    'tp_level': int(getattr(v, 'tp_level', 1)),
                                    'pnl': float(getattr(v, 'pnl', 0))
                                }
                                processed_list.append(pe_dict)
                            elif isinstance(v, (np.integer, np.floating)):
                                processed_list.append(float(v))
                            else:
                                processed_list.append(str(v))
                        trade_data[key] = processed_list
                    else:
                        trade_data[key] = value
            
            chart_data['trades'].append(trade_data)
        
        # Add performance metrics
        chart_data['metrics'] = {
            'total_trades': results.get('total_trades', 0),
            'winning_trades': results.get('winning_trades', 0),
            'losing_trades': results.get('losing_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'total_return': results.get('total_return', 0),
            'profit_factor': results.get('profit_factor', 0),
            'total_pnl': results.get('total_pnl', 0)
        }
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use custom JSON encoder to handle any remaining non-serializable objects
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, pd.Timestamp):
                        return str(obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            with open(output_path, 'w') as f:
                json.dump(chart_data, f, indent=2, cls=CustomEncoder)
            
            print(f"Chart data exported to: {output_path}")
        
        return chart_data
    
    @staticmethod
    def _detect_timeframe(df: pd.DataFrame) -> str:
        """Detect timeframe from DataFrame index"""
        if len(df) < 2:
            return 'Unknown'
        
        if isinstance(df.index[0], pd.Timestamp):
            time_diff = df.index[1] - df.index[0]
            seconds = time_diff.total_seconds()
            
            if seconds == 900:  # 15 minutes
                return '15M'
            elif seconds == 3600:  # 1 hour
                return '1H'
            elif seconds == 14400:  # 4 hours
                return '4H'
            elif seconds == 86400:  # 1 day
                return '1D'
            else:
                minutes = int(seconds / 60)
                return f'{minutes}M'
        
        return 'Unknown'
    
    def export_chart_data(self, df, results, output_path='../react_chart/public/chart_data.json'):
        """Export strategy data for React chart visualization with proper serialization"""
        
        # Prepare OHLC data with timestamps
        ohlc_data = []
        for idx, row in df.iterrows():
            ohlc_data.append({
                'time': int(idx.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close'])
            })
        
        # Export indicators
        indicators = {
            'neurotrend': {
                'fast_ema': [float(v) if not pd.isna(v) else None for v in df['NTI_FastEMA'].values],
                'slow_ema': [float(v) if not pd.isna(v) else None for v in df['NTI_SlowEMA'].values],
                'direction': [int(v) if not pd.isna(v) else 0 for v in df['NTI_Direction'].values],
                'confidence': [float(v) if not pd.isna(v) else 0 for v in df.get('NTI_Confidence', pd.Series([0]*len(df))).values]
            },
            'market_bias': {
                'bias': [int(v) if not pd.isna(v) else 0 for v in df['MB_Bias'].values],
                'o2': [float(v) if not pd.isna(v) else None for v in df['MB_o2'].values],
                'h2': [float(v) if not pd.isna(v) else None for v in df['MB_h2'].values],
                'l2': [float(v) if not pd.isna(v) else None for v in df['MB_l2'].values],
                'c2': [float(v) if not pd.isna(v) else None for v in df['MB_c2'].values]
            },
            'intelligent_chop': {
                'signal': [int(v) if not pd.isna(v) else 0 for v in df['IC_Signal'].values],
                'regime': [str(v) if not pd.isna(v) else 'Unknown' for v in df.get('IC_RegimeName', pd.Series(['Unknown']*len(df))).values]
            }
        }
        
        # Process trades with proper serialization
        trades_data = []
        if 'trades' in results and results['trades']:
            for trade in results['trades']:
                trade_dict = {}
                
                # Handle both Trade objects and dictionaries
                if hasattr(trade, '__dict__'):
                    # It's a Trade object - extract attributes
                    trade_dict = {
                        'entry_time': str(trade.entry_time) if hasattr(trade, 'entry_time') else '',
                        'exit_time': str(trade.exit_time) if hasattr(trade, 'exit_time') else '',
                        'entry_price': float(trade.entry_price) if hasattr(trade, 'entry_price') else 0,
                        'exit_price': float(trade.exit_price) if hasattr(trade, 'exit_price') and trade.exit_price else None,
                        'direction': str(trade.direction.value if hasattr(trade.direction, 'value') else trade.direction) if hasattr(trade, 'direction') else '',
                        'pnl': float(trade.pnl) if hasattr(trade, 'pnl') else 0,
                        'pnl_pct': float(trade.pnl_pct) if hasattr(trade, 'pnl_pct') else 0,
                        'exit_reason': str(trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason) if hasattr(trade, 'exit_reason') else '',
                        'position_size': float(trade.position_size) if hasattr(trade, 'position_size') else 1000000,
                        'stop_loss': float(trade.stop_loss) if hasattr(trade, 'stop_loss') and trade.stop_loss else None,
                        'take_profits': []
                    }
                    
                    # Handle take profits
                    if hasattr(trade, 'take_profits') and trade.take_profits:
                        trade_dict['take_profits'] = [float(tp) if tp else None for tp in trade.take_profits]
                    
                    # Handle partial exits - serialize properly
                    if hasattr(trade, 'partial_exits') and trade.partial_exits:
                        partial_exits = []
                        for pe in trade.partial_exits:
                            if hasattr(pe, '__dict__'):
                                # It's a PartialExit object
                                pe_dict = {
                                    'time': str(pe.time) if hasattr(pe, 'time') else '',
                                    'price': float(pe.price) if hasattr(pe, 'price') else 0,
                                    'size': float(pe.size) if hasattr(pe, 'size') else 0,
                                    'tp_level': int(pe.tp_level) if hasattr(pe, 'tp_level') else 1,
                                    'pnl': float(pe.pnl) if hasattr(pe, 'pnl') else 0
                                }
                                partial_exits.append(pe_dict)
                            elif isinstance(pe, dict):
                                # Already a dict
                                partial_exits.append(pe)
                        trade_dict['partial_exits'] = partial_exits
                    else:
                        trade_dict['partial_exits'] = []
                
                else:
                    # It's already a dictionary
                    trade_dict = trade
                    # Ensure all values are serializable
                    if 'partial_exits' in trade_dict:
                        serialized_pe = []
                        for pe in trade_dict['partial_exits']:
                            if isinstance(pe, dict):
                                serialized_pe.append(pe)
                            else:
                                # Try to convert object to dict
                                serialized_pe.append({
                                    'time': str(getattr(pe, 'time', '')),
                                    'price': float(getattr(pe, 'price', 0)),
                                    'size': float(getattr(pe, 'size', 0)),
                                    'tp_level': int(getattr(pe, 'tp_level', 1)),
                                    'pnl': float(getattr(pe, 'pnl', 0))
                                })
                        trade_dict['partial_exits'] = serialized_pe
                
                trades_data.append(trade_dict)
        
        # Export performance metrics
        performance = {
            'total_pnl': float(results.get('total_pnl', 0)),
            'total_return': float(results.get('total_return', 0)),
            'sharpe_ratio': float(results.get('sharpe_ratio', 0)),
            'win_rate': float(results.get('win_rate', 0)),
            'max_drawdown': float(results.get('max_drawdown', 0)),
            'profit_factor': float(results.get('profit_factor', 0)),
            'total_trades': int(results.get('total_trades', 0))
        }
        
        # Compile all data
        chart_data = {
            'metadata': {
                'symbol': results.get('symbol', 'UNKNOWN'),
                'timeframe': '15M',
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1]),
                'total_rows': len(df)
            },
            'ohlc': ohlc_data,
            'indicators': indicators,
            'trades': trades_data,
            'performance': performance
        }
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use custom JSON encoder to handle any remaining non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, pd.Timestamp):
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2, cls=CustomEncoder)
        
        return output_path

    @staticmethod
    def start_react_server(data_path: str = None, port: int = 5173):
        """
        Start the React development server with optional data path
        
        Parameters:
        -----------
        data_path : str, optional
            Path to JSON data file to load
        port : int
            Port for React dev server (default: 5173)
        """
        import subprocess
        import os
        
        # Set environment variable for data path if provided
        env = os.environ.copy()
        if data_path:
            env['VITE_DATA_PATH'] = data_path
        
        # Change to react_chart directory
        react_dir = Path(__file__).parent.parent / 'react_chart'
        
        # Start React dev server
        subprocess.run(['npm', 'run', 'dev'], cwd=react_dir, env=env)