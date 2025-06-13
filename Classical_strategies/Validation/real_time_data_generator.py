"""
Real-Time Market Data Generator
Simulates live market conditions by yielding one row at a time
"""

import pandas as pd
import numpy as np
from typing import Generator, Dict, Any
import os
import sys
from datetime import datetime

# Add parent directory to path to import technical indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from technical_indicators_custom import TIC


class RealTimeDataGenerator:
    """
    Generates market data one row at a time to simulate real trading conditions
    """
    
    def __init__(self, currency_pair: str = 'AUDUSD', data_path: str = None):
        self.currency_pair = currency_pair
        self.data_path = data_path or self._find_data_path()
        self.full_data = None
        self.current_index = 0
        self.historical_buffer = pd.DataFrame()
        
        # Load the full dataset
        self._load_data()
        
    def _find_data_path(self) -> str:
        """Find the data file path"""
        possible_paths = [
            f'../data/{self.currency_pair}_MASTER_15M.csv',
            f'../../data/{self.currency_pair}_MASTER_15M.csv',
            f'../../../data/{self.currency_pair}_MASTER_15M.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(f"Could not find data file for {self.currency_pair}")
    
    def _load_data(self):
        """Load the full dataset"""
        print(f"Loading {self.currency_pair} data from {self.data_path}")
        self.full_data = pd.read_csv(self.data_path)
        self.full_data['DateTime'] = pd.to_datetime(self.full_data['DateTime'])
        print(f"Loaded {len(self.full_data):,} rows of data")
        print(f"Date range: {self.full_data['DateTime'].iloc[0]} to {self.full_data['DateTime'].iloc[-1]}")
    
    def reset(self, start_index: int = 0):
        """Reset generator to start from a specific index"""
        self.current_index = start_index
        self.historical_buffer = pd.DataFrame()
        print(f"Generator reset to index {start_index}")
    
    def get_sample_period(self, start_date: str = None, rows: int = 8000, start_idx: int = None) -> tuple:
        """Get a specific sample period for testing"""
        if start_idx is not None:
            # Use specific start index
            pass
        elif start_date:
            # Find index for start date
            start_idx = self.full_data[self.full_data['DateTime'] >= start_date].index[0]
        else:
            # Random start point ensuring we have enough data
            max_start = len(self.full_data) - rows - 500  # Buffer for indicators
            start_idx = np.random.randint(500, max_start)  # Start after 500 rows for indicators
        
        end_idx = min(start_idx + rows, len(self.full_data))
        
        print(f"Sample period: Index {start_idx} to {end_idx} ({end_idx - start_idx} rows)")
        print(f"Date range: {self.full_data['DateTime'].iloc[start_idx]} to {self.full_data['DateTime'].iloc[end_idx-1]}")
        
        return start_idx, end_idx
    
    def stream_data(self, start_index: int, end_index: int, 
                   min_history: int = 200) -> Generator[Dict[str, Any], None, None]:
        """
        Stream data one row at a time with incremental indicator calculation
        
        Args:
            start_index: Starting row index
            end_index: Ending row index  
            min_history: Minimum rows needed before yielding data
        
        Yields:
            Dict containing current row data with calculated indicators
        """
        
        # Reset state
        self.reset(start_index)
        
        # Build initial historical buffer
        buffer_start = max(0, start_index - min_history)
        self.historical_buffer = self.full_data.iloc[buffer_start:start_index].copy()
        
        print(f"\\nStarting real-time simulation...")
        print(f"Historical buffer: {len(self.historical_buffer)} rows")
        print(f"Will stream from index {start_index} to {end_index}")
        print("=" * 60)
        
        # Stream data row by row
        for idx in range(start_index, end_index):
            current_row = self.full_data.iloc[idx].copy()
            
            # Add current row to historical buffer
            self.historical_buffer = pd.concat([
                self.historical_buffer, 
                pd.DataFrame([current_row])
            ], ignore_index=True)
            
            # Calculate indicators on the growing buffer
            # Only calculate if we have enough history
            if len(self.historical_buffer) >= min_history:
                # Calculate all indicators on the complete buffer
                buffer_with_indicators = self._calculate_incremental_indicators(
                    self.historical_buffer.copy()
                )
                
                # Get the latest row with indicators
                latest_row_with_indicators = buffer_with_indicators.iloc[-1].copy()
                
                # Create result dictionary
                result = {
                    'row_index': idx,
                    'current_time': current_row['DateTime'],
                    'data': latest_row_with_indicators,
                    'historical_length': len(self.historical_buffer),
                    'price': current_row['Close'],
                    'ohlc': {
                        'Open': current_row['Open'],
                        'High': current_row['High'], 
                        'Low': current_row['Low'],
                        'Close': current_row['Close']
                    }
                }
                
                # Print progress every 100 rows
                if (idx - start_index) % 100 == 0:
                    print(f"Row {idx:5d} | {current_row['DateTime']} | "
                          f"Price: {current_row['Close']:.5f} | "
                          f"Buffer: {len(self.historical_buffer)} rows")
                
                yield result
            
            # Keep buffer size manageable (keep last 1000 rows)
            if len(self.historical_buffer) > 1000:
                self.historical_buffer = self.historical_buffer.iloc[-800:].reset_index(drop=True)
    
    def _calculate_incremental_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators incrementally on the growing dataset
        
        This simulates real-time calculation where indicators are computed
        only on data available up to the current point in time
        """
        
        # Set DateTime as index for indicator calculations
        df = df.copy()
        df.set_index('DateTime', inplace=True)
        
        # Calculate indicators using the same method as the main strategy
        try:
            # Add Neuro Trend Intelligent
            df = TIC.add_neuro_trend_intelligent(df)
            
            # Add Market Bias
            df = TIC.add_market_bias(df)
            
            # Add Intelligent Chop
            df = TIC.add_intelligent_chop(df)
            
        except Exception as e:
            # If indicators fail (not enough data), fill with neutral values
            if 'NTI_Direction' not in df.columns:
                df['NTI_Direction'] = 0
            if 'MB_Bias' not in df.columns:
                df['MB_Bias'] = 0
            if 'IC_Regime' not in df.columns:
                df['IC_Regime'] = 3  # Neutral regime
            if 'IC_ATR_Normalized' not in df.columns:
                df['IC_ATR_Normalized'] = 0.0001  # Small default ATR
            if 'IC_RegimeName' not in df.columns:
                df['IC_RegimeName'] = 'Unknown'
        
        # Reset index to get DateTime back as column
        df.reset_index(inplace=True)
        
        return df
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset"""
        if self.full_data is None:
            return {}
        
        return {
            'currency_pair': self.currency_pair,
            'total_rows': len(self.full_data),
            'date_range': {
                'start': self.full_data['DateTime'].iloc[0],
                'end': self.full_data['DateTime'].iloc[-1]
            },
            'columns': list(self.full_data.columns),
            'price_range': {
                'min': self.full_data['Low'].min(),
                'max': self.full_data['High'].max(),
                'current': self.full_data['Close'].iloc[-1]
            }
        }


def test_data_generator():
    """Test the real-time data generator"""
    print("Testing Real-Time Data Generator")
    print("=" * 50)
    
    # Create generator
    generator = RealTimeDataGenerator('AUDUSD')
    
    # Print data info
    info = generator.get_data_info()
    print(f"Currency: {info['currency_pair']}")
    print(f"Total rows: {info['total_rows']:,}")
    print(f"Date range: {info['date_range']['start']} to {info['date_range']['end']}")
    print(f"Price range: {info['price_range']['min']:.5f} to {info['price_range']['max']:.5f}")
    
    # Get a small sample for testing
    start_idx, end_idx = generator.get_sample_period(rows=500)
    
    print(f"\\nTesting with {end_idx - start_idx} rows...")
    
    # Stream first 10 rows
    count = 0
    for data_point in generator.stream_data(start_idx, end_idx):
        count += 1
        
        # Print detailed info for first few rows
        if count <= 5:
            print(f"\\nRow {count}:")
            print(f"  Index: {data_point['row_index']}")
            print(f"  Time: {data_point['current_time']}")
            print(f"  Price: {data_point['price']:.5f}")
            print(f"  NTI: {data_point['data'].get('NTI_Direction', 'N/A')}")
            print(f"  MB: {data_point['data'].get('MB_Bias', 'N/A')}")
            print(f"  IC: {data_point['data'].get('IC_Regime', 'N/A')}")
            print(f"  Historical buffer: {data_point['historical_length']} rows")
        
        # Stop after 10 rows for testing
        if count >= 10:
            break
    
    print(f"\\nSuccessfully streamed {count} data points!")


if __name__ == "__main__":
    test_data_generator()