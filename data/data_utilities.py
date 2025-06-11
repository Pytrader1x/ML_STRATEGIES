#!/usr/bin/env python3
"""
Data Utilities - Helper functions for FX data processing and management
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse


class DataUtilities:
    """Collection of utilities for FX data processing"""
    
    @staticmethod
    def filter_by_year(input_file: str, year: int, output_file: Optional[str] = None) -> pd.DataFrame:
        """Filter data to keep only specific year"""
        df = pd.read_csv(input_file, index_col='Time', parse_dates=True)
        df_filtered = df[df.index.year >= year]
        
        if output_file:
            df_filtered.to_csv(output_file)
            print(f"Filtered {len(df)} -> {len(df_filtered)} rows, saved to {output_file}")
            
        return df_filtered
    
    @staticmethod
    def validate_data(file_path: str) -> Dict[str, any]:
        """Comprehensive data validation"""
        df = pd.read_csv(file_path, index_col='Time', parse_dates=True)
        
        validation = {
            'file': file_path,
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_timestamps': df.index.duplicated().sum(),
        }
        
        # Check time gaps
        time_diff = df.index.to_series().diff()
        expected_interval = pd.Timedelta('15 minutes')
        gaps = time_diff[time_diff > expected_interval * 2]
        validation['time_gaps'] = len(gaps)
        
        # Price statistics
        if 'Close' in df.columns:
            close_prices = df['Close']
            validation['price_stats'] = {
                'min': close_prices.min(),
                'max': close_prices.max(),
                'mean': close_prices.mean(),
                'std': close_prices.std()
            }
            
            # Check for anomalies
            pct_changes = close_prices.pct_change().abs()
            validation['large_price_moves'] = (pct_changes > 0.1).sum()  # >10% moves
            validation['zero_prices'] = (close_prices == 0).sum()
            validation['negative_prices'] = (close_prices < 0).sum()
        
        # OHLC consistency
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            validation['ohlc_errors'] = {
                'high_less_than_low': (df['High'] < df['Low']).sum(),
                'close_outside_range': ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum(),
                'open_outside_range': ((df['Open'] > df['High']) | (df['Open'] < df['Low'])).sum()
            }
        
        return validation
    
    @staticmethod
    def resample_data(input_file: str, timeframe: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Resample data to different timeframe"""
        df = pd.read_csv(input_file, index_col='Time', parse_dates=True)
        
        # Ensure data is sorted
        df = df.sort_index()
        
        # Resample based on OHLC
        resampled = df.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if output_file:
            resampled.to_csv(output_file)
            print(f"Resampled {len(df)} -> {len(resampled)} rows at {timeframe}")
            
        return resampled
    
    @staticmethod
    def merge_files(file1: str, file2: str, output_file: str) -> pd.DataFrame:
        """Merge two data files, removing duplicates"""
        df1 = pd.read_csv(file1, index_col='Time', parse_dates=True)
        df2 = pd.read_csv(file2, index_col='Time', parse_dates=True)
        
        # Merge
        merged = pd.concat([df1, df2])
        
        # Remove duplicates, keeping first occurrence
        merged = merged[~merged.index.duplicated(keep='first')]
        
        # Sort by time
        merged = merged.sort_index()
        
        # Save
        merged.to_csv(output_file)
        
        print(f"Merged {len(df1)} + {len(df2)} = {len(merged)} unique rows")
        return merged
    
    @staticmethod
    def clean_data(input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Clean data by removing anomalies and fixing common issues"""
        df = pd.read_csv(input_file, index_col='Time', parse_dates=True)
        original_len = len(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df = df.sort_index()
        
        # Remove rows with any missing values
        df = df.dropna()
        
        # Fix OHLC consistency
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Ensure High is the maximum
            df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
            # Ensure Low is the minimum
            df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Remove extreme price movements (likely errors)
        if 'Close' in df.columns:
            pct_change = df['Close'].pct_change().abs()
            # Remove rows where price changed more than 20% (likely data error)
            df = df[pct_change <= 0.2]
        
        # Remove zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        if output_file:
            df.to_csv(output_file)
            
        print(f"Cleaned data: {original_len} -> {len(df)} rows ({original_len - len(df)} removed)")
        return df
    
    @staticmethod
    def create_summary_report(data_dir: str, output_file: str = 'data_summary.txt'):
        """Create a summary report of all data files in directory"""
        report_lines = []
        report_lines.append("FX Data Summary Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now()}")
        report_lines.append("")
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'MASTER' in f]
        csv_files.sort()
        
        for file in csv_files:
            filepath = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(filepath, index_col='Time', parse_dates=True)
                
                report_lines.append(f"\n{file}")
                report_lines.append("-" * len(file))
                report_lines.append(f"Rows: {len(df):,}")
                report_lines.append(f"Date range: {df.index.min()} to {df.index.max()}")
                report_lines.append(f"Columns: {', '.join(df.columns)}")
                
                if 'Close' in df.columns:
                    report_lines.append(f"Price range: {df['Close'].min():.5f} - {df['Close'].max():.5f}")
                    
            except Exception as e:
                report_lines.append(f"\n{file}: Error reading file - {e}")
        
        # Write report
        report_path = os.path.join(data_dir, output_file)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        print(f"Summary report written to {report_path}")
        return report_lines


def main():
    parser = argparse.ArgumentParser(description='FX Data Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter data by year')
    filter_parser.add_argument('input_file', help='Input CSV file')
    filter_parser.add_argument('year', type=int, help='Year to filter from')
    filter_parser.add_argument('-o', '--output', help='Output file (optional)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data file')
    validate_parser.add_argument('file', help='File to validate')
    
    # Resample command
    resample_parser = subparsers.add_parser('resample', help='Resample data')
    resample_parser.add_argument('input_file', help='Input CSV file')
    resample_parser.add_argument('timeframe', help='Target timeframe (e.g., 1H, 4H, 1D)')
    resample_parser.add_argument('-o', '--output', help='Output file (optional)')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge two data files')
    merge_parser.add_argument('file1', help='First file')
    merge_parser.add_argument('file2', help='Second file')
    merge_parser.add_argument('output', help='Output file')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean data file')
    clean_parser.add_argument('input_file', help='Input CSV file')
    clean_parser.add_argument('-o', '--output', help='Output file (optional)')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Create summary report')
    summary_parser.add_argument('data_dir', help='Data directory')
    summary_parser.add_argument('-o', '--output', default='data_summary.txt',
                               help='Output report file')
    
    args = parser.parse_args()
    
    utils = DataUtilities()
    
    if args.command == 'filter':
        utils.filter_by_year(args.input_file, args.year, args.output)
        
    elif args.command == 'validate':
        validation = utils.validate_data(args.file)
        for key, value in validation.items():
            print(f"{key}: {value}")
            
    elif args.command == 'resample':
        utils.resample_data(args.input_file, args.timeframe, args.output)
        
    elif args.command == 'merge':
        utils.merge_files(args.file1, args.file2, args.output)
        
    elif args.command == 'clean':
        utils.clean_data(args.input_file, args.output)
        
    elif args.command == 'summary':
        utils.create_summary_report(args.data_dir, args.output)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()