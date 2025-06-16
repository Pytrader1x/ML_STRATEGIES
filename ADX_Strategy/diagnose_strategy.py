#!/usr/bin/env python3
"""
Diagnostic script to understand why the ADX strategy isn't trading.
"""

import pandas as pd
import backtrader as bt
import backtrader.indicators as btind
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DiagnosticADXStrategy(bt.Strategy):
    """Modified strategy to output diagnostic information."""
    
    params = (
        ('adx_period', 14),
        ('adx_threshold', 50),
        ('williams_period', 14),
        ('williams_oversold', -80),
        ('williams_overbought', -20),
        ('sma_period', 50),
        ('tp_lookback', 30),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        
        # Indicators
        self.dmi = btind.DirectionalMovementIndex(
            self.datas[0], 
            period=self.params.adx_period
        )
        self.adx = self.dmi.adx
        self.plus_di = self.dmi.plusDI
        self.minus_di = self.dmi.minusDI
        
        self.williams = btind.WilliamsR(
            self.datas[0],
            period=self.params.williams_period
        )
        
        self.sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=self.params.sma_period
        )
        
        # Track statistics (renamed to avoid conflict with backtrader's stats)
        self.diag_stats = {
            'bars_processed': 0,
            'adx_above_threshold': 0,
            'long_signals': 0,
            'short_signals': 0,
            'adx_values': [],
            'williams_values': []
        }
        
    def next(self):
        self.diag_stats['bars_processed'] += 1
        
        # Record indicator values
        if self.adx[0] > 0:  # Valid ADX value
            self.diag_stats['adx_values'].append(self.adx[0])
            self.diag_stats['williams_values'].append(self.williams[0])
            
            if self.adx[0] > self.params.adx_threshold:
                self.diag_stats['adx_above_threshold'] += 1
            
            # Check long conditions
            if (self.plus_di[0] > self.minus_di[0] and 
                self.adx[0] > self.params.adx_threshold and 
                self.williams[0] < self.params.williams_oversold):
                self.diag_stats['long_signals'] += 1
                
            # Check short conditions
            if (self.minus_di[0] > self.plus_di[0] and 
                self.adx[0] > self.params.adx_threshold and 
                self.williams[0] > self.params.williams_overbought):
                self.diag_stats['short_signals'] += 1
                
    def stop(self):
        """Print diagnostic information."""
        print("\n=== DIAGNOSTIC REPORT ===")
        print(f"Total bars processed: {self.diag_stats['bars_processed']}")
        print(f"Bars with ADX > {self.params.adx_threshold}: {self.diag_stats['adx_above_threshold']} ({self.diag_stats['adx_above_threshold']/self.diag_stats['bars_processed']*100:.1f}%)")
        print(f"Long signals generated: {self.diag_stats['long_signals']}")
        print(f"Short signals generated: {self.diag_stats['short_signals']}")
        
        if self.diag_stats['adx_values']:
            import numpy as np
            adx_array = np.array(self.diag_stats['adx_values'])
            williams_array = np.array(self.diag_stats['williams_values'])
            
            print(f"\nADX Statistics:")
            print(f"  Mean: {np.mean(adx_array):.2f}")
            print(f"  Max: {np.max(adx_array):.2f}")
            print(f"  Min: {np.min(adx_array):.2f}")
            print(f"  Std: {np.std(adx_array):.2f}")
            
            print(f"\nWilliams %R Statistics:")
            print(f"  Mean: {np.mean(williams_array):.2f}")
            print(f"  Max: {np.max(williams_array):.2f}")
            print(f"  Min: {np.min(williams_array):.2f}")
            
            # Count how often Williams is in extreme zones
            williams_oversold = np.sum(williams_array < self.params.williams_oversold)
            williams_overbought = np.sum(williams_array > self.params.williams_overbought)
            
            print(f"\nWilliams %R Extremes:")
            print(f"  Oversold (< {self.params.williams_oversold}): {williams_oversold} times ({williams_oversold/len(williams_array)*100:.1f}%)")
            print(f"  Overbought (> {self.params.williams_overbought}): {williams_overbought} times ({williams_overbought/len(williams_array)*100:.1f}%)")


def run_diagnostics(data_path, start_date='2020-01-01', end_date='2023-12-31'):
    """Run diagnostic analysis on the strategy."""
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DiagnosticADXStrategy)
    
    # Load data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Handle datetime
        for col in ['DateTime', 'Date', 'datetime', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        
        # Capitalize columns
        df.columns = [col.capitalize() for col in df.columns]
        
        # Add Volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Filter date range
        df = df[(df.index >= pd.to_datetime(start_date)) & 
                (df.index <= pd.to_datetime(end_date))]
        
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        
        print(f"Data loaded: {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
    cerebro.run()
    
    # Also test with different ADX thresholds
    print("\n\n=== TESTING DIFFERENT ADX THRESHOLDS ===")
    
    for threshold in [20, 30, 40, 50]:
        cerebro2 = bt.Cerebro()
        cerebro2.addstrategy(DiagnosticADXStrategy, adx_threshold=threshold)
        cerebro2.adddata(data)
        
        print(f"\nWith ADX threshold = {threshold}:")
        results = cerebro2.run()


if __name__ == '__main__':
    # Test with 1H data
    print("Running diagnostics on AUDUSD 1H data...")
    run_diagnostics('../data/AUDUSD_MASTER_1H.csv')
    
    # Also test with more recent data only
    print("\n\n" + "="*60)
    print("Testing with 2023 data only...")
    run_diagnostics('../data/AUDUSD_MASTER_1H.csv', start_date='2023-01-01', end_date='2023-12-31')