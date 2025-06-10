#!/usr/bin/env python3
"""
Filter 1M data to only include records from 2024 onwards for git storage
"""

import pandas as pd
import os
from datetime import datetime

# Currency pairs
pairs = [
    'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 
    'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 
    'NZDUSD', 'USDCAD'
]

# Start date for filtering
start_date = '2024-01-01'

print(f"Filtering 1M data to include only records from {start_date} onwards...")
print("-" * 60)

for pair in pairs:
    input_file = f"{pair}_MASTER.csv"
    output_file = f"git_data/{pair}_MASTER_2024.csv"
    
    if not os.path.exists(input_file):
        print(f"✗ {input_file} not found, skipping...")
        continue
    
    print(f"Processing {pair}...")
    
    try:
        # Read the full data
        df = pd.read_csv(input_file)
        
        # Convert DateTime column to datetime type
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Filter data from 2024 onwards
        df_filtered = df[df['DateTime'] >= start_date].copy()
        
        # Save filtered data
        df_filtered.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(df_filtered):,} records to {output_file}")
        print(f"    Date range: {df_filtered['DateTime'].min()} to {df_filtered['DateTime'].max()}")
        
    except Exception as e:
        print(f"  ✗ Error processing {pair}: {str(e)}")

print("-" * 60)
print("Filtering complete!")