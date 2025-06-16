"""Test script to understand indicator behavior"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tic import TIC

# Load data
print("Loading data...")
df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv", parse_dates=['DateTime'], index_col='DateTime')
print(f"Original data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Test with a smaller sample first
sample_size = 5000
test_df = df.iloc[:sample_size].copy()
print(f"\nTesting with {sample_size} rows...")

# Add indicators one by one to see which one causes the issue
print("\n1. Adding Intelligent Chop...")
test_df1 = test_df.copy()
test_df1 = TIC.add_intelligent_chop(test_df1, inplace=True)
print(f"   After IC: {len(test_df1)} rows (dropped {sample_size - len(test_df1)})")

print("\n2. Adding Market Bias...")
test_df2 = test_df1.copy()
test_df2 = TIC.add_market_bias(test_df2, inplace=True)
print(f"   After MB: {len(test_df2)} rows (dropped {len(test_df1) - len(test_df2)})")

print("\n3. Adding NeuroTrend Intelligent...")
test_df3 = test_df2.copy()
test_df3 = TIC.add_neuro_trend_intelligent(test_df3, inplace=True)
print(f"   After NTI: {len(test_df3)} rows (dropped {len(test_df2) - len(test_df3)})")

print("\n4. Adding SuperTrend...")
test_df4 = test_df3.copy()
test_df4 = TIC.add_super_trend(test_df4, inplace=True)
print(f"   After ST: {len(test_df4)} rows (dropped {len(test_df3) - len(test_df4)})")

print("\n5. Adding Fractal SR...")
test_df5 = test_df4.copy()
test_df5 = TIC.add_fractal_sr(test_df5, inplace=True)
print(f"   After SR: {len(test_df5)} rows (dropped {len(test_df4) - len(test_df5)})")

# Check for NaN values
print("\n\nChecking NaN values in final dataframe:")
nan_counts = test_df5.isna().sum()
print(f"Total columns: {len(test_df5.columns)}")
print(f"Columns with NaN: {(nan_counts > 0).sum()}")
print("\nTop columns with most NaN values:")
print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))

# Try without dropping NaN to see what we get
print("\n\nTrying without dropna()...")
test_nodrop = df.iloc[:sample_size].copy()
test_nodrop = TIC.add_intelligent_chop(test_nodrop, inplace=True)
test_nodrop = TIC.add_market_bias(test_nodrop, inplace=True)
test_nodrop = TIC.add_neuro_trend_intelligent(test_nodrop, inplace=True)
test_nodrop = TIC.add_super_trend(test_nodrop, inplace=True)
test_nodrop = TIC.add_fractal_sr(test_nodrop, inplace=True)

print(f"Shape without dropna: {test_nodrop.shape}")
print(f"Rows with any NaN: {test_nodrop.isna().any(axis=1).sum()}")
print(f"Rows with all valid data: {(~test_nodrop.isna().any(axis=1)).sum()}")

# Find the minimum rows needed
print("\n\nFinding minimum data requirement...")
for size in [1000, 2000, 3000, 5000, 10000, 20000]:
    if size > len(df):
        break
    test = df.iloc[:size].copy()
    test = TIC.add_intelligent_chop(test, inplace=True)
    test = TIC.add_market_bias(test, inplace=True)
    test = TIC.add_neuro_trend_intelligent(test, inplace=True)
    test = TIC.add_super_trend(test, inplace=True)
    test = TIC.add_fractal_sr(test, inplace=True)
    valid_rows = (~test.isna().any(axis=1)).sum()
    print(f"Size {size}: {valid_rows} valid rows ({valid_rows/size*100:.1f}%)")