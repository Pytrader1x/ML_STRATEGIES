"""Debug data types issue"""
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tic import TIC

# Load a small sample
df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv", parse_dates=['DateTime'], index_col='DateTime')
sample = df.iloc[:1000].copy()

# Add indicators
sample = TIC.add_intelligent_chop(sample, inplace=True)
sample = TIC.add_market_bias(sample, inplace=True)
sample = TIC.add_neuro_trend_intelligent(sample, inplace=True)
sample = TIC.add_super_trend(sample, inplace=True)
sample = TIC.add_fractal_sr(sample, inplace=True)

# Check data types
print("Data types for indicator columns:")
print("-" * 50)

# Check IC columns
ic_cols = [col for col in sample.columns if col.startswith('IC_')]
for col in ic_cols:
    print(f"{col}: {sample[col].dtype}")
    # Check for string values
    if sample[col].dtype == 'object':
        print(f"  Sample values: {sample[col].dropna().head()}")
        print(f"  Unique values: {sample[col].unique()[:10]}")

print("\n" + "-" * 50)

# Check NTI columns  
nti_cols = [col for col in sample.columns if col.startswith('NTI_')]
for col in nti_cols:
    print(f"{col}: {sample[col].dtype}")
    if sample[col].dtype == 'object':
        print(f"  Sample values: {sample[col].dropna().head()}")
        print(f"  Unique values: {sample[col].unique()[:10]}")

# Save a small sample for inspection
sample.head(20).to_csv('debug_sample.csv')
print("\nSaved debug_sample.csv for inspection")