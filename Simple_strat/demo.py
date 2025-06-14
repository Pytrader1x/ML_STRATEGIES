
# Import our custom indicators
from technical_indicators_custom.tic import TIC
from technical_indicators_custom.plotting import IndicatorPlotter
import pandas as pd
from pathlib import Path
from technical_indicators_custom.plotting import plot_neurotrend_market_bias_chop
import time
# Add all indicators

# Load the AUDUSD data
data_path = Path("../data/AUDUSD_MASTER_15M.csv")
df = pd.read_csv(data_path)

# Convert DateTime column to datetime and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

df = df.iloc[-5000:]  # Use last 5000 rows for the analysis

# Add indicators
print("Calculating indicators...")

# Neuro Trend Intelligent
print("  Calculating Neuro Trend Intelligent...")
start_time = time.time()
df = TIC.add_neuro_trend_intelligent(df)
elapsed_time = time.time() - start_time
print(f"  ✓ Completed Neuro Trend Intelligent in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")

# Market Bias
print("  Calculating Market Bias...")
start_time = time.time()
df = TIC.add_market_bias(df)
elapsed_time = time.time() - start_time
print(f"  ✓ Completed Market Bias in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")

# Intelligent Chop
print("  Calculating Intelligent Chop...")
start_time = time.time()
df = TIC.add_intelligent_chop(df)
elapsed_time = time.time() - start_time
print(f"  ✓ Completed Intelligent Chop in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")

print(f"Data ready: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")




# TIC.plot(df_combined.tail(500), title="NeuroTrend Intelligent Analysis")

# # Create the combined plot
fig = plot_neurotrend_market_bias_chop(
    df.tail(500),  # Use the last 500 rows for the example
    title="Combined Technical Analysis",
    figsize=(16, 12),
    save_path='combined_analysis.png',
    show=True, use_chop_background=True,show_chop_subplots=False,simplified_regime_colors=True,
)