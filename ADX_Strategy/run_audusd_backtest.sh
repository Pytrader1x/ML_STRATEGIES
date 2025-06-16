#!/bin/bash

# Run backtest on AUDUSD data
echo "Running ADX Strategy on AUDUSD 15M data..."

# Using the main.py with proper data path
python main.py backtest \
    --data ../data/AUDUSD_MASTER_15M.csv \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --cash 10000 \
    --no-plot

# Uncomment to run optimization
# echo -e "\n\nRunning parameter optimization..."
# python main.py optimize \
#     --data ../data/AUDUSD_MASTER_15M.csv \
#     --start 2015-01-01 \
#     --end 2020-12-31