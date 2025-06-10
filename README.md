# ML_Strategies

Machine Learning Trading Strategies Repository

## Overview

This repository contains various machine learning approaches for trading strategies, including:
- Dueling DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- XGBoost and Random Forest models

## Data

The `data/` directory contains historical FX data for multiple currency pairs:
- 1-minute interval data (raw)
- 15-minute interval data (resampled)

### Available Currency Pairs
- AUD/JPY, AUD/NZD, AUD/USD
- CAD/JPY, CHF/JPY
- EUR/GBP, EUR/JPY, EUR/USD
- GBP/JPY, GBP/USD
- NZD/USD, USD/CAD

### Data Download

To download or update FX data:
```bash
cd data
python download_fx_data.py
```

## Project Structure

```
ML_Strategies/
├── Dueling_DQN/       # Dueling Deep Q-Network implementation
├── PPO/               # Proximal Policy Optimization implementation
├── XG_boost_RForest/  # XGBoost and Random Forest models
└── data/              # Historical FX data and download scripts
```

## Requirements

- Python 3.x
- fx_data_downloader
- pandas
- numpy
- (additional requirements to be added based on specific implementations)

## Getting Started

1. Clone the repository
2. Install required dependencies
3. Download historical data using the provided script
4. Explore the different ML strategy implementations

## License

[To be determined]