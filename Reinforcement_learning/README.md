# Reinforcement Learning Trading Agent for AUDUSD

This directory contains an advanced reinforcement learning trading agent specifically designed for AUDUSD currency pair trading. The agent uses state-of-the-art deep reinforcement learning techniques optimized for Apple Silicon (M3 Pro) hardware.

## Overview

The trading agent (`rl_audusd_advanced_trading_pytorch_v2.py`) implements a **Dueling Double Deep Q-Network (Dueling DQN)** with prioritized experience replay to learn optimal trading strategies from historical AUDUSD price data.

### Key Features

- **Apple MPS Optimization**: Fully optimized for M3 Pro with Metal Performance Shaders (MPS) acceleration
- **Simplified Action Space**: Clean 3-action system (Hold, Buy, Sell) for more stable learning
- **USD-Based P&L**: Fixed 1M AUDUSD position sizing with USD profit/loss tracking
- **Advanced Technical Indicators**: NeuroTrend, Market Bias, and Intelligent Chop indicators
- **Real-time Trade Visualization**: Episode plotting shows entry/exit points during training
- **Efficient State Management**: JIT-compiled state building with zero CPU transfers

## Architecture

### Neural Network: Dueling DQN

The agent uses a Dueling DQN architecture that separates value and advantage streams:

```
Input (State) → Hidden Layers → Value Stream    → Q-values
                              → Advantage Stream →
```

- **State Space**: Window of 50 bars × 7 features + position information
- **Action Space**: 3 discrete actions (0=Hold, 1=Buy, 2=Sell)
- **Hidden Layers**: 512 → 256 → 128 with ReLU activation and dropout

### Memory: Prioritized Experience Replay

- Stores experiences with TD-error based priorities
- Samples important experiences more frequently
- Capacity: 50,000 transitions

## Performance Optimizations

### MPS (Metal Performance Shaders) Optimization

Achieved ~2x speedup through:

1. **Complete Tensor Pipeline**: All data stays on MPS device
   - Pre-loaded feature tensors
   - No CPU↔GPU transfers during training
   - Vectorized operations throughout

2. **JIT Compilation**: Critical functions compiled with `@torch.jit.script`
   - State building
   - Tensor operations

3. **Efficient Batching**: Optimized batch size (512) for MPS architecture

### Training Efficiency

- **Async Episode Plotting**: Non-blocking chart generation
- **Matplotlib Agg Backend**: Faster non-interactive rendering
- **ThreadPoolExecutor**: Parallel file I/O operations

## Technical Indicators

The agent uses custom technical indicators from the parent directory:

1. **NeuroTrend**: Advanced trend detection (fast=10, slow=50)
2. **Market Bias**: Multi-timeframe bias indicator (len1=350, len2=30)
3. **Intelligent Chop**: Market choppiness detection

## Trading Logic

### Position Management
- **Initial Balance**: USD 1,000,000
- **Position Size**: Fixed 1M AUDUSD units per trade
- **Max Positions**: 1 (single position at a time)
- **Risk Management**: Stop-loss and take-profit based on market conditions

### Action Execution
- **Hold (0)**: Maintain current position or stay flat
- **Buy (1)**: Open long position if no position exists
- **Sell (2)**: Open short position if no position exists

### Exit Tracking
The agent tracks three exit types:
- Stop-loss exits
- Take-profit exits
- Agent-initiated exits

## Training Process

### Hyperparameters
- **Episodes**: 200
- **Learning Rate**: 0.0001
- **Gamma**: 0.995 (high for long-term rewards)
- **Epsilon**: 1.0 → 0.01 (decay: 0.99995 per step)
- **Target Network Update**: Every 100 steps

### Window Sampling
- Random episode windows: 10,000 - 30,000 bars
- Ensures diverse market conditions
- Prevents overfitting to specific periods

### Metrics
- **Sharpe Ratio**: Calculated from equity curve (risk-adjusted returns)
- **Total Return**: Final balance / initial balance
- **Win Rate**: Percentage of profitable trades
- **Exit Type Distribution**: SL%, TP%, Agent% exits

## Usage

### Training
```bash
python rl_audusd_advanced_trading_pytorch_v2.py
```

### Outputs
1. **Model Checkpoints**: Saved every 10 episodes
   - `dqn_audusd_v2_episode_*.pth`
   
2. **Episode Plots**: Trade visualization
   - `plots/episode_*.png`
   - Shows price with entry (green ^) and exit (red v) markers
   
3. **Training Metrics**: Console output with episode performance

## File Structure

```
Reinforcement_learning/
├── rl_audusd_advanced_trading_pytorch_v2.py  # Main agent implementation
├── models/                                     # Saved model checkpoints
├── plots/                                      # Episode trade visualizations
├── v2_mps_optimization_summary.md             # MPS optimization details
├── v2_final_improvements.md                   # Epsilon & Sharpe improvements
├── v2_simplified_actions_usd.md              # Action space simplification
├── v2_episode_plotting.md                     # Plotting implementation
└── README.md                                  # This file
```

## Recent Improvements

### Version 2 Enhancements
1. **MPS Optimization** (2x speedup)
   - Eliminated CPU transfers
   - JIT compilation
   - Optimized tensor operations

2. **Simplified Action Space**
   - Reduced from 5 to 3 actions
   - Cleaner Buy/Sell/Hold logic
   - More stable learning

3. **USD-Based P&L**
   - Fixed 1M lot sizing
   - Clear USD profit tracking
   - Realistic trading simulation

4. **Enhanced Metrics**
   - Proper Sharpe ratio from equity curve
   - Exit type tracking
   - Per-episode trade visualization

5. **Slower Epsilon Decay**
   - Changed from 0.9995 to 0.99995
   - More exploration during training
   - Better convergence

## Future Enhancements

- Multi-currency pair support
- Dynamic position sizing
- Advanced risk management features
- Real-time trading integration
- Ensemble models with multiple agents

## Requirements

- Python 3.8+
- PyTorch with MPS support
- NumPy, Pandas, Matplotlib
- Custom technical indicators module (parent directory)

## Notes

- Optimized specifically for Apple M3 Pro hardware
- Uses non-interactive matplotlib backend for speed
- Thread-safe implementation for async operations
- Reproducible results with fixed random seeds (42)