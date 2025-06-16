"""
Advanced Reinforcement Learning Trading Agent for AUDUSD (PyTorch Version 2.0)
Major improvements for consistent profitability
"""

# %% Load Libraries
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque, namedtuple
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("TensorBoard available - logging enabled")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from technical_indicators_custom import TIC

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# For reproducibility only - no threading optimizations

# Check device availability - prioritize MPS for Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

# JIT-compiled state builder for performance
@torch.jit.script
def build_state(window: torch.Tensor, pos_info: torch.Tensor) -> torch.Tensor:
    """JIT-compiled state builder to reduce Python overhead"""
    return torch.cat([window.flatten(), pos_info], dim=0)

# %% Configuration
class Config:
    """Enhanced configuration with adaptive parameters"""
    # Data
    CURRENCY_PAIR = 'AUDUSD'
    TRAIN_TEST_SPLIT = 0.8
    
    # Indicators
    NEUROTREND_FAST = 10
    NEUROTREND_SLOW = 50
    MARKET_BIAS_LEN1 = 350
    MARKET_BIAS_LEN2 = 30
    
    # RL Parameters - IMPROVED
    WINDOW_SIZE = 50  # Larger window for better context
    ACTION_SIZE = 4   # Extended action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
    NUM_ENVS = 2      # Number of parallel environments (optimized for MPS)
    
    # Model - ENHANCED
    HIDDEN_LAYER_1 = 512
    HIDDEN_LAYER_2 = 256
    HIDDEN_LAYER_3 = 128
    LEARNING_RATE = 0.0001  # Lower learning rate
    
    # Training - OPTIMIZED
    EPISODES = 200
    BATCH_SIZE = 512  # Optimal batch size for MPS
    MEMORY_SIZE = 50000
    UPDATE_TARGET_EVERY = 500  # P2: Slower target network sync
    
    # Window sampling
    MIN_WINDOW_SIZE = 10000  # Larger windows
    MAX_WINDOW_SIZE = 30000
    
    # RL Hyperparameters - TUNED
    GAMMA = 0.995  # Higher for long-term
    EPSILON = 0.9   # Start at 0.9
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99  # Per-episode decay (moved to end of episode)
    MIN_HOLDING_BARS = 0  # No minimum holding period with Close action
    COOLDOWN_BARS = 0  # No cooldown with Close action 
    
    # N-step returns
    N_STEPS = 3  # Number of steps for n-step returns (reduced for faster startup)
    
    # PER hyperparameters - IMPROVED
    PER_ALPHA = 0.75  # Increased from 0.6 for stronger prioritization
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_STEPS = 20000  # Faster beta schedule
    PER_WARMUP_STEPS = 5000  # Delay PER activation
    
    # Trading Parameters - USD based with 1M lots
    INITIAL_BALANCE = 1_000_000  # Start with USD 1M
    POSITION_SIZE = 1_000_000    # Always trade 1M AUDUSD units
    MAX_POSITIONS = 1            # One position at a time
    
    # Risk Management - ADAPTIVE
    BASE_SL_ATR_MULT = 2.0  # Stop loss = 2 * ATR
    BASE_TP_ATR_MULT = 3.0  # Take profit = 3 * ATR
    MIN_RR_RATIO = 1.5      # Minimum risk/reward ratio
    
    # Reward shaping
    # Note: Drawdown penalty is now quadratic: -0.02 * (drawdown^2)
    
    # C51 Distributional RL parameters
    C51_ATOMS = 51  # Number of atoms for value distribution
    C51_VMIN = -10.0  # Minimum value support
    C51_VMAX = 10.0  # Maximum value support
    USE_C51 = True  # Enable C51 distributional RL
    
    # NoisyNet parameters
    USE_NOISY_NET = True  # Enable NoisyNet for exploration
    NOISY_STD = 0.5  # Initial noise standard deviation
    
    # LSTM parameters for temporal features
    USE_LSTM = True  # Enable LSTM after CNN
    LSTM_HIDDEN_SIZE = 64  # LSTM hidden state size
    LSTM_NUM_LAYERS = 2  # Number of LSTM layers
    
    # Performance optimization
    USE_TORCH_COMPILE = True  # Enable torch.compile for faster execution (auto-disabled on MPS)
    USE_AMP = False  # Enable Automatic Mixed Precision (AMP) - disabled on MPS

# %% Enhanced Data Loader
class DataLoader:
    """Enhanced data loader with better feature engineering"""
    
    def __init__(self, currency_pair: str):
        self.currency_pair = currency_pair
        self.df = None
        self.feature_cols = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        data_path = 'data' if os.path.exists('data') else '../data'
        file_path = os.path.join(data_path, f'{self.currency_pair}_MASTER_15M.csv')
        
        print(f"Loading {self.currency_pair} data...")
        self.df = pd.read_csv(file_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # Calculate returns and volatility
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        print(f"Loaded {len(self.df)} rows of data")
        return self.df
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        print("\nAdding technical indicators...")
        
        # Price action features
        self.df['High_Low_Pct'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['Close_Open_Pct'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        
        # Volatility
        self.df['ATR'] = self.calculate_atr(14)
        self.df['ATR_Pct'] = self.df['ATR'] / self.df['Close']
        
        # Moving averages
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(200).mean()
        
        # Price relative to MAs
        self.df['Close_to_SMA20'] = (self.df['Close'] - self.df['SMA_20']) / self.df['SMA_20']
        self.df['Close_to_SMA50'] = (self.df['Close'] - self.df['SMA_50']) / self.df['SMA_50']
        
        # Momentum
        self.df['RSI'] = self.calculate_rsi(14)
        self.df['RSI_Signal'] = np.where(self.df['RSI'] > 70, -1, 
                                       np.where(self.df['RSI'] < 30, 1, 0))
        
        # Add NeuroTrend Intelligent
        print("- Adding NeuroTrend Intelligent...")
        self.df = TIC.add_neuro_trend_intelligent(
            self.df,
            base_fast=Config.NEUROTREND_FAST,
            base_slow=Config.NEUROTREND_SLOW,
            confirm_bars=3,
            dynamic_thresholds=True,
            enable_diagnostics=False
        )
        
        # Create trading signals from NeuroTrend
        self.df['NTI_Signal'] = np.where(
            (self.df['NTI_Direction'] > 0) & (self.df['NTI_Confidence'] > 0.7), 1,
            np.where((self.df['NTI_Direction'] < 0) & (self.df['NTI_Confidence'] > 0.7), -1, 0)
        )
        
        # Add Market Bias
        print("- Adding Market Bias...")
        self.df = TIC.add_market_bias(
            self.df,
            ha_len=Config.MARKET_BIAS_LEN1,
            ha_len2=Config.MARKET_BIAS_LEN2
        )
        
        # Add Intelligent Chop
        print("- Adding Intelligent Chop...")
        self.df = TIC.add_intelligent_chop(self.df)
        
        # Market regime from Intelligent Chop
        self.df['Trending'] = np.where(self.df['IC_Regime'].isin([1, -1]), 1, 0)
        
        # Composite signal
        self.df['Composite_Signal'] = (
            self.df['NTI_Signal'] * 0.4 +
            np.sign(self.df['MB_Bias']) * 0.3 +
            self.df['RSI_Signal'] * 0.3
        )
        
        # Drop NaN values
        self.df.dropna(inplace=True)
        
        print(f"Indicators added. Final shape: {self.df.shape}")
        return self.df
    
    def calculate_atr(self, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_rsi(self, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, train_df: Optional[pd.DataFrame] = None) -> List[str]:
        """Select and normalize features for the model
        
        Args:
            train_df: Training dataframe to compute normalization stats from.
                     If provided, these stats will be stored and used for all normalization.
        """
        # Core features
        self.feature_cols = [
            # Price action
            'Close', 'Returns', 'High_Low_Pct', 'Close_Open_Pct',
            'ATR_Pct', 'Close_to_SMA20', 'Close_to_SMA50',
            
            # Indicators
            'RSI', 'NTI_Direction', 'NTI_Confidence', 'NTI_SlopePower',
            'NTI_ReversalRisk', 'MB_Bias', 'IC_Regime', 'IC_Confidence',
            'IC_Signal', 'Trending', 'Composite_Signal'
        ]
        
        # Normalize features
        print("\nNormalizing features...")
        
        # If train_df provided, compute and store normalization stats
        if train_df is not None:
            self.norm_stats = {}
            for col in self.feature_cols:
                if col in train_df.columns:
                    self.norm_stats[col] = {
                        'mean': train_df[col].mean(),
                        'std': train_df[col].std() + 1e-8
                    }
            print("Computed normalization stats from training data")
        
        # Apply normalization using stored stats (no look-ahead bias)
        for col in self.feature_cols:
            if col in self.df.columns and hasattr(self, 'norm_stats') and col in self.norm_stats:
                # Use train-only statistics
                self.df[f'{col}_norm'] = (
                    (self.df[col] - self.norm_stats[col]['mean']) / 
                    self.norm_stats[col]['std']
                )
            elif col in self.df.columns:
                # Fallback - compute from training set if norm_stats not available
                if train_df is not None:
                    mean_val = train_df[col].mean()
                    std_val = train_df[col].std() + 1e-8
                else:
                    # Last resort - use full dataset stats
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std() + 1e-8
                
                self.df[f'{col}_norm'] = (
                    (self.df[col] - mean_val) / std_val
                ).fillna(0)
        
        return [f'{col}_norm' for col in self.feature_cols if col in self.df.columns]
    

# %% Enhanced DQN Model
# NoisyLinear layer for exploration
class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (Fortunato et al. 2017)"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingC51DQN_CNN(nn.Module):
    """Dueling C51 DQN with 1D CNN for distributional RL"""
    
    def __init__(self, state_size: int, action_size: int, num_features: int = 18, window_size: int = 50,
                 n_atoms: int = Config.C51_ATOMS, v_min: float = Config.C51_VMIN, v_max: float = Config.C51_VMAX):
        super(DuelingC51DQN_CNN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Support for categorical distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        
        # 1D CNN for temporal feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal features if enabled
        if Config.USE_LSTM:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=Config.LSTM_HIDDEN_SIZE,
                num_layers=Config.LSTM_NUM_LAYERS,
                batch_first=True,
                dropout=0.1 if Config.LSTM_NUM_LAYERS > 1 else 0
            )
            # Calculate flattened size after convolutions and LSTM
            conv_out_size = window_size // 2  # After one MaxPool1d(2)
            flat_size = Config.LSTM_HIDDEN_SIZE * conv_out_size + 9  # 9 position features
        else:
            self.lstm = None
            # Calculate flattened size after convolutions
            conv_out_size = window_size // 2  # After one MaxPool1d(2)
            flat_size = 128 * conv_out_size + 9  # 9 position features
        
        # Value stream - outputs distribution over atoms
        self.value = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_atoms)
        )
        
        # Advantage stream - outputs distribution over atoms for each action
        self.advantage = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_size * n_atoms)
        )
        
    def forward(self, x, return_dist=False):
        # Split window and position features
        batch_size = x.shape[0]
        window_features = x[:, :-9].view(batch_size, self.num_features, self.window_size)
        pos_features = x[:, -9:]
        
        # Apply CNN to window features
        conv_out = self.conv(window_features)
        
        # Apply LSTM if enabled
        if self.lstm is not None:
            # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
            conv_out = conv_out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(conv_out)
            # Use the LSTM output: (batch, seq_len, hidden_size)
            lstm_flat = lstm_out.flatten(1)
            combined = torch.cat([lstm_flat, pos_features], dim=1)
        else:
            conv_flat = conv_out.flatten(1)
            combined = torch.cat([conv_flat, pos_features], dim=1)
        
        # Compute value and advantage distributions
        value_dist = self.value(combined).view(batch_size, 1, self.n_atoms)
        advantage_dist = self.advantage(combined).view(batch_size, self.action_size, self.n_atoms)
        
        # Combine streams using dueling architecture
        q_dist = value_dist + (advantage_dist - advantage_dist.mean(dim=1, keepdim=True))
        
        # Apply softmax to get probabilities
        q_probs = F.softmax(q_dist, dim=-1)
        
        if return_dist:
            return q_probs
        
        # Return expected Q-values for action selection
        q_values = (q_probs * self.support.view(1, 1, -1)).sum(dim=-1)
        return q_values


class DuelingDQN_CNN(nn.Module):
    """Original Dueling DQN with 1D CNN for feature extraction (non-distributional)"""
    
    def __init__(self, state_size: int, action_size: int, num_features: int = 18, window_size: int = 50):
        super(DuelingDQN_CNN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # 1D CNN for temporal feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal features if enabled
        if Config.USE_LSTM:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=Config.LSTM_HIDDEN_SIZE,
                num_layers=Config.LSTM_NUM_LAYERS,
                batch_first=True,
                dropout=0.1 if Config.LSTM_NUM_LAYERS > 1 else 0
            )
            # Calculate flattened size after convolutions and LSTM
            conv_out_size = window_size // 2  # After one MaxPool1d(2)
            flat_size = Config.LSTM_HIDDEN_SIZE * conv_out_size + 9  # 9 position features
        else:
            self.lstm = None
            # Calculate flattened size after convolutions
            conv_out_size = window_size // 2  # After one MaxPool1d(2)
            flat_size = 128 * conv_out_size + 9  # 9 position features
        
        # Value stream
        if Config.USE_NOISY_NET:
            self.value = nn.Sequential(
                NoisyLinear(flat_size, 128, Config.NOISY_STD),
                nn.ReLU(),
                nn.Dropout(0.2),
                NoisyLinear(128, 1, Config.NOISY_STD)
            )
            
            # Advantage stream
            self.advantage = nn.Sequential(
                NoisyLinear(flat_size, 128, Config.NOISY_STD),
                nn.ReLU(),
                nn.Dropout(0.2),
                NoisyLinear(128, action_size, Config.NOISY_STD)
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(flat_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            
            # Advantage stream
            self.advantage = nn.Sequential(
                nn.Linear(flat_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, action_size)
            )
    
    def forward(self, x):
        # Split window and position features
        batch_size = x.shape[0]
        window_features = x[:, :-9].view(batch_size, self.num_features, self.window_size)
        pos_features = x[:, -9:]
        
        # Apply CNN to window features
        conv_out = self.conv(window_features)
        
        # Apply LSTM if enabled
        if self.lstm is not None:
            # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
            conv_out = conv_out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(conv_out)
            # Use the LSTM output: (batch, seq_len, hidden_size)
            lstm_flat = lstm_out.flatten(1)
            combined = torch.cat([lstm_flat, pos_features], dim=1)
        else:
            conv_flat = conv_out.flatten(1)
            combined = torch.cat([conv_flat, pos_features], dim=1)
        
        # Compute value and advantage
        value = self.value(combined)
        advantage = self.advantage(combined)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        if Config.USE_NOISY_NET:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class DuelingDQN(nn.Module):
    """Original Dueling DQN (fallback option)"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, Config.HIDDEN_LAYER_1)
        self.bn1 = nn.BatchNorm1d(Config.HIDDEN_LAYER_1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(Config.HIDDEN_LAYER_1, Config.HIDDEN_LAYER_2)
        self.bn2 = nn.BatchNorm1d(Config.HIDDEN_LAYER_2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Value stream
        if Config.USE_NOISY_NET:
            self.value_fc = NoisyLinear(Config.HIDDEN_LAYER_2, Config.HIDDEN_LAYER_3, Config.NOISY_STD)
            self.value_out = NoisyLinear(Config.HIDDEN_LAYER_3, 1, Config.NOISY_STD)
            
            # Advantage stream
            self.advantage_fc = NoisyLinear(Config.HIDDEN_LAYER_2, Config.HIDDEN_LAYER_3, Config.NOISY_STD)
            self.advantage_out = NoisyLinear(Config.HIDDEN_LAYER_3, action_size, Config.NOISY_STD)
        else:
            self.value_fc = nn.Linear(Config.HIDDEN_LAYER_2, Config.HIDDEN_LAYER_3)
            self.value_out = nn.Linear(Config.HIDDEN_LAYER_3, 1)
            
            # Advantage stream
            self.advantage_fc = nn.Linear(Config.HIDDEN_LAYER_2, Config.HIDDEN_LAYER_3)
            self.advantage_out = nn.Linear(Config.HIDDEN_LAYER_3, action_size)
        
        # Noisy linear layers for exploration
        self.noisy_std = 0.1
        
    def forward(self, x):
        # Shared network
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        if Config.USE_NOISY_NET:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

# %% Enhanced Trading Environment
class TradingEnvironment:
    """Improved trading environment with better position management"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df
        self.feature_cols = feature_cols
        self.equity_curve = []  # Track equity for proper Sharpe calculation
        self.state_size = len(feature_cols) * Config.WINDOW_SIZE + 9  # +9 for enhanced position info
        # Pre-convert feature data to tensor for fast access
        self.feature_tensor = torch.from_numpy(df[feature_cols].values).float().to(device)
        
        # Cache frequently accessed columns as numpy arrays for speed
        self.close_prices = df['Close'].values
        self.atr_values = df['ATR'].values
        self.composite_signals = df['Composite_Signal'].values
        self.trending_values = df['Trending'].values
        self.nti_confidence = df['NTI_Confidence'].values if 'NTI_Confidence' in df else None
        
        # Pre-allocate position info tensor for reuse
        self.pos_info_buffer = torch.zeros(9, device=device, dtype=torch.float32)
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.balance = Config.INITIAL_BALANCE
        self.initial_balance = Config.INITIAL_BALANCE
        self.position = None  # Single position tracking
        self.trade_history = []
        self.current_step = 0
        self.max_drawdown = 0
        self.peak_balance = Config.INITIAL_BALANCE
        self.equity_curve = [Config.INITIAL_BALANCE]  # Initialize equity curve
        self.last_exit_bar = -Config.COOLDOWN_BARS  # Track last exit for cooldown
        self.atr_entry = None  # Store ATR at position entry
        
    def get_state(self, index: int, window_size: int) -> torch.Tensor:
        """Get enhanced state representation - returns torch.Tensor on device"""
        if index < window_size:
            return torch.zeros(self.state_size, device=device)
        
        # Get windowed features directly from MPS tensor - no CPU transfer!
        window = self.feature_tensor[index-window_size+1:index+1]  # [W, F] on MPS
        
        # Use cached numpy array instead of pandas iloc
        current_atr = self.atr_values[index] if index < len(self.atr_values) else 0.0
        atr_ratio = (current_atr / self.atr_entry) if self.atr_entry and self.atr_entry > 0 else 1.0
        
        # Reuse pre-allocated buffer instead of creating new tensor
        self.pos_info_buffer[0] = 1.0 if self.position else 0.0
        self.pos_info_buffer[1] = self.position['unrealized_pnl'] / self.initial_balance if self.position else 0.0
        self.pos_info_buffer[2] = self.position['holding_time'] / 100.0 if self.position else 0.0
        self.pos_info_buffer[3] = (self.balance / self.initial_balance - 1.0)
        self.pos_info_buffer[4] = self.max_drawdown
        self.pos_info_buffer[5] = len(self.trade_history) / 100.0
        self.pos_info_buffer[6] = self.get_win_rate()
        self.pos_info_buffer[7] = self.atr_entry if self.atr_entry else current_atr
        self.pos_info_buffer[8] = atr_ratio
        
        pos_info = self.pos_info_buffer
        
        # Use JIT-compiled state builder
        state = build_state(window, pos_info)
        return state
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        if not self.trade_history:
            return 0.5
        wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
        return wins / len(self.trade_history)
    
    def calculate_adaptive_sl_tp(self, entry_price: float, direction: int, 
                                conservative: bool = False) -> Tuple[float, float]:
        """Calculate adaptive SL/TP based on ATR and market conditions"""
        current_atr = self.atr_values[self.current_step]
        
        # Adjust multipliers based on market regime
        if self.trending_values[self.current_step] == 1:
            # Trending market - wider stops, bigger targets
            sl_mult = Config.BASE_SL_ATR_MULT * 1.2
            tp_mult = Config.BASE_TP_ATR_MULT * 1.5
        else:
            # Ranging market - tighter stops
            sl_mult = Config.BASE_SL_ATR_MULT * 0.8
            tp_mult = Config.BASE_TP_ATR_MULT * 0.8
        
        if conservative:
            sl_mult *= 0.7
            tp_mult *= 0.7
        
        # P2: Minimum SL distance (≥ 0.0005 or 5 pips)
        sl_distance = max(current_atr * sl_mult, 0.0005)
        tp_distance = current_atr * tp_mult
        
        if direction == 1:  # Long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def execute_action(self, action: int, index: int) -> Tuple[float, Dict]:
        """Execute trading action with NAV-Δ (Mark-to-Market) reward system
        
        Reward Philosophy:
        - Reward = scaled change in net asset value (NAV)
        - NAV = cash balance + unrealized P&L
        - No junk rewards, no entry bonuses, no asymmetric scaling
        - Pure alignment with actual trading profitability
        """
        self.current_step = index
        current_price = self.close_prices[index]
        info = {'action': action, 'price': current_price}
        
        # Calculate NAV before action (B_{t-1} + U_{t-1})
        nav_before = self.balance
        if self.position:
            # Add unrealized P&L before update
            nav_before += self.position['unrealized_pnl']
        
        # Update existing position's unrealized P&L
        if self.position:
            self.position['unrealized_pnl'] = (
                (current_price - self.position['entry_price']) * 
                self.position['direction'] * self.position['size']
            )
            self.position['holding_time'] += 1
            
            # Check stop loss and take profit
            if self.position['direction'] == 1:  # Long
                if current_price <= self.position['stop_loss']:
                    self._close_position(current_price, 'stop_loss')
                    info['exit_type'] = 'stop_loss'
                elif current_price >= self.position['take_profit']:
                    self._close_position(current_price, 'take_profit')
                    info['exit_type'] = 'take_profit'
            else:  # Short
                if current_price >= self.position['stop_loss']:
                    self._close_position(current_price, 'stop_loss')
                    info['exit_type'] = 'stop_loss'
                elif current_price <= self.position['take_profit']:
                    self._close_position(current_price, 'take_profit')
                    info['exit_type'] = 'take_profit'
        
        # Execute new actions with strict one-trade-at-a-time enforcement
        if action == 0:  # Hold
            # No action taken
            pass
            
        elif action == 3:  # Close position
            if self.position:
                self._close_position(current_price, 'early_close')
                info['exit_type'] = 'early_close'
            # Ignore close action if no position
            
        elif action == 1 and not self.position:  # Buy - only if no position
            # Open long position
                    stop_loss, take_profit = self.calculate_adaptive_sl_tp(
                        current_price, 1, False
                    )
                    
                    self.position = {
                        'entry_price': current_price,
                        'size': Config.POSITION_SIZE,
                        'direction': 1,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_index': index,
                        'unrealized_pnl': 0,
                        'holding_time': 0,
                        'entry_signal_strength': self.composite_signals[index]
                    }
                    
                    # Store ATR at entry
                    self.atr_entry = self.atr_values[index]
                    
                    info['position_opened'] = 'long'
            # Ignore buy action if already have position (enforcing one-trade-at-a-time)
                    
        elif action == 2 and not self.position:  # Sell - only if no position
            # Open short position
                    stop_loss, take_profit = self.calculate_adaptive_sl_tp(
                        current_price, -1, False
                    )
                    
                    self.position = {
                        'entry_price': current_price,
                        'size': Config.POSITION_SIZE,
                        'direction': -1,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_index': index,
                        'unrealized_pnl': 0,
                        'holding_time': 0,
                        'entry_signal_strength': self.composite_signals[index]
                    }
                    
                    # Store ATR at entry
                    self.atr_entry = self.atr_values[index]
                    
                    info['position_opened'] = 'short'
            # Ignore sell action if already have position (enforcing one-trade-at-a-time)
        
        # Update metrics
        self._update_metrics()
        
        # Calculate NAV after action (B_t + U_t)
        nav_after = self.balance
        if self.position:
            # Add current unrealized P&L
            nav_after += self.position['unrealized_pnl']
        
        # Track equity curve at each step (using NAV)
        self.equity_curve.append(nav_after)
        
        # Risk-scaled reward normalization
        nav_delta = nav_after - nav_before
        
        # Get current ATR for scaling - use cached array
        current_atr = self.atr_values[index] if index < len(self.atr_values) else 0.0
        atr_scale = max(current_atr, 1e-5)
        
        # Scale reward by ATR and position size for normalization
        # Option: Sharper reward scaling with larger multiplier
        reward = 200 * nav_delta / (atr_scale * Config.POSITION_SIZE)
        
        # Add reward clipping/smoothing
        reward = float(torch.tanh(torch.tensor(reward)))
        
        # Enhanced reward shaping
        # 1. Boost early-close bonus significantly
        if info.get('exit_type') == 'early_close':
            reward += 1.0  # Increased from 0.2 to 1.0 to make Close action more attractive
            
        # 2. Remove over-trading penalty (double-counting with transaction cost)
        # Removed: if info.get('exit_type'): reward -= 0.01
            
        # 3. Amplified holding fee on losers
        if self.position and self.position['unrealized_pnl'] < 0:
            reward -= 0.02  # Increased from 0.005 to discourage holding losers
        
        # 4. Reward positive holding on winners
        if self.position and self.position['unrealized_pnl'] > 0:
            reward += 0.005  # Small bonus per bar to encourage letting winners run
        
        # 5. Quadratic drawdown penalty curve - hits harder on large drawdowns
        current_drawdown = self.max_drawdown
        reward -= 0.02 * (current_drawdown ** 2)  # Changed from linear to quadratic
        
        # 6. Small time-in-market cost (regardless of P&L)
        if self.position:
            reward -= 0.001  # Tiny per-bar penalty to avoid unnecessary churn
        
        return reward, info
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series for Sharpe calculation"""
        return pd.Series(self.equity_curve)
    
    def _check_entry_conditions(self, direction: int) -> bool:
        """Check if entry conditions are met"""
        # Get current signals - use cached arrays
        composite_signal = self.composite_signals[self.current_step]
        nti_confidence = self.nti_confidence[self.current_step] if self.nti_confidence is not None else 0.0
        trending = self.trending_values[self.current_step]
        
        if direction == 1:  # Long
            return (composite_signal > 0.3 and nti_confidence > 0.6) or \
                   (composite_signal > 0.5 and trending == 1)
        else:  # Short
            return (composite_signal < -0.3 and nti_confidence > 0.6) or \
                   (composite_signal < -0.5 and trending == 1)
    
    def _calculate_position_size(self) -> float:
        """Return fixed position size - always 1M units"""
        return Config.POSITION_SIZE
    
    def _close_position(self, exit_price: float, exit_type: str) -> None:
        """Close position and update balance
        
        Note: Rewards are now calculated in execute_action() using NAV-Δ.
        This method only handles position closure and balance updates.
        """
        if not self.position:
            return
        
        # Calculate P&L
        if self.position['direction'] == 1:  # Long
            pnl = (exit_price - self.position['entry_price']) * self.position['size']
        else:  # Short
            pnl = (self.position['entry_price'] - exit_price) * self.position['size']
        
        # P0-1: Real transaction cost - $20 round trip (0.2 pip per 1M AUDUSD)
        # This is the FULL round-trip cost, applied only at exit
        transaction_cost = 20.0  # $20 per 1M units round trip (entry + exit spreads)
        net_pnl = pnl - transaction_cost
        
        # Update balance with net P&L after costs
        self.balance += net_pnl
        
        # Record trade with enhanced metrics including costs
        self.trade_history.append({
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'direction': self.position['direction'],
            'pnl': net_pnl,  # Net P&L after costs
            'pnl_usd': net_pnl,  # Net in USD
            'gross_pnl_usd': pnl,  # P0-1: Gross P&L before costs
            'transaction_cost': transaction_cost,  # P0-1: Cost tracking
            'pnl_pct': net_pnl / self.initial_balance,
            'holding_time': self.position['holding_time'],
            'exit_type': exit_type,
            'tp_exit': exit_type == 'take_profit',
            'sl_exit': exit_type == 'stop_loss',
            'manual_exit': exit_type == 'manual',
            'early_exit': exit_type == 'early_close',  # Track early close exits
            'drawdown': self.max_drawdown,
            'entry_index': self.position['entry_index']
        })
        
        self.position = None
        self.atr_entry = None  # Clear ATR at entry
        self.last_exit_bar = self.current_step  # Record exit time for cooldown
    
    def _update_metrics(self):
        """Update performance metrics"""
        # Update max drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

# %% Enhanced Trading Agent
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

NStepExperience = namedtuple('NStepExperience',
    ['state', 'action', 'n_reward', 'n_state', 'done', 'actual_n'])

class SumTree:
    """Binary sum tree for O(log n) prioritized sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity  # Use list instead of numpy array to avoid type issues
        self.write = 0
        self.n_entries = 0
        
        # Initialize leaf nodes with small non-zero values to avoid NaN
        # Leaf nodes start at index capacity-1
        self.tree[capacity-1:] = 1e-5
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority"""
        total = self.tree[0]
        # Check for NaN and reset if necessary
        if np.isnan(total):
            print("WARNING: Total priority is NaN, resetting tree")
            # Reset all priorities to small value
            self.tree[:] = 1e-5
            # Recalculate internal nodes
            for i in range(self.capacity - 1, 2 * self.capacity - 1):
                self._propagate(i, 0)
            total = self.tree[0]
        return total
    
    def add(self, priority: float, data: Experience):
        """Add new experience with priority"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of existing node"""
        # Ensure priority is never zero to avoid NaN
        priority = max(priority, 1e-5)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """Get experience by priority sum"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        # Ensure dataIdx is within bounds and handle circular buffer wrap
        if 0 <= dataIdx < self.capacity:
            # Check if this position has been written to
            if dataIdx < self.n_entries or (self.n_entries == self.capacity and self.data[dataIdx] is not None):
                return idx, self.tree[idx], self.data[dataIdx]
        
        # Return None experience if out of bounds or not yet written
        return idx, 0.0, None


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer - simplified implementation"""
    
    def __init__(self, capacity: int, alpha: float = Config.PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        self.beta = Config.PER_BETA_START
        self.beta_increment_per_sampling = 0.001
        self.total_steps = 0  # Track total steps for warmup
        
    def push(self, experience: Experience):
        """Add experience with maximum priority"""
        self.total_steps += 1
        # Use lower priority during warmup period
        if self.total_steps < Config.PER_WARMUP_STEPS:
            priority = 1.0  # Uniform priority during warmup
        else:
            priority = self.max_priority ** self.alpha
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with priority-based probabilities"""
        n = len(self.buffer)
        
        # Ensure we have enough experiences to sample
        if n < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {n} < {batch_size}")
        
        # Convert to numpy for efficient sampling
        priorities = np.array(self.priorities, dtype=np.float32)
        
        # Ensure no negative or zero priorities
        priorities = np.maximum(priorities, 1e-7)
        
        # Calculate probabilities
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Ensure probabilities are valid (no NaN or inf)
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            # Fallback to uniform sampling
            probabilities = np.ones(n, dtype=np.float32) / n
        
        # Sample indices
        indices = np.random.choice(n, batch_size, p=probabilities)
        
        experiences = []
        sampled_priorities = []
        
        for idx in indices:
            experiences.append(self.buffer[idx])
            sampled_priorities.append(priorities[idx])
        
        # Calculate importance sampling weights
        sampling_probabilities = (priorities[indices] / priorities.sum()) ** self.alpha
        is_weight = (n * sampling_probabilities) ** (-beta)
        is_weight /= is_weight.max()
        
        return experiences, indices, torch.FloatTensor(is_weight).to(device)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            priority = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        """Return buffer length"""
        return len(self.buffer)

class TradingAgent:
    """Enhanced RL Trading Agent with advanced features"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(Config.MEMORY_SIZE)
        
        # N-step buffer for storing transitions
        self.n_step_buffer = deque(maxlen=Config.N_STEPS)
        
        # RL parameters
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.beta = Config.PER_BETA_START  # For importance sampling
        self.beta_increment = (Config.PER_BETA_END - Config.PER_BETA_START) / Config.PER_BETA_STEPS
        
        # Neural networks - use CNN version
        self.use_cnn = True  # Flag to enable CNN
        self.use_c51 = Config.USE_C51  # Flag for C51 distributional RL
        
        if self.use_cnn:
            num_features = len([col for col in ['Close', 'Returns', 'High_Low_Pct', 'Close_Open_Pct',
                                               'ATR_Pct', 'Close_to_SMA20', 'Close_to_SMA50',
                                               'RSI', 'NTI_Direction', 'NTI_Confidence', 'NTI_SlopePower',
                                               'NTI_ReversalRisk', 'MB_Bias', 'IC_Regime', 'IC_Confidence',
                                               'IC_Signal', 'Trending', 'Composite_Signal'] if True])  # 18 features
            if self.use_c51:
                self.q_network = DuelingC51DQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
                self.target_network = DuelingC51DQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
            else:
                self.q_network = DuelingDQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
                self.target_network = DuelingDQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
        else:
            self.q_network = DuelingDQN(state_size, action_size).to(device)
            self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.update_target_model()
        
        # Compile models for performance if enabled (skip on MPS due to compatibility issues)
        if Config.USE_TORCH_COMPILE and hasattr(torch, 'compile') and device.type != 'mps':
            try:
                self.q_network = torch.compile(self.q_network, mode='reduce-overhead')
                self.target_network = torch.compile(self.target_network, mode='reduce-overhead')
                print("Models compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}, using regular models")
        elif device.type == 'mps':
            print("Skipping torch.compile on MPS device due to compatibility issues")
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # AMP scaler for mixed precision training (only on CUDA)
        self.scaler = torch.amp.GradScaler('cuda') if Config.USE_AMP and device.type == 'cuda' else None
        
        self.update_counter = 0
        
        # Per-step epsilon decay parameters
        # Calculate total steps for 7 episodes (approx 10k-15k steps per episode)
        # Episode 0: 20k steps, Episodes 1-6: ~10k * 1.2^(episode/10) each
        estimated_steps_per_episode = [20000] + [int(10000 * (1.2 ** (ep/10))) for ep in range(1, 7)]
        self.total_steps_7_episodes = sum(estimated_steps_per_episode)
        self.epsilon_start = Config.EPSILON
        self.epsilon_end = Config.EPSILON_MIN
        self.epsilon_decay_rate = -math.log(self.epsilon_end / self.epsilon_start) / self.total_steps_7_episodes
        self.global_step = 0
        print(f"Per-step epsilon decay initialized: {self.epsilon_start:.3f} → {self.epsilon_end:.3f} over ~{self.total_steps_7_episodes:,} steps")
        
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in n-step buffer and compute n-step returns"""
        # Ensure states are tensors on device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        
        # For early training, push both 1-step and n-step experiences
        # This ensures we have enough data to start training quickly
        if len(self.memory) < Config.BATCH_SIZE * 2:
            # Push immediate experience for bootstrapping
            immediate_exp = Experience(state, action, reward, next_state, done)
            self.memory.push(immediate_exp)
        
        # Add to n-step buffer
        self.n_step_buffer.append(Experience(state, action, reward, next_state, done))
        
        # Only process if we have enough steps or episode is done
        if len(self.n_step_buffer) >= Config.N_STEPS or done:
            self._process_n_step_buffer()
    
    def act(self, state: torch.Tensor, signal: float = 0.0, has_position: bool = False) -> int:
        """Choose action using epsilon-greedy policy with strict one-trade-at-a-time masking
        
        Action masking for one trade at a time:
        - If has position: can only Hold (0) or Close (3)
        - If no position: can only Hold (0), Buy (1), or Sell (2)
        - Additional signal-based filtering when opening positions
        """
        # Epsilon-greedy with masked actions (reduced if using NoisyNet)
        effective_epsilon = self.epsilon * 0.1 if Config.USE_NOISY_NET else self.epsilon
        if random.random() <= effective_epsilon:
            if has_position:
                # Can only hold or close when we have a position
                valid_actions = [0, 3]
            else:
                # Can only hold or open new position when flat
                valid_actions = [0, 1, 2]
                
                # Apply signal-based filtering for new positions
                if signal > 0.35 and 2 in valid_actions:
                    valid_actions.remove(2)  # Remove Sell when strongly bullish
                elif signal < -0.35 and 1 in valid_actions:
                    valid_actions.remove(1)  # Remove Buy when strongly bearish
            
            return random.choice(valid_actions)
        
        # Get Q-values - state is already a tensor on device!
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        self.q_network.eval()
        with torch.no_grad():
            # Use autocast for inference if AMP is enabled
            if Config.USE_AMP and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    q_values = self.q_network(state)
            else:
                q_values = self.q_network(state)
        self.q_network.train()
        
        # Add exploration bonus
        q_values_np = q_values.cpu().numpy()[0]
        exploration_bonus = np.random.normal(0, 0.01, self.action_size)
        q_values_np += exploration_bonus
        
        # Apply strict one-trade-at-a-time masking
        if has_position:
            # Can only hold or close
            q_values_np[1] = -1e9  # Mask Buy
            q_values_np[2] = -1e9  # Mask Sell
        else:
            # Can only hold or open new
            q_values_np[3] = -1e9  # Mask Close
            
            # Additional signal-based filtering for new positions
            if signal > 0.35:
                q_values_np[2] = -1e9  # Mask Sell when strongly bullish
            elif signal < -0.35:
                q_values_np[1] = -1e9  # Mask Buy when strongly bearish
        
        return np.argmax(q_values_np)
    
    def replay(self, batch_size: int):
        """Train model on batch of experiences with prioritized replay"""
        # Ensure we have enough experiences (considering n-step delay)
        min_required = max(batch_size, Config.N_STEPS * 10)
        if len(self.memory) < min_required:
            return
        
        # Sample from memory
        experiences, indices, weights = self.memory.sample(batch_size, self.beta)
        
        # Prepare batch - states are already tensors on device!
        states = torch.stack([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences]).to(device)
        
        # Use AMP autocast for forward passes if enabled
        if Config.USE_AMP and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                if self.use_c51:
                    # C51 Distributional loss
                    loss, priorities = self._compute_c51_loss(states, actions, rewards, next_states, dones, weights)
                else:
                    # Standard DQN loss
                    # Current Q-values
                    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                    
                    # Next Q-values using Double DQN
                    with torch.no_grad():
                        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
                        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                    
                    # Calculate loss with importance sampling
                    td_errors = target_q_values.unsqueeze(1) - current_q_values
                    loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
                    
                    # Update priorities
                    priorities = td_errors.detach().abs().cpu().numpy().squeeze()
        else:
            if self.use_c51:
                # C51 Distributional loss
                loss, priorities = self._compute_c51_loss(states, actions, rewards, next_states, dones, weights)
            else:
                # Standard DQN loss
                # Current Q-values
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # Next Q-values using Double DQN
                with torch.no_grad():
                    next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                    next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Calculate loss with importance sampling
                td_errors = target_q_values.unsqueeze(1) - current_q_values
                loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
                
                # Update priorities
                priorities = td_errors.detach().abs().cpu().numpy().squeeze()
        
        self.memory.update_priorities(indices, priorities)
        
        # Optimize model with AMP if enabled
        self.optimizer.zero_grad()
        
        if self.scaler is not None:  # Using AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % Config.UPDATE_TARGET_EVERY == 0:
            self.update_target_model()
        
        # Per-step exponential epsilon decay
        self.global_step += 1
        self.epsilon = max(self.epsilon_end, self.epsilon_start * math.exp(-self.epsilon_decay_rate * self.global_step))
        
        # Reset noise in networks if using NoisyNet
        if Config.USE_NOISY_NET and hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return loss.item()
    
    def _process_n_step_buffer(self):
        """Process n-step buffer to compute n-step returns"""
        if not self.n_step_buffer:
            return
            
        # Get the first transition
        first_exp = self.n_step_buffer[0]
        
        # Calculate n-step return
        n_step_reward = 0
        gamma_power = 1
        
        for i, exp in enumerate(self.n_step_buffer):
            n_step_reward += gamma_power * exp.reward
            gamma_power *= self.gamma
            
            if exp.done:
                # Episode ended, use this as final state
                # Convert to regular experience for compatibility
                final_exp = Experience(
                    first_exp.state, first_exp.action, n_step_reward,
                    exp.next_state, True
                )
                self.memory.push(final_exp)
                
                # Clear the buffer since episode ended
                self.n_step_buffer.clear()
                return
        
        # If buffer is full (n steps), create n-step experience
        if len(self.n_step_buffer) == Config.N_STEPS:
            last_exp = self.n_step_buffer[-1]
            # Convert to regular experience
            final_exp = Experience(
                first_exp.state, first_exp.action, n_step_reward,
                last_exp.next_state, False
            )
            self.memory.push(final_exp)
            
            # Remove only the first element to maintain sliding window
            self.n_step_buffer.popleft()
    
    def _compute_c51_loss(self, states, actions, rewards, next_states, dones, weights):
        """Compute C51 distributional loss"""
        batch_size = states.size(0)
        
        # Get current action distributions
        current_dist = self.q_network(states, return_dist=True)
        current_action_dist = current_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, self.q_network.n_atoms)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN action selection
            next_q_values = self.q_network(next_states, return_dist=False)
            next_actions = next_q_values.argmax(1)
            
            # Get target distributions
            target_dist = self.target_network(next_states, return_dist=True)
            target_action_dist = target_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, self.target_network.n_atoms)).squeeze(1)
            
            # Compute projected distribution
            support = self.q_network.support
            delta_z = self.q_network.delta_z
            v_min = self.q_network.v_min
            v_max = self.q_network.v_max
            
            # Compute Tz (projected support)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            gamma = self.gamma ** self.n_step_buffer.maxlen if hasattr(self, 'n_step_buffer') else self.gamma
            
            tz = rewards + (1 - dones) * gamma * support.unsqueeze(0)
            tz = tz.clamp(min=v_min, max=v_max)
            
            # Compute projection indices
            b = (tz - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability mass
            m = torch.zeros_like(target_action_dist)
            offset = torch.linspace(0, (batch_size - 1) * self.q_network.n_atoms, batch_size,
                                  dtype=torch.long, device=device).unsqueeze(1).expand(batch_size, self.q_network.n_atoms)
            
            # Lower projection
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                 (target_action_dist * (u.float() - b)).view(-1))
            # Upper projection
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                 (target_action_dist * (b - l.float())).view(-1))
        
        # Cross-entropy loss with importance weights
        loss = -(m * current_action_dist.clamp(min=1e-8).log()).sum(1)
        weighted_loss = (weights * loss).mean()
        
        # Compute priorities as KL divergence
        with torch.no_grad():
            kl_div = (m * (m.clamp(min=1e-8).log() - current_action_dist.clamp(min=1e-8).log())).sum(1)
            priorities = kl_div.cpu().numpy()
        
        return weighted_loss, priorities

# %% Training Functions
def train_agent(agent: TradingAgent, env: TradingEnvironment, df_train: pd.DataFrame,
                window_size: int, episodes: int, fast_mode: bool = False,
                writer=None, activations=None):
    """Enhanced training loop with better monitoring and proper Sharpe calculation"""
    
    episode_rewards = []
    episode_profits = []
    episode_sharpes = []
    best_sharpe = -float('inf')
    no_improvement_count = 0
    
    # Setup plotting directory
    os.makedirs('plots', exist_ok=True)
    
    # Store plotting data for deferred processing
    plotting_data = []
    
    for episode in range(episodes):
        # Reset environment
        env.reset()
        
        # Sequential first-epoch training for coherent experience
        if episode == 0:
            # First episode: use chronological data but cap at 20k for faster training
            start_idx = window_size
            end_idx = min(len(df_train) - 1, start_idx + 20000)  # Reduced from 50k
            window_length = end_idx - start_idx
        else:
            # Subsequent episodes: sample random windows
            window_length = min(
                int(Config.MIN_WINDOW_SIZE * (1.2 ** (episode / 10))),
                Config.MAX_WINDOW_SIZE,
                len(df_train) - window_size - 1
            )
            
            if fast_mode:
                window_length = min(window_length, 5000)
            
            start_idx = random.randint(window_size, len(df_train) - window_length - 1)
            end_idx = start_idx + window_length
        
        # Get initial state - now returns torch.Tensor
        state = env.get_state(start_idx, window_size)
        total_reward = 0
        
        # Track position history for visualization
        position_history = []
        
        # Track step rewards for distribution logging
        step_rewards = []
        
        # Cache frequently accessed data
        composite_signals_train = df_train['Composite_Signal'].values
        
        # Training loop with ETA tracking
        start_time_episode = time.time()
        pbar = tqdm(range(start_idx, end_idx), 
                   desc=f"Episode {episode+1}/{episodes}",
                   mininterval=1.0)  # Update progress bar only once per second
        
        # Local buffer and training frequency settings
        local_buffer = []
        COLLECT_STEPS = 256  # Reduced from 512 for more frequent training
        TRAIN_ITERATIONS = 16  # Increased from 8 for more gradient steps
        
        for t in pbar:
            # Get composite signal for action masking - use cached array
            composite_signal = composite_signals_train[t]
            
            # Agent takes action with signal-based masking
            action = agent.act(state, composite_signal, has_position=bool(env.position))
            
            # Execute action
            reward, _ = env.execute_action(action, t)
            total_reward += reward
            step_rewards.append(reward)
            
            # Track position for visualization
            if env.position:
                position_history.append({
                    'bar': t - start_idx,
                    'size': env.position['size'] * env.position['direction'],
                    'direction': env.position['direction']
                })
            else:
                position_history.append({
                    'bar': t - start_idx,
                    'size': 0,
                    'direction': 0
                })
            
            # Get next state
            next_state = env.get_state(t + 1, window_size)
            done = (t == end_idx - 1)
            
            # Store experience in the LOCAL buffer first, not the main agent memory
            experience = Experience(state, action, reward, next_state, done)
            local_buffer.append(experience)
            state = next_state
            
            # Check if it's time to train
            if len(local_buffer) >= COLLECT_STEPS:
                # 1. Push all collected experiences to the main replay buffer
                for exp in local_buffer:
                    agent.remember(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
                
                # 2. Clear the local buffer
                local_buffer = []
                
                # 3. Run a concentrated training phase
                # Ensure we have enough experiences before training (considering n-step delay)
                min_required = max(Config.BATCH_SIZE, Config.N_STEPS * 10)
                if len(agent.memory) > min_required:
                    pbar.set_description(f"Episode {episode+1}/{episodes} [Training...]")
                    for _ in range(TRAIN_ITERATIONS):
                        agent.replay(Config.BATCH_SIZE)
                    pbar.set_description(f"Episode {episode+1}/{episodes}")
            
            # Update progress bar less frequently
            if (t - start_idx) % 250 == 0 and t > start_idx:
                elapsed = time.time() - start_time_episode
                avg_per_bar = elapsed / (t - start_idx + 1)
                remaining_bars = end_idx - 1 - t
                eta_seconds = remaining_bars * avg_per_bar
                
                # Use equity curve for live P&L
                live_profit = env.equity_curve[-1] - env.initial_balance if env.equity_curve else 0
                
                pbar.set_postfix({
                    'Profit': f"${live_profit:,.2f}",
                    'Trades': len(env.trade_history),
                    'ε': f"{agent.epsilon:.3f}",
                    'ETA': f"{eta_seconds/60:.1f}m"
                })
                pbar.refresh()  # Force refresh
        
        # Push any remaining experiences in local buffer to main memory
        if local_buffer:
            for exp in local_buffer:
                agent.remember(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
            local_buffer = []
        
        # Force final postfix update with actual end-of-episode values
        final_profit = env.equity_curve[-1] - env.initial_balance if env.equity_curve else env.balance - env.initial_balance
        pbar.set_postfix({
            'Profit': f"${final_profit:,.2f}",
            'Trades': len(env.trade_history),
            'ε': f"{agent.epsilon:.3f}",
            'ETA': "0.0m"
        })
        pbar.refresh()  # Force final refresh
        pbar.close()
        
        # Close any open position before computing summary
        if env.position:
            final_price = df_train.iloc[end_idx - 1]['Close']
            env._close_position(final_price, 'manual')
        
        # Episode summary with USD profits
        final_profit_usd = env.balance - env.initial_balance
        final_profit_pct = (env.balance / env.initial_balance - 1) * 100
        trades = env.trade_history
        
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades) * 100
            
            # Exit type statistics - including early close
            tp_exits = sum(1 for t in trades if t.get('tp_exit', False))
            sl_exits = sum(1 for t in trades if t.get('sl_exit', False))
            manual_exits = sum(1 for t in trades if t.get('manual_exit', False))
            early_exits = sum(1 for t in trades if t.get('early_exit', False))
            
            # Exit type percentages
            pct_tp = (tp_exits / len(trades)) * 100 if trades else 0
            pct_sl = (sl_exits / len(trades)) * 100 if trades else 0
            pct_manual = (manual_exits / len(trades)) * 100 if trades else 0
            pct_early = (early_exits / len(trades)) * 100 if trades else 0
            
            # Average drawdown
            avg_drawdown = np.mean([t.get('drawdown', 0) for t in trades]) * 100
            
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if len(trades) > len(wins) else 0
        else:
            win_rate = 0
            tp_exits = sl_exits = manual_exits = early_exits = 0
            pct_tp = pct_sl = pct_manual = pct_early = 0
            avg_drawdown = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate proper Sharpe ratio from equity curve
        if len(env.equity_curve) > 1:
            equity_series = pd.Series(env.equity_curve)
            # Calculate per-bar returns
            bar_returns = equity_series.pct_change().dropna()
            if len(bar_returns) > 0 and bar_returns.std() > 0:
                # Annualize: 96 bars per day (15-minute bars)
                sharpe = bar_returns.mean() / bar_returns.std() * np.sqrt(252 * 96)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        print(f"\nEpisode {episode+1} Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Profit: ${final_profit_usd:,.0f} ({final_profit_pct:.2f}%)")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Total Trades: {len(trades)}")
        
        if trades:
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
            total_exit_pct = pct_tp + pct_sl + pct_manual + pct_early
            print(f"  Exit Types: TP={pct_tp:.1f}%, SL={pct_sl:.1f}%, Manual={pct_manual:.1f}%, Early={pct_early:.1f}% (Total: {total_exit_pct:.1f}%)")
            print(f"  Avg Drawdown: {avg_drawdown:.1f}%")
        
        episode_rewards.append(total_reward)
        episode_profits.append(final_profit_pct)  # Keep percentage for compatibility
        episode_sharpes.append(sharpe)
        
        # Learning rate scheduling based on Sharpe
        agent.scheduler.step(sharpe)
        
        # Early stopping based on Sharpe
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            no_improvement_count = 0
            # Save best model
            torch.save(agent.q_network.state_dict(), 'best_model.pth')
        else:
            no_improvement_count += 1
        
        # Epsilon is now decayed per-step in the replay() method
        # Calculate expected epsilon for verification
        expected_eps_at_ep1 = agent.epsilon_start * math.exp(-agent.epsilon_decay_rate * agent.total_steps_7_episodes / 7)
        expected_eps_at_ep3 = agent.epsilon_start * math.exp(-agent.epsilon_decay_rate * agent.total_steps_7_episodes * 3 / 7)
        expected_eps_at_ep7 = agent.epsilon_start * math.exp(-agent.epsilon_decay_rate * agent.total_steps_7_episodes)
        
        print(f"  Current Epsilon: {agent.epsilon:.4f} (Global Step: {agent.global_step:,})")
        if episode == 0:
            print(f"  Expected ε trajectory: Episode 1 end: ~{expected_eps_at_ep1:.3f}, Episode 3 end: ~{expected_eps_at_ep3:.3f}, Episode 7 end: ~{expected_eps_at_ep7:.3f}")
        
        # Anneal PER beta: linear from 0.4 to 1.0 based on total steps
        total_steps = (episode + 1) * window_length
        agent.beta = min(1.0, 0.4 + 0.8 * total_steps / 80000)
        
        # TensorBoard logging
        if writer is not None:
            ep_num = episode + 1
            
            # Log scalar metrics
            writer.add_scalar('Profit/Final_Profit_USD', final_profit_usd, ep_num)
            writer.add_scalar('Profit/Final_Profit_Pct', final_profit_pct, ep_num)
            writer.add_scalar('Performance/Sharpe_Ratio', sharpe, ep_num)
            writer.add_scalar('Performance/Win_Rate', win_rate, ep_num)
            writer.add_scalar('Performance/Total_Trades', len(trades), ep_num)
            writer.add_scalar('Performance/Max_Drawdown', env.max_drawdown * 100, ep_num)
            writer.add_scalar('Hyperparameters/Epsilon', agent.epsilon, ep_num)
            writer.add_scalar('Hyperparameters/Beta', agent.beta, ep_num)
            writer.add_scalar('Hyperparameters/Learning_Rate', agent.optimizer.param_groups[0]['lr'], ep_num)
            
            # Log step reward distribution
            if step_rewards:
                writer.add_histogram('Rewards/Step_Rewards_Distribution', np.array(step_rewards), ep_num)
                writer.add_scalar('Rewards/Mean_Step_Reward', np.mean(step_rewards), ep_num)
                writer.add_scalar('Rewards/Std_Step_Reward', np.std(step_rewards), ep_num)
            
            # Log exit type percentages
            if trades:
                writer.add_scalar('ExitTypes/TP_Percentage', pct_tp, ep_num)
                writer.add_scalar('ExitTypes/SL_Percentage', pct_sl, ep_num)
                writer.add_scalar('ExitTypes/Manual_Percentage', pct_manual, ep_num)
                writer.add_scalar('ExitTypes/Early_Percentage', pct_early, ep_num)
                writer.add_scalar('Trades/Avg_Win', avg_win, ep_num)
                writer.add_scalar('Trades/Avg_Loss', avg_loss, ep_num)
            
            # Log weight histograms
            for name, param in agent.q_network.named_parameters():
                if 'weight' in name:
                    writer.add_histogram(f'Weights/{name}', param, ep_num)
                if 'bias' in name:
                    writer.add_histogram(f'Biases/{name}', param, ep_num)
            
            # Log activation histograms
            if activations is not None and state is not None:
                # Do one forward pass to capture activations
                agent.q_network.eval()
                with torch.no_grad():
                    _ = agent.q_network(state.unsqueeze(0) if state.dim() == 1 else state[:1])
                agent.q_network.train()
                
                # Log captured activations
                for act_name, act_tensor in activations.items():
                    writer.add_histogram(f'Activations/{act_name}', act_tensor, ep_num)
                    # Also log percentage of dead neurons (zeros)
                    dead_neurons_pct = (act_tensor == 0).float().mean().item() * 100
                    writer.add_scalar(f'Activations/{act_name}_dead_pct', dead_neurons_pct, ep_num)
        
        # Defer heavy plotting - only create charts every 5 episodes
        if len(trades) > 0 and episode % 5 == 0:
            # Get price series for this window
            prices = df_train['Close'].iloc[start_idx:end_idx].values
            
            # Build color-coded entry/exit data
            win_entries = []
            win_exits = []
            lose_entries = []
            lose_exits = []
            
            for i, trade in enumerate(trades):
                entry_idx = trade.get('entry_index', 0) - start_idx
                exit_idx = entry_idx + trade['holding_time']
                
                # Check if trade was profitable
                win = trade['pnl_usd'] >= 0
                
                if 0 <= entry_idx < len(prices):
                    if win:
                        win_entries.append((entry_idx, prices[entry_idx], i, trade['pnl_usd'], trade['direction']))
                    else:
                        lose_entries.append((entry_idx, prices[entry_idx], i, trade['pnl_usd'], trade['direction']))
                        
                if 0 <= exit_idx < len(prices):
                    if win:
                        win_exits.append((exit_idx, prices[exit_idx], i, trade['pnl_usd'], trade['direction']))
                    else:
                        lose_exits.append((exit_idx, prices[exit_idx], i, trade['pnl_usd'], trade['direction']))
            
            # Create simple 2-row subplot for readability
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],  # Focus on price chart
                vertical_spacing=0.05,
                subplot_titles=(
                    f"Episode {episode+1} - AUDUSD Price & Trades",
                    "Cumulative P&L (USD)"
                )
            )
            
            
            # Add simple price line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(prices))),
                    y=prices,
                    mode="lines",
                    name="AUDUSD",
                    line=dict(width=1.5, color='#1f77b4'),  # Default blue
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Removed unused marker_size variable - using fixed size of 8 in markers
            
            # Add simple entry/exit markers
            # Entries (up triangles)
            entry_indices = []
            entry_prices = []
            entry_colors = []
            entry_texts = []
            
            # Exits (down triangles)  
            exit_indices = []
            exit_prices = []
            exit_colors = []
            exit_texts = []
            
            for i, trade in enumerate(trades):
                entry_idx = trade.get('entry_index', 0) - start_idx
                exit_idx = entry_idx + trade['holding_time']
                
                if 0 <= entry_idx < len(prices):
                    entry_indices.append(entry_idx)
                    entry_prices.append(prices[entry_idx])
                    entry_colors.append('green' if trade['direction'] == 1 else 'red')
                    entry_texts.append(f"{'Long' if trade['direction'] == 1 else 'Short'} Entry<br>Trade #{i+1}")
                    
                if 0 <= exit_idx < len(prices):
                    exit_indices.append(exit_idx)
                    exit_prices.append(prices[exit_idx])
                    exit_colors.append('green' if trade['direction'] == 1 else 'red')
                    exit_texts.append(f"{'Long' if trade['direction'] == 1 else 'Short'} Exit<br>Trade #{i+1}<br>P&L: ${trade['pnl_usd']:,.0f}")
            
            # Add all entries as one trace
            if entry_indices:
                fig.add_trace(
                    go.Scatter(
                        x=entry_indices,
                        y=entry_prices,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color=entry_colors,
                            line=dict(width=1, color='black')
                        ),
                        text=entry_texts,
                        hovertemplate="%{text}<br>Price: %{y:.5f}<extra></extra>",
                        name="Entries",
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add all exits as one trace
            if exit_indices:
                fig.add_trace(
                    go.Scatter(
                        x=exit_indices,
                        y=exit_prices,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color=exit_colors,
                            line=dict(width=1, color='black')
                        ),
                        text=exit_texts,
                        hovertemplate="%{text}<br>Price: %{y:.5f}<extra></extra>",
                        name="Exits",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
            
            
            # Add cumulative P&L from equity curve
            if len(env.equity_curve) > 0:
                cum_pnl = np.array(env.equity_curve) - env.equity_curve[0]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(cum_pnl))),
                        y=cum_pnl,
                        mode="lines",
                        name="Cumulative P&L",
                        line=dict(color='darkgreen' if cum_pnl[-1] > 0 else 'darkred', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0,128,0,0.1)' if cum_pnl[-1] > 0 else 'rgba(128,0,0,0.1)',
                        hovertemplate="Bar: %{x}<br>P&L: $%{y:,.0f}<extra></extra>",
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
                # Add zero line for P&L
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
            
            # Update layout for simplicity
            fig.update_layout(
                height=700,
                width=1400,
                title_text=f"Episode {episode+1}: P&L ${final_profit_usd:,.0f} ({final_profit_pct:.1f}%) | Sharpe: {sharpe:.2f} | Trades: {len(trades)} | Win Rate: {win_rate:.1f}%",
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # P0: Clamp price y-axis to actual price range
            price_margin = (prices.max() - prices.min()) * 0.02  # 2% margin
            fig.update_yaxes(
                range=[prices.min() - price_margin, prices.max() + price_margin],
                title_text="Price",
                row=1, col=1
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (bars)", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
            
            # Note: Plotly doesn't support 'layer' property for scatter traces
            # Position fill transparency already ensures markers are visible
            
            # Store figure data instead of saving immediately
            plotting_data.append((fig, f'plots/episode_{episode+1:03d}.html'))
        
        if no_improvement_count >= 30:
            print(f"\nEarly stopping: No improvement for 30 episodes")
            break
    
    # Save all plots after training completes
    print("\nSaving training visualizations...")
    for fig, path in plotting_data:
        fig.write_html(path)
    
    return episode_rewards, episode_profits, episode_sharpes

# %% Main Execution
def main(fast_mode: bool = False):
    """Main training and testing pipeline"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data loader
    loader = DataLoader(Config.CURRENCY_PAIR)
    
    # Load and prepare data
    df = loader.load_data()
    df = loader.add_technical_indicators()
    
    # Split data BEFORE normalization to avoid look-ahead bias
    train_size = int(len(df) * Config.TRAIN_TEST_SPLIT)
    df_train = df[:train_size].copy()
    df_test = df[train_size:].copy()
    
    # Store splits back in loader.df before normalization
    loader.df = pd.concat([df_train, df_test])
    
    # Compute normalization stats from training data only
    feature_cols = loader.prepare_features(train_df=df_train)
    
    # Now extract the normalized data
    df_train = loader.df[:train_size].copy()
    df_test = loader.df[train_size:].copy()
    
    print(f"\nUsing {len(feature_cols)} features")
    
    print(f"\nTraining data: {len(df_train)} samples")
    print(f"Testing data: {len(df_test)} samples")
    
    # Initialize environment and agent
    env = TradingEnvironment(df_train, feature_cols)
    state_size = len(feature_cols) * Config.WINDOW_SIZE + 9  # +9 for enhanced position info
    agent = TradingAgent(state_size, Config.ACTION_SIZE)
    
    print(f"\nState size: {state_size}")
    print(f"Action size: {Config.ACTION_SIZE}")
    
    # Initialize TensorBoard if available
    writer = None
    activations = None
    # Disable TensorBoard logging for speed optimization
    # if TENSORBOARD_AVAILABLE:
    #     writer = SummaryWriter(f'runs/audusd_agent_v2_{time.strftime("%Y%m%d_%H%M%S")}')
    #     
    #     # Log model graph (optional)
    #     dummy_state = torch.zeros((1, state_size), device=device)
    #     try:
    #         writer.add_graph(agent.q_network, dummy_state)
    #     except:
    #         print("Could not log model graph to TensorBoard")
    #     
    #     # Register forward hooks to capture activations
    #     activations = {}
    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activations[name] = output.detach()
    #         return hook
    #     
    #     # Attach hooks to key layers
    #     if agent.use_cnn:
    #         agent.q_network.conv[1].register_forward_hook(get_activation('conv1_relu'))
    #         agent.q_network.conv[4].register_forward_hook(get_activation('conv2_relu'))
    #     agent.q_network.value[0].register_forward_hook(get_activation('value_fc'))
    #     agent.q_network.advantage[0].register_forward_hook(get_activation('advantage_fc'))
    
    # Start training
    print("\nStarting training...")
    episodes = 20 if fast_mode else Config.EPISODES
    _, episode_profits, episode_sharpes = train_agent(
        agent, env, df_train, Config.WINDOW_SIZE, episodes, fast_mode,
        writer=writer, activations=activations
    )
    
    # Load best model for testing
    agent.q_network.load_state_dict(torch.load('best_model.pth'))
    agent.epsilon = 0  # No exploration during testing
    
    print("\n" + "="*50)
    print("Testing on unseen data...")
    print("="*50)
    
    # Test on test set
    env_test = TradingEnvironment(df_test, feature_cols)
    env_test.reset()
    
    # Get initial state as tensor
    state = env_test.get_state(Config.WINDOW_SIZE, Config.WINDOW_SIZE)
    test_trades = []
    
    # Cache test data for speed
    composite_signals_test = df_test['Composite_Signal'].values
    
    for t in tqdm(range(Config.WINDOW_SIZE, len(df_test) - 1)):
        # Get composite signal for action masking - use cached array
        composite_signal = composite_signals_test[t]
        
        # Agent takes action with signal-based masking
        action = agent.act(state, composite_signal, has_position=bool(env_test.position))
        _, info = env_test.execute_action(action, t)
        
        next_state = env_test.get_state(t + 1, Config.WINDOW_SIZE)
        state = next_state
        
        if 'position_opened' in info:
            test_trades.append({
                'time': df_test.index[t],
                'action': info['position_opened'],
                'price': info['price']
            })
    
    # Final test results
    final_balance = env_test.balance
    if env_test.position:
        # Close any open position
        final_price = df_test['Close'].iloc[-1]
        env_test._close_position(final_price, 'manual')
        final_balance = env_test.balance
    
    test_profit_usd = final_balance - Config.INITIAL_BALANCE
    test_profit_pct = (final_balance / Config.INITIAL_BALANCE - 1) * 100
    test_trades_data = env_test.trade_history
    
    # Calculate proper Sharpe ratio from equity curve
    equity_curve = env_test.get_equity_curve()
    
    if len(equity_curve) > 1:
        # Calculate per-bar returns
        bar_returns = equity_curve.pct_change().dropna()
        
        if len(bar_returns) > 0 and bar_returns.std() > 0:
            # Annualize: 96 bars per day (15-minute bars), 252 trading days per year
            sharpe = bar_returns.mean() / bar_returns.std() * np.sqrt(252 * 96)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    if test_trades_data:
        test_wins = [t for t in test_trades_data if t['pnl'] > 0]
        test_win_rate = len(test_wins) / len(test_trades_data) * 100
        max_dd = env_test.max_drawdown * 100
        
        # Exit type statistics for test
        test_tp_exits = sum(1 for t in test_trades_data if t.get('tp_exit', False))
        test_sl_exits = sum(1 for t in test_trades_data if t.get('sl_exit', False))
        test_manual_exits = sum(1 for t in test_trades_data if t.get('manual_exit', False))
        test_early_exits = sum(1 for t in test_trades_data if t.get('early_exit', False))
        
        # Exit type percentages
        test_pct_tp = (test_tp_exits / len(test_trades_data)) * 100
        test_pct_sl = (test_sl_exits / len(test_trades_data)) * 100
        test_pct_manual = (test_manual_exits / len(test_trades_data)) * 100
        test_pct_early = (test_early_exits / len(test_trades_data)) * 100
        
        # Average drawdown
        test_avg_drawdown = np.mean([t.get('drawdown', 0) for t in test_trades_data]) * 100
        
        print("\nTest Results:")
        print(f"Final Profit: ${test_profit_usd:,.0f} ({test_profit_pct:.2f}%)")
        print(f"Total Trades: {len(test_trades_data)}")
        print(f"Win Rate: {test_win_rate:.1f}%")
        print(f"Sharpe Ratio (Daily): {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.1f}%")
        total_test_exit_pct = test_pct_tp + test_pct_sl + test_pct_manual + test_pct_early
        print(f"Exit Types: TP={test_pct_tp:.1f}%, SL={test_pct_sl:.1f}%, Manual={test_pct_manual:.1f}%, Early={test_pct_early:.1f}% (Total: {total_test_exit_pct:.1f}%)")
        print(f"Avg Drawdown per Trade: {test_avg_drawdown:.1f}%")
    else:
        print("\nNo trades executed during testing")
        print(f"Sharpe Ratio (Daily): {sharpe:.2f}")
        test_win_rate = 0
    
    # Save results
    results = {
        'train_profits': episode_profits,
        'train_sharpes': episode_sharpes,
        'test_profit_usd': test_profit_usd,
        'test_profit_pct': test_profit_pct,
        'test_sharpe': sharpe,
        'test_trades': len(test_trades_data) if test_trades_data else 0,
        'test_win_rate': test_win_rate if test_trades_data else 0,
        'equity_curve': equity_curve.tolist()
    }
    
    torch.save(results, 'training_results.pth')
    
    # Close TensorBoard writer if available
    if writer is not None:
        writer.close()
        print("\nTraining complete! Results and TensorBoard logs saved.")
        print("\nTo view TensorBoard logs, run:")
        print("  python -m tensorboard.main --logdir=runs --host=localhost --port=6006")
        print("Then open http://localhost:6006/ in your browser.")
    else:
        print("\nTraining complete! Results saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Fast mode for testing')
    args = parser.parse_args()
    
    main(fast_mode=args.fast)