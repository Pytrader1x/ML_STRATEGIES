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
    ACTION_SIZE = 3   # Clean action space: 0=Hold, 1=Buy, 2=Sell
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
    MIN_HOLDING_BARS = 2  # Minimum holding period
    COOLDOWN_BARS = 1  # Cooldown after any exit 
    
    # Trading Parameters - USD based with 1M lots
    INITIAL_BALANCE = 1_000_000  # Start with USD 1M
    POSITION_SIZE = 1_000_000    # Always trade 1M AUDUSD units
    MAX_POSITIONS = 1            # One position at a time
    
    # Risk Management - ADAPTIVE
    BASE_SL_ATR_MULT = 2.0  # Stop loss = 2 * ATR
    BASE_TP_ATR_MULT = 3.0  # Take profit = 3 * ATR
    MIN_RR_RATIO = 1.5      # Minimum risk/reward ratio

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
                # Fallback for backward compatibility
                self.df[f'{col}_norm'] = (
                    (self.df[col] - self.df[col].rolling(200).mean()) / 
                    (self.df[col].rolling(200).std() + 1e-8)
                ).fillna(0)
        
        return [f'{col}_norm' for col in self.feature_cols if col in self.df.columns]
    

# %% Enhanced DQN Model
class DuelingDQN_CNN(nn.Module):
    """Dueling DQN with 1D CNN for feature extraction"""
    
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
        
        # Calculate flattened size after convolutions
        conv_out_size = window_size // 2  # After one MaxPool1d(2)
        flat_size = 128 * conv_out_size + 7  # 7 position features
        
        # Value stream
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
        window_features = x[:, :-7].view(batch_size, self.num_features, self.window_size)
        pos_features = x[:, -7:]
        
        # Apply CNN to window features
        conv_out = self.conv(window_features)
        conv_flat = conv_out.flatten(1)
        
        # Concatenate with position features
        combined = torch.cat([conv_flat, pos_features], dim=1)
        
        # Compute value and advantage
        value = self.value(combined)
        advantage = self.advantage(combined)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


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

# %% Enhanced Trading Environment
class TradingEnvironment:
    """Improved trading environment with better position management"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df
        self.feature_cols = feature_cols
        self.equity_curve = []  # Track equity for proper Sharpe calculation
        self.state_size = len(feature_cols) * Config.WINDOW_SIZE + 7
        # Pre-convert feature data to tensor for fast access
        self.feature_tensor = torch.from_numpy(df[feature_cols].values).float().to(device)
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
        
    def get_state(self, index: int, window_size: int) -> torch.Tensor:
        """Get enhanced state representation - returns torch.Tensor on device"""
        if index < window_size:
            return torch.zeros(self.state_size, device=device)
        
        # Get windowed features directly from MPS tensor - no CPU transfer!
        window = self.feature_tensor[index-window_size+1:index+1]  # [W, F] on MPS
        
        # Build position info tensor directly on device
        pos_info = torch.tensor([
            1.0 if self.position else 0.0,
            self.position['unrealized_pnl'] / self.initial_balance if self.position else 0.0,
            self.position['holding_time'] / 100.0 if self.position else 0.0,
            (self.balance / self.initial_balance - 1.0),
            self.max_drawdown,
            len(self.trade_history) / 100.0,
            self.get_win_rate()
        ], device=device, dtype=torch.float32)
        
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
        current_atr = self.df['ATR'].iloc[self.current_step]
        
        # Adjust multipliers based on market regime
        if self.df['Trending'].iloc[self.current_step] == 1:
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
        current_price = self.df['Close'].iloc[index]
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
                elif current_price >= self.position['take_profit']:
                    self._close_position(current_price, 'take_profit')
            else:  # Short
                if current_price >= self.position['stop_loss']:
                    self._close_position(current_price, 'stop_loss')
                elif current_price <= self.position['take_profit']:
                    self._close_position(current_price, 'take_profit')
        
        # Execute new actions
        if action == 0:  # Hold
            # No action taken
            pass
            
        elif action == 1:  # Buy
            if not self.position:
                # Check cooldown after exit
                if self.current_step - self.last_exit_bar < Config.COOLDOWN_BARS:
                    action = 0  # Convert to Hold during cooldown
                    info['action_blocked'] = 'cooldown'
                else:
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
                        'entry_signal_strength': self.df['Composite_Signal'].iloc[index]
                    }
                    
                    info['position_opened'] = 'long'
                    
            elif self.position['direction'] == -1:
                # P1: Minimum holding bars - prevent instant flip
                if self.position['holding_time'] <= Config.MIN_HOLDING_BARS:
                    # BUG FIX: Don't return early - let method continue to update metrics
                    action = 0  # Convert to Hold action
                    info['action_blocked'] = 'min_hold_time'
                
                else:
                    # Close short and open long
                    self._close_position(current_price, 'manual')
                    
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
                        'entry_signal_strength': self.df['Composite_Signal'].iloc[index]
                    }
                    
                    info['position_opened'] = 'long'
                    
        elif action == 2:  # Sell
            if not self.position:
                # Check cooldown after exit
                if self.current_step - self.last_exit_bar < Config.COOLDOWN_BARS:
                    action = 0  # Convert to Hold during cooldown
                    info['action_blocked'] = 'cooldown'
                else:
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
                        'entry_signal_strength': self.df['Composite_Signal'].iloc[index]
                    }
                    
                    info['position_opened'] = 'short'
                    
            elif self.position['direction'] == 1:
                # P1: Minimum holding bars - prevent instant flip
                if self.position['holding_time'] <= Config.MIN_HOLDING_BARS:
                    # BUG FIX: Don't return early - let method continue to update metrics
                    action = 0  # Convert to Hold action
                    info['action_blocked'] = 'min_hold_time'
                
                else:
                    # Close long position (manual exit)
                    self._close_position(current_price, 'manual')
                    
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
                        'entry_signal_strength': self.df['Composite_Signal'].iloc[index]
                    }
                    
                    info['position_opened'] = 'short'
        
        # Update metrics
        self._update_metrics()
        
        # Calculate NAV after action (B_t + U_t)
        nav_after = self.balance
        if self.position:
            # Add current unrealized P&L
            nav_after += self.position['unrealized_pnl']
        
        # Track equity curve at each step (using NAV)
        self.equity_curve.append(nav_after)
        
        # P0-2: Pure NAV-Δ reward only
        # Calculate NAV-Δ reward: r_t = (B_t + U_t) - (B_{t-1} + U_{t-1})
        nav_delta = nav_after - nav_before
        
        # Scale by 1000 to get typical rewards in [-1, 1] range
        # This symmetric scaling ensures no bias toward long or short
        reward = (nav_delta / self.initial_balance) * 1000 - 0.0001
        
        return reward, info
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series for Sharpe calculation"""
        return pd.Series(self.equity_curve)
    
    def _check_entry_conditions(self, direction: int) -> bool:
        """Check if entry conditions are met"""
        # Get current signals
        composite_signal = self.df['Composite_Signal'].iloc[self.current_step]
        nti_confidence = self.df['NTI_Confidence'].iloc[self.current_step]
        trending = self.df['Trending'].iloc[self.current_step]
        
        if direction == 1:  # Long
            return (composite_signal > 0.3 and nti_confidence > 0.6) or \
                   (composite_signal > 0.5 and trending == 1)
        else:  # Short
            return (composite_signal < -0.3 and nti_confidence > 0.6) or \
                   (composite_signal < -0.5 and trending == 1)
    
    def _calculate_position_size(self, conservative: bool = False) -> float:
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
            'drawdown': self.max_drawdown,
            'entry_index': self.position['entry_index']
        })
        
        self.position = None
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

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, experience: Experience):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with priority-based probabilities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, torch.FloatTensor(weights).to(device)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

class TradingAgent:
    """Enhanced RL Trading Agent with advanced features"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(Config.MEMORY_SIZE)
        
        # RL parameters
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.beta = 0.4  # For importance sampling
        self.beta_increment = 0.001
        
        # Neural networks - use CNN version
        self.use_cnn = True  # Flag to enable CNN
        if self.use_cnn:
            num_features = len([col for col in ['Close', 'Returns', 'High_Low_Pct', 'Close_Open_Pct',
                                               'ATR_Pct', 'Close_to_SMA20', 'Close_to_SMA50',
                                               'RSI', 'NTI_Direction', 'NTI_Confidence', 'NTI_SlopePower',
                                               'NTI_ReversalRisk', 'MB_Bias', 'IC_Regime', 'IC_Confidence',
                                               'IC_Signal', 'Trending', 'Composite_Signal'] if True])  # 18 features
            self.q_network = DuelingDQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
            self.target_network = DuelingDQN_CNN(state_size, action_size, num_features, Config.WINDOW_SIZE).to(device)
        else:
            self.q_network = DuelingDQN(state_size, action_size).to(device)
            self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        self.update_counter = 0
        
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory - now with torch tensors"""
        # Ensure states are tensors on device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def act(self, state: torch.Tensor, signal: float = 0.0) -> int:
        """Choose action using epsilon-greedy policy with signal-based masking
        
        P0-3: Action masking based on Composite_Signal
        - If signal > 0.2: disable Sell (action 2)
        - If signal < -0.2: disable Buy (action 1)
        """
        # Epsilon-greedy with masked actions
        if random.random() <= self.epsilon:
            # Create action mask based on signal (tightened thresholds)
            valid_actions = [0, 1, 2]  # Hold, Buy, Sell
            if signal > 0.35:  # Stronger bullish signal required
                valid_actions.remove(2)  # Remove Sell when strongly bullish
            elif signal < -0.35:  # Stronger bearish signal required
                valid_actions.remove(1)  # Remove Buy when strongly bearish
            return random.choice(valid_actions)
        
        # Get Q-values - state is already a tensor on device!
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
        self.q_network.train()
        
        # Add exploration bonus
        q_values_np = q_values.cpu().numpy()[0]
        exploration_bonus = np.random.normal(0, 0.01, self.action_size)
        q_values_np += exploration_bonus
        
        # P0-3: Apply action masking (tightened thresholds)
        if signal > 0.35:  # Stronger bullish signal required
            q_values_np[2] = -1e9  # Mask Sell when strongly bullish
        elif signal < -0.35:  # Stronger bearish signal required
            q_values_np[1] = -1e9  # Mask Buy when strongly bearish
        
        return np.argmax(q_values_np)
    
    def replay(self, batch_size: int):
        """Train model on batch of experiences with prioritized replay"""
        if len(self.memory.buffer) < batch_size:
            return
        
        # Sample from memory
        experiences, indices, weights = self.memory.sample(batch_size, self.beta)
        
        # Prepare batch - states are already tensors on device!
        states = torch.stack([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences]).to(device)
        
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
        
        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % Config.UPDATE_TARGET_EVERY == 0:
            self.update_target_model()
        
        # Removed per-step epsilon decay - moved to episode level
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return loss.item()

# %% Training Functions
def train_agent(agent: TradingAgent, env: TradingEnvironment, df_train: pd.DataFrame,
                window_size: int, episodes: int, fast_mode: bool = False):
    """Enhanced training loop with better monitoring and proper Sharpe calculation"""
    
    episode_rewards = []
    episode_profits = []
    episode_sharpes = []
    best_sharpe = -float('inf')
    no_improvement_count = 0
    
    # Setup plotting directory
    os.makedirs('plots', exist_ok=True)
    
    # Thread pool for async HTML saving
    executor = ThreadPoolExecutor(max_workers=2)
    
    def async_save_html(fig, path):
        """Save Plotly figure as HTML asynchronously"""
        fig.write_html(path)    
    for episode in range(episodes):
        # Reset environment
        env.reset()
        
        # Sample random window
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
        
        # Training loop
        pbar = tqdm(range(start_idx, end_idx - 1), 
                   desc=f"Episode {episode+1}/{episodes}")
        
        for t in pbar:
            # P0-3: Get composite signal for action masking
            composite_signal = df_train['Composite_Signal'].iloc[t]
            
            # Agent takes action with signal-based masking
            action = agent.act(state, composite_signal)
            
            # Execute action
            reward, info = env.execute_action(action, t)
            total_reward += reward
            
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
            done = (t == end_idx - 2)
            
            # Store experience - states are already tensors
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train on mini-batch
            if len(agent.memory.buffer) > Config.BATCH_SIZE:
                _ = agent.replay(Config.BATCH_SIZE)
            
            # Update progress bar with USD profit
            if env.position:
                pbar.set_postfix({
                    'Profit': f"${env.balance - env.initial_balance:,.0f}",
                    'Trades': len(env.trade_history),
                    'ε': f"{agent.epsilon:.3f}"
                })
        
        # Episode summary with USD profits
        final_profit_usd = env.balance - env.initial_balance
        final_profit_pct = (env.balance / env.initial_balance - 1) * 100
        trades = env.trade_history
        
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades) * 100
            
            # Exit type statistics
            tp_exits = sum(1 for t in trades if t.get('tp_exit', False))
            sl_exits = sum(1 for t in trades if t.get('sl_exit', False))
            manual_exits = sum(1 for t in trades if t.get('manual_exit', False))
            
            # Exit type percentages
            pct_tp = (tp_exits / len(trades)) * 100 if trades else 0
            pct_sl = (sl_exits / len(trades)) * 100 if trades else 0
            pct_manual = (manual_exits / len(trades)) * 100 if trades else 0
            
            # Average drawdown
            avg_drawdown = np.mean([t.get('drawdown', 0) for t in trades]) * 100
            
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if len(trades) > len(wins) else 0
        else:
            win_rate = 0
            tp_exits = sl_exits = manual_exits = 0
            pct_tp = pct_sl = pct_manual = 0
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
            print(f"  Exit Types: TP={pct_tp:.1f}%, SL={pct_sl:.1f}%, Manual={pct_manual:.1f}%")
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
        
        # Epsilon decay at end of episode
        agent.epsilon = max(Config.EPSILON_MIN, agent.epsilon * Config.EPSILON_DECAY)
        print(f"  Current Epsilon: {agent.epsilon:.4f}")
        
        # Anneal beta over episodes
        agent.beta = min(1.0, 0.4 + (0.6 * episode / episodes))  # Linear from 0.4 to 1.0
        
        # Create interactive Plotly chart for episode
        if len(trades) > 0 and episode % 1 == 0:  # Save every 1th episode to avoid too many files
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
            
            # Save HTML asynchronously
            executor.submit(async_save_html, fig, f'plots/episode_{episode+1:03d}.html')
        
        if no_improvement_count >= 30:
            print(f"\nEarly stopping: No improvement for 30 episodes")
            break
    
    # Cleanup
    executor.shutdown(wait=True)
    
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
    
    # Compute normalization stats from training data only
    feature_cols = loader.prepare_features(train_df=df_train)
    
    print(f"\nUsing {len(feature_cols)} features")
    
    print(f"\nTraining data: {len(df_train)} samples")
    print(f"Testing data: {len(df_test)} samples")
    
    # Initialize environment and agent
    env = TradingEnvironment(df_train, feature_cols)
    state_size = len(feature_cols) * Config.WINDOW_SIZE + 7  # +7 for position info
    agent = TradingAgent(state_size, Config.ACTION_SIZE)
    
    print(f"\nState size: {state_size}")
    print(f"Action size: {Config.ACTION_SIZE}")
    
    # Start training
    print("\nStarting training...")
    episodes = 10 if fast_mode else Config.EPISODES
    _, episode_profits, episode_sharpes = train_agent(
        agent, env, df_train, Config.WINDOW_SIZE, episodes, fast_mode
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
    
    for t in tqdm(range(Config.WINDOW_SIZE, len(df_test) - 1)):
        # P0-3: Get composite signal for action masking
        composite_signal = df_test['Composite_Signal'].iloc[t]
        
        # Agent takes action with signal-based masking
        action = agent.act(state, composite_signal)
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
        
        # Exit type percentages
        test_pct_tp = (test_tp_exits / len(test_trades_data)) * 100
        test_pct_sl = (test_sl_exits / len(test_trades_data)) * 100
        test_pct_manual = (test_manual_exits / len(test_trades_data)) * 100
        
        # Average drawdown
        test_avg_drawdown = np.mean([t.get('drawdown', 0) for t in test_trades_data]) * 100
        
        print("\nTest Results:")
        print(f"Final Profit: ${test_profit_usd:,.0f} ({test_profit_pct:.2f}%)")
        print(f"Total Trades: {len(test_trades_data)}")
        print(f"Win Rate: {test_win_rate:.1f}%")
        print(f"Sharpe Ratio (Daily): {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.1f}%")
        print(f"Exit Types: TP={test_pct_tp:.1f}%, SL={test_pct_sl:.1f}%, Manual={test_pct_manual:.1f}%")
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
    print("\nTraining complete! Results saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Fast mode for testing')
    args = parser.parse_args()
    
    main(fast_mode=args.fast)