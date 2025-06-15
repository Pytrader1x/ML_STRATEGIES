"""
Advanced Reinforcement Learning Trading Agent for AUDUSD
Uses NeuroTrend Intelligent, Market Bias, and Intelligent Chop indicators
"""

# %% Load Libraries
import os
import sys
import math
import keras
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from technical_indicators_custom import TIC

# For reproducibility
keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

# %% Constants and Configuration
class Config:
    """Configuration class for trading parameters"""
    # Data
    CURRENCY_PAIR = 'AUDUSD'
    TRAIN_TEST_SPLIT = 0.8
    
    # Indicators
    NEUROTREND_FAST = 10
    NEUROTREND_SLOW = 50
    MARKET_BIAS_LEN1 = 350
    MARKET_BIAS_LEN2 = 30
    
    # RL Parameters
    WINDOW_SIZE = 10  # Number of time steps to look back
    ACTION_SIZE = 3   # Hold, Buy, Sell
    
    # Model
    HIDDEN_LAYER_1 = 128
    HIDDEN_LAYER_2 = 64
    HIDDEN_LAYER_3 = 32
    LEARNING_RATE = 0.001
    
    # Training
    EPISODES = 20
    BATCH_SIZE = 32
    MEMORY_SIZE = 2000
    
    # RL Hyperparameters
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    
    # Trading
    INITIAL_BALANCE = 10000
    POSITION_SIZE = 0.02  # 2% of balance per trade
    MAX_POSITIONS = 5     # Maximum simultaneous positions
    STOP_LOSS_PCT = 0.01  # 1% stop loss
    TAKE_PROFIT_PCT = 0.02  # 2% take profit


# %% Data Loading and Preprocessing
class DataLoader:
    """Class for loading and preparing trading data"""
    
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
        
        # Keep only OHLCV columns initially
        self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Loaded {len(self.df)} rows of data")
        print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        return self.df
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add advanced technical indicators"""
        print("\nAdding technical indicators...")
        
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
        
        # Add basic indicators for additional features
        print("- Adding price features...")
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Price_Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['Close_Position'] = (self.df['Close'] - self.df['Low']) / (self.df['High'] - self.df['Low'])
        
        # Volume features
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume'].rolling(20).mean()
        
        # Drop NaN values
        self.df.dropna(inplace=True)
        
        print(f"Indicators added. Final shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self) -> List[str]:
        """Select and normalize features for the model"""
        # Define feature columns to use
        self.feature_cols = [
            'Close', 'Volume_Ratio', 'Returns', 'Price_Range', 'Close_Position',
            # NeuroTrend features
            'NTI_SlopePower', 'NTI_Direction', 'NTI_Confidence',
            'NTI_ReversalRisk', 'NTI_StallDetected',
            # Market Bias features
            'MB_Bias', 'MB_ha_avg',
            # Intelligent Chop features
            'IC_Regime', 'IC_Confidence', 'IC_ADX', 'IC_ChoppinessIndex',
            'IC_Signal', 'IC_ATR_Normalized'
        ]
        
        # Normalize features
        print("\nNormalizing features...")
        for col in self.feature_cols:
            if col in self.df.columns:
                # Use robust scaling to handle outliers
                median = self.df[col].median()
                mad = (self.df[col] - median).abs().median()
                if mad > 0:
                    self.df[f'{col}_norm'] = (self.df[col] - median) / (1.4826 * mad)
                else:
                    self.df[f'{col}_norm'] = 0
        
        # Return normalized column names
        return [f'{col}_norm' for col in self.feature_cols if col in self.df.columns]


# %% Deep Q-Network Model
@keras.saving.register_keras_serializable()
class DQN(keras.Model):
    """Enhanced Deep Q-Network for trading"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        
        # Build model
        self.model = keras.Sequential([
            # Input layer
            keras.layers.Dense(Config.HIDDEN_LAYER_1, input_dim=state_size),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.2),
            
            # Hidden layers
            keras.layers.Dense(Config.HIDDEN_LAYER_2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(Config.HIDDEN_LAYER_3),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            # Output layer
            keras.layers.Dense(action_size, activation='linear')
        ])
        
        # Compile model
        self.model.compile(
            loss='huber',  # More robust to outliers
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        )
    
    def call(self, x):
        return self.model(x)


# %% Trading Environment
class TradingEnvironment:
    """Trading environment with realistic constraints"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df
        self.feature_cols = feature_cols
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.balance = Config.INITIAL_BALANCE
        self.positions = []  # List of open positions
        self.trade_history = []
        self.current_step = 0
        
    def get_state(self, index: int, window_size: int) -> np.ndarray:
        """Get state representation for given index"""
        if index < window_size:
            # Pad with first values if not enough history
            padding = window_size - index
            data = self.df[self.feature_cols].iloc[:index+1].values
            if len(data) == 0:
                data = np.zeros((1, len(self.feature_cols)))
            padded = np.vstack([data[0:1]] * padding + [data])
            return padded[-window_size:].flatten()
        else:
            return self.df[self.feature_cols].iloc[index-window_size+1:index+1].values.flatten()
    
    def calculate_position_size(self) -> float:
        """Calculate position size based on Kelly Criterion and risk management"""
        # Simple fixed position sizing for now
        return self.balance * Config.POSITION_SIZE
    
    def execute_action(self, action: int, index: int) -> Tuple[float, Dict]:
        """Execute trading action and return reward and info"""
        reward = 0
        info = {'action': action, 'price': self.df['Close'].iloc[index]}
        
        if action == 1:  # Buy
            if len(self.positions) < Config.MAX_POSITIONS:
                position_size = self.calculate_position_size()
                entry_price = self.df['Close'].iloc[index]
                
                position = {
                    'entry_price': entry_price,
                    'size': position_size,
                    'entry_index': index,
                    'stop_loss': entry_price * (1 - Config.STOP_LOSS_PCT),
                    'take_profit': entry_price * (1 + Config.TAKE_PROFIT_PCT)
                }
                
                self.positions.append(position)
                self.balance -= position_size
                info['position_opened'] = True
                
        elif action == 2 and len(self.positions) > 0:  # Sell
            # Close oldest position
            position = self.positions.pop(0)
            exit_price = self.df['Close'].iloc[index]
            
            # Calculate P&L
            price_change = (exit_price - position['entry_price']) / position['entry_price']
            pnl = position['size'] * price_change
            self.balance += position['size'] + pnl
            
            # Reward based on profit
            reward = pnl / Config.INITIAL_BALANCE * 100  # Percentage return
            
            trade = {
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return': price_change,
                'duration': index - position['entry_index']
            }
            self.trade_history.append(trade)
            info['trade_closed'] = trade
        
        # Check stop loss and take profit for all positions
        current_price = self.df['Close'].iloc[index]
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                positions_to_close.append(i)
        
        # Close positions that hit SL/TP
        for i in reversed(positions_to_close):
            position = self.positions.pop(i)
            price_change = (current_price - position['entry_price']) / position['entry_price']
            pnl = position['size'] * price_change
            self.balance += position['size'] + pnl
            
            # Penalty for stop loss, reward for take profit
            if current_price <= position['stop_loss']:
                reward -= abs(pnl) / Config.INITIAL_BALANCE * 50  # Penalty
            else:
                reward += pnl / Config.INITIAL_BALANCE * 100  # Reward
        
        return reward, info


# %% Reinforcement Learning Agent
class TradingAgent:
    """Advanced RL Trading Agent"""
    
    def __init__(self, state_size: int, action_size: int, is_eval: bool = False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.is_eval = is_eval
        
        # RL parameters
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON if not is_eval else 0.0
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        
        # Neural networks
        self.model = DQN(state_size, action_size).model
        self.target_model = DQN(state_size, action_size).model
        self.update_target_model()
        
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape state for prediction
        state = state.reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int):
        """Train model on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Predict Q-values for starting states
        current_qs = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network (Double DQN)
        next_qs = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                current_qs[i][actions[i]] = rewards[i]
            else:
                current_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(next_qs[i])
        
        # Train model
        self.model.fit(states, current_qs, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# %% Training Functions
def train_agent(agent: TradingAgent, env: TradingEnvironment, df_train: pd.DataFrame, 
                window_size: int, episodes: int):
    """Train the RL agent"""
    
    episode_rewards = []
    episode_profits = []
    
    for e in range(episodes):
        print(f"\nEpisode {e+1}/{episodes}")
        env.reset()
        
        state = env.get_state(window_size, window_size)
        total_reward = 0
        
        # Progress bar for episode
        for t in tqdm(range(window_size, len(df_train) - 1), desc=f"Episode {e+1}"):
            # Agent takes action
            action = agent.act(state)
            
            # Execute action
            reward, info = env.execute_action(action, t)
            total_reward += reward
            
            # Get next state
            next_state = env.get_state(t + 1, window_size)
            done = (t == len(df_train) - 2)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train on mini-batch
            if len(agent.memory) > Config.BATCH_SIZE:
                agent.replay(Config.BATCH_SIZE)
            
            if done:
                # Update target network
                agent.update_target_model()
                
                # Calculate episode metrics
                final_balance = env.balance
                for pos in env.positions:  # Close remaining positions
                    final_balance += pos['size']
                
                profit = final_balance - Config.INITIAL_BALANCE
                profit_pct = (profit / Config.INITIAL_BALANCE) * 100
                
                print(f"Episode {e+1} - Total Reward: {total_reward:.2f}")
                print(f"Final Balance: ${final_balance:.2f} ({profit_pct:.2f}%)")
                print(f"Total Trades: {len(env.trade_history)}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                
                episode_rewards.append(total_reward)
                episode_profits.append(profit_pct)
                
                # Save model periodically
                if (e + 1) % 5 == 0:
                    agent.model.save(f'models/rl_audusd_ep{e+1}.keras')
                
                break
    
    return episode_rewards, episode_profits


# %% Testing Functions
def test_agent(agent: TradingAgent, env: TradingEnvironment, df_test: pd.DataFrame, 
               window_size: int):
    """Test the trained agent"""
    
    env.reset()
    state = env.get_state(window_size, window_size)
    
    states_buy = []
    states_sell = []
    total_profit = 0
    
    print("\nTesting agent...")
    for t in tqdm(range(window_size, len(df_test) - 1)):
        action = agent.act(state)
        reward, info = env.execute_action(action, t)
        
        if action == 1:  # Buy
            states_buy.append(t)
        elif action == 2:  # Sell
            states_sell.append(t)
            if 'trade_closed' in info:
                total_profit += info['trade_closed']['pnl']
        
        next_state = env.get_state(t + 1, window_size)
        state = next_state
    
    # Close remaining positions
    for pos in env.positions:
        exit_price = df_test['Close'].iloc[-1]
        pnl = pos['size'] * ((exit_price - pos['entry_price']) / pos['entry_price'])
        total_profit += pnl
    
    return states_buy, states_sell, total_profit, env.trade_history


# %% Visualization Functions
def plot_results(df: pd.DataFrame, states_buy: List[int], states_sell: List[int], 
                 total_profit: float, title: str = "RL Trading Results"):
    """Plot trading results"""
    
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1)
    
    # Plot buy signals
    if states_buy:
        buy_prices = [df['Close'].iloc[i] for i in states_buy]
        buy_dates = [df.index[i] for i in states_buy]
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', 
                   s=100, label=f'Buy ({len(states_buy)})')
    
    # Plot sell signals
    if states_sell:
        sell_prices = [df['Close'].iloc[i] for i in states_sell]
        sell_dates = [df.index[i] for i in states_sell]
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', 
                   s=100, label=f'Sell ({len(states_sell)})')
    
    plt.title(f'{title} - Total Profit: ${total_profit:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot indicators
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['NTI_Direction'], label='NeuroTrend Direction', alpha=0.7)
    plt.plot(df.index, df['IC_Signal'], label='Intelligent Chop Signal', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_progress(episode_rewards: List[float], episode_profits: List[float]):
    """Plot training progress"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards, label='Episode Reward')
    ax1.plot(pd.Series(episode_rewards).rolling(5).mean(), label='5-Episode MA', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress - Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot profits
    ax2.plot(episode_profits, label='Episode Profit %')
    ax2.plot(pd.Series(episode_profits).rolling(5).mean(), label='5-Episode MA', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Profit %')
    ax2.set_title('Training Progress - Profits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()


# %% Main Execution
def main():
    """Main training and testing pipeline"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize data loader
    loader = DataLoader(Config.CURRENCY_PAIR)
    
    # Load and prepare data
    df = loader.load_data()
    df = loader.add_technical_indicators()
    feature_cols = loader.prepare_features()
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Split data
    train_size = int(len(df) * Config.TRAIN_TEST_SPLIT)
    df_train = df[:train_size].copy()
    df_test = df[train_size:].copy()
    
    print(f"\nTraining data: {len(df_train)} samples")
    print(f"Testing data: {len(df_test)} samples")
    
    # Initialize environment and agent
    env_train = TradingEnvironment(df_train, feature_cols)
    state_size = Config.WINDOW_SIZE * len(feature_cols)
    agent = TradingAgent(state_size, Config.ACTION_SIZE)
    
    print(f"\nState size: {state_size}")
    print(f"Action size: {Config.ACTION_SIZE}")
    
    # Train agent
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    episode_rewards, episode_profits = train_agent(
        agent, env_train, df_train, Config.WINDOW_SIZE, Config.EPISODES
    )
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_profits)
    
    # Test agent
    print("\n" + "="*50)
    print("Starting testing...")
    print("="*50)
    
    env_test = TradingEnvironment(df_test, feature_cols)
    agent_test = TradingAgent(state_size, Config.ACTION_SIZE, is_eval=True)
    agent_test.model = agent.model  # Use trained model
    
    states_buy, states_sell, total_profit, trades = test_agent(
        agent_test, env_test, df_test, Config.WINDOW_SIZE
    )
    
    # Calculate metrics
    profit_pct = (total_profit / Config.INITIAL_BALANCE) * 100
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
    
    print(f"\n" + "="*50)
    print("Test Results:")
    print(f"Total Profit: ${total_profit:.2f} ({profit_pct:.2f}%)")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Buy Signals: {len(states_buy)}")
    print(f"Sell Signals: {len(states_sell)}")
    print("="*50)
    
    # Plot results
    plot_results(df_test, states_buy, states_sell, total_profit, 
                 f"{Config.CURRENCY_PAIR} RL Trading Results")
    
    # Save final model
    agent.model.save('models/rl_audusd_final.keras')
    print("\nModel saved to models/rl_audusd_final.keras")


if __name__ == "__main__":
    main()