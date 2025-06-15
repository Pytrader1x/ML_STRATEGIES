Part 1: Reinforcement Learning for Stock Trading using Dueling Double Deep Q-Networks (Dueling DQN)

Sushant Kumar Jha
Sushant Kumar Jha

Follow
9 min read
·
Apr 14, 2024
65





Hey there! If you’re new here, no worries — you can jump right in. Today, we’re diving deeper into reinforcement learning (RL). We’ll be exploring how to implement Dueling Double Deep Q-Networks (Dueling DQN) to create an agent that can make trading decisions within a simulated stock trading environment. From data preprocessing to model training and evaluation, we’ll walk through the entire process step by step. Let’s get started!
Follow this repo. for dataset and code refrence from here.

Dueling Double Deep Q-Networks (Dueling DQN)
Before we go to code let deep dive into
Dueling Double Deep Q-Networks (Dueling DQN) is an extension of the traditional Deep Q-Network (DQN) architecture, specifically designed to improve the efficiency and stability of Q-learning in reinforcement learning tasks. It separates the representation of state values and action advantages, allowing the agent to learn more efficiently and effectively, especially in environments with many actions or where the advantages of different actions can vary significantly.
Components of Dueling DQN:
Value Stream (V(s)): This stream estimates the value of being in a particular state without committing to any specific action. It learns to predict the expected return (or total future reward) from a given state, irrespective of the action taken. The value stream helps the agent understand the overall desirability of different states in the environment.
Advantage Stream (A(s, a)): This stream estimates the advantage of taking a specific action in a given state compared to other actions. It learns to predict the additional reward gained by taking a particular action over the average reward obtained from all possible actions in the same state. The advantage stream helps the agent understand the relative importance of different actions in each state. example, Let’s say our agent is in a state where it can choose between three actions: A, B, and C. The Advantage Stream evaluates each action and estimates how much better (or worse) each action is compared to the average action in that state. If action A has a higher Advantage value compared to actions B and C, action A is expected to lead to a higher reward in that state.
Combining Value and Advantage Streams: The outputs of the value stream and advantage stream are combined to produce the final Q-values for each action in each state. This is achieved by adding the value estimates to the advantages after centring the advantages by subtracting their mean. By doing so, the network can learn both the absolute value of being in a state and the relative advantages of different actions within that state.
Advantages of Dueling DQN:
Efficient Learning: By separating the estimation of state values and action advantages, Dueling DQN reduces the redundancy in the learning process. It allows the agent to focus on learning the critical aspects of the environment more efficiently.
Improved Stability: Dueling DQN stabilizes the learning process by decoupling the estimation of state values and action advantages. This separation prevents the network from overestimating the Q-values or being biased towards certain actions, leading to more stable and reliable learning.
Effective Exploration: The advantage stream helps the agent prioritize actions that are likely to lead to higher rewards in each state. This facilitates more effective exploration of the environment and enables the agent to discover optimal policies more quickly.
Ok, Let's Implement and see what we can get.
Prerequisites
Make sure you have the following libraries installed.
pip install pandas numpy matplotlib gym torch tensorboardX ptan black flake8
Step 1: Implementation Dueling DQN:
In practice, Dueling DQN is implemented by splitting the last hidden layer of the neural network into two streams — one for estimating the state value and the other for estimating the action advantages. These streams are then combined to produce the final Q-values using a special aggregation function.
By leveraging the benefits of Dueling DQN, agents can achieve superior performance and faster convergence in a wide range of reinforcement learning tasks, including complex environments with high-dimensional state and action spaces.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        """
        Initialize the Simple Feedforward Dueling Deep Q-Network.

        Parameters:
        obs_len (int): Length of the input observation vector.
        actions_n (int): Number of possible actions.

        """
        super(SimpleFFDQN, self).__init__()

        # Value stream network
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream network
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor containing the estimated Q-values.

        """
        # Compute value and advantage streams
        val = self.fc_val(x)
        adv = self.fc_adv(x)

        # Combine value and advantage streams to produce the final Q-values
        # The advantage values are centered by subtracting their mean to improve stability
        return val + (adv - adv.mean(dim=1, keepdim=True))
Step 2: Preprocessing the Data
First, let’s preprocess our stock price data. We’ll normalize the data and convert it into a format suitable for training our RL agent. We’ll define functions to perform this preprocessing, including converting prices to relative values concerning the opening price.
import pandas as pd
import os
import csv
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt

Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])
def bar2rel(df,tolerance):
    prev_vals = None
    fix_open_price  = True
    open, high, low, close, volume = [], [], [], [], []
    count_out = 0
    count_filter = 0
    count_fixed = 0
    for row in df.itertuples():
        val = (row._3,row._4,row._5,row._6,row._7)
        po, ph, pl,pc,pv = val
        if fix_open_price and prev_vals is not None:
            ppo, pph, ppl, ppc, ppv = prev_vals
            if abs(po - ppc) > 1e-8:
                count_fixed += 1
                po = ppc
                pl = min(pl, po)
                ph = max(ph, po)
                count_out += 1
        open.append(po)
        close.append(pc)
        high.append(ph)
        low.append(pl)
        volume.append(pv)
        prev_vals = val
    prices=Prices(open=np.array(open, dtype=np.float32),
                  high=np.array(high, dtype=np.float32),
                  low=np.array(low, dtype=np.float32),
                  close=np.array(close, dtype=np.float32),
                  volume=np.array(volume, dtype=np.float32))
    return prices_to_relative(prices)

def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

def preprocess(path):
    df = pd.read_csv(os.path.abspath(train_path))

    index = ['<OPEN>', "<HIGH>", "<LOW>","<CLOSE>","<VOL>"]
    df[index] = df[index].astype(float)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    # Define the tolerance value
    tolerance = 1e-8

    # Apply the lambda function to check if each value is within the tolerance of the first value
    df_normalized.applymap(lambda v: abs(v - df_normalized.iloc[0]) < tolerance)
    return bar2rel(df_normalized,tolerance)
Step 3:Environment (StocksEnv) and State Representation
Environment (StocksEnv)
The trading environment simulates the stock market and provides an interface for the RL agent to interact with. Here’s a breakdown of the key components of the StocksEnv:
Observation Space: This defines the shape and range of possible observations the agent receives from the environment. In our case, the observation space represents the historical price data (e.g., open, high, low, close prices, and volume) for a specific number of bars (time periods) in the past.
Action Space: This defines the possible actions the agent can take in the environment. In our case, the agent can choose among three actions: Buy, Close (Sell), or Skip (Hold).
Step Function: This function executes a single step in the environment based on the action chosen by the agent. It updates the state of the environment and returns the new observation, reward, and whether the episode is done (terminated).
Reward Mechanism: This defines how the agent is rewarded based on its actions and the current state of the environment. In trading, the reward can be defined in various ways, such as the profit/loss made from trades, risk-adjusted returns, or other performance metrics.
Reset Function: This resets the environment to its initial state, typically at the beginning of each episode. It’s essential to ensure reproducibility and independence between episodes.
class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices: Prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False,
                 volumes=False):
        """
        Initializes the StocksEnv environment.
        
        Parameters:
        - prices (Prices): Named tuple containing open, high, low, close, and volume prices.
        - bars_count (int): Number of bars (time periods) in the state representation.
        - commission (float): Commission percentage applied to each trade.
        - reset_on_close (bool): Whether to reset the environment when a position is closed.
        - state_1d (bool): Whether to represent the state in 1D format.
        - random_ofs_on_reset (bool): Whether to reset the environment with a random offset.
        - reward_on_close (bool): Whether to reward the agent immediately when a position is closed.
        - volumes (bool): Whether to include volume information in the state representation.
        """
        self._prices = prices
        self._state = State(
            bars_count, commission, reset_on_close,
            reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        
    def seed(self, seed=None):
        """
        Sets the random seed for reproducibility.
        """
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
    
    def reset(self):
        """
        Resets the environment to its initial state.
        
        Returns:
        - observation (np.ndarray): Initial observation/state of the environment.
        """
        self._instrument = self.np_random.choice(
            list(self._prices._fields))
        if self._instrument is "open":
            prices = self._prices.open
        if self._instrument is "close":
            prices = self
State Representation
The state represents the current snapshot of the environment that the agent observes. It contains all the relevant information necessary for the agent to make decisions. In our case, the state representation includes:
Historical Price Data: This includes the historical prices of the stock (e.g., open, high, low, close prices, and volume) for a certain number of bars (time periods) in the past. These prices provide context for the agent to analyze and make trading decisions.
Position Information: This indicates whether the agent currently holds a position in the market (e.g., whether it has bought or sold the stock). It also includes information about the open price of the position, which is crucial for calculating profits/losses.
Other Relevant Information: Depending on the specific trading strategy and environment setup, additional information may be included in the state representation. For example, technical indicators, economic indicators, or market sentiment data may be incorporated to provide more context for decision-making.
The state representation is crucial for the agent’s learning process, as it determines the information available to the agent when making decisions. A well-designed state representation should capture the essential features of the environment while avoiding unnecessary complexity or redundancy.
By carefully designing the environment and state representation, we can create a realistic and informative setting for training our reinforcement learning agent to make profitable trading decisions.
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np


DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class State:
    def __init__(self, bars_count, commission_perc,
                 reset_on_close, reward_on_close=True,
                 volumes=True):
        """
        Initializes the State object.
        
        Parameters:
        - bars_count (int): Number of bars (time periods) in the state representation.
        - commission_perc (float): Commission percentage applied to each trade.
        - reset_on_close (bool): Whether to reset the environment when a position is closed.
        - reward_on_close (bool): Whether to reward the agent immediately when a position is closed.
        - volumes (bool): Whether to include volume information in the state representation.
        """
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        """
        Resets the state with new prices and offset.
        
        Parameters:
        - prices (Prices): Named tuple containing open, high, low, close, and volume prices.
        - offset (int): Offset index for the starting position in the price data.
        """
        assert isinstance(prices, Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        """
        Returns the shape of the state representation.
        """
        # [h, l, c] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3*self.bars_count + 1 + 1,

    def encode(self):
        """
        Converts the current state into a numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self):
        """
        Calculates the real close price for the current bar.
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Performs one step in the environment.
        
        Parameters:
        - action (Actions): Action taken by the agent.
        
        Returns:
        - reward (float): Reward obtained from the action.
        - done (bool): Whether the episode is done.
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done
Run and test StockEnv
give a test run on your dataset CSV file and check StockEnv environment is working properly.
train_path = "./YNDX_150101_151231.csv"
rp=preprocess(train_path)
env  = StocksEnv(rp, bars_count=10,
                 commission=0.1,
                 reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False,
                 volumes=True)
obs = env.reset()
action_idx = env.action_space.sample()
Training
Follow here for the training pipeline using ptan. Follow here for training.
thanks.

Part 2: Reinforcement Learning for Stock Trading using Dueling Double Deep Q-Networks (Dueling DQN)

Sushant Kumar Jha
Sushant Kumar Jha

Follow
5 min read
·
Apr 25, 2024
41

4




In our journey to explore the intersection of finance and artificial intelligence, we delve deeper into the world of Reinforcement Learning (RL) applied to stock trading. Building upon the foundation laid in Part 1, where we introduced the basics of RL and its application to financial markets, we now advance our understanding by implementing Dueling Double Deep Q-Networks (Dueling DQN) for stock trading.

Importing requirements
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Union
from datetime import datetime, timedelta
import ptan
import pathlib
import argparse
import gym.wrappers
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter

# Importing classes from local environment discussed in part1
from mymodule import SimpleFFDQN, EpsilonTracker, Actions, preprocess, StocksEnv

Bellman Equation Loss Calculation
The Bellman equation is used in dynamic programming algorithms like value iteration and policy iteration to iteratively compute the optimal value function or policy for a given Markov decision process (MDP). In reinforcement learning, it serves as a key equation for estimating the value function of states or the Q-function of state-action pairs, which are essential for learning optimal policies through Q-learning and deep Q-networks (DQN).
𝑉(𝑠)=max⁡𝑎(𝑅(𝑠,𝑎)+𝛾∑𝑠′𝑃(𝑠′∣𝑠,𝑎)𝑉(𝑠′))
Where:
𝑉(𝑠) discussed in Part1
The reward function R(s,a): R(s,a) provides the reward RiR_iRi​ for taking action aia_iai​ in state sis_isi​. This reward signals how good or bad the action was in that state, guiding the agent towards better actions over time.,
𝑃(𝑠′∣𝑠,𝑎) is the probability of transitioning to state 𝑠′s′ from state 𝑠s after taking action 𝑎a,
𝛾γ is the discount factor of future rewards to immediate rewards.
def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    if dones is not None:
        done_mask = torch.BoolTensor(dones.astype(bool)).to(device)
    else:
        done_mask =torch.BoolTensor(np.array([0],dtype =np.uint8))


    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # get action performed in next state . i will take max score
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net.target_model(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0
    # The Bellman equation is used to compute the expected Q-values for the current state-action pairs.
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    # calculate loss between state_action_values and expected_state_action_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)
Variables and Constants
These parameters include batch size, exploration parameters, discount factor, and more, which significantly influence the learning dynamics of our model.
# Constants and parameters
BATCH_SIZE = 32
BARS_COUNT = 10
EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000
GAMMA = 0.99
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000

# Initialize Tensorboard writer
writer = SummaryWriter(log_dir='logs')

# Preprocessing data
train_path = "\ch08-small-quotes\YNDX_150101_151231.csv"
val_path = "\ch08-small-quotes\YNDX_150101_151231.csv"

The heart of our experiment lies in the training loop, where we orchestrate the process of learning optimal trading strategies. Defining batch generators and training batch functions, we optimize our neural network models using the Bellman equation loss, a key component in updating Q-values based on observed transitions in the environment.
tp = preprocess(train_path)
vp = preprocess(val_path)

# Creating environments
env = StocksEnv(tp, bars_count=10, commission=0.1, reset_on_close=True, state_1d=False, random_ofs_on_reset=True, reward_on_close=False, volumes=True)
env_val = StocksEnv(vp, bars_count=10, commission=0.1, reset_on_close=True, state_1d=False, random_ofs_on_reset=True, reward_on_close=False, volumes=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing neural network models
net = SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
tgt_net = ptan.agent.TargetNet(net)

# Initializing action selector and epsilon tracker
selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
eps_tracker = EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)

# Initializing DQN agent
agent = ptan.agent.DQNAgent(net, selector, device=device)

# Initializing experience source
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

# Initializing optimizer
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Initializing Engine for training
trainer = Engine(train_batch)
Let’s set the generator from exp_source and quickly set train and validate function
# Define batch generator function
def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)

# Define training batch function
def train_batch(engine, batch):
    optimizer.zero_grad()
    loss_v = calc_loss(batch=batch, net=net, tgt_net=tgt_net, gamma=GAMMA ** REWARD_STEPS, device=device)
    loss_v.backward()
    optimizer.step()
    eps_tracker.frame(engine.state.iteration)
    if getattr(engine.state, "eval_states", None) is None:
        eval_states = buffer.sample(STATES_TO_EVALUATE)
        eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
        engine.state.eval_states = np.array(eval_states, copy=False)
    writer.add_scalar("training/loss", loss_v, engine.state.epoch)
    return {"loss": loss_v.item(), "epsilon": selector.epsilon}

# Validation function
def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, commission=0.1):
    stats = {metric: [] for metric in METRICS}

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)
            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = Actions(action_idx)
            close_price = env._state._cur_close()

            if action == Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * commission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * commission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return {key: np.mean(vals) for key, vals in stats.items()}
Setup ignites the handler and fires your training.
# Ignite event handlers
@trainer.on(Events.COMPLETED | Events.EPOCH_COMPLETED(every=10))
def log_training_results(engine):
    if engine.state.epoch % 10 == 0:
        res = validation_run(env_val, net, episodes=100, device="cpu", epsilon=0.02, commission=0.1)
        for key, value in res.items():
            writer.add_scalar("Agent Metrics", key, value)

@trainer.on(Events.ITERATION_COMPLETED)
def log_something(engine):
    out_dict = engine.state.output
    for key, value in out_dict.items():
        if value is None:
            value = 0.0
        elif isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(f"Iteration Metrics{engine.state.epoch}/{key}", value, engine.state.iteration)

# Checkpointing
checkpoint_handler = ModelCheckpoint(dirname='saved_models', filename_prefix='checkpoint', n_saved=2, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': net})

# Training
trainer.run(batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE), max_epochs=100)
writer.close()
torch.save(net.state_dict(), 'model_state_dict.pth')
res = validation_run(env_val, net, episodes=100, device="cpu", epsilon=0.02, commission=0.1)
print(res)
# Write a blog post on this script as the second part
Start trainer
We will quickly run the model train the model and test logs.
# Checkpointing
checkpoint_handler = ModelCheckpoint(dirname='saved_models', filename_prefix='checkpoint', n_saved=2, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': net})
trainer.run(batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE),max_epochs=100)
writer.close() 
torch.save(net.state_dict(), 'model_state_dict.pth')
res=validation_run(env_val, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1)
print(res) 
Conclusion
When I trained the model on a GPU with 8GB of memory (specifically, a T2-medium instance), it took 12 hours to complete. The agent needed around 30,000 more episodes to converge and start exhibiting satisfactory performance. I’m completely open to exploring deep learning models that can learn the relationships within our data and potentially achieve higher accuracy in fewer episodes. I’ll be testing such models in my next post. Until then, happy reading!