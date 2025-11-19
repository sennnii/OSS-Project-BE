import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import N_AGENTS, WINDOW_SIZE, REWARD_SCALE

class MARLStockEnv(gym.Env):
    def __init__(self, features_df, prices_df, 
                 agent_0_cols, agent_1_cols, agent_2_cols, 
                 n_agents=N_AGENTS, window_size=WINDOW_SIZE):
        super().__init__()
        
        self.df = features_df
        self.prices = prices_df
        self.window_size = window_size
        self.n_agents = n_agents
        self.max_steps = len(self.df) - self.window_size - 1
        
        all_feature_cols = list(features_df.columns)
        self.agent_0_indices = [all_feature_cols.index(col) for col in agent_0_cols if col in all_feature_cols]
        self.agent_1_indices = [all_feature_cols.index(col) for col in agent_1_cols if col in all_feature_cols]
        self.agent_2_indices = [all_feature_cols.index(col) for col in agent_2_cols if col in all_feature_cols]
        
        self.n_features_agent_0 = len(self.agent_0_indices)
        self.n_features_agent_1 = len(self.agent_1_indices)
        self.n_features_agent_2 = len(self.agent_2_indices)
        self.n_features_global = len(all_feature_cols)

        self.observation_dim_0 = self.window_size * self.n_features_agent_0 + 2
        self.observation_dim_1 = self.window_size * self.n_features_agent_1 + 2
        self.observation_dim_2 = self.window_size * self.n_features_agent_2 + 2
        
        self.state_dim = self.window_size * self.n_features_global + (self.n_agents * 2)
        
        self.observation_space = spaces.Dict({
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim_0,), dtype=np.float32),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim_1,), dtype=np.float32),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim_2,), dtype=np.float32)
        })
        
        self.action_dim = 3
        self.action_space = spaces.Dict({
            f'agent_{i}': spaces.Discrete(self.action_dim) for i in range(self.n_agents)
        })
        
        self.current_step = 0
        self.positions = [0] * self.n_agents
        self.entry_prices = [0.0] * self.n_agents

    def _get_obs_and_state(self):
        start = self.current_step
        end = start + self.window_size
        
        market_data_global_windowed = self.df.iloc[start:end].values
        
        market_data_agent_0 = market_data_global_windowed[:, self.agent_0_indices]
        market_data_agent_1 = market_data_global_windowed[:, self.agent_1_indices]
        market_data_agent_2 = market_data_global_windowed[:, self.agent_2_indices]

        market_data_global_flat = market_data_global_windowed.flatten()
        market_data_agent_0_flat = market_data_agent_0.flatten()
        market_data_agent_1_flat = market_data_agent_1.flatten()
        market_data_agent_2_flat = market_data_agent_2.flatten()
            
        current_price = self.prices.iloc[self.current_step + self.window_size - 1]
        
        global_portfolio_state = []
        observations = {}
        
        for i in range(self.n_agents):
            pos_signal = self.positions[i]
            entry_price = self.entry_prices[i]
            
            unrealized_return_pct = 0.0
            if pos_signal == 1 and entry_price != 0:
                unrealized_return_pct = (current_price - entry_price) / (entry_price + 1e-9)
            elif pos_signal == -1 and entry_price != 0:
                unrealized_return_pct = (entry_price - current_price) / (entry_price + 1e-9)
            unrealized_return_pct = np.clip(unrealized_return_pct, -1.0, 1.0)
            
            own_portfolio_state = np.array([pos_signal, unrealized_return_pct], dtype=np.float32)
            
            if i == 0:
                obs_flat = market_data_agent_0_flat
            elif i == 1:
                obs_flat = market_data_agent_1_flat
            elif i == 2:
                obs_flat = market_data_agent_2_flat
            else:
                obs_flat = market_data_global_flat
                
            observations[f'agent_{i}'] = np.concatenate([obs_flat, own_portfolio_state])
            global_portfolio_state.append(own_portfolio_state)
            
        global_state = np.concatenate([market_data_global_flat, np.concatenate(global_portfolio_state)])
        return observations, global_state

    def reset(self, seed=None, initial_portfolio=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if initial_portfolio:
            self.positions = initial_portfolio['positions']
            self.entry_prices = initial_portfolio['entry_prices']
        else:
            self.positions = [0] * self.n_agents
            self.entry_prices = [0.0] * self.n_agents
            
        obs, state = self._get_obs_and_state()
        return obs, {"global_state": state}

    def get_state(self):
        _, state = self._get_obs_and_state()
        return state
    
    def step(self, actions):
        old_price = self.prices.iloc[self.current_step + self.window_size - 1]
        self.current_step += 1
        new_price = self.prices.iloc[self.current_step + self.window_size - 1]
        
        price_return = (new_price - old_price) / (old_price + 1e-9)
        
        # ==================== 💎 초단순 보상 ====================
        
        # 거래 처리
        for i in range(self.n_agents):
            action = actions[f'agent_{i}']
            
            if action == 0:  # Buy
                self.positions[i] = 1
                self.entry_prices[i] = float(new_price)
            elif action == 2:  # Sell
                self.positions[i] = -1
                self.entry_prices[i] = float(new_price)
        
        # 보상 = 포지션 합 × 가격 변화율
        joint_position = sum(self.positions)
        team_reward = float(joint_position * price_return * REWARD_SCALE)
        
        # =======================================================
        
        rewards = {f'agent_{i}': team_reward for i in range(self.n_agents)}
        
        next_obs, next_state = self._get_obs_and_state()
        done = self.current_step >= self.max_steps
        dones = {f'agent_{i}': done for i in range(self.n_agents)}
        dones['__all__'] = done
        
        info = {
            "global_state": next_state,
            "raw_pnl": team_reward,
            "price_return": price_return
        }
        
        return next_obs, rewards, dones, False, info