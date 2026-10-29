"""
env_wrapper.py

Regime-Aware Supply Chain Environment Wrapper:
- 4-echelon serial inventory system (Retailer -> Warehouse -> Distributor -> Manufacturer).
- Actor observation: 8-D [I/20, B/20, O_down/20, P_4/20, P_3/20, P_2/20, P_1/20, alpha_t^i].
- Centralized Critic observation: 28-D (4 agents x 7 physical state features, strictly NO alpha).
- Discrete Cooperation Regimes:
    1. Cruiser (alpha = 0.375): Steady-state anchor (70% training prior).
    2. Shock Absorber (alpha = 0.100): System-wide volatility damping (15% training prior).
    3. Backlog Clearer (alpha = 0.650): Local stockout recovery gradient (15% training prior).
- Normalized Reward Equation matching released reference implementation:
    r_t^i = - [ alpha_t^i * C_t^i + (1 - alpha_t^i) * (1 / M) * C_t^total ], M = 4.
"""

import os
import sys
import numpy as np
import gym
from gym import spaces

# Ensure repo root and MADRL are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from envs.serial import Env as BaseSerialEnv


class RegimeAwareSerialEnv(BaseSerialEnv):
    """
    Serial Multi-Echelon Supply Chain Environment with Regime-Aware Cooperation Weights.
    """
    REGIMES = [0.375, 0.100, 0.650]
    CURRICULUM_PROBS = [0.70, 0.15, 0.15]

    def __init__(self, all_args):
        super().__init__(all_args)
        self.obs_dim = 8
        self.alpha_randomized = False
        self.current_regime_name = "Cruiser (0.375)"

    def reset(self, train=True, normalize=True, fixed_alphas=None):
        if fixed_alphas is not None:
            self.alphas = list(fixed_alphas)
            self.current_regime_name = f"Fixed ({self.alphas[0]})"
        elif train:
            chosen = float(np.random.choice(self.REGIMES, p=self.CURRICULUM_PROBS))
            self.alphas = [chosen for _ in range(self.level_num)]
            if np.isclose(chosen, 0.375):
                self.current_regime_name = "Cruiser (0.375)"
            elif np.isclose(chosen, 0.100):
                self.current_regime_name = "Shock Absorber (0.100)"
            else:
                self.current_regime_name = "Backlog Clearer (0.650)"
        else:
            self.alphas = [0.375 for _ in range(self.level_num)]
            self.current_regime_name = "Eval Cruiser (0.375)"

        self.alpha_randomized = False
        obs = super().reset(train=train, normalize=normalize, fixed_alphas=self.alphas)
        return obs

    def get_processed_rewards(self, reward):
        """
        Calculates cooperative reward per echelon:
            r_t^i = alpha_t^i * reward[i] + (1 - alpha_t^i) * (total / M)
        where reward[i] = -C_t^i, total = sum(reward) = -C_t^total, M = 4.
        """
        processed = []
        if self.train:
            total = sum(reward)
            for i in range(self.level_num):
                a = self.alphas[i]
                r = a * reward[i] + (1.0 - a) * (total / float(self.level_num))
                processed.append([r])
        else:
            processed = [[r] for r in reward]
        return processed


class RegimeAwareSubprocVecEnv(object):
    """
    Vectorized parallel environment runner for HAPPO curriculum training.
    """
    def __init__(self, all_args):
        self.env_list = [RegimeAwareSerialEnv(all_args) for _ in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = 8
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        for agent in range(self.num_agent):
            u_action_space = spaces.Discrete(self.signal_action_dim)
            self.action_space.append(u_action_space)
            share_obs_dim += 7  # strictly 7 physical features per agent, NO alpha for critic
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            )

        # 4 agents x 7 physical features = 28-D centralized critic observation
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
        inv = [env.get_inventory() for env in self.env_list]
        demand = [env.get_demand() for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders

    def reset(self):
        obs = [env.reset() for env in self.env_list]
        return np.stack(obs), None

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass
