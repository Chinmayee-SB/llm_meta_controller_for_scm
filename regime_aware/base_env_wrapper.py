"""
base_env_wrapper.py

Reproduced Base Paper Specialist Environment:
- 4-echelon serial inventory system (Liu et al., 2022/2025).
- Actor observation: Strictly 7-D physical state [I/20, B/20, O_down/20, P_4/20, P_3/20, P_2/20, P_1/20].
  Strictly NO alpha in observation vector.
- Centralized Critic observation: 28-D (4 agents x 7 physical state features).
- Fixed Alpha: alpha = 0.375 fixed across all agents and training episodes.
- Reward: Exact released implementation equation:
    r_t^i = - [ 0.375 * C_t^i + (1 - 0.375) * (1 / M) * C_t^total ], M = 4.
"""

import os
import sys
import numpy as np
import gym
from gym import spaces

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from envs.serial import Env as BaseSerialEnv


class BaseSpecialistSerialEnv(BaseSerialEnv):
    """
    Serial Environment for the Base Paper Specialist baseline (alpha = 0.375 fixed).
    """
    def __init__(self, all_args):
        super().__init__(all_args)
        self.obs_dim = 7
        self.alpha_randomized = False
        self.alphas = [0.375 for _ in range(self.level_num)]

    def reset(self, train=True, normalize=True, fixed_alphas=None):
        self.alphas = [0.375 for _ in range(self.level_num)]
        self.alpha_randomized = False
        obs = super().reset(train=train, normalize=normalize, fixed_alphas=self.alphas)
        return obs

    def get_processed_rewards(self, reward):
        processed = []
        if self.train:
            total = sum(reward)  # raw negative costs (-C_t^total)
            for i in range(self.level_num):
                a = self.alphas[i]  # 0.375
                r = a * reward[i] + (1.0 - a) * (total / float(self.level_num))
                processed.append([r])
        else:
            processed = [[r] for r in reward]
        return processed


class BaseSpecialistSubprocVecEnv(object):
    """
    Vectorized parallel environment wrapper for Base Specialist training.
    """
    def __init__(self, all_args):
        self.env_list = [BaseSpecialistSerialEnv(all_args) for _ in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = 7
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
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            )
            share_obs_dim += self.signal_obs_dim

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
