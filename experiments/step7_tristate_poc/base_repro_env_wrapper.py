import sys
import os
import numpy as np
import gym
from gym import spaces

# Ensure parent and root dirs are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from envs.serial import Env as BaseSerialEnv, S_I, S_O, LEAD_TIME, LEVEL_NUM, ACTION_DIM, OBS_DIM, H, B
from envs.env_wrappers import MultiDiscrete

class BasePaperReproEnv(BaseSerialEnv):
    def __init__(self, all_args):
        super().__init__(all_args)
        self.alpha_randomized = False
        
        # Strictly load the 20 frozen Merton demand traces as specified in the base paper
        eval_files = sorted([f for f in os.listdir(self.EVAL_PTH if hasattr(self, 'EVAL_PTH') else "./test_data/test_demand_merton/") if f.startswith("demand_")])[:20]
        eval_data = []
        for file in eval_files:
            data = []
            pth = os.path.join("./test_data/test_demand_merton/", file)
            with open(pth, "rb") as f:
                for line in f.readlines():
                    data.append(int(line.strip()))
            eval_data.append(data)
        self.n_eval = len(eval_data)
        self.eval_data = eval_data
        self.eval_index = 0

    def reset(self, train=True, normalize=True, fixed_alphas=None):
        self.alphas = [0.375 for _ in range(self.level_num)]
        sub_agent_obs = super().reset(train=train, normalize=normalize, fixed_alphas=self.alphas)
        return sub_agent_obs

    def get_processed_rewards(self, reward):
        """
        Fix 1: Exact Equation (7) from the base paper:
        r_t^i = - alpha * C_t^i - (1 - alpha) * C_t^total
        (Without dividing total by self.level_num)
        """
        processed = []
        if self.train:
            total = sum(reward)  # reward[i] is already negative cost (-C_t^i)
            for i in range(self.level_num):
                a = self.alphas[i]  # fixed 0.375
                r = a * reward[i] + (1.0 - a) * total
                processed.append([r])
        else:
            # Eval mode: raw negative costs for clean KPI measurement
            processed = [[r] for r in reward]
        return processed

class BasePaperReproSubprocVecEnv(object):
    def __init__(self, all_args):
        self.env_list = [BasePaperReproEnv(all_args) for _ in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []
            u_action_space = spaces.Discrete(self.signal_action_dim)
            total_action_space.append(u_action_space)

            act_space = total_action_space[0]
            self.action_space.append(act_space)

            # Fix 2: Retain all features for the centralized critic
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]

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

    def close(self): pass

class BasePaperReproDummyVecEnv(object):
    def __init__(self, all_args):
        self.env_list = [BasePaperReproEnv(all_args) for _ in range(all_args.n_eval_rollout_threads)]
        self.num_envs = all_args.n_eval_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []
            u_action_space = spaces.Discrete(self.signal_action_dim)
            total_action_space.append(u_action_space)

            act_space = total_action_space[0]
            self.action_space.append(act_space)

            # Fix 2: Retain all features for the centralized critic
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]

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
        obs = [env.reset(train=False, normalize=True) for env in self.env_list]
        return np.stack(obs), None

    def close(self): pass

