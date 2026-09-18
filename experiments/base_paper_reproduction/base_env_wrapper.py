import os
import sys
import numpy as np
import gym
from gym import spaces

# Ensure repo root and MADRL are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from envs.serial import Env as BaseSerialEnv, S_I, S_O, LEAD_TIME, LEVEL_NUM, ACTION_DIM, OBS_DIM, H, B

class BasePaperSerialEnv(BaseSerialEnv):
    """
    Official Base Paper Supply Chain Environment:
    - Actor observation: Strictly 7 dimensions [I/20, B/20, O_down/20, P_4/20, P_3/20, P_2/20, P_1/20].
      NO alpha in observation vector.
    - Centralized Critic observation: 28 dimensions (4 agents x 7 physical features).
    - Fixed Alpha: 0.375 across all agents and all episodes.
    - Reward: Official Base Paper Equation (7):
        r_t^i = 0.375 * r_indiv^i + (1 - 0.375) * (1/4 * sum(r_indiv^j))
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
                # Exact Equation (7) with level_num averaging as in official repo
                r = a * reward[i] + (1.0 - a) * (total / self.level_num)
                processed.append([r])
        else:
            # Eval mode: raw negative costs for exact system cost accounting
            processed = [[r] for r in reward]
        return processed


class BasePaperSubprocVecEnv(object):
    """
    Vectorized parallel environment wrapper running n_rollout_threads (default 5).
    - Actor observation space: 7 dimensions (strictly physical state).
    - Centralized Critic observation space: 28 dimensions (4 agents x 7 features).
    """
    def __init__(self, all_args):
        self.env_list = [BasePaperSerialEnv(all_args) for _ in range(all_args.n_rollout_threads)]
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

            # Actor sees 7 physical state features
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            )

            share_obs_dim += self.signal_obs_dim

        # 4 agents x 7 features = 28 dimensions for the centralized critic
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


class BasePaperDummyVecEnv(object):
    """
    Single-thread / dummy vector env for evaluation.
    """
    def __init__(self, all_args):
        n_threads = getattr(all_args, "n_eval_rollout_threads", 1)
        self.env_list = [BasePaperSerialEnv(all_args) for _ in range(n_threads)]
        self.num_envs = n_threads
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
        obs = [env.reset(train=False, normalize=True) for env in self.env_list]
        return np.stack(obs), None

    def close(self):
        pass
