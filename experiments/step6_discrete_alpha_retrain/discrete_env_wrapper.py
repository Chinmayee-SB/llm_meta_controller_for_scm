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

DISCRETE_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class DiscreteSerialSupplyChainEnv(BaseSerialEnv):
    """
    Subclass of SerialSupplyChainEnvironment that strictly samples alpha values
    from the discrete set {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}.
    """
    def __init__(self, all_args):
        super().__init__(all_args)
        self.discrete_alphas = list(DISCRETE_ALPHAS)
        self.alpha_randomized = True

    def reset(self, train=True, normalize=True, fixed_alphas=None):
        sub_agent_obs = super().reset(train=train, normalize=normalize, fixed_alphas=fixed_alphas)
        
        # Override continuous uniform alpha with discrete choice if randomized
        if fixed_alphas is not None:
            self.set_alphas(fixed_alphas)
        elif getattr(self, 'alpha_randomized', True):
            self.alphas = [float(np.random.choice(self.discrete_alphas)) for _ in range(self.level_num)]
        
        # Re-compute initial observation with discrete alphas
        sub_agent_obs = self.get_reset_obs(normalize)
        return sub_agent_obs

    def set_alphas(self, alphas_list):
        assert len(alphas_list) == self.level_num, f"Expected {self.level_num} alphas, got {len(alphas_list)}"
        self.alphas = [float(a) for a in alphas_list]
        self.alpha_randomized = False

    def update_alphas_mid_episode(self, alphas_list):
        """Allows dynamic mid-episode regime switching without resetting state."""
        assert len(alphas_list) == self.level_num, f"Expected {self.level_num} alphas, got {len(alphas_list)}"
        self.alphas = [float(a) for a in alphas_list]


class DiscreteSubprocVecEnv(object):
    """
    Vectorized environment wrapper for multi-threaded rollout of DiscreteSerialSupplyChainEnv.
    """
    def __init__(self, all_args):
        self.env_list = [DiscreteSerialSupplyChainEnv(all_args) for _ in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            share_obs_dim += (self.signal_obs_dim - 1)
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

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


class DiscreteDummyVecEnv(object):
    """
    Single-threaded dummy wrapper for evaluation.
    """
    def __init__(self, all_args):
        self.env_list = [DiscreteSerialSupplyChainEnv(all_args) for _ in range(all_args.n_eval_rollout_threads)]
        self.num_envs = all_args.n_eval_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            share_obs_dim += (self.signal_obs_dim - 1)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = [env.reset() for env in self.env_list]
        return np.stack(obs), None

    def close(self):
        pass
