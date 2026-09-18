import os
import sys
import random
import numpy as np

# Add project root and MADRL to sys.path if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from MADRL.envs import generator

# Exogenous parameters matching base serial environment
H = [1, 1, 1, 1]
B = [1, 1, 1, 1]
S_I = 10
S_O = 10
LEAD_TIME = 4
LEVEL_NUM = 4
ACTION_DIM = 21
OBS_DIM = 8
EPISODE_LEN = 200

MERTON_DEMAND = True
EVAL_PTH = os.path.join(MADRL_DIR, "test_data", "test_demand_merton") + "/"

RANDOM_SHIPPING_LOSS = False
LOST_RATE = 0.1
PRICE_DISCOUNT = False
DISCOUNT = [1, 0.95, 0.9, 0.85, 0.8]
C = [1, 1, 1, 1, 1]
FIXED_COST = False
F_C = 5

def get_eval_data():
    eval_files = os.listdir(EVAL_PTH)
    eval_data = []
    n_eval = len(eval_files)
    for file in sorted(eval_files):
        data = []
        with open(os.path.join(EVAL_PTH, file), "rb") as f:
            lines = f.readlines()
            for line in lines:
                data.append(int(line))
        eval_data.append(data)
    return n_eval, eval_data

def get_training_data():
    demand_gen = generator.merton(EPISODE_LEN, ACTION_DIM - 1, mu=5e-5, sigma=1e-2, lamda=15, l1=10, l2=0, l3=0.01)
    demand_list = [int(max(1, min(40, d))) for d in demand_gen.demand_list]
    return demand_list

class DynamicSerialEnv(object):
    """
    4-Echelon Serial Supply Chain Environment with Mid-Episode Dynamic Alpha Switching.
    
    Key enhancement over Phase 1:
      During training, alpha_i shifts at configured intra-episodic intervals (default N=50),
      forcing the recurrent policy to learn smooth adaptations across shift boundaries.
    """
    def __init__(self, all_args=None):
        self.all_args = all_args
        self.agent_num = all_args.num_agents if all_args else LEVEL_NUM
        self.obs_dim = all_args.obs_dim if all_args else OBS_DIM
        self.action_dim = ACTION_DIM
        self.level_num = self.agent_num

        self.inventory = []
        self.backlog = []
        self.order = []
        self.record_act_sta = [[] for _ in range(self.level_num)]
        self.eval_episode_len = EPISODE_LEN
        self.episode_max_steps = EPISODE_LEN

        # Dynamic Alpha configuration
        self.dynamic_alpha_enabled = True
        self.dynamic_alpha_interval = 50  # Steps between alpha shifts (e.g. t=50, 100, 150)
        self._alpha_low = 0.1
        self._alpha_high = 0.9

        self.n_eval, self.eval_data = get_eval_data()
        self.eval_index = 0
        self.eval_bw_res = [0.0] * self.level_num
        self.current_orders = [0] * self.level_num

    def reset(self, train=True, normalize=True, fixed_alphas=None):
        self.step_num = 0
        self.backlog = [0 for _ in range(self.level_num)]
        self.action_history = [[] for _ in range(self.level_num)]
        self.train = train
        self.normalize = normalize
        self.inventory = [S_I for _ in range(self.level_num)]
        self.order = [[S_O for _ in range(LEAD_TIME)] for _ in range(self.level_num)]

        # Initial Alpha assignment at t=0
        if fixed_alphas is not None:
            self.alphas = list(fixed_alphas)
            self.alpha_randomized = False
        elif getattr(self, 'alpha_randomized', True):
            self.alphas = [float(np.random.uniform(self._alpha_low, self._alpha_high))
                           for _ in range(self.level_num)]
        else:
            pass
        self.alpha_randomized = True

        # Track full alpha history across timesteps
        self.alpha_step_history = [list(self.alphas)]

        # KPI tracking
        self._kpi_total_cost = 0.0
        self._kpi_agent_costs = [0.0] * self.level_num
        self._kpi_fulfilled = 0.0
        self._kpi_demanded = 0.0
        self._kpi_action_history = [[] for _ in range(self.level_num)]
        self._kpi_step_rewards = []

        if train:
            self.demand_list = get_training_data()
        else:
            self.demand_list = self.eval_data[self.eval_index]
            self.eval_index = (self.eval_index + 1) % self.n_eval

        sub_agent_obs = self.get_reset_obs(normalize)
        return sub_agent_obs

    def set_alphas(self, alphas_list):
        assert len(alphas_list) == self.level_num, f"Expected {self.level_num} alphas, got {len(alphas_list)}"
        self.alphas = list(alphas_list)
        self.alpha_randomized = False

    def set_alpha_range(self, low, high):
        self._alpha_low = max(0.1, float(low))
        self._alpha_high = min(0.9, float(high))

    def step(self, actions, one_hot=True):
        if one_hot:
            action_ = [np.argmax(i) for i in actions]
        else:
            action_ = actions

        action = self.action_map(action_)
        reward = self.state_update(action)

        # Dynamic Alpha Transition Logic:
        # If in training mode and dynamic alpha is enabled, switch alpha at step intervals (e.g., t=50, 100, 150)
        if self.train and self.dynamic_alpha_enabled:
            if self.step_num < self.episode_max_steps and self.step_num % self.dynamic_alpha_interval == 0:
                self.alphas = [float(np.random.uniform(self._alpha_low, self._alpha_high))
                               for _ in range(self.level_num)]

        self.alpha_step_history.append(list(self.alphas))

        sub_agent_obs = self.get_step_obs(action)
        sub_agent_reward = self.get_processed_rewards(reward)

        if self.step_num > self.episode_max_steps:
            sub_agent_done = [True for _ in range(self.agent_num)]
        else:
            sub_agent_done = [False for _ in range(self.agent_num)]
        sub_agent_info = [[] for _ in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def get_eval_num(self):
        return self.n_eval

    def get_eval_bw_res(self):
        return self.eval_bw_res

    def get_demand(self):
        return [self.demand_list[self.step_num - 1]]

    def get_orders(self):
        return self.current_orders

    def get_inventory(self):
        return self.inventory

    def action_map(self, action):
        self.current_orders = action
        return [i for i in action]

    def get_reset_obs(self, normalize):
        sub_agent_obs = []
        for i in range(self.level_num):
            if normalize:
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i], dtype=np.float32) / (ACTION_DIM - 1)
                if self.obs_dim == 8:
                    arr = np.append(arr, self.alphas[i])
            else:
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i], dtype=np.float32)
                if self.obs_dim == 8:
                    arr = np.append(arr, self.alphas[i])
            arr = np.reshape(arr, (self.obs_dim,))
            sub_agent_obs.append(arr)
        return sub_agent_obs

    def get_step_obs(self, action):
        sub_agent_obs = []
        if self.normalize:
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0], dtype=np.float32) / (ACTION_DIM - 1)
            if self.obs_dim == 8:
                arr = np.append(arr, self.alphas[0])
        else:
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0], dtype=np.float32)
            if self.obs_dim == 8:
                arr = np.append(arr, self.alphas[0])
        arr = np.reshape(arr, (self.obs_dim,))
        sub_agent_obs.append(arr)

        for i in range(1, self.level_num):
            if self.normalize:
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i], dtype=np.float32) / (ACTION_DIM - 1)
                if self.obs_dim == 8:
                    arr = np.append(arr, self.alphas[i])
            else:
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i], dtype=np.float32)
                if self.obs_dim == 8:
                    arr = np.append(arr, self.alphas[i])
            arr = np.reshape(arr, (self.obs_dim,))
            sub_agent_obs.append(arr)

        return sub_agent_obs

    def get_processed_rewards(self, reward):
        processed = []
        if self.train:
            total = sum(reward)  # reward[i] is raw negative cost
            for i in range(self.level_num):
                a = self.alphas[i]
                r = a * reward[i] + (1 - a) * (total / self.level_num)
                processed.append([r])
        else:
            processed = [[r] for r in reward]
        return processed

    def state_update(self, action):
        cur_demand_1 = [self.demand_list[self.step_num]] + action[:-1]
        cur_demand = [cur_demand_1[i] + self.backlog[i] for i in range(self.level_num)]

        lost_rate = [1 for _ in range(self.level_num)]
        self.step_num += 1
        rewards = []

        for i in range(self.level_num):
            unmet = - self.inventory[i] - int(self.order[i][0] * lost_rate[i]) + cur_demand[i]
            self.inventory[i] = np.max([-unmet, 0])
            self.backlog[i] = np.max([unmet, 0])

            if i == self.level_num - 1:
                self.order[i].append(action[i])
            else:
                self.order[i].append(np.min([cur_demand[i+1], self.inventory[i+1] + int(self.order[i+1][0] * lost_rate[i+1])]))
            
            actual_order = self.order[i][-1]
            self.action_history[i].append(action[i])
            self.order[i] = self.order[i][1:]

            ordering_cost = 0
            fixed_cost = 0
            reward = - self.inventory[i] * H[i] - self.backlog[i] * B[i] - ordering_cost - fixed_cost
            rewards.append(reward)

        self._kpi_step_rewards.append(rewards)

        for i in range(self.level_num):
            c_i = H[i] * self.inventory[i] + B[i] * self.backlog[i]
            self._kpi_agent_costs[i] += c_i
            self._kpi_total_cost += c_i

        for i in range(self.level_num):
            self._kpi_action_history[i].append(action[i])

        t = self.step_num - 1
        demanded_t = cur_demand[0]
        fulfilled_t = demanded_t - self.backlog[0]
        self._kpi_demanded += max(demanded_t, 1)
        self._kpi_fulfilled += min(fulfilled_t, demanded_t)

        if not self.train and self.step_num == self.eval_episode_len - 1:
            for k in range(self.level_num):
                if np.mean(self.action_history[k]) < 1e-6:
                    self.record_act_sta[k].append(0)
                else:
                    self.record_act_sta[k].append(np.std(self.action_history[k]) / max(np.mean(self.action_history[k]), 1e-6))

        if not self.train and self.eval_index == 0 and self.step_num == self.eval_episode_len - 1:
            self.eval_bw_res = []
            for i in range(self.level_num):
                self.eval_bw_res.append(float(np.mean(self.record_act_sta[i])))
            self.record_act_sta = [[] for _ in range(self.level_num)]

        return rewards

    def get_episode_kpis(self):
        sl = self._kpi_fulfilled / max(self._kpi_demanded, 1.0)
        bullwhip = self._compute_bullwhip()
        rv = self._compute_reward_variance()
        return {
            "total_cost": self._kpi_total_cost,
            "per_agent_costs": dict(zip(
                ["retailer", "warehouse", "distributor", "manufacturer"],
                self._kpi_agent_costs)),
            "service_level": sl,
            "bullwhip_ratio": bullwhip,
            "reward_variance": rv
        }

    def _compute_bullwhip(self):
        if len(self._kpi_action_history[0]) < 2:
            return 1.0
        cust = np.array(self.demand_list[:self.step_num], dtype=float)
        mfr = np.array(self._kpi_action_history[self.level_num - 1], dtype=float)
        cv_c = np.std(cust) / max(np.mean(cust), 1e-6)
        cv_m = np.std(mfr) / max(np.mean(mfr), 1e-6)
        return float(cv_m / max(cv_c, 1e-6))

    def _compute_reward_variance(self):
        if not self._kpi_step_rewards:
            return 0.0
        totals = [sum(r[i] for r in self._kpi_step_rewards) for i in range(self.level_num)]
        return float(np.var(totals))
