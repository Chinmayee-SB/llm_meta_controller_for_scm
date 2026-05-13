import pytest
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.serial import Env
from algorithms.actor_critic import Actor, Critic
from config import get_config
from gym import spaces

@pytest.fixture
def args():
    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.n_rollout_threads = 1
    # Ensure RNN/GRU and hidden sizes match spec
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.use_naive_recurrent_policy = True
    all_args.use_recurrent_policy = False
    all_args.use_feature_normalization = False
    all_args.use_orthogonal = True
    all_args.gain = 0.01
    all_args.use_policy_active_masks = False
    all_args.use_ReLU = True
    all_args.stacked_frames = 1
    all_args.layer_N = 2
    all_args.algorithm_name = "happo"
    return all_args

@pytest.fixture
def env(args):
    e = Env(args)
    e.train = True
    yield e

class TestEnvironment:
    def test_agent_count(self, env):
        obs = env.reset()
        assert len(obs) == 4

    def test_actor_obs_dim(self, env):
        obs = env.reset()
        for i, o in enumerate(obs):
            assert len(o) == 8, f"Agent {i}: expected dim=8, got {len(o)}"

    def test_alpha_is_last_obs_element(self, env):
        obs = env.reset()
        for i, o in enumerate(obs):
            assert abs(float(o[-1]) - env.alphas[i]) < 1e-6

    def test_alphas_per_agent_independent(self, env):
        env.reset()
        a1 = env.alphas[:]
        env.reset()
        a2 = env.alphas[:]
        # Very unlikely all 4 are identical across two resets
        assert a1 != a2 or not all(x == a1[0] for x in a1)

    def test_alpha_range(self, env):
        for _ in range(20):
            env.reset()
            for a in env.alphas:
                assert 0.0 <= a <= 1.0

    def test_critic_obs_excludes_alpha(self, env):
        obs = env.reset()
        critic_obs = np.concatenate([o[:-1] for o in obs])
        assert critic_obs.shape == (28,), f"Expected (28,), got {critic_obs.shape}"

    def test_set_alphas(self, env):
        env.reset()
        env.set_alphas([0.1, 0.3, 0.6, 0.9])
        assert env.alphas == [0.1, 0.3, 0.6, 0.9]

    def test_backlog_nonnegative(self, env):
        env.reset()
        for _ in range(200):
            actions = [np.random.randint(0, 21) for _ in range(4)]
            env.step(actions, one_hot=False)
        for b in env.backlog:
            assert b >= 0

    def test_pipeline_length(self, env):
        env.reset()
        for pipeline in env.order:
            assert len(pipeline) == 4, "Pipeline length must equal LEAD_TIME=4"

    def test_kpi_keys(self, env):
        env.reset()
        for _ in range(200):
            env.step([10, 10, 10, 10], one_hot=False)
        kpis = env.get_episode_kpis()
        required = {"total_cost","per_agent_costs","service_level","bullwhip_ratio","reward_variance"}
        assert required.issubset(kpis.keys()), f"Missing: {required - set(kpis.keys())}"

    def test_service_level_range(self, env):
        env.reset()
        for _ in range(200):
            env.step([10, 10, 10, 10], one_hot=False)
        assert 0.0 <= env.get_episode_kpis()["service_level"] <= 1.0

    def test_bullwhip_positive(self, env):
        env.reset()
        for _ in range(200):
            env.step([np.random.randint(0,21) for _ in range(4)], one_hot=False)
        assert env.get_episode_kpis()["bullwhip_ratio"] > 0

    def test_total_cost_positive(self, env):
        env.reset()
        for _ in range(200):
            env.step([10, 10, 10, 10], one_hot=False)
        assert env.get_episode_kpis()["total_cost"] > 0


class TestNetworks:
    def test_actor_output_is_distribution(self, args):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        act_space = spaces.Discrete(21)
        actor = Actor(args, obs_space, act_space)
        obs = torch.randn(1, 8)
        h0  = torch.zeros(1, 2, 128)
        masks = torch.ones(1, 1)
        
        actor_features = actor.base(obs)
        actor_features, h_new = actor.rnn(actor_features, h0, masks)
        probs = actor.act.get_probs(actor_features)
        
        assert abs(probs.sum().item() - 1.0) < 1e-4
        assert probs.shape[-1] == 21

    def test_critic_output_scalar(self, args):
        cent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,))
        critic = Critic(args, cent_obs_space)
        g_obs = torch.randn(1, 28)
        h0    = torch.zeros(1, 2, 128)
        masks = torch.ones(1, 1)
        val, _ = critic(g_obs, h0, masks)
        assert val.shape == (1, 1)

    def test_gru_hidden_shape(self, args):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        act_space = spaces.Discrete(21)
        actor = Actor(args, obs_space, act_space)
        obs   = torch.randn(1, 8)
        h0    = torch.zeros(1, 2, 128)
        masks = torch.ones(1, 1)
        _, _, h_new = actor(obs, h0, masks)
        assert h_new.shape == (1, 2, 128), "GRU must be 2-layer, hidden=128"
