"""
train_curriculum.py

Curriculum Pretraining for Regime-Aware Cooperation in Multi-Echelon Supply Chains.
Trains a single HAPPO policy across three discrete cooperation regimes:
- Cruiser (alpha = 0.375): 70% probability
- Shock Absorber (alpha = 0.100): 15% probability
- Backlog Clearer (alpha = 0.650): 15% probability

Algorithm: HAPPO (Heterogeneous-Agent Proximal Policy Optimization)
Environment: 4-echelon serial inventory system (L=4, T=200, action cap 20)
Total Steps: 3,000,000 environment steps (3,000 updates over 5 parallel rollout threads)
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import time
import yaml
import argparse
import numpy as np
import torch
import wandb
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from runners.separated.runner import CRunner
from regime_aware.env_wrapper import RegimeAwareSubprocVecEnv

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def make_train_env(all_args):
    return RegimeAwareSubprocVecEnv(all_args)


class RegimeAwareRunner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.ep = 0
        self.best_cruiser_cost = float("inf")
        self.best_overall_cost = float("inf")
        self.cruiser_costs = []
        self.mode_counts = {"Cruiser": 0, "Shock": 0, "Backlog": 0}
        self.MAX_UPDATES = int(self.num_env_steps // self.episode_length // self.n_rollout_threads)

    def train(self):
        train_infos = super().train()
        self.ep += 1

        # Collect KPIs across the parallel rollout threads
        thread_costs = []
        thread_sls = []
        for env in self.envs.env_list:
            kpis = env.get_episode_kpis()
            alpha_val = env.alphas[0]
            per_period_cost = kpis["total_cost"] / float(self.episode_length)
            thread_costs.append(per_period_cost)
            thread_sls.append(kpis["service_level"])

            if np.isclose(alpha_val, 0.375):
                self.cruiser_costs.append(per_period_cost)
                self.mode_counts["Cruiser"] += 1
            elif np.isclose(alpha_val, 0.100):
                self.mode_counts["Shock"] += 1
            else:
                self.mode_counts["Backlog"] += 1

        mean_update_cost = float(np.mean(thread_costs))
        mean_update_sl = float(np.mean(thread_sls))
        rolling_cruiser = float(np.mean(self.cruiser_costs[-50:])) if self.cruiser_costs else mean_update_cost

        # Save best checkpoint tracking nominal Cruiser performance
        if len(self.cruiser_costs) >= 20 and rolling_cruiser < self.best_cruiser_cost:
            self.best_cruiser_cost = rolling_cruiser
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            best_path = os.path.join(CHECKPOINT_DIR, "best_regime_aware_policy.pt")
            torch.save({
                "update": self.ep,
                "best_cruiser_cost": self.best_cruiser_cost,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, best_path)

        if self.ep % 50 == 0 or self.ep == 1:
            tot = sum(self.mode_counts.values()) or 1
            c_pct = (self.mode_counts["Cruiser"] / tot) * 100.0
            s_pct = (self.mode_counts["Shock"] / tot) * 100.0
            b_pct = (self.mode_counts["Backlog"] / tot) * 100.0
            best_str = f"{self.best_cruiser_cost:.2f}" if self.best_cruiser_cost < float("inf") else "N/A"
            print(f"--- [Regime-Aware Pretraining Update {self.ep:04d}/{self.MAX_UPDATES}] ---")
            print(f"  Cruiser (0.375) Rolling Cost (last 50): {rolling_cruiser:.2f} (Best: {best_str})")
            print(f"  Current Update Mean Cost: {mean_update_cost:.2f} | Mean SL: {mean_update_sl:.3f}")
            print(f"  Curriculum Distribution: Cruiser {c_pct:.1f}% | Shock {s_pct:.1f}% | Backlog {b_pct:.1f}%")

        if self.ep >= self.MAX_UPDATES:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            final_path = os.path.join(CHECKPOINT_DIR, "final_regime_aware_policy.pt")
            torch.save({
                "update": self.ep,
                "best_cruiser_cost": self.best_cruiser_cost,
                "final_cruiser_rolling": rolling_cruiser,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, final_path)
            print(f"\nTraining complete! Target {self.MAX_UPDATES} updates reached.")
            raise StopIteration("Target 3,000,000 Environment Steps Reached")

        return train_infos


def main():
    parser = get_config()
    all_args = parser.parse_known_args(sys.argv[1:])[0]

    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)

    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8  # 7 physical + alpha
    all_args.n_rollout_threads = 5
    all_args.n_training_threads = 1
    all_args.num_env_steps = 3000000
    all_args.ppo_epoch = 15
    all_args.num_mini_batch = 1
    all_args.data_chunk_length = 10
    all_args.lr = 1e-4
    all_args.critic_lr = 1e-4
    all_args.clip_param = 0.2
    all_args.entropy_coef = 0.01
    all_args.gamma = 0.95
    all_args.gae_lambda = 0.95
    all_args.use_huber_loss = True
    all_args.huber_delta = 10.0
    all_args.use_linear_lr_decay = False
    all_args.use_eval = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(all_args.seed, list):
        all_args.seed = int(all_args.seed[0])
    else:
        all_args.seed = int(all_args.seed)

    print("=" * 80)
    print(" REGIME-AWARE COOPERATION: CURRICULUM PRETRAINING (70/15/15)")
    print(f" Device: {device}")
    print(f" Rollout Threads: {all_args.n_rollout_threads}")
    print(f" Total Environment Steps: {all_args.num_env_steps:,} (3,000 updates)")
    print("=" * 80)

    envs = make_train_env(all_args)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(os.path.dirname(__file__))
    }
    runner = RegimeAwareRunner(config)
    try:
        runner.run()
    except StopIteration as e:
        print(f"\n{str(e)}")
    envs.close()


if __name__ == "__main__":
    main()
