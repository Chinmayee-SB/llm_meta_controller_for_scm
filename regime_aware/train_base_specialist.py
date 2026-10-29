"""
train_base_specialist.py

Training script for the Base Paper Specialist baseline (Liu et al., 2022/2025).
Trains HAPPO policy at fixed alpha = 0.375 with 7-D decentralized actor observations.

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
from regime_aware.base_env_wrapper import BaseSpecialistSubprocVecEnv

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def make_train_env(all_args):
    return BaseSpecialistSubprocVecEnv(all_args)


class BaseSpecialistRunner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.ep = 0
        self.best_cost = float("inf")
        self.all_costs = []
        self.MAX_UPDATES = int(self.num_env_steps // self.episode_length // self.n_rollout_threads)

    def train(self):
        train_infos = super().train()
        self.ep += 1

        thread_costs = []
        thread_sls = []
        for env in self.envs.env_list:
            kpis = env.get_episode_kpis()
            per_period_cost = kpis["total_cost"] / float(self.episode_length)
            thread_costs.append(per_period_cost)
            thread_sls.append(kpis["service_level"])
            self.all_costs.append(per_period_cost)

        mean_update_cost = float(np.mean(thread_costs))
        mean_update_sl = float(np.mean(thread_sls))
        rolling_cost = float(np.mean(self.all_costs[-250:])) if len(self.all_costs) >= 50 else mean_update_cost

        if len(self.all_costs) >= 20 and rolling_cost < self.best_cost:
            self.best_cost = rolling_cost
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            best_path = os.path.join(CHECKPOINT_DIR, "best_base_paper_policy.pt")
            torch.save({
                "update": self.ep,
                "best_cost": self.best_cost,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, best_path)

        if self.ep % 50 == 0 or self.ep == 1:
            best_str = f"{self.best_cost:.2f}" if self.best_cost < float("inf") else "N/A"
            print(f"--- [Base Specialist Update {self.ep:04d}/{self.MAX_UPDATES}] ---")
            print(f"  Rolling Cost: {rolling_cost:.2f} (Best: {best_str}) | Mean SL: {mean_update_sl:.3f}")

        if self.ep >= self.MAX_UPDATES:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            final_path = os.path.join(CHECKPOINT_DIR, "final_base_paper_policy.pt")
            torch.save({
                "update": self.ep,
                "best_cost": self.best_cost,
                "final_rolling_cost": rolling_cost,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, final_path)
            print(f"\nBase Specialist Training complete! Target {self.MAX_UPDATES} updates reached.")
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
    all_args.obs_dim = 7  # strictly physical state, no alpha
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
    print(" BASE PAPER SPECIALIST TRAINING (alpha = 0.375 fixed)")
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
    runner = BaseSpecialistRunner(config)
    try:
        runner.run()
    except StopIteration as e:
        print(f"\n{str(e)}")
    envs.close()


if __name__ == "__main__":
    main()
