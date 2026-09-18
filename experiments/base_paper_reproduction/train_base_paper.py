import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import time
import yaml
import numpy as np
import torch
import wandb
from pathlib import Path

# Add repo root and MADRL to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from runners.separated.runner import CRunner
from experiments.base_paper_reproduction.base_env_wrapper import BasePaperSubprocVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def make_train_env(all_args):
    return BasePaperSubprocVecEnv(all_args)

class BasePaperRunner(CRunner):
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

        # Collect KPIs across the parallel rollout threads
        thread_costs = []
        thread_sls = []
        thread_bws = []
        for env in self.envs.env_list:
            kpis = env.get_episode_kpis()
            per_period_cost = kpis["total_cost"] / self.episode_length
            thread_costs.append(per_period_cost)
            thread_sls.append(kpis["service_level"])
            thread_bws.append(kpis["bullwhip_ratio"])
            self.all_costs.append(per_period_cost)

        mean_update_cost = float(np.mean(thread_costs))
        mean_update_sl = float(np.mean(thread_sls))
        mean_update_bw = float(np.mean(thread_bws))

        # Rolling cost over the last 50 updates (250 episode rollouts)
        rolling_cost = float(np.mean(self.all_costs[-250:])) if len(self.all_costs) >= 50 else mean_update_cost

        # Save best checkpoint
        if self.ep >= 20 and rolling_cost < self.best_cost:
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

        # Periodic checkpoint every 500 updates
        if self.ep % 500 == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            period_path = os.path.join(CHECKPOINT_DIR, f"base_paper_update_{self.ep:04d}.pt")
            torch.save({
                "update": self.ep,
                "rolling_cost": rolling_cost,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, period_path)

        # Diagnostic progress print every 50 updates
        if self.ep % 50 == 0 or self.ep == 1 or self.ep == self.MAX_UPDATES:
            best_str = f"{self.best_cost:.2f}" if self.best_cost < float("inf") else "N/A"
            print(f"--- [Base Paper Update {self.ep:04d}/{self.MAX_UPDATES}] ---")
            print(f"  Rolling Per-Period Cost (last 50 updates): {rolling_cost:.2f} (Best: {best_str})")
            print(f"  Current Update Cost: {mean_update_cost:.2f} | Mean SL: {mean_update_sl:.3f} | Bullwhip: {mean_update_bw:.2f}")

        # Wandb offline log
        try:
            wandb.log({
                "update": self.ep,
                "cost/rolling_per_period": rolling_cost,
                "cost/update_mean": mean_update_cost,
                "kpi/service_level": mean_update_sl,
                "kpi/bullwhip_ratio": mean_update_bw,
                "train/entropy": train_infos[0].get("dist_entropy", 0.0) if train_infos else 0.0,
                "train/actor_grad_norm": train_infos[0].get("actor_grad_norm", 0.0) if train_infos else 0.0,
            })
        except Exception:
            pass

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
            print(f"\n==================================================================")
            print(f"Base Paper Training complete! Target {self.MAX_UPDATES} updates reached.")
            print(f"Final model saved to {final_path}")
            print(f"Best model saved to {os.path.join(CHECKPOINT_DIR, 'best_base_paper_policy.pt')}")
            print(f"==================================================================\n")
            raise StopIteration("Target 3,000,000 Environment Steps Reached")

        return train_infos

def parse_args():
    parser = get_config()
    parser.add_argument("--smoke_test", action="store_true", default=False)
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args

def main():
    all_args = parse_args()

    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)

    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']

    # EXACT BASE PAPER ARCHITECTURE
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2

    # CRUCIAL: Actor sees strictly 7 dimensions (NO alpha)
    all_args.obs_dim = 7

    # OFFICIAL BASE PAPER HYPERPARAMETERS
    all_args.n_rollout_threads = 5
    all_args.n_training_threads = 1
    
    if all_args.smoke_test:
        all_args.num_env_steps = 5000  # 5 updates
        os.environ["WANDB_MODE"] = "offline"
    else:
        if not hasattr(all_args, "num_env_steps") or all_args.num_env_steps == 10000000:
            all_args.num_env_steps = 3000000  # 3M steps = 3000 updates

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

    device = torch.device("cuda:0" if torch.cuda.is_available() and all_args.cuda else "cpu")

    if isinstance(all_args.seed, list):
        all_args.seed = int(all_args.seed[0])
    else:
        all_args.seed = int(all_args.seed)

    print("==================================================================")
    print(" OFFICIAL BASE PAPER SPECIALIST REPRODUCTION TRAINING")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Rollout Threads: {all_args.n_rollout_threads} parallel environments")
    print(f" Total Environment Steps: {all_args.num_env_steps:,} ({all_args.num_env_steps // (all_args.episode_length * all_args.n_rollout_threads)} updates)")
    print(f" Actor Obs Dim: {all_args.obs_dim} (strictly physical, NO alpha) | Critic Dim: 28")
    print(f" Fixed Alpha: 0.375 across all agents and episodes")
    print(f" PPO Epoch: {all_args.ppo_epoch} | Chunk Length: {all_args.data_chunk_length} | LR: {all_args.lr}")
    print("==================================================================")

    wandb.init(
        project="SCM_Episodic_Meta_LLM",
        name=f"base_paper_repro_5threads_seed_{all_args.seed}",
        tags=["base_paper", "official_repro", "parallel_5threads", "fixed_0375"],
        config=vars(all_args),
        mode="offline"
    )

    envs = make_train_env(all_args)
    eval_envs = None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(os.path.dirname(__file__))
    }

    runner = BasePaperRunner(config)

    # Enforce base paper entropy coefficient
    for t in runner.trainer:
        t.entropy_coef = 0.01

    try:
        runner.run()
    except StopIteration as e:
        print(f"\nTraining completed normally: {str(e)}")

    envs.close()
    wandb.finish()

if __name__ == "__main__":
    main()
