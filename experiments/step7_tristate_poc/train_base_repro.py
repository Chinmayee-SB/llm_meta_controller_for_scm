import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["WANDB_MODE"] = "offline"

import argparse
import yaml
import numpy as np
import torch
import wandb
from pathlib import Path

# Ensure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from runners.separated.runner import CRunner
from experiments.step7_tristate_poc.base_repro_env_wrapper import BasePaperReproSubprocVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_base_repro")

def make_train_env(all_args):
    return BasePaperReproSubprocVecEnv(all_args)

class BaseReproRunner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.ep = 0
        self.best_cost = float("inf")
        self.MAX_EPISODES = 360  # Base paper specifies 327 ± 35 episodes

    def train(self):
        train_infos = super().train()
        self.ep += 1
        
        env = self.envs.env_list[0]
        kpis = env.get_episode_kpis()
        self.kpi_history.append(kpis)

        # Log to wandb
        log_dict = {
            "episode": self.ep,
            "kpi/total_cost": kpis["total_cost"],
            "kpi/per_period_cost": kpis["total_cost"] / self.episode_length,
            "kpi/service_level": kpis["service_level"],
            "kpi/bullwhip_ratio": kpis["bullwhip_ratio"],
            "train/entropy": train_infos[0].get("dist_entropy", 0.0) if train_infos else 0.0,
            "train/lr": self.policy[0].actor_optimizer.param_groups[0]['lr'],
        }

        try:
            wandb.log(log_dict)
        except Exception:
            pass

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        per_period_c = kpis['total_cost'] / self.episode_length
        print(f"Episode {self.ep}/{self.MAX_EPISODES} | Alpha: 0.375 Fixed | Cost: {kpis['total_cost']:.1f} (Per Period: {per_period_c:.2f}) | SL: {kpis['service_level']:.3f}")

        # Rolling cost check & best checkpoint
        if self.ep >= 50:
            recent_costs = [k["total_cost"] / self.episode_length for k in self.kpi_history[-30:]]
            mean_cost = np.mean(recent_costs)
            
            if mean_cost < self.best_cost:
                self.best_cost = mean_cost
                best_path = os.path.join(CHECKPOINT_DIR, "best_base_repro_policy.pt")
                torch.save({
                    "episode": self.ep,
                    "best_cost": self.best_cost,
                    "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                    "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                    "config": vars(self.all_args)
                }, best_path)
            
            if self.ep % 25 == 0:
                print(f"\n--- [Episode {self.ep} Diagnostic] ---")
                print(f"Rolling Per-Period Cost (last 30): {mean_cost:.2f} (Best: {self.best_cost:.2f})")

        if self.ep >= self.MAX_EPISODES:
            final_path = os.path.join(CHECKPOINT_DIR, "final_base_repro_policy.pt")
            torch.save({
                "episode": self.ep,
                "best_cost": self.best_cost,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, final_path)
            print(f"\nTraining successfully reached target of {self.MAX_EPISODES} episodes. Final checkpoint saved to {final_path}.")
            raise StopIteration("Target 360 Episodes Reached")

        return train_infos

def parse_args():
    parser = get_config()
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--n_episodes", type=int, default=5)
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args

def main():
    all_args = parse_args()
    
    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    
    # EXACT BASE PAPER HYPERPARAMETERS
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    
    # CRUCIAL: Remove Alpha from observation state (7 instead of 8)
    all_args.obs_dim = 7
    
    all_args.ppo_epoch = 15
    # The default_config.yaml might override some, but we rely on arg parser defaults for lr
    # the arg parser defaults in config.py already have lr=1e-4, epsilon=0.2
    
    all_args.use_eval = False

    if all_args.smoke_test:
        all_args.num_env_steps = all_args.n_episodes * all_args.episode_length * all_args.n_rollout_threads
        os.environ["WANDB_MODE"] = "offline"

    if isinstance(all_args.seed, list):
        all_args.seed = int(all_args.seed[0])
    else:
        all_args.seed = int(all_args.seed)

    wandb.init(
        project="SCM_Episodic_Meta_LLM",
        name=f"base_repro_128gru_seed_{all_args.seed}",
        tags=["step7", "base_repro"],
        config=vars(all_args),
        mode="offline"
    )

    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
        
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Base Paper Repro (Fixed 0.375, No Alpha in OBS)")

    envs = make_train_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(CHECKPOINT_DIR)
    }

    runner = BaseReproRunner(config)
    
    # Force base paper entropy
    for t in runner.trainer:
        t.entropy_coef = 0.01
        
    try:
        runner.run()
    except StopIteration as e:
        print(f"Training loop finished: {e}")
    finally:
        envs.close()
        wandb.finish()

if __name__ == "__main__":
    main()
