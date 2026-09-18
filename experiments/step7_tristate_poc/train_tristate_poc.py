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
from experiments.step7_tristate_poc.tristate_env_wrapper import (
    TriStateSubprocVecEnv,
    TriStateDummyVecEnv,
    TRISTATE_ALPHAS
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def make_train_env(all_args):
    return TriStateSubprocVecEnv(all_args)

def make_eval_env(all_args):
    return TriStateDummyVecEnv(all_args)

class TriStatePoCRunner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.alpha_history = []
        self.ep = 0
        self.best_cost_0375 = float("inf")
        
        # Max episodes for PoC
        self.MAX_EPISODES = 1200
        
        # Decay Boundaries
        self.start_lr = 1e-4
        self.end_lr = 1e-5
        
        self.start_entropy = 0.05
        self.end_entropy = 0.01

        # Force initial LR/Entropy
        self._set_lr(self.start_lr)
        self._set_entropy(self.start_entropy)

    def _set_lr(self, lr):
        for p in self.policy:
            for param_group in p.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in p.critic_optimizer.param_groups:
                param_group['lr'] = lr

    def _set_entropy(self, entropy):
        for t in self.trainer:
            t.entropy_coef = entropy

    def train(self):
        train_infos = super().train()
        self.ep += 1
        
        env = self.envs.env_list[0]
        kpis = env.get_episode_kpis()
        alphas = list(env.alphas)
        current_alpha = alphas[0]  # Since all echelons share the same alpha
        
        action_stats = []
        for i in range(self.num_agents):
            try:
                acts = self.buffer[i].actions
                if torch.is_tensor(acts):
                    acts = acts.cpu().numpy()
                mean_mag = float(np.mean(np.abs(acts)))
                var = float(np.var(acts))
            except Exception:
                mean_mag, var = 0.0, 0.0
            action_stats.append((mean_mag, var))
            
        kpis["action_stats"] = action_stats
        
        self.kpi_history.append(kpis)
        self.alpha_history.append(alphas)

        # Log to wandb
        log_dict = {
            "episode": self.ep,
            "kpi/total_cost": kpis["total_cost"],
            "kpi/service_level": kpis["service_level"],
            "kpi/bullwhip_ratio": kpis["bullwhip_ratio"],
            "alphas/active": current_alpha,
            "train/entropy": train_infos[0].get("dist_entropy", 0.0) if train_infos else 0.0,
            "train/lr": self.policy[0].actor_optimizer.param_groups[0]['lr'],
        }
        
        # Specifically track 0.375 cost for convergence check
        is_optimum_alpha = abs(current_alpha - 0.375) < 0.01
        if is_optimum_alpha:
            log_dict["kpi_0375_only/total_cost"] = kpis["total_cost"]
            log_dict["kpi_0375_only/service_level"] = kpis["service_level"]

        try:
            wandb.log(log_dict)
        except Exception:
            pass

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        if self.ep % 50 == 0 or self.ep == 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ep_{self.ep:04d}.pt")
            torch.save({
                "episode": self.ep,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, ckpt_path)

        print(f"Episode {self.ep}/{self.MAX_EPISODES} | Alpha: {current_alpha:.3f} | Cost: {kpis['total_cost']:.2f} | SL: {kpis['service_level']:.3f} | Bullwhip: {kpis['bullwhip_ratio']:.2f}")

        # --- Update Decay Schedules ---
        progress = min(1.0, self.ep / self.MAX_EPISODES)
        
        # Linear decay
        new_lr = self.start_lr - progress * (self.start_lr - self.end_lr)
        new_entropy = self.start_entropy - progress * (self.start_entropy - self.end_entropy)
        
        self._set_lr(new_lr)
        self._set_entropy(new_entropy)

        # Convergence Check based purely on 0.375 episodes
        if self.ep >= 100:
            # Get the last 50 episodes that were alpha 0.375
            recent_0375_costs = [k["total_cost"] for k, a in zip(self.kpi_history, self.alpha_history) if abs(a[0] - 0.375) < 0.01][-50:]
            
            if len(recent_0375_costs) >= 20:
                mean_0375_cost = np.mean(recent_0375_costs)
                
                # Save best 0.375 model
                if mean_0375_cost < self.best_cost_0375:
                    self.best_cost_0375 = mean_0375_cost
                    best_path = os.path.join(CHECKPOINT_DIR, "best_0375_policy.pt")
                    torch.save({
                        "episode": self.ep,
                        "best_cost_0375": self.best_cost_0375,
                        "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                        "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                        "config": vars(self.all_args)
                    }, best_path)
                
                if self.ep % 50 == 0:
                    print(f"\n--- [Episode {self.ep} Diagnostic] ---")
                    print(f"Rolling Mean 0.375 Cost (last 50): {mean_0375_cost:.2f} (Best: {self.best_cost_0375:.2f})")
                    print(f"Current LR: {new_lr:.2e} | Current Entropy: {new_entropy:.4f}\n")

        if self.ep >= self.MAX_EPISODES:
            print(f"Training reached max episode cap ({self.MAX_EPISODES}).")
            final_path = os.path.join(CHECKPOINT_DIR, "final_poc_policy.pt")
            torch.save({
                "episode": self.ep,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "config": vars(self.all_args)
            }, final_path)
            raise StopIteration("Max Episode Cap Reached")

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
    
    # 1. Load yaml config
    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    
    # GRU=256 for PoC test 2
    all_args.hidden_size = 256
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8  # [inv, backlog, demand/order_in, lead_time_orders..., alpha]
    
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
        name=f"tristate_poc_256gru_1200ep_seed_{all_args.seed}",
        tags=["step7", "tristate_poc", "256gru", f"seed_{all_args.seed}"],
        config=vars(all_args),
        mode="offline"
    )

    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.cuda and not torch.cuda.is_available():
        print("Error: --cuda was specified but no GPU is available.")
        sys.exit(1)
        
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Network: GRU hidden={all_args.hidden_size}, layers={all_args.recurrent_N}")
    print(f"PoC Starting: Tri-State interleaved training [0.375, 0.1, 0.65]")

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(CHECKPOINT_DIR)
    }

    runner = TriStatePoCRunner(config)
    
    try:
        runner.run()
    except StopIteration as e:
        print(f"Training loop finished: {e}")
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        envs.close()
        wandb.finish()

if __name__ == "__main__":
    main()
