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
from experiments.step6_discrete_alpha_retrain.discrete_env_wrapper import (
    DiscreteSubprocVecEnv,
    DiscreteDummyVecEnv,
    DISCRETE_ALPHAS
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def make_train_env(all_args):
    return DiscreteSubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DiscreteDummyVecEnv(all_args)

class DiscretePhase1Runner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.alpha_history = []
        self.ep = 0
        self.converged = False
        self.best_cost = float("inf")
        self.best_ep = 0

    def train(self):
        train_infos = super().train()
        self.ep += 1
        
        env = self.envs.env_list[0]
        kpis = env.get_episode_kpis()
        alphas = list(env.alphas)
        
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

        try:
            wandb.log({
                "episode": self.ep,
                "kpi/total_cost": kpis["total_cost"],
                "kpi/service_level": kpis["service_level"],
                "kpi/bullwhip_ratio": kpis["bullwhip_ratio"],
                "kpi/reward_variance": kpis["reward_variance"],
                "costs/retailer": kpis["per_agent_costs"]["retailer"],
                "costs/warehouse": kpis["per_agent_costs"]["warehouse"],
                "costs/distributor": kpis["per_agent_costs"]["distributor"],
                "costs/manufacturer": kpis["per_agent_costs"]["manufacturer"],
                "alphas/retailer": alphas[0],
                "alphas/warehouse": alphas[1],
                "alphas/distributor": alphas[2],
                "alphas/manufacturer": alphas[3],
                "train/entropy": train_infos[0].get("dist_entropy", 0.0) if train_infos else 0.0,
                "train/grad_norm": train_infos[0].get("actor_grad_norm", 0.0) if train_infos else 0.0,
            })
        except Exception:
            pass

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        if self.ep % 50 == 0 or self.ep == 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ep_{self.ep:04d}.pt")
            torch.save({
                "episode": self.ep,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "kpi_history": self.kpi_history,
                "alpha_history": self.alpha_history,
                "config": vars(self.all_args)
            }, ckpt_path)

        total_episodes = int(self.num_env_steps // self.episode_length // self.n_rollout_threads)
        print(f"Episode {self.ep}/{total_episodes} | Cost: {kpis['total_cost']:.2f} | SL: {kpis['service_level']:.3f} | Bullwhip: {kpis['bullwhip_ratio']:.2f} | alphas: [{alphas[0]:.1f}, {alphas[1]:.1f}, {alphas[2]:.1f}, {alphas[3]:.1f}]")

        # PPO Epoch reduction schedule
        original_ppo_epoch = self.all_args.ppo_epoch
        new_ppo_epoch = original_ppo_epoch
        if self.ep > 2200:
            new_ppo_epoch = 4
        elif self.ep > 1500:
            new_ppo_epoch = 6
        elif self.ep > 800:
            new_ppo_epoch = max(10, original_ppo_epoch // 2)
            
        if self.trainer[0].ppo_epoch != new_ppo_epoch:
            for t in self.trainer:
                t.ppo_epoch = new_ppo_epoch
            print(f"Episode {self.ep}: PPO epochs reduced to {new_ppo_epoch}")

        # Entropy decay
        for t in self.trainer:
            t.entropy_coef = max(0.001, t.entropy_coef * 0.999)

        # LR decay
        for p in self.policy:
            for param_group in p.actor_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.999)
            for param_group in p.critic_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.999)

        # Convergence and Best Model Check
        if self.ep >= 100 and not self.converged:
            recent_kpis = self.kpi_history[-100:]
            recent_costs = [k["total_cost"] for k in recent_kpis]
            recent_sls   = [k["service_level"] for k in recent_kpis]
            
            mean_cost = np.mean(recent_costs)
            std_cost = np.std(recent_costs)
            ratio = std_cost / mean_cost if mean_cost > 0 else 0
            
            mean_sl = np.mean(recent_sls)
            min_sl  = np.min(recent_sls)
            catastrophic_pct = sum(1 for c in recent_costs if c > 40000) / len(recent_costs)
            
            # Save best checkpoint
            if mean_cost < self.best_cost and mean_sl > 0.60 and catastrophic_pct == 0:
                self.best_cost = mean_cost
                self.best_ep = self.ep
                best_path = os.path.join(CHECKPOINT_DIR, "best_policy.pt")
                torch.save({
                    "episode": self.ep,
                    "best_cost": self.best_cost,
                    "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                    "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                    "config": vars(self.all_args)
                }, best_path)
                print(f"[BEST MODEL SAVED] Ep {self.ep}: Rolling Mean Cost = {mean_cost:.2f}, SL = {mean_sl:.3f}")

            target_ratio = 0.15
            is_converged = (
                mean_cost > 0 and 
                ratio < target_ratio and 
                catastrophic_pct < 0.05 and 
                mean_sl > 0.65 and 
                min_sl > 0.20 and
                self.ep >= 1500
            )

            if self.ep % 100 == 0:
                current_lr = self.policy[0].actor_optimizer.param_groups[0]['lr']
                print(f"\n--- [Episode {self.ep} Diagnostic] ---")
                print(f"Rolling Mean Cost (100 ep): {mean_cost:.2f} | Std/Mean: {ratio:.4f}")
                print(f"Rolling Mean SL: {mean_sl:.3f} | Min SL: {min_sl:.3f} | Catastrophic %: {catastrophic_pct*100:.1f}%")
                print(f"LR: {current_lr:.2e} | Entropy: {self.trainer[0].entropy_coef:.5f} | Best Cost So Far: {self.best_cost:.2f} (ep {self.best_ep})")
                print(f"Converged: {is_converged}\n")
            
            if is_converged:
                self.converged = True
                print(f"[CONVERGENCE ACHIEVED] Training stopped at episode {self.ep}.")
                # Save final converged checkpoint
                converged_path = os.path.join(CHECKPOINT_DIR, "converged_policy.pt")
                torch.save({
                    "episode": self.ep,
                    "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                    "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                    "config": vars(self.all_args)
                }, converged_path)
                raise StopIteration("Phase 1 Discrete Converged")

        if self.ep >= 2500:
            print("Training reached max episode cap (2500).")
            # Save final checkpoint as fallback
            final_path = os.path.join(CHECKPOINT_DIR, "final_cap_policy.pt")
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
    
    # UPGRADED ARCHITECTURE: 256 Hidden Units, 2 GRU layers
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
        name=f"discrete_256gru_seed_{all_args.seed}",
        tags=["step6", "discrete_alpha", "256gru", f"seed_{all_args.seed}"],
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

    runner = DiscretePhase1Runner(config)
    
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
