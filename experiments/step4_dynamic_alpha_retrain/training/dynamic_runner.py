import os
import sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import torch
from pathlib import Path

# Add project root and MADRL to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from MADRL.runners.separated.runner import CRunner

class DynamicPhase1Runner(CRunner):
    """
    HAPPO Training Runner with dynamic mid-episode alpha tracking,
    recurrent memory preservation across transitions, action variance tracking,
    and adaptive learning rate / entropy cooling schedules.
    """
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.alpha_history = []
        self.ep = 0
        self.converged = False
        self.retry_ep = -1
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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

        # Periodic Checkpointing (every 100 episodes)
        if self.ep % 100 == 0:
            ckpt_path = os.path.join(self.checkpoint_dir, f"ep_{self.ep:04d}.pt")
            torch.save({
                "episode": self.ep,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "kpi_history": self.kpi_history,
                "alpha_history": self.alpha_history,
                "config": vars(self.all_args)
            }, ckpt_path)

        current_lr = self.policy[0].actor_optimizer.param_groups[0]['lr']
        print(f"Episode {self.ep:04d}/{int(self.num_env_steps // self.episode_length // self.n_rollout_threads)} | Cost: {kpis['total_cost']:.2f} | SL: {kpis['service_level']:.3f} | Bullwhip: {kpis['bullwhip_ratio']:.2f} | alphas: [{alphas[0]:.2f},{alphas[1]:.2f},{alphas[2]:.2f},{alphas[3]:.2f}] | LR: {current_lr:.2e} | Entropy: {self.trainer[0].entropy_coef:.5f}")

        # PPO Epoch Reduction Schedule
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
            print(f"  -> Episode {self.ep}: PPO epochs reduced to {new_ppo_epoch}")

        # Entropy decay
        for t in self.trainer:
            t.entropy_coef = max(0.001, t.entropy_coef * 0.999)

        # LR decay
        for p in self.policy:
            for param_group in p.actor_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.999)
            for param_group in p.critic_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.999)

        # Convergence Check (rolling window of 100 episodes after episode 100)
        if self.ep >= 100 and not self.converged:
            recent_kpis = self.kpi_history[-100:]
            recent_costs = [k["total_cost"] for k in recent_kpis]
            recent_sls = [k["service_level"] for k in recent_kpis]

            mean_cost = float(np.mean(recent_costs))
            std_cost = float(np.std(recent_costs))
            ratio = std_cost / mean_cost if mean_cost > 0 else 0

            mean_sl = float(np.mean(recent_sls))
            min_sl = float(np.min(recent_sls))
            catastrophic_pct = sum(1 for c in recent_costs if c > 40000) / len(recent_costs)

            target_ratio = 0.18  # Slightly wider threshold for dynamic transitions
            is_converged = (
                mean_cost > 0 and
                ratio < target_ratio and
                catastrophic_pct < 0.05 and
                mean_sl > 0.60 and
                min_sl > 0.15
            )

            if self.ep % 100 == 0:
                print(f"\n--- [Convergence Check @ Episode {self.ep}] ---")
                print(f"Mean Cost: {mean_cost:.2f} | Std: {std_cost:.2f} | Std/Mean: {ratio:.4f} (target < {target_ratio})")
                print(f"Mean SL: {mean_sl:.3f} | Min SL: {min_sl:.3f} | Catastrophic %: {catastrophic_pct:.3f}")
                print(f"Converged: {is_converged}\n")

            if is_converged and self.ep >= 600:
                self.converged = True
                print(f"Training stopped: convergence achieved at Episode {self.ep}!")
                raise StopIteration("Dynamic Phase 1 Converged")

        if self.ep >= 2500 and not self.converged:
            print("Training stopped: reached max episode cap (2500)")
            raise StopIteration("Max Episode Cap Reached")

        if self.retry_ep != -1 and self.ep >= self.retry_ep:
            raise StopIteration("Retry Complete")

        # Exploration range decay
        self.alpha_half = max(0.05, self.alpha_half * self.alpha_decay)
        alpha_low = max(0.1, self.alpha_center - self.alpha_half)
        alpha_high = min(0.9, self.alpha_center + self.alpha_half)
        for e in self.envs.env_list:
            e.set_alpha_range(alpha_low, alpha_high)

        return train_infos
