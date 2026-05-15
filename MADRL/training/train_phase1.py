import sys
import os
import argparse
import yaml
import numpy as np
import torch
import wandb
from pathlib import Path

# Add parent directory to path so we can import MADRL modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runners.separated.runner import CRunner

def make_train_env(all_args):
    return SubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DummyVecEnv(all_args)

class Phase1Runner(CRunner):
    def __init__(self, config):
        super().__init__(config)
        self.kpi_history = []
        self.alpha_history = []
        self.ep = 0
        self.converged = False
        self.retry_ep = -1

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
            "train/entropy": train_infos[0].get("dist_entropy", 0.0),
            "train/grad_norm": train_infos[0].get("actor_grad_norm", 0.0),
        })

        if self.ep % 100 == 0:
            os.makedirs("MADRL/results/checkpoints", exist_ok=True)
            torch.save({
                "episode": self.ep,
                "actor_state_dicts": [a.actor.state_dict() for a in self.policy],
                "critic_state_dicts": [c.critic.state_dict() for c in self.policy],
                "kpi_history": self.kpi_history,
                "alpha_history": self.alpha_history,
                "config": vars(self.all_args)
            }, f"MADRL/results/checkpoints/ep_{self.ep:04d}.pt")
            
        print(f"Episode {self.ep}/{int(self.num_env_steps // self.episode_length // self.n_rollout_threads)} | Cost: {kpis['total_cost']:.2f} | SL: {kpis['service_level']:.3f} | Bullwhip: {kpis['bullwhip_ratio']:.2f} | alpha: [{alphas[0]:.2f},{alphas[1]:.2f},{alphas[2]:.2f},{alphas[3]:.2f}]")

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

        if self.ep % 100 == 0:
            current_lr = self.policy[0].actor_optimizer.param_groups[0]['lr']
            print(f"Episode {self.ep} Checkpoint: Entropy Coef: {self.trainer[0].entropy_coef:.5f} | LR: {current_lr:.2e}")

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
            
            recent_acts = [k.get("action_stats", [(0,0)]*4) for k in recent_kpis]
            mean_mags = [np.mean([ep[i][0] for ep in recent_acts]) for i in range(4)]
            mean_vars = [np.mean([ep[i][1] for ep in recent_acts]) for i in range(4)]
            
            target_ratio = 0.15
            is_converged = (
                mean_cost > 0 and 
                ratio < target_ratio and 
                catastrophic_pct < 0.05 and 
                mean_sl > 0.65 and 
                min_sl > 0.20
            )

            current_lr = self.policy[0].actor_optimizer.param_groups[0]['lr']
            alpha_low  = max(0.1, self.alpha_center - self.alpha_half)
            alpha_high = min(0.9, self.alpha_center + self.alpha_half)
            
            print(f"\n[Convergence Check]")
            print(f"Episodes Window: {len(recent_costs)}")
            print(f"Mean Cost: {mean_cost:.2f}")
            print(f"Std Cost: {std_cost:.2f}")
            print(f"Std/Mean: {ratio:.4f}")
            print(f"Mean SL: {mean_sl:.3f}")
            print(f"Min SL: {min_sl:.3f}")
            print(f"Catastrophic %: {catastrophic_pct:.3f}")
            print(f"Threshold Target: {target_ratio:.3f}")
            print(f"Converged: {is_converged}")
            print(f"Params -> PPO Epochs: {self.trainer[0].ppo_epoch} | Entropy: {self.trainer[0].entropy_coef:.5f} | LR: {current_lr:.2e} | Alpha: [{alpha_low:.2f}, {alpha_high:.2f}]")
            print(f"Action Mags: {[f'{m:.2f}' for m in mean_mags]}")
            print(f"Action Vars: {[f'{v:.2f}' for v in mean_vars]}\n")
            
            if is_converged:
                self.converged = True
                print("Training stopped: convergence achieved")
                raise StopIteration("Phase 1 Converged")
        
        if self.ep >= 2500 and not self.converged:
            print("Training stopped: reached max episode cap")
            raise StopIteration("Max Episode Cap Reached")
            
        if self.retry_ep != -1 and self.ep >= self.retry_ep:
            raise StopIteration("Retry Complete")

        # Exploration decay — narrow alpha range over time
        self.alpha_half = max(0.05, self.alpha_half * self.alpha_decay)
        alpha_low  = max(0.1, self.alpha_center - self.alpha_half)
        alpha_high = min(0.9, self.alpha_center + self.alpha_half)
        env.set_alpha_range(alpha_low, alpha_high)

        # Switch from curriculum to uniform after curriculum_episodes
        if self.curriculum_on and self.ep == self.curriculum_eps:
            env.disable_curriculum()
            print(f"Episode {self.ep}: Alpha curriculum complete — switching to uniform random")

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
    
    # 1. Load yaml
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    # 2. Merge with argparse
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    
    alpha_decay       = proj_config["alpha"]["exploration_decay"]    # 0.995
    curriculum_on     = proj_config["alpha"]["curriculum_enabled"]   # False
    curriculum_eps    = proj_config["alpha"].get("curriculum_episodes", 100)
    curriculum_anchors= proj_config["alpha"].get("curriculum_anchor_values", [0.2,0.5,0.8])
    
    if all_args.mode == "baseline":
        all_args.obs_dim = 7
    else:
        all_args.obs_dim = 8

    # Explicitly turn off built-in evaluation to prevent NaN crashes
    # and incorrect early stopping (we use our own convergence + stability gate)
    all_args.use_eval = False

    if all_args.smoke_test:
        all_args.num_env_steps = all_args.n_episodes * all_args.episode_length * all_args.n_rollout_threads
        os.environ["WANDB_MODE"] = "offline"

    # 3. Initialize wandb
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="scm_episodic",
        # Set the wandb project where this run will be logged.
        project="SCM_Episodic_Meta_LLM",
        tags=["phase1", f"seed_{all_args.seed}", all_args.mode],
        config=vars(all_args)
    )

    # Set seeds
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.cuda and not torch.cuda.is_available():
        print("Error: --cuda was specified but no GPU is available. Stopping training to prevent CPU fallback.")
        sys.exit(1)
        
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    print(device)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # Enforce baseline mode explicitly
    if all_args.mode == "baseline":
        for e in envs.env_list:
            e.set_alphas([0.5, 0.5, 0.5, 0.5])

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path("MADRL/results/phase1")
    }

    # 4. Use existing runner
    runner = Phase1Runner(config)
    runner.alpha_center = 0.5
    runner.alpha_half   = 0.4   # initial half-range: samples from [0.1, 0.9]
    runner.alpha_decay = alpha_decay
    runner.curriculum_on = curriculum_on
    runner.curriculum_eps = curriculum_eps

    # Enable curriculum if configured
    if curriculum_on:
        envs.env_list[0].enable_curriculum(curriculum_anchors)
        print(f"Alpha curriculum ENABLED for first {curriculum_eps} episodes")
    
    if all_args.resume:
        checkpoint = torch.load(all_args.resume, map_location=device)
        for i, p in enumerate(runner.policy):
            p.actor.load_state_dict(checkpoint["actor_state_dicts"][i])
            p.critic.load_state_dict(checkpoint["critic_state_dicts"][i])
        runner.ep = checkpoint["episode"]
        runner.kpi_history = checkpoint["kpi_history"]
        runner.alpha_history = checkpoint["alpha_history"]

    try:
        runner.run()
    except StopIteration as e:
        print(f"\nConvergence detected: {str(e)}")
        
    if all_args.smoke_test:
        print("Smoke test completed successfully. Exiting.")
        envs.close()
        if eval_envs:
            eval_envs.close()
        wandb.finish()
        sys.exit(0)
        
    # 8. On convergence: freeze weights
    for p in runner.policy:
        for param in p.actor.parameters():
            param.requires_grad = False
        for param in p.critic.parameters():
            param.requires_grad = False
            
    # Save frozen policy BEFORE calling the gate
    os.makedirs(Path("MADRL/results/phase1_weights"), exist_ok=True)
    torch.save([p.actor.state_dict() for p in runner.policy], Path("MADRL/results/phase1_weights/frozen_policy.pt"))
            
    # Call StabilityGate
    print("Calling Stability Gate...")
    try:
        import subprocess
        gate_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stability_gate.py")
        subprocess.run([sys.executable, gate_script], check=True)
        passed = True 
    except Exception as e:
        print(f"Gate evaluation exception/fail: {e}")
        passed = False

    # 9. If gate passes
    if passed:
        print("Gate passed. Frozen policy saved to MADRL/results/phase1_weights/frozen_policy.pt")
    else:
        # 10. If gate fails: unfreeze weights, continue 500 more episodes, retry gate once
        print("Gate failed. Unfreezing weights and continuing for 500 episodes.")
        for p in runner.policy:
            for param in p.actor.parameters():
                param.requires_grad = True
            for param in p.critic.parameters():
                param.requires_grad = True
        
        runner.retry_ep = runner.ep + 500
        runner.all_args.num_env_steps = 500 * runner.all_args.episode_length * runner.all_args.n_rollout_threads
        try:
            runner.run()
        except StopIteration:
            pass
        
        os.makedirs("MADRL/results/phase1_weights", exist_ok=True)
        torch.save([p.actor.state_dict() for p in runner.policy], "MADRL/results/phase1_weights/frozen_policy.pt")
        print("Retry complete. Frozen policy saved.")

    envs.close()
    if eval_envs:
        eval_envs.close()
    wandb.finish()

if __name__ == "__main__":
    main()
