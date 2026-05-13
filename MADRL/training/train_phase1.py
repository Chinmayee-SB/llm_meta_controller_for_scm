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

        if self.ep >= 50 and not self.converged:
            recent_costs = [k["total_cost"] for k in self.kpi_history[-50:]]
            if np.mean(recent_costs) > 0 and (np.std(recent_costs) / np.mean(recent_costs)) < 0.05:
                self.converged = True
                raise StopIteration("Phase 1 Converged")
        
        if self.retry_ep != -1 and self.ep >= self.retry_ep:
            raise StopIteration("Retry Complete")

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

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

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
