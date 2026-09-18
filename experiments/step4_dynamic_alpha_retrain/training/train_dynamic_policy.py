import os
import sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

# Add project root and MADRL to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
STEP4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)
if STEP4_DIR not in sys.path:
    sys.path.insert(0, STEP4_DIR)

from MADRL.config import get_config
from envs.dynamic_wrappers import DynamicSubprocVecEnv, DynamicDummyVecEnv
from training.dynamic_runner import DynamicPhase1Runner
from training.stability_gate_dynamic import StabilityGateDynamic

def make_train_env(all_args):
    return DynamicSubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DynamicDummyVecEnv(all_args)

def parse_args():
    parser = get_config()
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--n_episodes", type=int, default=5)
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args

def main():
    all_args = parse_args()

    # 1. Load Step 4 yaml config
    yaml_path = os.path.join(STEP4_DIR, "config", "dynamic_retrain_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)

    # 2. Merge config
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    all_args.obs_dim = 8
    all_args.use_eval = False

    if all_args.smoke_test:
        all_args.num_env_steps = all_args.n_episodes * all_args.episode_length * all_args.n_rollout_threads
        print(f"[SMOKE TEST] Running {all_args.n_episodes} episodes ({all_args.num_env_steps} steps total)...")

    # Seeds
    seed = all_args.seed[0] if isinstance(all_args.seed, list) else int(all_args.seed)
    all_args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    weights_dir = os.path.join(STEP4_DIR, "weights")
    results_dir = os.path.join(STEP4_DIR, "results")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(results_dir)
    }

    runner = DynamicPhase1Runner(config)
    runner.alpha_center = 0.5
    runner.alpha_half = 0.4
    runner.alpha_decay = 0.995

    if all_args.resume and os.path.isfile(all_args.resume):
        print(f"Resuming checkpoint from: {all_args.resume}")
        checkpoint = torch.load(all_args.resume, map_location=device)
        for i, p in enumerate(runner.policy):
            p.actor.load_state_dict(checkpoint["actor_state_dicts"][i])
            p.critic.load_state_dict(checkpoint["critic_state_dicts"][i])
        runner.ep = checkpoint.get("episode", 0)
        runner.kpi_history = checkpoint.get("kpi_history", [])
        runner.alpha_history = checkpoint.get("alpha_history", [])

    print("\n" + "=" * 60)
    print("  STARTING STEP 4 DYNAMIC ALPHA RETRAINING")
    print("  Mid-Episode Transitions: Every 50 steps (t=0, 50, 100, 150)")
    print("  Alpha Space: Continuous U(0.1, 0.9)")
    print("  Rollout Threads: 5 | Chunk Length: 10 | Device: " + str(device))
    print("=" * 60 + "\n")

    try:
        runner.run()
    except StopIteration as e:
        print(f"\nConvergence event: {str(e)}")

    if all_args.smoke_test:
        print("\n[SMOKE TEST COMPLETE] Step 4 pipeline verified successfully!")
        envs.close()
        if eval_envs:
            eval_envs.close()
        return

    # Freeze weights
    for p in runner.policy:
        for param in p.actor.parameters():
            param.requires_grad = False
        for param in p.critic.parameters():
            param.requires_grad = False

    frozen_policy_path = os.path.join(weights_dir, "dynamic_frozen_policy.pt")
    torch.save([p.actor.state_dict() for p in runner.policy], frozen_policy_path)
    print(f"\nSaved frozen policy weights to: {frozen_policy_path}")

    # Run Stability Gate
    print("\nRunning Automated Stability Gate on holdout traces...")
    frozen_actors = [p.actor for p in runner.policy]
    gate = StabilityGateDynamic(envs, frozen_actors, config=proj_config, training_history=runner.kpi_history)
    gate_passed = gate.run(save_dir=results_dir)

    if gate_passed:
        print("\n Stability Gate PASSED. Step 4 dynamic policy is certified and ready for evaluation benchmark.")
    else:
        print("\n Stability Gate flagged potential variance. Weights saved for inspection.")

    envs.close()
    if eval_envs:
        eval_envs.close()

if __name__ == "__main__":
    main()
