import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import yaml
import numpy as np
import torch
from pathlib import Path

# Ensure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from experiments.step7_tristate_poc.base_repro_env_wrapper import BasePaperReproDummyVecEnv
from algorithms.actor_critic import Actor

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_base_repro")

def parse_args():
    parser = get_config()
    parser.add_argument("--checkpoint", type=str, default="")
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args

def main():
    all_args = parse_args()
    
    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    
    # Base Paper Repro parameters
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 7  # No alpha in observations
    all_args.n_eval_rollout_threads = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================================================================")
    print(f" EVALUATION: Base Paper Specialist Reproduction")
    print(f" Protocol: Base_Paper_Evaluation_Process_and_Results.md")
    print(f" Target: 20 Merton Traces | Cost ~34.00 | F ~0.758 | Amp ~92.37%")
    print(f" Device: {device}")
    print(f"================================================================")
    
    # Locate checkpoint
    if all_args.checkpoint and os.path.exists(all_args.checkpoint):
        ckpt_path = all_args.checkpoint
    else:
        final_p = os.path.join(CHECKPOINT_DIR, "final_base_repro_policy.pt")
        best_p = os.path.join(CHECKPOINT_DIR, "best_base_repro_policy.pt")
        if os.path.exists(final_p):
            ckpt_path = final_p
        elif os.path.exists(best_p):
            ckpt_path = best_p
        else:
            print(f"Error: No checkpoint found in {CHECKPOINT_DIR}")
            sys.exit(1)
            
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    env = BasePaperReproDummyVecEnv(all_args)
    
    # Initialize Actor networks
    actors = []
    for i in range(all_args.num_agents):
        obs_space = env.observation_space[i]
        act_space = env.action_space[i]
        actor = Actor(all_args, obs_space, act_space, device)
        actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        actor.eval()
        actors.append(actor)
        
    num_eval_episodes = 20
    eval_costs = []
    trace_fluctuations = [[] for _ in range(all_args.num_agents)]  # [agent][trace]
    
    print("\nStarting evaluation across 20 frozen Merton demand traces...")
    print(f"{'Trace':<8}{'Per-Period Cost':<18}{'F_Retailer':<12}{'F_Whs':<10}{'F_Dist':<10}{'F_Mfr':<10}{'Trace F':<10}{'Amp (%)':<10}")
    print("-" * 88)
    
    for ep in range(num_eval_episodes):
        obs, _ = env.reset()
        rnn_states = np.zeros((1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
        
        trace_actions = [[] for _ in range(all_args.num_agents)]
        
        for step in range(all_args.episode_length):
            one_hot_actions = []
            with torch.no_grad():
                for i in range(all_args.num_agents):
                    obs_tensor = torch.tensor(obs[:, i, :], dtype=torch.float32).to(device)
                    rnn_tensor = torch.tensor(rnn_states[:, i, :, :], dtype=torch.float32).to(device)
                    mask_tensor = torch.tensor(masks[:, i, :], dtype=torch.float32).to(device)
                    
                    action, _, rnn_state = actors[i](obs_tensor, rnn_tensor, mask_tensor, deterministic=True)
                    act_idx = int(action.cpu().numpy()[0, 0])
                    trace_actions[i].append(act_idx)
                    rnn_states[:, i, :, :] = rnn_state.cpu().numpy()
                    
                    # One-hot encode action for environment step
                    one_hot = np.zeros(21, dtype=np.float32)
                    one_hot[act_idx] = 1.0
                    one_hot_actions.append(one_hot)
            
            # Step environment with one-hot formatted actions
            env_actions = [one_hot_actions]
            obs, rews, dones, infos = env.step(env_actions)
        
        # Calculate trace cost
        base_env = env.env_list[0]
        kpis = base_env.get_episode_kpis()
        cost_per_period = kpis["total_cost"] / all_args.episode_length
        eval_costs.append(cost_per_period)
        
        # Calculate echelon-level fluctuation for this trace: F_i = std(O_i) / mean(O_i)
        f_trace_echelons = []
        for i in range(all_args.num_agents):
            acts = trace_actions[i]
            mean_a = np.mean(acts)
            std_a = np.std(acts)
            f_i = (std_a / mean_a) if mean_a > 1e-6 else 0.0
            trace_fluctuations[i].append(f_i)
            f_trace_echelons.append(f_i)
            
        trace_avg_f = np.mean(f_trace_echelons)
        trace_amp = ((f_trace_echelons[-1] / f_trace_echelons[0]) - 1.0) * 100.0 if f_trace_echelons[0] > 1e-6 else 0.0
        
        print(f"Trace {ep+1:<3} {cost_per_period:<18.2f}{f_trace_echelons[0]:<12.3f}{f_trace_echelons[1]:<10.3f}{f_trace_echelons[2]:<10.3f}{f_trace_echelons[3]:<10.3f}{trace_avg_f:<10.3f}{trace_amp:<10.2f}%")
        
    # Aggregate over all 20 traces as defined in Base_Paper_Evaluation_Process_and_Results.md
    mean_cost = np.mean(eval_costs)
    std_cost = np.std(eval_costs)
    
    # Echelon-level average fluctuations
    mean_f_per_echelon = [np.mean(trace_fluctuations[i]) for i in range(all_args.num_agents)]
    f_downstream = mean_f_per_echelon[0]
    f_upstream = mean_f_per_echelon[-1]
    
    # Average fluctuation across all 4 echelons: F = (F1 + F2 + F3 + F4) / 4
    overall_f = np.mean(mean_f_per_echelon)
    
    # Bullwhip Amplification: A = (F_upstream / F_downstream - 1) * 100%
    overall_amp = ((f_upstream / f_downstream) - 1.0) * 100.0 if f_downstream > 1e-6 else 0.0
    
    print("\n" + "=" * 65)
    print(" FINAL EVALUATION SUMMARY (20 MERTON TRACES)")
    print("=" * 65)
    print(f"{'Metric':<30}{'Our Base Repro':<18}{'Base Paper':<15}")
    print("-" * 65)
    print(f"{'Overall System Cost':<30}{mean_cost:<18.2f}{'~34.00':<15}")
    print(f"{'F_1 (Retailer Fluctuation)':<30}{mean_f_per_echelon[0]:<18.3f}{'-':<15}")
    print(f"{'F_2 (Warehouse Fluctuation)':<30}{mean_f_per_echelon[1]:<18.3f}{'-':<15}")
    print(f"{'F_3 (Distributor Fluctuation)':<30}{mean_f_per_echelon[2]:<18.3f}{'-':<15}")
    print(f"{'F_4 (Manufacturer Fluctuation)':<30}{mean_f_per_echelon[3]:<18.3f}{'-':<15}")
    print(f"{'Average Fluctuation (F)':<30}{overall_f:<18.3f}{'0.758':<15}")
    print(f"{'Bullwhip Amplification':<30}{f'{overall_amp:.2f}%':<18}{'92.37%':<15}")
    print("=" * 65)

if __name__ == "__main__":
    main()
