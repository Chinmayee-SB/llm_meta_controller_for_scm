import os
import sys

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import yaml
import numpy as np
import torch
import argparse
from pathlib import Path

# Ensure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from experiments.base_paper_reproduction.base_env_wrapper import BasePaperDummyVecEnv
from algorithms.actor_critic import Actor
from experiments.common.demand_loader import load_evaluation_demand_traces

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def parse_args():
    parser = get_config()
    parser.add_argument("--checkpoint", type=str, default="", help="Path to policy checkpoint")
    parser.add_argument("--demand_dir", type=str, default=None, help="Directory of evaluation demand traces")
    parser.add_argument("--use_generated_demand", action="store_true", help="Use test_demand_merton_generated")
    parser.add_argument("--num_traces", type=int, default=20, help="Number of traces to evaluate")
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args

def main():
    all_args = parse_args()
    
    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 7  # strictly physical state, no alpha
    all_args.n_eval_rollout_threads = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and all_args.cuda else "cpu")

    # Locate checkpoint
    if all_args.checkpoint and os.path.exists(all_args.checkpoint):
        ckpt_path = all_args.checkpoint
    else:
        best_p = os.path.join(CHECKPOINT_DIR, "best_base_paper_policy.pt")
        final_p = os.path.join(CHECKPOINT_DIR, "final_base_paper_policy.pt")
        if os.path.exists(best_p):
            ckpt_path = best_p
        elif os.path.exists(final_p):
            ckpt_path = final_p
        else:
            print(f"Error: No checkpoint found in {CHECKPOINT_DIR}")
            sys.exit(1)
            
    # Demand directory
    if all_args.use_generated_demand and not all_args.demand_dir:
        demand_dir = os.path.join(BASE_DIR, "test_data", "test_demand_merton_generated")
        demand_name = "Generated Merton Traces (Matching Training Distribution)"
    elif all_args.demand_dir:
        demand_dir = all_args.demand_dir
        demand_name = f"Custom ({os.path.basename(all_args.demand_dir)})"
    else:
        demand_dir = os.path.join(BASE_DIR, "test_data", "test_demand_merton")
        demand_name = "Frozen Merton Traces (Base Paper Exact 20 Traces)"

    print("================================================================")
    print(" EVALUATION: Official Base Paper Specialist Reproduction")
    print(f" Checkpoint:  {ckpt_path}")
    print(f" Demand Set:  {demand_name}")
    print(f" Target:      Cost ~34.00 | Fluctuation ~0.758 | Bullwhip ~92.37%")
    print(f" Device:      {device}")
    print("================================================================")

    checkpoint = torch.load(ckpt_path, map_location=device)
    
    env = BasePaperDummyVecEnv(all_args)
    
    # Initialize Actor networks
    actors = []
    for i in range(all_args.num_agents):
        obs_space = env.observation_space[i]
        act_space = env.action_space[i]
        actor = Actor(all_args, obs_space, act_space, device)
        actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        actor.eval()
        actors.append(actor)

    traces = load_evaluation_demand_traces(demand_dir=demand_dir)[:all_args.num_traces]
    
    eval_costs = []
    trace_f1 = []
    trace_f2 = []
    trace_f3 = []
    trace_f4 = []
    trace_avg_f = []
    trace_amp = []
    trace_sl = []
    trace_cfr = []
    
    print(f"\nEvaluating across {len(traces)} traces...")
    print(f"{'Trace':<8}{'Per-Period Cost':<18}{'F1 (Ret)':<10}{'F2 (Whs)':<10}{'F3 (Dist)':<10}{'F4 (Mfr)':<10}{'Avg F':<10}{'Amp (%)':<12}{'SL (%)':<10}{'CFR (%)':<10}")
    print("-" * 102)

    for trace_item in traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]
        
        # Reset environment with specific demand trace
        base_env = env.env_list[0]
        obs, _ = env.reset()
        base_env.demand_list = list(demand_series)
        
        rnn_states = np.zeros((1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
        
        trace_actions = [[] for _ in range(all_args.num_agents)]
        trace_ret_backlog = []
        
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
                    
                    one_hot = np.zeros(21, dtype=np.float32)
                    one_hot[act_idx] = 1.0
                    one_hot_actions.append(one_hot)
            
            env_actions = [one_hot_actions]
            obs, rews, dones, infos = env.step(env_actions)
            trace_ret_backlog.append(float(base_env.backlog[0]))
        
        kpis = base_env.get_episode_kpis()
        cost_per_period = kpis["total_cost"] / all_args.episode_length
        eval_costs.append(cost_per_period)
        
        f_echelons = []
        for i in range(all_args.num_agents):
            acts = trace_actions[i]
            mean_a = np.mean(acts)
            std_a = np.std(acts)
            f_i = (std_a / mean_a) if mean_a > 1e-6 else 0.0
            f_echelons.append(f_i)
            
        trace_f1.append(f_echelons[0])
        trace_f2.append(f_echelons[1])
        trace_f3.append(f_echelons[2])
        trace_f4.append(f_echelons[3])
        
        avg_f = float(np.mean(f_echelons))
        trace_avg_f.append(avg_f)
        
        amp = ((f_echelons[3] / f_echelons[0]) - 1.0) * 100.0 if f_echelons[0] > 1e-6 else 0.0
        trace_amp.append(amp)
        
        # Calculate Service Level and Customer Fill Rate identically to TriState
        total_demanded = float(np.sum(demand_series))
        total_backlog_r = float(np.sum(trace_ret_backlog))
        sl_pct = max(0.0, float(1.0 - (total_backlog_r / (total_demanded + 1e-4)))) * 100.0
        trace_sl.append(sl_pct)
        
        fulfilled = base_env._kpi_fulfilled
        demanded = base_env._kpi_demanded
        cfr_pct = (fulfilled / max(demanded, 1.0)) * 100.0
        trace_cfr.append(cfr_pct)
        
        amp_str = f"{amp:.2f}%"
        sl_str = f"{sl_pct:.1f}%"
        cfr_str = f"{cfr_pct:.1f}%"
        print(f"Trace {trace_idx+1:<3} {cost_per_period:<18.2f}{f_echelons[0]:<10.3f}{f_echelons[1]:<10.3f}{f_echelons[2]:<10.3f}{f_echelons[3]:<10.3f}{avg_f:<10.3f}{amp_str:<12}{sl_str:<10}{cfr_str:<10}")

    mean_cost = float(np.mean(eval_costs))
    median_cost = float(np.median(eval_costs))
    m_f1 = float(np.mean(trace_f1))
    m_f2 = float(np.mean(trace_f2))
    m_f3 = float(np.mean(trace_f3))
    m_f4 = float(np.mean(trace_f4))
    m_avg_f = float(np.mean(trace_avg_f))
    m_amp = float(np.mean(trace_amp))
    m_sl = float(np.mean(trace_sl))
    m_cfr = float(np.mean(trace_cfr))

    print("\n" + "=" * 90)
    print(f" FINAL REPRODUCTION BENCHMARK ({len(traces)} TRACES)")
    print("=" * 90)
    print(f"{'Metric':<32}{'Reproduction Result':<22}{'Base Paper Target':<20}{'Tri-State + LLM':<16}")
    print("-" * 90)
    print(f"{'Per-Period Cost (Mean)':<32}{mean_cost:<22.2f}{'~34.00':<20}{'85.76':<16}")
    print(f"{'Per-Period Cost (Median)':<32}{median_cost:<22.2f}{'-':<20}{'78.89':<16}")
    print(f"{'F_1 Retailer Fluctuation':<32}{m_f1:<22.3f}{'-':<20}{'0.586':<16}")
    print(f"{'F_2 Warehouse Fluctuation':<32}{m_f2:<22.3f}{'-':<20}{'0.772':<16}")
    print(f"{'F_3 Distributor Fluctuation':<32}{m_f3:<22.3f}{'-':<20}{'0.901':<16}")
    print(f"{'F_4 Manufacturer Fluctuation':<32}{m_f4:<22.3f}{'-':<20}{'1.124':<16}")
    print(f"{'Average Fluctuation (F_bar)':<32}{m_avg_f:<22.3f}{'0.758':<20}{'0.846':<16}")
    print(f"{'Bullwhip Amplification (A)':<32}{f'{m_amp:.2f}%':<22}{'92.37%':<20}{'93.11%':<16}")
    print(f"{'Customer Service Level (SL)':<32}{f'{m_sl:.1f}%':<22}{'-':<20}{'36.3%':<16}")
    print(f"{'Customer Fill Rate (CFR)':<32}{f'{m_cfr:.1f}%':<22}{'-':<20}{'60.8%':<16}")
    print("=" * 90)

if __name__ == "__main__":
    main()
