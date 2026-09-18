import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import gym
from gym import spaces
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from algorithms.actor_critic import Actor
from experiments.step6_discrete_alpha_retrain.discrete_env_wrapper import (
    DiscreteSerialSupplyChainEnv,
    DISCRETE_ALPHAS
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def load_discrete_policy(checkpoint_path, device="cpu"):
    parser = get_config()
    all_args = parser.parse_known_args(["--mode", "phase1", "--obs_dim", "8"])[0]
    
    all_args.hidden_size = 256
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8
    all_args.gain = 0.01
    all_args.use_orthogonal = True
    all_args.use_policy_active_masks = False
    all_args.use_naive_recurrent_policy = True
    all_args.use_recurrent_policy = False
    all_args.use_ReLU = True
    all_args.stacked_frames = 1
    all_args.use_feature_normalization = False
    
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    act_space = spaces.Discrete(21)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dicts = ckpt.get("actor_state_dicts", None)
    if state_dicts is None:
        raise ValueError(f"No actor_state_dicts found in {checkpoint_path}")
        
    actors = []
    for i in range(4):
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(state_dicts[i])
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        actors.append(actor)
        
    print(f"[Loaded Policy] From {checkpoint_path} on {device} (hidden={all_args.hidden_size}, recurrent_N={all_args.recurrent_N})", flush=True)
    return actors, all_args

def evaluate_static_alpha(actors, alpha_val, n_traces=20, device="cpu"):
    parser = get_config()
    all_args = parser.parse_known_args(["--mode", "phase1", "--obs_dim", "8"])[0]
    all_args.hidden_size = 256
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8
    all_args.num_agents = 4
    all_args.episode_length = 200

    env = DiscreteSerialSupplyChainEnv(all_args)
    
    hidden_size = getattr(actors[0], 'hidden_size', 256)
    recurrent_N = getattr(actors[0], '_recurrent_N', 2)
    dev = torch.device(device)
    
    total_costs = []
    service_levels = []
    bullwhip_ratios = []
    per_agent_costs = {0: [], 1: [], 2: [], 3: []}
    order_variances = {0: [], 1: [], 2: [], 3: []}
    
    for trace_idx in range(n_traces):
        env.eval_index = trace_idx % env.n_eval
        env.set_alphas([alpha_val] * 4)
        obs = env.reset(train=False, normalize=True)
        
        rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=dev) for _ in range(4)]
        masks = torch.ones((1, 1), dtype=torch.float32, device=dev)
        
        with torch.no_grad():
            for t in range(200):
                actions = []
                for i in range(4):
                    obs_t = torch.from_numpy(obs[i]).float().unsqueeze(0).to(dev)
                    act, _, next_rnn = actors[i](obs_t, rnn_states[i], masks, deterministic=True)
                    rnn_states[i] = next_rnn
                    act_int = int(act.item())
                    one_hot = np.eye(21)[act_int]
                    actions.append(one_hot)
                obs, rews, done, info = env.step(actions)
                if all(done) if isinstance(done, list) else done:
                    break
                    
        kpis = env.get_episode_kpis()
        total_costs.append(kpis["total_cost"])
        service_levels.append(kpis["service_level"])
        bullwhip_ratios.append(kpis["bullwhip_ratio"])
        
        c_dict = kpis["per_agent_costs"]
        per_agent_costs[0].append(c_dict["retailer"])
        per_agent_costs[1].append(c_dict["warehouse"])
        per_agent_costs[2].append(c_dict["distributor"])
        per_agent_costs[3].append(c_dict["manufacturer"])
        
        for i in range(4):
            hist = env.action_history[i]
            order_variances[i].append(float(np.var(hist)) if len(hist) > 0 else 0.0)
            
    res = {
        "alpha": alpha_val,
        "mean_cost": float(np.mean(total_costs)),
        "std_cost": float(np.std(total_costs)),
        "mean_sl": float(np.mean(service_levels)),
        "mean_bullwhip": float(np.mean(bullwhip_ratios)),
        "agent_costs": {k: float(np.mean(v)) for k, v in per_agent_costs.items()},
        "order_vars": {k: float(np.mean(v)) for k, v in order_variances.items()}
    }
    return res

def main():
    final_cap_ckpt = os.path.join(CHECKPOINT_DIR, "final_cap_policy.pt")
    converged_ckpt = os.path.join(CHECKPOINT_DIR, "converged_policy.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_policy.pt")
    
    if os.path.exists(converged_ckpt):
        ckpt_path = converged_ckpt
    elif os.path.exists(final_cap_ckpt):
        ckpt_path = final_cap_ckpt
    elif os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    else:
        ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and f.startswith("ep_")])
        if not ckpts:
            print(f"No checkpoints found in {CHECKPOINT_DIR}", flush=True)
            return
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        
    device = "cpu"
    actors, _ = load_discrete_policy(ckpt_path, device=device)
    
    print("\n" + "="*85, flush=True)
    print(f"EVALUATING DISCRETE 256-GRU POLICY ACROSS ALL 9 STATIC ALPHA REGIMES (20 TRACES)", flush=True)
    print(f"Checkpoint: {os.path.basename(ckpt_path)} | Device: {device}", flush=True)
    print("="*85, flush=True)
    
    results = []
    for a in DISCRETE_ALPHAS:
        res = evaluate_static_alpha(actors, a, n_traces=20, device=device)
        results.append(res)
        print(f"Alpha: {a:.1f} | Cost: {res['mean_cost']:7.2f} ± {res['std_cost']:6.2f} | SL: {res['mean_sl']:.3f} | Bullwhip: {res['mean_bullwhip']:5.2f} | R_Cost: {res['agent_costs'][0]:6.1f} | M_Cost: {res['agent_costs'][3]:6.1f}", flush=True)
        
    out_json = os.path.join(os.path.dirname(__file__), "discrete_static_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}", flush=True)

if __name__ == "__main__":
    main()
