import os
import sys
import json
import torch
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MADRL.training.run_phase2 import load_frozen_policy, EVAL_SEEDS
from MADRL.envs.serial import Env as SerialEnv
from MADRL.config import get_config

def run_logged_episode(env, frozen_actors, alpha_value, demand_seed, log_timesteps):
    """
    Runs a single 200-timestep episode with frozen actors, and logs specifically.
    """
    alphas = [alpha_value, alpha_value, alpha_value, alpha_value]
    env.set_alphas(alphas)
    
    np.random.seed(demand_seed)
    torch.manual_seed(demand_seed)
        
    obs = env.reset()
    
    if len(frozen_actors) > 0:
        hidden_size = getattr(frozen_actors[0], 'hidden_size', 64)
        recurrent_N = getattr(frozen_actors[0], '_recurrent_N', 1)
    else:
        hidden_size, recurrent_N = 64, 1
        
    rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32) for _ in range(4)]
    masks = torch.ones((1, 1), dtype=torch.float32)
    
    logs = []
    actions_taken = {i: [] for i in range(4)}
    
    with torch.no_grad():
        for t in range(200):
            actions = []
            
            step_logs = {}
            if t in log_timesteps:
                step_logs['t'] = t
                step_logs['agents'] = {}

            for i in range(4):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                
                if i < len(frozen_actors):
                    action, _, next_rnn_state = frozen_actors[i](
                        obs_tensor, 
                        rnn_states[i], 
                        masks, 
                        deterministic=True
                    )
                    rnn_states[i] = next_rnn_state
                    act_int = int(action.item())
                    one_hot = np.eye(21)[act_int]
                    actions.append(one_hot)
                else:
                    one_hot = np.zeros(21)
                    one_hot[10] = 1.0
                    actions.append(one_hot)
                    act_int = 10
                
                actions_taken[i].append(act_int)
                
                if t in log_timesteps:
                    step_logs['agents'][i] = {
                        'obs': [float(x) for x in obs[i]],
                        'alpha': float(obs[i][-1]),
                        'action': act_int
                    }
                    
            if t in log_timesteps:
                logs.append(step_logs)
            
            obs, rewards, done, info = env.step(actions)
            
            if all(done) if isinstance(done, list) else done:
                break
                
    kpis_dict = env.get_episode_kpis()
    return kpis_dict, logs, actions_taken

def main():
    print("======================================================================")
    print("FROZEN POLICY ALPHA SENSITIVITY AUDIT")
    print("======================================================================")
    
    parser = get_config()
    args, _ = parser.parse_known_args([])
    args.num_agents = 4
    args.obs_dim = 8
    
    with open("MADRL/config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env = SerialEnv(args)
    frozen_actors = load_frozen_policy("MADRL/results/phase1_weights/frozen_policy.pt", config)
    
    test_alphas = [0.10, 0.30, 0.50, 0.70, 0.90]
    results_dir = "MADRL/results/alpha_sensitivity_audit"
    os.makedirs(results_dir, exist_ok=True)
    
    summary_table = []
    agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
    
    # Structure to hold avg order quantity analysis
    avg_order_quantities = {}
    
    print("\n[1/2] RUNNING FULL EVALUATION ACROSS ALPHAS...")
    
    for alpha in test_alphas:
        alpha_kpis = []
        alpha_actions = {i: [] for i in range(4)}
        
        for seed in EVAL_SEEDS:
            kpis, _, actions_taken = run_logged_episode(env, frozen_actors, alpha, seed, log_timesteps=[])
            alpha_kpis.append(kpis)
            for i in range(4):
                alpha_actions[i].extend(actions_taken[i])
                
        costs = [r.get("total_cost", 0) for r in alpha_kpis]
        sls = [r.get("service_level", 0) for r in alpha_kpis]
        bws = [r.get("bullwhip_ratio", 0) for r in alpha_kpis]
        
        mean_cost, std_cost = np.mean(costs), np.std(costs)
        mean_sl, std_sl = np.mean(sls), np.std(sls)
        mean_bw, std_bw = np.mean(bws), np.std(bws)
        
        summary_table.append({
            "alpha": alpha,
            "mean_cost": mean_cost, "std_cost": std_cost,
            "mean_sl": mean_sl, "std_sl": std_sl,
            "mean_bw": mean_bw, "std_bw": std_bw
        })
        
        with open(os.path.join(results_dir, f"results_alpha_{alpha:.2f}.json"), "w") as f:
            json.dump({"alpha": alpha, "traces": alpha_kpis}, f, indent=2)
            
        avg_order_quantities[alpha] = {}
        for i in range(4):
            agent_actions = np.array(alpha_actions[i])
            # Actions mapped from 0-20 to actual order quantity if it's 0-20.
            # Assuming action corresponds to order quantity 0 to 20 directly.
            avg_order_quantities[alpha][agent_names[i]] = {
                "mean": float(np.mean(agent_actions)),
                "std": float(np.std(agent_actions))
            }
            
    print("\n[2/2] RUNNING POLICY REACTION TEST & ACTION DIVERGENCE ANALYSIS...")
    
    test_seed = EVAL_SEEDS[0]
    log_times = [0, 50, 100, 150]
    
    _, logs_010, actions_010 = run_logged_episode(env, frozen_actors, 0.10, test_seed, log_timesteps=log_times)
    _, logs_090, actions_090 = run_logged_episode(env, frozen_actors, 0.90, test_seed, log_timesteps=log_times)
    
    print("\n======================================================================")
    print("OUTPUT TABLE")
    print("======================================================================")
    print(f"| {'Alpha':<5} | {'Mean Cost':<12} | {'Mean SL':<10} | {'Mean BW':<10} |")
    print(f"| {'-----':<5} | {'---------':<12} | {'-------':<10} | {'-------':<10} |")
    for row in summary_table:
        print(f"| {row['alpha']:<5.2f} | {row['mean_cost']:<12.1f} | {row['mean_sl']:<10.3f} | {row['mean_bw']:<10.3f} |")
        
    print("\n======================================================================")
    print("AVERAGE ORDER QUANTITY ANALYSIS")
    print("======================================================================")
    for alpha in test_alphas:
        print(f"Alpha: {alpha:.2f}")
        for ag_name in agent_names:
            print(f"  {ag_name.capitalize():<12}: Mean = {avg_order_quantities[alpha][ag_name]['mean']:.2f}, Std = {avg_order_quantities[alpha][ag_name]['std']:.2f}")
            
    print("\n======================================================================")
    print("ADDITIONAL POLICY REACTION TEST (Seed: 42)")
    print("======================================================================")
    for log1, log2 in zip(logs_010, logs_090):
        t = log1['t']
        print(f"Step {t}")
        for i in range(4):
            print(f"  Agent {i} (Retailer={i==0}, Warehouse={i==1}, Dist={i==2}, Mfg={i==3}):")
            print(f"    [Alpha 0.10] Obs: {[round(x,2) for x in log1['agents'][i]['obs']]} | Alpha Component: {log1['agents'][i]['alpha']:.2f} | Action: {log1['agents'][i]['action']}")
            print(f"    [Alpha 0.90] Obs: {[round(x,2) for x in log2['agents'][i]['obs']]} | Alpha Component: {log2['agents'][i]['alpha']:.2f} | Action: {log2['agents'][i]['action']}")
        print()
        
    print("\n======================================================================")
    print("ACTION DIVERGENCE ANALYSIS (Seed: 42, Alpha=0.10 vs Alpha=0.90)")
    print("======================================================================")
    for i in range(4):
        a1 = actions_010[i]
        a2 = actions_090[i]
        diffs = sum(1 for x, y in zip(a1, a2) if x != y)
        pct = (diffs / len(a1)) * 100
        print(f"  {agent_names[i].capitalize():<12}: {diffs}/{len(a1)} actions different = {pct:.1f}%")

if __name__ == "__main__":
    main()
