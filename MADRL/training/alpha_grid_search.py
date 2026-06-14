import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MADRL.training.run_phase2 import load_frozen_policy, run_single_episode, EVAL_SEEDS

GRID = [
    {"retailer": 0.1, "warehouse": 0.1, "distributor": 0.1, "manufacturer": 0.1},
    {"retailer": 0.1, "warehouse": 0.2, "distributor": 0.3, "manufacturer": 0.5},
    {"retailer": 0.1, "warehouse": 0.1, "distributor": 0.5, "manufacturer": 0.9},
    {"retailer": 0.2, "warehouse": 0.2, "distributor": 0.3, "manufacturer": 0.3},
    {"retailer": 0.1, "warehouse": 0.3, "distributor": 0.5, "manufacturer": 0.7},
    {"retailer": 0.3, "warehouse": 0.1, "distributor": 0.1, "manufacturer": 0.5},
    {"retailer": 0.1, "warehouse": 0.1, "distributor": 0.3, "manufacturer": 0.7},
    {"retailer": 0.2, "warehouse": 0.1, "distributor": 0.2, "manufacturer": 0.4},
]

def run_grid_search():
    from MADRL.envs.serial import Env as SerialEnv
    import yaml
    with open("MADRL/config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    class DummyArgs:
        pass
    env_args = DummyArgs()
    env_args.num_agents = 4
    env_args.obs_dim = 8
    env = SerialEnv(env_args)
    
    frozen_actors = load_frozen_policy("MADRL/results/phase1_weights/frozen_policy.pt", config)
    
    results_summary = []
    
    print(f"Starting grid search over {len(GRID)} combinations. (This may take a minute...)")
    
    # We will use torch.no_grad() but run_single_episode already does this.
    for idx, alphas in enumerate(GRID):
        results = []
        for seed in EVAL_SEEDS:
            kpis, _ = run_single_episode(env, frozen_actors, alphas, demand_seed=seed)
            results.append(kpis)
            
        costs = [r.get("total_cost", 0) for r in results]
        sls = [r.get("service_level", 0) for r in results]
        bws = [r.get("bullwhip_ratio", 0) for r in results]
        
        mean_cost = np.mean(costs)
        mean_sl = np.mean(sls)
        mean_bw = np.mean(bws)
        
        results_summary.append({
            "alphas": alphas,
            "mean_cost": float(mean_cost),
            "mean_sl": float(mean_sl),
            "mean_bw": float(mean_bw)
        })
        
    results_summary.sort(key=lambda x: x["mean_cost"])
    
    print("\nALPHA GRID SEARCH RESULTS (20 traces each, sorted by cost)")
    print(f"{'Rank':<6} {'R':<5} {'W':<5} {'D':<5} {'M':<5} {'Cost':<9} {'SL':<7} {'BW':<7}")
    
    for i, res in enumerate(results_summary):
        a = res["alphas"]
        rank = i + 1
        print(f"{rank:<6} {a['retailer']:<5} {a['warehouse']:<5} {a['distributor']:<5} {a['manufacturer']:<5} {res['mean_cost']:<9.2f} {res['mean_sl']:<7.3f} {res['mean_bw']:<7.3f}")
        
    os.makedirs("MADRL/results/phase2", exist_ok=True)
    with open("MADRL/results/phase2/alpha_grid_search.json", "w") as f:
        json.dump(results_summary, f, indent=2)

if __name__ == "__main__":
    import sys
    # Redirect stdout to devnull temporarily if we want to avoid massive logs, 
    # but the user might want to see them. I'll let it print, but run_single_episode is very noisy.
    # To keep it clean for the grid search, we can patch the print inside the loop.
    import builtins
    original_print = builtins.print
    
    def quiet_print(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            if args[0].startswith("===") or args[0].startswith("Alpha values") or args[0].startswith("Observation vectors") or args[0].startswith("  Agent"):
                return
        original_print(*args, **kwargs)
        
    builtins.print = quiet_print
    
    try:
        run_grid_search()
    finally:
        builtins.print = original_print
