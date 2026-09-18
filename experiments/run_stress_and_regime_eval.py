"""
run_stress_and_regime_eval.py

Unified evaluation runner for Stress Test (Black Swan Pulse Shock) and
Sustained Demand Response (Step-UP and Step-DOWN) across:
1. Official Base Paper Reproduction Policy (Fixed alpha=0.375 specialist)
2. Tri-State Fixed Cruiser (Fixed alpha=0.375)
3. Tri-State Rule-Based Heuristic
4. Tri-State Meta-LLM Supervisor (Gemini 2.5 Flash-Lite, N=20)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import yaml

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from experiments.common.demand_loader import load_evaluation_demand_traces
from experiments.base_paper_reproduction.base_env_wrapper import BasePaperDummyVecEnv
from experiments.tristate_parallel_threads.tristate_intra_controller import (
    load_tristate_actors,
    TriStateIntraSimulationEngine,
    FixedTriStateController,
    RuleBasedTriStateController,
    LLMTriStateController
)
from algorithms.actor_critic import Actor

SCENARIO_DIRS = {
    "stress_shock": os.path.join(PROJECT_ROOT, "test_data", "test_demand_stress_shock"),
    "sustained_up": os.path.join(PROJECT_ROOT, "test_data", "test_demand_sustained_up"),
    "sustained_down": os.path.join(PROJECT_ROOT, "test_data", "test_demand_sustained_down")
}

def evaluate_base_paper(traces, scenario_name, output_dir, device="cpu"):
    """Evaluates the Base Paper Reproduction Policy on given traces."""
    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "base_paper_reproduction", "checkpoints", "best_base_paper_policy.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "base_paper_reproduction", "checkpoints", "final_base_paper_policy.pt")
        
    print(f"\n[BASE PAPER] Evaluating on '{scenario_name}' ({len(traces)} traces)...")
    print(f"  Checkpoint: {ckpt_path}")
    
    yaml_path = os.path.join(MADRL_DIR, "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    all_args = get_config().parse_known_args()[0]
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.episode_length = proj_config['environment']['episode_length']
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 7
    all_args.n_eval_rollout_threads = 1
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    env = BasePaperDummyVecEnv(all_args)
    actors = []
    for i in range(all_args.num_agents):
        obs_space = env.observation_space[i]
        act_space = env.action_space[i]
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        if "actor_state_dicts" in checkpoint:
            actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        else:
            actor.load_state_dict(checkpoint[f"actor_{i}"])
        actor.eval()
        actors.append(actor)
        
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for trace_item in traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]
        
        base_env = env.env_list[0]
        obs, _ = env.reset()
        base_env.demand_list = list(demand_series)
        
        rnn_states = np.zeros((1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
        
        trace_actions = [[] for _ in range(all_args.num_agents)]
        trace_inventories = [[] for _ in range(all_args.num_agents)]
        trace_backlogs = [[] for _ in range(all_args.num_agents)]
        trace_costs = []
        
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
            
            obs, rews, dones, infos = env.step([one_hot_actions])
            for i in range(all_args.num_agents):
                trace_inventories[i].append(float(base_env.inventory[i]))
                trace_backlogs[i].append(float(base_env.backlog[i]))
            step_c = sum(float(base_env.inventory[i]) + float(base_env.backlog[i]) for i in range(all_args.num_agents))
            trace_costs.append(step_c)
            
        kpis = base_env.get_episode_kpis()
        cost_per_period = kpis["total_cost"] / all_args.episode_length
        
        f_echelons = []
        for i in range(all_args.num_agents):
            acts = trace_actions[i]
            mean_a = np.mean(acts)
            std_a = np.std(acts)
            f_i = (std_a / mean_a) if mean_a > 1e-6 else 0.0
            f_echelons.append(float(f_i))
            
        avg_f = float(np.mean(f_echelons))
        amp = ((f_echelons[3] / f_echelons[0]) - 1.0) * 100.0 if f_echelons[0] > 1e-6 else 0.0
        
        total_demanded = float(np.sum(demand_series))
        total_backlog_r = float(np.sum(trace_backlogs[0]))
        sl_pct = max(0.0, float(1.0 - (total_backlog_r / (total_demanded + 1e-4)))) * 100.0
        cfr_pct = (base_env._kpi_fulfilled / max(base_env._kpi_demanded, 1.0)) * 100.0
        
        # Scenario specific metrics
        post_shock_holding_cost = float(np.sum([np.sum(trace_inventories[i][80:]) for i in range(all_args.num_agents)]))
        
        # Recovery time after t=75 (shock end)
        t_rec = 0
        if scenario_name == "stress_shock":
            t_rec = 125  # default if not recovered
            for t_step in range(75, 200):
                if trace_backlogs[0][t_step] <= 0.01:
                    t_rec = t_step - 75
                    break
                    
        record = {
            "trace_idx": trace_idx,
            "cost_per_period": float(cost_per_period),
            "f1": f_echelons[0],
            "f2": f_echelons[1],
            "f3": f_echelons[2],
            "f4": f_echelons[3],
            "avg_f": avg_f,
            "amp": float(amp),
            "service_level": sl_pct,
            "cfr": cfr_pct,
            "post_shock_holding_cost": post_shock_holding_cost,
            "t_rec": t_rec,
            "trajectory": {
                "demands": list(demand_series),
                "actions": trace_actions,
                "inventories": trace_inventories,
                "backlogs": trace_backlogs,
                "costs": trace_costs
            }
        }
        results.append(record)
        print(f"  Trace {trace_idx+1:02d} | Cost: {cost_per_period:6.2f} | SL: {sl_pct:5.1f}% | CFR: {cfr_pct:5.1f}% | Amp: {amp:6.1f}% | T_rec: {t_rec} | Post-Hold: {post_shock_holding_cost:.1f}")
        
    out_file = os.path.join(output_dir, f"base_paper_{scenario_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        # save summary without full trajectory to keep summary light, and separate trajectory file
        summary_data = [{k: v for k, v in r.items() if k != "trajectory"} for r in results]
        json.dump(summary_data, f, indent=2)
        
    traj_file = os.path.join(output_dir, f"base_paper_{scenario_name}_trajectories.json")
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump(results, f)
        
    return results

def evaluate_tristate_controller(traces, scenario_name, controller_type, output_dir, model_name="gemini-2.5-flash-lite", api_key=None, interval=20, device="cpu"):
    """Evaluates Tri-State MARL under Fixed, Rule-Based, or LLM supervision."""
    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "tristate_parallel_threads", "checkpoints", "best_tristate_policy.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "tristate_parallel_threads", "checkpoints", "final_tristate_policy.pt")
        
    print(f"\n[TRI-STATE: {controller_type.upper()}] Evaluating on '{scenario_name}' ({len(traces)} traces)...")
    print(f"  Checkpoint: {ckpt_path}")
    
    actors = load_tristate_actors(ckpt_path, device=device)
    engine = TriStateIntraSimulationEngine(actors, device=device, initial_alpha=0.375)
    
    llm_log_dir = os.path.join(output_dir, f"llm_logs_{scenario_name}")
    if controller_type == "fixed_cruiser":
        controller = FixedTriStateController(fixed_alpha=0.375, name="Fixed Cruiser (0.375)")
    elif controller_type == "rule_based":
        controller = RuleBasedTriStateController(name="Rule-Based Tri-State")
    elif controller_type == "llm":
        controller = LLMTriStateController(api_key=api_key, model=model_name, log_dir=llm_log_dir)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")
        
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for trace_item in traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]
        
        out = engine.run_trace(
            trace_id=trace_idx,
            demand_trace=demand_series,
            controller=controller,
            interval=interval
        )
        
        inventories = out["history_inventory"] # [agent][t]
        backlogs = out["history_backlog"]       # [agent][t]
        orders = out["history_orders"]
        post_shock_holding_cost = float(np.sum([np.sum(inventories[i][80:]) for i in range(4)]))
        
        t_rec = 0
        if scenario_name == "stress_shock":
            t_rec = 125
            for t_step in range(75, 200):
                if backlogs[0][t_step] <= 0.01:
                    t_rec = t_step - 75
                    break
                    
        record = {
            "trace_idx": trace_idx,
            "cost_per_period": float(out["per_period_cost"]),
            "f1": float(out["fluctuations"][0]),
            "f2": float(out["fluctuations"][1]),
            "f3": float(out["fluctuations"][2]),
            "f4": float(out["fluctuations"][3]),
            "avg_f": float(out["avg_fluctuation_F"]),
            "amp": float(out["bullwhip_amplification_pct"]),
            "service_level": float(out["service_level"] * 100.0),
            "cfr": float(out["customer_fill_rate"] * 100.0),
            "post_shock_holding_cost": post_shock_holding_cost,
            "t_rec": t_rec,
            "interventions": out["interventions"],
            "trajectory": {
                "demands": list(demand_series),
                "orders": orders,
                "inventories": inventories,
                "backlogs": backlogs,
                "alphas": out["history_alphas"]
            }
        }
        results.append(record)
        print(f"  Trace {trace_idx+1:02d} | Cost: {record['cost_per_period']:6.2f} | SL: {record['service_level']:5.1f}% | CFR: {record['cfr']:5.1f}% | Amp: {record['amp']:6.1f}% | T_rec: {t_rec} | Post-Hold: {post_shock_holding_cost:.1f}")
        
    out_file = os.path.join(output_dir, f"{controller_type}_{scenario_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        summary_data = [{k: v for k, v in r.items() if k != "trajectory"} for r in results]
        json.dump(summary_data, f, indent=2)
        
    traj_file = os.path.join(output_dir, f"{controller_type}_{scenario_name}_trajectories.json")
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump(results, f)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Stress Shock and Sustained Step Evaluations")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "stress_shock", "sustained_up", "sustained_down"])
    parser.add_argument("--controller", type=str, default="all", choices=["all", "base_paper", "fixed_cruiser", "rule_based", "llm"])
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "results_stress_and_regime")
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios_to_run = list(SCENARIO_DIRS.keys()) if args.scenario == "all" else [args.scenario]
    controllers_to_run = ["base_paper", "fixed_cruiser", "rule_based", "llm"] if args.controller == "all" else [args.controller]
    
    print("=" * 85)
    print(" STRESS TEST & SUSTAINED REGIME BENCHMARK SUITE")
    print(f" Scenarios:   {scenarios_to_run}")
    print(f" Controllers: {controllers_to_run}")
    print(f" Output Dir:  {output_dir}")
    print("=" * 85)
    
    all_benchmark_results = {}
    
    for sc in scenarios_to_run:
        traces = load_evaluation_demand_traces(demand_dir=SCENARIO_DIRS[sc])
        all_benchmark_results[sc] = {}
        
        for ctrl in controllers_to_run:
            if ctrl == "base_paper":
                res = evaluate_base_paper(traces, scenario_name=sc, output_dir=output_dir, device=device)
            else:
                res = evaluate_tristate_controller(
                    traces=traces,
                    scenario_name=sc,
                    controller_type=ctrl,
                    output_dir=output_dir,
                    model_name=args.model,
                    api_key=args.api_key,
                    interval=args.interval,
                    device=device
                )
            all_benchmark_results[sc][ctrl] = res
            
    print("\n" + "=" * 85)
    print(" ALL BENCHMARK RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    main()
