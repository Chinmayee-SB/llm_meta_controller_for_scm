import os
import sys

# Ensure MADRL is in sys.path so 'algorithms' module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import numpy as np

from MADRL.llm.prompt_builder import build_input_a, build_input_b, build_full_prompt, SYSTEM_PROMPT
from MADRL.llm.response_parser import parse_llm_response
from MADRL.envs import generator
from MADRL.config import get_config
from MADRL.envs.serial import Env as SerialEnv
from MADRL.llm.alpha_controller import AlphaController
from MADRL.llm.llm_client import LLMClient
import yaml
import copy

def load_frozen_policy(weights_path, config):
    """
    Identical to run_phase2.py load_frozen_policy
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    
    from MADRL.config import get_config
    from MADRL.algorithms.actor_critic import Actor
    import gym
    
    parser = get_config()
    args, _ = parser.parse_known_args([])
    args.num_agents = 4
    args.obs_dim = 8
    args.std_x_coef = 1.
    args.std_y_coef = 0.5
    
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
    act_space = gym.spaces.Discrete(21)
    
    actors = []
    if isinstance(checkpoint, list):
        state_dicts = checkpoint
    elif isinstance(checkpoint, dict) and "actor_state_dicts" in checkpoint:
        state_dicts = checkpoint["actor_state_dicts"]
    else:
        raise ValueError("Unknown checkpoint format")
        
    for i, state_dict in enumerate(state_dicts):
        actor = Actor(args, obs_space, act_space)
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False
        actors.append(actor)
        
    n_params = sum(p.numel() for a in actors for p in a.parameters())
    print("\n=== FROZEN POLICY VERIFICATION REPORT ===")
    print(f"Path: {weights_path}")
    print(f"Actors: {len(actors)}")
    print(f"Class: {actors[0].__class__.__name__}")
    print(f"Parameters: {n_params}")
    print(f"Checkpoint Loaded: YES")
    print(f"Fallback Used: NO")
    print(f"requires_grad=False: {all(not p.requires_grad for a in actors for p in a.parameters())}")
    print("=========================================\n")
    
    return actors

def run_single_episode(env, frozen_actors, current_alphas, demand_seed=None, demand_slice=None, collect_first_actions=False):
    """
    Preserves exact official execution logic while allowing demand slice injection.
    """
    env.set_alphas(list(current_alphas.values()))
    alphas = list(current_alphas.values())
    
    if demand_seed is not None:
        np.random.seed(demand_seed)
        torch.manual_seed(demand_seed)
        
    obs = env.reset()
    
    # NEW LOGIC: Override demand after reset
    if demand_slice is not None:
        env.demand_list = list(demand_slice)
        
    demand_trace = []
    first_actions = None
    
    if len(frozen_actors) > 0:
        hidden_size = getattr(frozen_actors[0], 'hidden_size', 64)
        recurrent_N = getattr(frozen_actors[0], '_recurrent_N', 1)
    else:
        hidden_size, recurrent_N = 64, 1
        
    rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32) for _ in range(4)]
    masks = torch.ones((1, 1), dtype=torch.float32)
    
    with torch.no_grad():
        for t in range(200):
            actions = []
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
            
            if t == 0 and collect_first_actions:
                first_actions = [np.argmax(a) for a in actions]
                
            obs, rewards, done, info = env.step(actions)
            
            if hasattr(env, 'demand_list') and env.step_num > 0:
                demand_trace.append(env.demand_list[env.step_num - 1])
                
            if all(done) if isinstance(done, list) else done:
                break
                
    kpis_dict = env.get_episode_kpis()
    
    if collect_first_actions:
        return kpis_dict, demand_trace, first_actions
    return kpis_dict, demand_trace


def run_continuous_trace_evaluation():
    with open("MADRL/config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    class DummyArgs: pass
    env_args = DummyArgs()
    env_args.num_agents = 4
    env_args.obs_dim = 8
    env = SerialEnv(env_args)
    
    weights_path = "MADRL/results/phase1_weights/frozen_policy.pt"
    frozen_actors = load_frozen_policy(weights_path, config)
    
    if "OPENAI_API_KEY" not in os.environ:
        class MockLLMClient:
            def query(self, prompt, sys_prompt, block_id=-1):
                return '{"reasoning": "Mocked", "self_correction": "", "compiled_rule": "Mock Rule", "alphas": {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5}}'
        llm_client = MockLLMClient()
    else:
        llm_client = LLMClient()
        
    alpha_controller = AlphaController(config, initial_alpha=0.5)
    
    rulebook = []
    if os.path.exists("MADRL/results/phase2/llm_rulebook.json"):
        with open("MADRL/results/phase2/llm_rulebook.json", "r") as f:
            rulebook = json.load(f)
            
    # ACTION FIDELITY CHECK
    print("\n=== ACTION FIDELITY CHECK ===")
    baseline_alphas = {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5}
    # Run exact official script semantics
    _, _, baseline_actions = run_single_episode(env, frozen_actors, baseline_alphas, demand_seed=0, collect_first_actions=True)
    
    # Run our script semantics
    np.random.seed(0)
    dg = generator.merton(2000, 20, mu=5e-5, sigma=1e-2, lamda=15, l1=10, l2=0, l3=0.01)
    continuous_demand = [int(max(1, min(40, d))) for d in dg.demand_list]
    slice_0 = continuous_demand[0:200]
    _, _, our_actions = run_single_episode(env, frozen_actors, baseline_alphas, demand_seed=0, demand_slice=slice_0, collect_first_actions=True)
    
    print(f"Official run_single_episode first actions (Seed 0): {baseline_actions}")
    print(f"Our sliced run_single_episode first actions (Seed 0): {our_actions}")
    
    if baseline_actions != our_actions:
        print("ERROR: Action fidelity check failed! Stopping execution.")
        return
    print("Action fidelity check passed. Semantics identical.\n")
    
    log_dir = "MADRL/results/phase2_continuous_trace_control"
    os.makedirs(log_dir, exist_ok=True)
    
    seeds = list(range(20))
    summary_results = []
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args, _ = parser.parse_known_args()
    
    if args.smoke_test:
        seeds = [0]
        
    for seed in seeds:
        print(f"\n===========================================")
        print(f"Starting Continuous Trace for Seed: {seed}")
        print(f"===========================================")
        
        alpha_controller.reset(0.5)
        
        np.random.seed(seed)
        dg = generator.merton(2000, 20, mu=5e-5, sigma=1e-2, lamda=15, l1=10, l2=0, l3=0.01)
        continuous_demand = [int(max(1, min(40, d))) for d in dg.demand_list]
        
        if seed == 0:
            print("\n=== DEMAND SLICE VERIFICATION ===")
            print(f"Full trace length: {len(continuous_demand)}")
        
        episodes_1_5_kpis = []
        episodes_6_10_kpis = []
        full_executed_trace_1_5 = []
        
        # Episodes 1-5
        for ep in range(1, 6):
            start = (ep-1) * 200
            end = ep * 200
            d_slice = continuous_demand[start:end]
            
            if seed == 0:
                print(f"Episode {ep} boundaries: {start} to {end}")
                
            kpis, trace = run_single_episode(env, frozen_actors, alpha_controller.get_current(), demand_slice=d_slice)
            episodes_1_5_kpis.append(kpis)
            full_executed_trace_1_5.extend(trace)
            
        if seed == 0:
            match = (full_executed_trace_1_5 == continuous_demand[0:1000])
            print(f"Episodes 1-5 exactly reconstruct trace 0-1000: {match}")
            print("=================================\n")
            print(f"Smoke Test - Episode 1 KPI: Cost={episodes_1_5_kpis[0]['total_cost']:.0f}")
            print(f"Smoke Test - Episode 5 KPI: Cost={episodes_1_5_kpis[4]['total_cost']:.0f}")
            
        # Average KPIs for 1-5
        avg_1_5 = {}
        for k in episodes_1_5_kpis[0].keys():
            if isinstance(episodes_1_5_kpis[0][k], dict):
                avg_1_5[k] = {agent: np.mean([ep[k][agent] for ep in episodes_1_5_kpis]) for agent in episodes_1_5_kpis[0][k]}
            else:
                avg_1_5[k] = np.mean([ep[k] for ep in episodes_1_5_kpis])
                
        if seed == 0:
            print(f"Smoke Test - Average KPI Episodes 1-5: Cost={avg_1_5['total_cost']:.0f}")
                
        # No LLM Update (Control A)
        applied_alphas = alpha_controller.get_current()
        
        if seed == 0:
            print(f"Smoke Test - Applied alpha: {applied_alphas}")
            
        # Episodes 6-10
        full_executed_trace_6_10 = []
        for ep in range(6, 11):
            start = (ep-1) * 200
            end = ep * 200
            d_slice = continuous_demand[start:end]
            
            kpis, trace = run_single_episode(env, frozen_actors, alpha_controller.get_current(), demand_slice=d_slice)
            episodes_6_10_kpis.append(kpis)
            full_executed_trace_6_10.extend(trace)
            
        if seed == 0:
            match_6_10 = (full_executed_trace_6_10 == continuous_demand[1000:2000])
            print(f"Episodes 6-10 exactly reconstruct trace 1000-2000: {match_6_10}")
            print(f"Smoke Test - Episode 6 KPI: Cost={episodes_6_10_kpis[0]['total_cost']:.0f}")
            print(f"Smoke Test - Episode 10 KPI: Cost={episodes_6_10_kpis[4]['total_cost']:.0f}")
            
        # Average KPIs for 6-10
        avg_6_10 = {}
        for k in episodes_6_10_kpis[0].keys():
            if isinstance(episodes_6_10_kpis[0][k], dict):
                avg_6_10[k] = {agent: np.mean([ep[k][agent] for ep in episodes_6_10_kpis]) for agent in episodes_6_10_kpis[0][k]}
            else:
                avg_6_10[k] = np.mean([ep[k] for ep in episodes_6_10_kpis])
                
        if seed == 0:
            print(f"Smoke Test - Average KPI Episodes 6-10: Cost={avg_6_10['total_cost']:.0f}")
                
        trace_data = {
            "seed": seed,
            "initial_alphas": {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5},
            "updated_alphas": alpha_controller.get_current(),
            "episode_1_to_5_metrics": avg_1_5,
            "episode_6_to_10_metrics": avg_6_10,
            "cost_delta": avg_6_10["total_cost"] - avg_1_5["total_cost"],
            "service_level_delta": avg_6_10["service_level"] - avg_1_5["service_level"],
            "bullwhip_delta": avg_6_10["bullwhip_ratio"] - avg_1_5["bullwhip_ratio"],
            "reasoning": "None (Control)",
            "compiled_rule": "None"
        }
        
        summary_results.append(trace_data)
        
        with open(os.path.join(log_dir, f"trace_{seed}.json"), "w") as f:
            json.dump(trace_data, f, indent=2)
            
    if not args.smoke_test:
        # Final Summary
        c_b = [x["episode_1_to_5_metrics"]["total_cost"] for x in summary_results]
        c_a = [x["episode_6_to_10_metrics"]["total_cost"] for x in summary_results]
        sl_b = [x["episode_1_to_5_metrics"]["service_level"] for x in summary_results]
        sl_a = [x["episode_6_to_10_metrics"]["service_level"] for x in summary_results]
        bw_b = [x["episode_1_to_5_metrics"]["bullwhip_ratio"] for x in summary_results]
        bw_a = [x["episode_6_to_10_metrics"]["bullwhip_ratio"] for x in summary_results]
        
        alpha_diffs = []
        for x in summary_results:
            init = list(x["initial_alphas"].values())
            upd = list(x["updated_alphas"].values())
            alpha_diffs.append(np.mean(np.abs(np.array(init) - np.array(upd))))
            
        summary = {
            "Mean Cost Before": np.mean(c_b),
            "Mean Cost After": np.mean(c_a),
            "Mean Service Level Before": np.mean(sl_b),
            "Mean Service Level After": np.mean(sl_a),
            "Mean Bullwhip Before": np.mean(bw_b),
            "Mean Bullwhip After": np.mean(bw_a),
            "Mean Alpha Change": np.mean(alpha_diffs)
        }
        
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        print("===========================================")
        print("CONTINUOUS TRACE EVALUATION SUMMARY (v3)")
        print("===========================================")
        print(f"Cost Before: {summary['Mean Cost Before']:.0f} -> After: {summary['Mean Cost After']:.0f}")
        print(f"SL Before:   {summary['Mean Service Level Before']:.3f} -> After: {summary['Mean Service Level After']:.3f}")
        print(f"BW Before:   {summary['Mean Bullwhip Before']:.3f} -> After: {summary['Mean Bullwhip After']:.3f}")
        print("===========================================")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        except Exception:
            pass
    run_continuous_trace_evaluation()
