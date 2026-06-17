import os
import sys
import json
import numpy as np
import torch
import yaml

# Add MADRL to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MADRL.envs.serial import Env as SerialEnv
from MADRL.training.run_phase2 import load_frozen_policy
from MADRL.llm.llm_client import LLMClient
from MADRL.llm.alpha_controller import AlphaController
from MADRL.llm.response_parser import parse_llm_response
from MADRL.llm.prompt_builder import build_input_a, build_input_b, build_input_c, build_full_prompt, SYSTEM_PROMPT

AGENT_NAMES = ["retailer", "warehouse", "distributor", "manufacturer"]

def run_single_episode_with_behavior(env, frozen_actors, alphas):
    env.set_alphas([alphas['retailer'], alphas['warehouse'], alphas['distributor'], alphas['manufacturer']])
    obs = env.reset(train=False, normalize=True)
    
    demand_trace = []
    
    agent_orders = {0: [], 1: [], 2: [], 3: []}
    agent_inventories = {0: [], 1: [], 2: [], 3: []}
    agent_backlogs = {0: [], 1: [], 2: [], 3: []}
    
    hidden_size = getattr(frozen_actors[0], 'hidden_size', 64)
    recurrent_N = getattr(frozen_actors[0], '_recurrent_N', 1)
        
    rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32) for _ in range(4)]
    masks = torch.ones((1, 1), dtype=torch.float32)
    
    with torch.no_grad():
        for t in range(200):
            actions = []
            for i in range(4):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                
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
                
                agent_orders[i].append(act_int)
                agent_inventories[i].append(env.inventory[i])
                agent_backlogs[i].append(env.backlog[i])
            
            obs, rewards, done, info = env.step(actions)
            
            if hasattr(env, 'demand_list') and env.step_num > 0:
                demand_trace.append(env.demand_list[env.step_num - 1])
                
            if all(done) if isinstance(done, list) else done:
                break
                
    kpis_dict = env.get_episode_kpis()
    
    behavior_dict = {
        "orders": agent_orders,
        "inventories": agent_inventories,
        "backlogs": agent_backlogs
    }
    
    return kpis_dict, demand_trace, behavior_dict

def run_evaluation_block_with_behavior(env, frozen_actors, current_alphas, n_episodes=5):
    all_kpis = []
    full_demand_trace = []
    all_behaviors = []
    
    for _ in range(n_episodes):
        kpis, trace, behavior = run_single_episode_with_behavior(env, frozen_actors, current_alphas)
        all_kpis.append(kpis)
        full_demand_trace.extend(trace)
        all_behaviors.append(behavior)
        
    # Average all KPIs
    averaged_kpis = {}
    if all_kpis:
        for k in all_kpis[0].keys():
            if isinstance(all_kpis[0][k], dict):
                averaged_kpis[k] = {
                    agent: float(np.mean([kpis[k][agent] for kpis in all_kpis]))
                    for agent in all_kpis[0][k]
                }
            else:
                averaged_kpis[k] = float(np.mean([kpis[k] for kpis in all_kpis]))
                
    agent_behavior_raw = {}
    for i, name in enumerate(AGENT_NAMES):
        all_orders = np.concatenate([b["orders"][i] for b in all_behaviors])
        all_inv = np.concatenate([b["inventories"][i] for b in all_behaviors])
        all_back = np.concatenate([b["backlogs"][i] for b in all_behaviors])
        
        agent_behavior_raw[name] = {
            "avg_order_qty": float(np.mean(all_orders)),
            "order_variance": float(np.std(all_orders)),
            "avg_inventory": float(np.mean(all_inv)),
            "avg_backlog": float(np.mean(all_back)),
            "inventory_backlog_ratio": float(np.mean(all_inv) / max(1.0, np.mean(all_back)))
        }
        
    averaged_kpis["agent_behavior_raw"] = agent_behavior_raw
    
    # Calculate cost shares if not already strictly calculated
    total_cost = averaged_kpis["total_cost"]
    if total_cost > 0:
        averaged_kpis["cost_shares"] = {
            k: float(v / total_cost) for k, v in averaged_kpis["per_agent_costs"].items()
        }
    else:
        averaged_kpis["cost_shares"] = {k: 0.25 for k in AGENT_NAMES}
        
    return averaged_kpis, full_demand_trace

def run_demo2():
    with open("MADRL/config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    class DummyArgs:
        pass
    env_args = DummyArgs()
    env_args.num_agents = 4
    env_args.obs_dim = 8
    env = SerialEnv(env_args)
    
    # Redirect stdout to suppress print from load_frozen_policy
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    frozen_actors = load_frozen_policy("MADRL/results/phase1_weights/frozen_policy.pt", config)
    sys.stdout.close()
    sys.stdout = old_stdout
    
    llm_client = LLMClient()
    p2_config = config.setdefault("phase2", {})
    p2_config["alpha_smoothing"] = 0.0
    p2_config["max_alpha_change"] = 0.25
    alpha_controller = AlphaController(config=config, initial_alpha=0.5)
    
    initial_alpha = {agent: 0.5 for agent in AGENT_NAMES}
    alpha_controller.alphas = initial_alpha.copy()
    
    n_blocks = 5
    episodes_per_block = 5
    
    blocks_data = []
    
    llm_history = []
    block_metrics_history = []
    
    total_cost_traj = []
    sl_traj = []
    bw_traj = []
    alpha_traj = {agent: [] for agent in AGENT_NAMES}
    rules_traj = []
    
    print("Running Demo 2 (Alpha Impact Simulation)...")
    
    for block_id in range(1, n_blocks + 1):
        current_alphas = alpha_controller.get_current()
        
        averaged_kpis, full_demand_trace = run_evaluation_block_with_behavior(env, frozen_actors, current_alphas, n_episodes=episodes_per_block)
        
        input_a = build_input_a(full_demand_trace)
        input_b = build_input_b(block_metrics_history, averaged_kpis)
        input_c = build_input_c(llm_history)
        
        from MADRL.llm.rulebook import extract_rulebook
        dynamic_rulebook = extract_rulebook(llm_history)
        
        prompt = build_full_prompt(input_a, input_b, input_c, SYSTEM_PROMPT, rulebook=dynamic_rulebook)
        
        raw_response = llm_client.query(prompt, SYSTEM_PROMPT, block_id=block_id)
        llm_alphas, reasoning, self_correction, compiled_rule, confidence = parse_llm_response(raw_response, current_alphas)
        
        applied_alphas = alpha_controller.apply_llm_update(llm_alphas)
        
        block_data = {
            "block_id": block_id,
            "episodes_in_block": episodes_per_block,
            "alpha_used": current_alphas.copy(),
            "block_kpis": averaged_kpis,
            "llm_input": {
                "input_a_summary": input_a,
                "input_b_summary": input_b,
                "input_c_blocks": len(llm_history)
            },
            "llm_output": {
                "raw_response": raw_response,
                "reasoning": reasoning,
                "self_correction": self_correction,
                "compiled_rule": compiled_rule,
                "alpha_recommended": llm_alphas
            },
            "alpha_after_update": applied_alphas.copy()
        }
        blocks_data.append(block_data)
        
        c_r, c_w, c_d, c_m = [averaged_kpis['cost_shares'][a]*100 for a in ["retailer", "warehouse", "distributor", "manufacturer"]]
        
        print(f"\n  BLOCK {block_id}/{n_blocks} -------------------------------------------------")
        print(f"  Alpha used:   R={current_alphas['retailer']:.2f}  W={current_alphas['warehouse']:.2f}  D={current_alphas['distributor']:.2f}  M={current_alphas['manufacturer']:.2f}")
        print(f"  Cost:         {averaged_kpis['total_cost']:,.0f}  | SL: {averaged_kpis['service_level']:.3f} | BW: {averaged_kpis['bullwhip_ratio']:.2f}")
        print(f"  Cost shares:  R={c_r:.0f}%  W={c_w:.0f}%  D={c_d:.0f}%  M={c_m:.0f}%")
        
        reasoning_print = reasoning.replace('\n', ' ')
        if len(reasoning_print) > 60: reasoning_print = reasoning_print[:57] + "..."
        rule_print = compiled_rule.replace('\n', ' ')
        if len(rule_print) > 60: rule_print = rule_print[:57] + "..."
        
        print(f"  LLM reasoning: \"{reasoning_print}\"")
        print(f"  LLM rule:     \"{rule_print}\"")
        print(f"  New alpha:    R={applied_alphas['retailer']:.2f}  W={applied_alphas['warehouse']:.2f}  D={applied_alphas['distributor']:.2f}  M={applied_alphas['manufacturer']:.2f}")
        print("  ------------------------------------------------------------")
        
        block_metrics_history.append(averaged_kpis)
        
        llm_record = {
            "block_id": block_id,
            "reasoning": reasoning,
            "compiled_rule": compiled_rule,
            "alphas": current_alphas.copy(),
            "applied_alphas": applied_alphas.copy(),
            "confidence": confidence,
            "actual_total_cost": averaged_kpis.get("total_cost", 0.0),
            "actual_bullwhip": averaged_kpis.get("bullwhip_ratio", 1.0),
            "actual_service_level": averaged_kpis.get("service_level", 1.0),
            "prediction_correct": False
        }
        if block_id > 1:
            llm_record["prev_cost"] = block_metrics_history[-2].get("total_cost", float('inf'))
            llm_record["prev_sl"] = block_metrics_history[-2].get("service_level", 1.0)
        llm_history.append(llm_record)
        
        total_cost_traj.append(averaged_kpis['total_cost'])
        sl_traj.append(averaged_kpis['service_level'])
        bw_traj.append(averaged_kpis['bullwhip_ratio'])
        for agent in AGENT_NAMES:
            alpha_traj[agent].append(current_alphas[agent])
        rules_traj.append(compiled_rule)
        
    cost_improvement_pct = 0.0
    if len(total_cost_traj) > 0 and total_cost_traj[0] > 0:
        cost_improvement_pct = ((total_cost_traj[0] - total_cost_traj[-1]) / total_cost_traj[0]) * 100
        
    print("\n  FINAL SUMMARY")
    print(f"  Cost:    {total_cost_traj[0]:,.0f} -> {total_cost_traj[-1]:,.0f} ({cost_improvement_pct:+.1f}% change)")
    print(f"  SL:      {sl_traj[0]:.3f} -> {sl_traj[-1]:.3f}")
    print(f"  BW:      {bw_traj[0]:.2f} -> {bw_traj[-1]:.2f}")
    
    output = {
        "config": {
            "n_blocks": n_blocks,
            "episodes_per_block": episodes_per_block,
            "smoothing": 0.0,
            "initial_alpha": 0.5
        },
        "blocks": blocks_data,
        "summary": {
            "total_cost_trajectory": total_cost_traj,
            "service_level_trajectory": sl_traj,
            "bullwhip_trajectory": bw_traj,
            "alpha_trajectory": alpha_traj,
            "llm_rules_compiled": rules_traj,
            "cost_improvement_pct": float(cost_improvement_pct)
        }
    }
    
    os.makedirs("MADRL/simulations/output", exist_ok=True)
    out_path = "MADRL/simulations/output/demo2_alpha_impact.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    return output

if __name__ == "__main__":
    run_demo2()
