import os
import sys
import json
import yaml
import numpy as np

# Reusing Phase 2 components directly
from MADRL.training.run_phase2 import (
    load_frozen_policy,
    run_evaluation_block,
    build_input_a,
    build_input_b,
    build_input_c,
    build_full_prompt,
    parse_llm_response,
    run_single_episode,
    SYSTEM_PROMPT,
    EVAL_SEEDS
)
from MADRL.llm.rulebook import extract_rulebook
from MADRL.llm.llm_client import LLMClient
from MADRL.llm.alpha_controller import AlphaController
from MADRL.envs.serial import Env as SerialEnv

def ablation_warmup(env, frozen_actors, llm_client, alpha_controller, config, output_dir, n_blocks=10):
    llm_history = []
    block_metrics_history = []
    
    warmup_dir = os.path.join(output_dir, "warmup")
    os.makedirs(warmup_dir, exist_ok=True)
    
    for block_id in range(n_blocks):
        current_alphas = alpha_controller.get_current()
        
        # 1. Run evaluation block (5 episodes)
        averaged_kpis, full_demand_trace = run_evaluation_block(env, frozen_actors, current_alphas, n_episodes=5)
        
        # 2. Build Input A, B, C
        input_a = build_input_a(full_demand_trace)
        input_b = build_input_b(block_metrics_history, averaged_kpis)
        input_c = build_input_c(llm_history)
        
        # 3. Build full prompt
        prompt = build_full_prompt(input_a, input_b, input_c, SYSTEM_PROMPT, rulebook=None)
        
        # 4. Query LLM -> parse response
        raw_response = llm_client.query(prompt, SYSTEM_PROMPT, block_id=block_id)
        llm_alphas, reasoning, self_correction, compiled_rule, confidence = parse_llm_response(raw_response, current_alphas)
        
        # 5. On parse failure: skip update
        if reasoning == "PARSE_FAILURE":
            applied_alphas = current_alphas.copy()
            delta_before = {k: 0.0 for k in current_alphas}
            delta_after = {k: 0.0 for k in current_alphas}
            raw_llm_alphas = current_alphas.copy()
        else:
            raw_llm_alphas = {}
            for idx, agent in enumerate(["retailer", "warehouse", "distributor", "manufacturer"]):
                if idx in llm_alphas:
                    raw_llm_alphas[agent] = float(llm_alphas[idx])
                elif agent in llm_alphas:
                    raw_llm_alphas[agent] = float(llm_alphas[agent])
                else:
                    raw_llm_alphas[agent] = current_alphas[agent]
                    
            # 6. Apply alpha update
            applied_alphas = alpha_controller.apply_llm_update(llm_alphas)
            
            # calculate deltas
            delta_before = {}
            delta_after = {}
            for agent in current_alphas:
                # clip raw to min/max to be fair
                clipped = max(alpha_controller.alpha_min, min(alpha_controller.alpha_max, raw_llm_alphas[agent]))
                delta_before[agent] = clipped - current_alphas[agent]
                delta_after[agent] = applied_alphas[agent] - current_alphas[agent]
            
        # 7. Append to history
        block_metrics_history.append(averaged_kpis)
        
        record = {
            "block_id": block_id,
            "reasoning": reasoning,
            "compiled_rule": compiled_rule,
            "alphas": current_alphas.copy(),
            "raw_llm_alphas": raw_llm_alphas.copy(),
            "applied_alphas": applied_alphas.copy(),
            "delta_before_smoothing": delta_before.copy(),
            "delta_after_smoothing": delta_after.copy(),
            "confidence": confidence,
            "actual_total_cost": averaged_kpis.get("total_cost", 0.0),
            "actual_bullwhip": averaged_kpis.get("bullwhip_ratio", 1.0),
            "actual_service_level": averaged_kpis.get("service_level", 1.0)
        }
        llm_history.append(record)
        
        # 9. Save block log
        with open(os.path.join(warmup_dir, f"block_{block_id}.json"), "w") as f:
            json.dump(record, f, indent=2)
            
    return llm_history, block_metrics_history

def ablation_evaluation(env, frozen_actors, llm_client, rulebook, config, eval_seeds, output_dir):
    alpha_controller = AlphaController(config, initial_alpha=0.5)
    results = []
    
    eval_dir = os.path.join(output_dir, "official_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    for seed in eval_seeds:
        alpha_controller.reset(0.5)
        
        kpis_initial, demand_trace_initial = run_single_episode(env, frozen_actors, alpha_controller.get_current(), demand_seed=seed)
        input_a = build_input_a(demand_trace_initial)
        input_b = build_input_b([], kpis_initial)
        prompt = build_full_prompt(input_a, input_b, [], SYSTEM_PROMPT, rulebook=rulebook)
        
        raw_response = llm_client.query(prompt, SYSTEM_PROMPT, block_id=seed)
        llm_alphas, reasoning, _, _, _ = parse_llm_response(raw_response, alpha_controller.get_current())
        
        if reasoning != "PARSE_FAILURE":
            alpha_controller.apply_llm_update(llm_alphas)
            
        kpis_final, _ = run_single_episode(env, frozen_actors, alpha_controller.get_current(), demand_seed=seed)
        
        record = {
            "seed": seed,
            "initial_cost": kpis_initial.get("total_cost", 0),
            "final_cost": kpis_final.get("total_cost", 0),
            "final_service_level": kpis_final.get("service_level", 0.0),
            "final_bullwhip": kpis_final.get("bullwhip_ratio", 1.0),
            "applied_alphas": alpha_controller.get_current(),
            "reasoning": reasoning
        }
        results.append(record)
        
        with open(os.path.join(eval_dir, f"trace_{seed}.json"), "w") as f:
            json.dump(record, f, indent=2)
            
    costs = [r["final_cost"] for r in results]
    sls = [r["final_service_level"] for r in results]
    bws = [r["final_bullwhip"] for r in results]
    
    summary = {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_service_level": float(np.mean(sls)),
        "std_service_level": float(np.std(sls)),
        "mean_bullwhip": float(np.mean(bws)),
        "std_bullwhip": float(np.std(bws)),
    }
    
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

def main():
    print("\n" + "="*60)
    print("Running isolated smoothing ablation mode.")
    print("No production files will be modified.")
    print("="*60 + "\n")
    
    try:
        with open("MADRL/config/default_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)
        
    class DummyArgs:
        pass
    env_args = DummyArgs()
    env_args.num_agents = 4
    env_args.obs_dim = 8
    
    try:
        env = SerialEnv(env_args)
    except Exception as e:
        print(f"Failed to load SerialEnv: {e}")
        sys.exit(1)
        
    try:
        frozen_actors = load_frozen_policy("MADRL/results/phase1_weights/frozen_policy.pt", config)
    except Exception as e:
        print(f"Failed to load frozen policy: {e}")
        sys.exit(1)
        
    if "OPENAI_API_KEY" not in os.environ:
        class MockLLMClient:
            def query(self, prompt, sys_prompt, block_id=-1):
                return '{"reasoning": "Mocked", "self_correction": "", "compiled_rule": "Mock Rule", "alphas": {"retailer": 0.4, "warehouse": 0.4, "distributor": 0.4, "manufacturer": 0.4}}'
        llm_client = MockLLMClient()
    else:
        llm_client = LLMClient()
        
    base_ablation_dir = "MADRL/results/phase2_ablation_smoothing"
    os.makedirs(base_ablation_dir, exist_ok=True)
    
    smoothing_values = [0.8, 0.5, 0.0]
    comparison_results = {}
    rulebooks = {}
    histories = {}
    
    print("\n" + "="*60)
    print("PHASE 1: RUNNING WARMUPS")
    print("="*60)
    for smoothing in smoothing_values:
        print("\n" + "="*40)
        print(f"WARMUP EXPERIMENT: alpha_smoothing = {smoothing}")
        print("="*40)
        
        # Configure smoothing
        config["phase2"]["alpha_smoothing"] = smoothing
        alpha_controller = AlphaController(config)
        
        out_dir_name = f"smoothing_{str(smoothing).replace('.', '_')}"
        exp_dir = os.path.join(base_ablation_dir, out_dir_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print("Running Warmup...")
        llm_history, _ = ablation_warmup(env, frozen_actors, llm_client, alpha_controller, config, exp_dir, n_blocks=10)
        
        print("Extracting Rulebook...")
        rulebook = extract_rulebook(llm_history)
        with open(os.path.join(exp_dir, "rulebook.json"), "w") as f:
            json.dump(rulebook, f, indent=2)
            
        rulebooks[smoothing] = rulebook
        histories[smoothing] = llm_history
        
    print("\n" + "="*60)
    print("PHASE 2: RUNNING EVALUATIONS")
    print("="*60)
    for smoothing in smoothing_values:
        print("\n" + "="*40)
        print(f"EVAL EXPERIMENT: alpha_smoothing = {smoothing}")
        print("="*40)
        
        config["phase2"]["alpha_smoothing"] = smoothing
        out_dir_name = f"smoothing_{str(smoothing).replace('.', '_')}"
        exp_dir = os.path.join(base_ablation_dir, out_dir_name)
        
        rulebook = rulebooks[smoothing]
        llm_history = histories[smoothing]
        
        print("Running Official Evaluation...")
        eval_summary = ablation_evaluation(env, frozen_actors, llm_client, rulebook, config, EVAL_SEEDS, exp_dir)
        
        # Compute trajectory stats from llm_history
        all_alphas = {agent: [] for agent in ["retailer", "warehouse", "distributor", "manufacturer"]}
        absolute_movements = []
        
        for record in llm_history:
            applied = record["applied_alphas"]
            for a in applied:
                all_alphas[a].append(applied[a])
            
            delta_after = record["delta_after_smoothing"]
            mean_block_movement = np.mean([abs(v) for v in delta_after.values()])
            absolute_movements.append(mean_block_movement)
            
        traj_stats = {}
        for agent in all_alphas:
            traj_stats[agent] = {
                "avg": float(np.mean(all_alphas[agent])),
                "std": float(np.std(all_alphas[agent])),
                "min": float(np.min(all_alphas[agent])),
                "max": float(np.max(all_alphas[agent]))
            }
            
        mean_movement = float(np.mean(absolute_movements))
        
        comparison_results[f"smoothing_{smoothing}"] = {
            "mean_cost": eval_summary["mean_cost"],
            "std_cost": eval_summary["std_cost"],
            "mean_service_level": eval_summary["mean_service_level"],
            "std_service_level": eval_summary["std_service_level"],
            "mean_bullwhip": eval_summary["mean_bullwhip"],
            "std_bullwhip": eval_summary["std_bullwhip"],
            "alpha_trajectory_stats": traj_stats,
            "mean_alpha_movement_per_block": mean_movement
        }
        
    with open(os.path.join(base_ablation_dir, "comparison_table.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
        
    print("\n" + "="*60)
    print("FINAL ABLATION RESULTS")
    print("="*60)
    for smoothing in smoothing_values:
        key = f"smoothing_{smoothing}"
        res = comparison_results[key]
        print(f"\nSmoothing: {smoothing}")
        print(f"  Cost: {res['mean_cost']:.0f} +- {res['std_cost']:.0f}")
        print(f"  SL:   {res['mean_service_level']:.3f} +- {res['std_service_level']:.3f}")
        print(f"  BW:   {res['mean_bullwhip']:.3f} +- {res['std_bullwhip']:.3f}")
        print(f"  Mean Alpha Movement/Block: {res['mean_alpha_movement_per_block']:.4f}")
        print("  Alpha Trajectory (Retailer):")
        ts = res["alpha_trajectory_stats"]["retailer"]
        print(f"    avg={ts['avg']:.3f}, std={ts['std']:.3f}, min={ts['min']:.3f}, max={ts['max']:.3f}")
        
if __name__ == "__main__":
    main()
