import json
import os
import numpy as np

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def analyze_warmup_logs(log_path):
    logs = load_json(log_path)
    if not logs:
        return None
        
    alphas = [list(entry["applied_alphas"].values()) for entry in logs]
    mean_alphas = np.mean(alphas, axis=0)
    std_alphas = np.std(alphas, axis=0)
    
    alpha_shifts = []
    for i in range(1, len(alphas)):
        shift = np.mean(np.abs(np.array(alphas[i]) - np.array(alphas[i-1])))
        alpha_shifts.append(shift)
        
    return {
        "mean_alphas": mean_alphas,
        "std_alphas": std_alphas,
        "mean_shift": np.mean(alpha_shifts) if alpha_shifts else 0.0,
        "num_rules": len([entry for entry in logs if entry.get("compiled_rule")])
    }

def print_comparison():
    baseline_path = "MADRL/results/phase2/warmup/llm_warmup_log.json"
    diverse_path = "MADRL/results/phase2/warmup_diverse/llm_warmup_log.json"
    
    baseline_stats = analyze_warmup_logs(baseline_path)
    diverse_stats = analyze_warmup_logs(diverse_path)
    
    baseline_rulebook = load_json("MADRL/results/phase2/warmup/llm_rulebook.json") or []
    diverse_rulebook = load_json("MADRL/results/phase2/warmup_diverse/llm_rulebook.json") or []
    
    print("===========================================")
    print("WARMUP MODES COMPARISON REPORT")
    print("===========================================\n")
    
    if not baseline_stats or not diverse_stats:
        print("Error: Could not load logs for both warmup modes. Run both modes first.")
        return
        
    print(f"1. ALPHA TRAJECTORY & STABILITY")
    print("-------------------------------------------")
    np.set_printoptions(precision=3)
    print(f"Baseline Warmup:")
    print(f"  Mean Alphas (R,W,D,M): {baseline_stats['mean_alphas']}")
    print(f"  Std Alphas:  {baseline_stats['std_alphas']}")
    print(f"  Mean Shift Per Block:  {baseline_stats['mean_shift']:.4f}")
    
    print(f"\nDiverse-Seed Warmup:")
    print(f"  Mean Alphas (R,W,D,M): {diverse_stats['mean_alphas']}")
    print(f"  Std Alphas:  {diverse_stats['std_alphas']}")
    print(f"  Mean Shift Per Block:  {diverse_stats['mean_shift']:.4f}")
    print("\n")
    
    print(f"2. RULEBOOK DIFFERENCES")
    print("-------------------------------------------")
    print(f"Baseline Rulebook ({len(baseline_rulebook)} rules):")
    for r in baseline_rulebook:
        print(f"  - {r}")
        
    print(f"\nDiverse-Seed Rulebook ({len(diverse_rulebook)} rules):")
    for r in diverse_rulebook:
        print(f"  - {r}")
        
    print("\n===========================================")
    
if __name__ == "__main__":
    print_comparison()
