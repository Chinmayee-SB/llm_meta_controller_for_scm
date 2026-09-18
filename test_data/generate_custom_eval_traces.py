"""
generate_custom_eval_traces.py

Generates deterministic evaluation demand traces for:
1. Stress Test (Black Swan Pulse Shock): test_data/test_demand_stress_shock/
2. Sustained Demand Step-UP: test_data/test_demand_sustained_up/
3. Sustained Demand Step-DOWN: test_data/test_demand_sustained_down/
"""

import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

def generate_stress_shock_traces(n_traces=10, horizon=200, seed=1001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_stress_shock")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating {n_traces} Stress Pulse Shock traces in {out_dir}...")
    for i in range(n_traces):
        # Base Poisson demand around 10
        base = rng.poisson(lam=10, size=horizon)
        base = np.clip(base, 5, 15)
        
        # Shock pulse between t=60 and t=74 (15 timesteps)
        shock_start = 60
        shock_end = 75
        for t in range(shock_start, shock_end):
            # Extreme surge: 35 to 45 units
            base[t] = rng.randint(35, 46)
            
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in base:
                f.write(f"{int(val)}\n")
        print(f"  [Created] {os.path.basename(fpath)} (Mean: {np.mean(base):.2f}, Peak: {np.max(base)})")

def generate_sustained_step_up_traces(n_traces=10, horizon=200, seed=2001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_sustained_up")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nGenerating {n_traces} Sustained Step-UP traces in {out_dir}...")
    for i in range(n_traces):
        trace = np.zeros(horizon, dtype=int)
        # Phase 1: t=0..79, Normal mean=10
        p1 = rng.poisson(lam=10, size=80)
        trace[:80] = np.clip(p1, 6, 14)
        
        # Phase 2: t=80..199, High sustained demand mean=19 (close to max single-step action 20)
        p2 = rng.poisson(lam=19, size=120)
        trace[80:] = np.clip(p2, 16, 22)
        
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in trace:
                f.write(f"{int(val)}\n")
        print(f"  [Created] {os.path.basename(fpath)} (P1 Mean: {np.mean(trace[:80]):.2f}, P2 Mean: {np.mean(trace[80:]):.2f})")

def generate_sustained_step_down_traces(n_traces=10, horizon=200, seed=3001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_sustained_down")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nGenerating {n_traces} Sustained Step-DOWN traces in {out_dir}...")
    for i in range(n_traces):
        trace = np.zeros(horizon, dtype=int)
        # Phase 1: t=0..79, High demand mean=18
        p1 = rng.poisson(lam=18, size=80)
        trace[:80] = np.clip(p1, 14, 20)
        
        # Phase 2: t=80..199, Low collapsed demand mean=5
        p2 = rng.poisson(lam=5, size=120)
        trace[80:] = np.clip(p2, 2, 8)
        
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in trace:
                f.write(f"{int(val)}\n")
        print(f"  [Created] {os.path.basename(fpath)} (P1 Mean: {np.mean(trace[:80]):.2f}, P2 Mean: {np.mean(trace[80:]):.2f})")

if __name__ == "__main__":
    generate_stress_shock_traces()
    generate_sustained_step_up_traces()
    generate_sustained_step_down_traces()
    print("\nAll evaluation traces generated successfully!")
