"""
generate_280_eval_traces.py

Generates and freezes exactly 280 deterministic evaluation demand traces:
- S1 Merton In-Distribution: 100 traces in test_data/test_demand_merton/ (000..099)
- S2 Black Swan Pulse Shock: 30 traces in test_data/test_demand_stress_shock/ (000..029)
- S3 Sustained Step-Up (+200%): 120 traces in test_data/test_demand_sustained_up/ (000..119)
- S4 Sustained Step-Down (-67%): 30 traces in test_data/test_demand_sustained_down/ (000..029)
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from envs import generator

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

def generate_s1_merton_traces(target_count=100):
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_merton")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Ensuring {target_count} S1 Merton traces in {out_dir}...")
    
    # Check existing files
    existing = [f for f in os.listdir(out_dir) if f.startswith("demand_") and f.endswith(".txt")]
    print(f"  Found {len(existing)} existing Merton traces.")
    
    # Generate missing up to target_count
    for i in range(len(existing), target_count):
        np.random.seed(5000 + i)
        # Standard Merton generator from MADRL/envs/serial.py
        g = generator.merton(200, 20, mu=5e-5, sigma=1e-2, lamda=15, l1=10, l2=0, l3=0.01)
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in g.demand_list:
                f.write(f"{int(val)}\n")
        if (i + 1) % 10 == 0 or i == len(existing):
            print(f"  [Created] {os.path.basename(fpath)} (Mean: {np.mean(g.demand_list):.2f})")
    print(f"  [Complete] S1 Merton now has {len(os.listdir(out_dir))} traces.")

def generate_s2_stress_shock_traces(n_traces=30, horizon=200, seed=1001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_stress_shock")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating {n_traces} S2 Stress Pulse Shock traces in {out_dir}...")
    for i in range(n_traces):
        base = rng.poisson(lam=10, size=horizon)
        base = np.clip(base, 5, 15)
        # Pulse shock t=60..74
        for t in range(60, 75):
            base[t] = rng.randint(35, 46)
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in base:
                f.write(f"{int(val)}\n")
    print(f"  [Complete] S2 Pulse Shock has {n_traces} traces (000..{n_traces-1:03d}).")

def generate_s3_sustained_step_up_traces(n_traces=120, horizon=200, seed=2001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_sustained_up")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating {n_traces} S3 Sustained Step-UP traces in {out_dir}...")
    for i in range(n_traces):
        trace = np.zeros(horizon, dtype=int)
        # Phase 1: t=0..79, Normal mean=10
        p1 = rng.poisson(lam=10, size=80)
        trace[:80] = np.clip(p1, 6, 14)
        # Phase 2: t=80..199, High sustained mean=19
        p2 = rng.poisson(lam=19, size=120)
        trace[80:] = np.clip(p2, 16, 22)
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in trace:
                f.write(f"{int(val)}\n")
    print(f"  [Complete] S3 Step-Up has {n_traces} traces (000..{n_traces-1:03d}).")

def generate_s4_sustained_step_down_traces(n_traces=30, horizon=200, seed=3001):
    rng = np.random.RandomState(seed)
    out_dir = os.path.join(TEST_DATA_DIR, "test_demand_sustained_down")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating {n_traces} S4 Sustained Step-DOWN traces in {out_dir}...")
    for i in range(n_traces):
        trace = np.zeros(horizon, dtype=int)
        # Phase 1: t=0..79, High demand mean=18
        p1 = rng.poisson(lam=18, size=80)
        trace[:80] = np.clip(p1, 14, 20)
        # Phase 2: t=80..199, Low demand mean=5
        p2 = rng.poisson(lam=5, size=120)
        trace[80:] = np.clip(p2, 2, 8)
        fpath = os.path.join(out_dir, f"demand_{i:03d}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for val in trace:
                f.write(f"{int(val)}\n")
    print(f"  [Complete] S4 Step-Down has {n_traces} traces (000..{n_traces-1:03d}).")

if __name__ == "__main__":
    generate_s1_merton_traces(100)
    generate_s2_stress_shock_traces(30)
    generate_s3_sustained_step_up_traces(120)
    generate_s4_sustained_step_down_traces(30)
    print("\n[SUCCESS] Exactly 280 evaluation traces (100 + 30 + 120 + 30) generated and frozen!")
