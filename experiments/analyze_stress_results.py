"""
analyze_stress_results.py

Aggregates and formats comparative benchmark results across Base Paper,
Fixed Cruiser, Rule-Based, and Tri-State Meta-LLM for:
1. Stress Test (Black Swan Pulse Shock)
2. Sustained Demand Step-UP
3. Sustained Demand Step-DOWN
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results_stress_and_regime")

def load_scenario_results(scenario: str) -> Dict[str, List[Dict]]:
    controllers = ["base_paper", "fixed_cruiser", "rule_based", "llm"]
    data = {}
    for c in controllers:
        fpath = os.path.join(RESULTS_DIR, f"{c}_{scenario}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                data[c] = json.load(f)
        else:
            print(f"[NOTE] Missing {fpath}")
    return data

def compute_controller_summary(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    costs = [r["cost_per_period"] for r in records]
    sls = [r["service_level"] for r in records]
    cfrs = [r["cfr"] for r in records]
    f_bars = [r["avg_f"] for r in records]
    amps = [r["amp"] for r in records]
    post_holds = [r.get("post_shock_holding_cost", 0.0) for r in records]
    t_recs = [r.get("t_rec", 0) for r in records]
    
    return {
        "cost_mean": float(np.mean(costs)),
        "cost_median": float(np.median(costs)),
        "cost_std": float(np.std(costs)),
        "sl_mean": float(np.mean(sls)),
        "cfr_mean": float(np.mean(cfrs)),
        "f_bar_mean": float(np.mean(f_bars)),
        "amp_mean": float(np.mean(amps)),
        "post_hold_mean": float(np.mean(post_holds)),
        "t_rec_mean": float(np.mean(t_recs))
    }

def format_scenario_table(scenario_name: str, title: str, data: Dict[str, List[Dict]]) -> str:
    lines = []
    lines.append(f"### {title}\n")
    lines.append("| Controller Strategy | Per-Period Cost | SL (%) | CFR (%) | Avg Fluctuation ($\\bar{F}$) | Bullwhip Amp ($A$) | Recovery Time ($T_{\\text{rec}}$) | Post-Shock Holding |")
    lines.append("|---|---|---|---|---|---|---|---|")
    
    name_map = {
        "base_paper": "Base Paper Specialist ($\\alpha=0.375$)",
        "fixed_cruiser": "Tri-State Fixed Cruiser ($\\alpha=0.375$)",
        "rule_based": "Tri-State Rule-Based Heuristic",
        "llm": "**Tri-State Meta-LLM Supervisor**"
    }
    
    for c in ["base_paper", "fixed_cruiser", "rule_based", "llm"]:
        if c not in data:
            continue
        stats = compute_controller_summary(data[c])
        if not stats:
            continue
        c_name = name_map.get(c, c)
        lines.append(
            f"| {c_name} | **{stats['cost_mean']:.2f}** $\\pm$ {stats['cost_std']:.1f} | "
            f"{stats['sl_mean']:.1f}% | {stats['cfr_mean']:.1f}% | {stats['f_bar_mean']:.3f} | "
            f"{stats['amp_mean']:+.1f}% | {stats['t_rec_mean']:.1f} steps | {stats['post_hold_mean']:.1f} |"
        )
    lines.append("\n")
    return "\n".join(lines)

def main():
    scenarios = [
        ("stress_shock", "Scenario 1: Stress Test (Black Swan Pulse Shock)"),
        ("sustained_up", "Scenario 2A: Sustained Demand Response (Step-UP Surge)"),
        ("sustained_down", "Scenario 2B: Sustained Demand Response (Step-DOWN Collapse)")
    ]
    
    full_report = ["# Comprehensive Stress Test & Sustained Demand Response Benchmark\n"]
    full_report.append("This document benchmarks the **Tri-State Meta-LLM Supervisor** against the **Base Paper Reproduction Policy** and fixed/heuristic baselines under extreme non-stationary disruptions.\n")
    
    for sc, title in scenarios:
        data = load_scenario_results(sc)
        table_str = format_scenario_table(sc, title, data)
        full_report.append(table_str)
        
    final_md = "\n".join(full_report)
    print(final_md)
    
    out_path = os.path.join(RESULTS_DIR, "stress_and_sustained_benchmark_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_md)
    print(f"\n[INFO] Summary saved to {out_path}")

if __name__ == "__main__":
    main()
