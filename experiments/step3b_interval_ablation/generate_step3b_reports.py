import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def generate_detailed_plots(summary_path: str, plots_dir: str):
    if not os.path.exists(summary_path):
        print(f"Summary path not found: {summary_path}")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Multi-strategy interval cost bar chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r["name"] for r in results.values()]
    costs = [r["summary"]["per_period_cost"]["mean"] for r in results.values()]
    stds = [r["summary"]["per_period_cost"]["std"] for r in results.values()]
    
    colors = ["#2980B9", "#27AE60", "#2ECC71", "#1ABC9C", "#8E44AD", "#9B59B6", "#AF7AC5", "#E67E22"][:len(names)]
    
    bars = ax.bar(names, costs, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=1.1)
    ax.set_title("Version B: Per-Period Cost Across Intervention Frequencies & Strategies", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Per-Period Cost (Holding + Backlog)", fontsize=11)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_interval_cost_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Bullwhip Amplification Across Intervals
    fig, ax = plt.subplots(figsize=(12, 6))
    amps = [r["summary"]["bullwhip_amplification_pct"]["mean"] for r in results.values()]
    amp_stds = [r["summary"]["bullwhip_amplification_pct"]["std"] for r in results.values()]
    
    bars = ax.bar(names, amps, yerr=amp_stds, capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=1.1)
    ax.set_title("Version B: Bullwhip Amplification ((F4/F1 - 1) %) by Strategy & Interval", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Amplification (%)", fontsize=11)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:+.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_interval_bullwhip_amplification.png"), dpi=300)
    plt.close()
    
    print(f"Generated detailed plots in: {plots_dir}")

if __name__ == "__main__":
    res_dir = os.path.join(os.path.dirname(__file__), "results")
    plt_dir = os.path.join(os.path.dirname(__file__), "plots")
    generate_detailed_plots(os.path.join(res_dir, "step3b_summary.json"), plt_dir)
