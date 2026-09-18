import os
import sys
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Filter non-critical warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from experiments.common.policy_loader import load_frozen_phase1_policy
from experiments.common.demand_loader import load_evaluation_demand_traces
from experiments.common.env_wrapper import SupplyChainSimulation
from experiments.common.metrics import aggregate_trace_metrics

# Base Paper Published Benchmarks (Merton Demand + No Fixed Cost + 4 Echelons)
BASE_PAPER_BENCHMARKS = {
    "Base Paper HAPPO (Specialist, α=0.625)": {
        "per_period_cost": 34.0,
        "avg_fluctuation": 0.758,
        "amplification_pct": 92.37,
        "service_level": None
    },
    "Base Paper PPO (Specialist)": {
        "per_period_cost": 37.0,
        "avg_fluctuation": 0.890,
        "amplification_pct": 122.8,
        "service_level": None
    },
    "Base Paper Non-Stationary Base Stock": {
        "per_period_cost": 42.4,
        "avg_fluctuation": 0.764,
        "amplification_pct": 101.9,
        "service_level": None
    }
}

DEFAULT_ALPHA_SWEEP = [
    {"name": "α = 0.1 (Cooperative)", "alphas": [0.1, 0.1, 0.1, 0.1], "tag": "alpha_0.1"},
    {"name": "α = 0.3", "alphas": [0.3, 0.3, 0.3, 0.3], "tag": "alpha_0.3"},
    {"name": "α = 0.5 (Balanced)", "alphas": [0.5, 0.5, 0.5, 0.5], "tag": "alpha_0.5"},
    {"name": "α = 0.625 (Base Paper Val)", "alphas": [0.625, 0.625, 0.625, 0.625], "tag": "alpha_0.625"},
    {"name": "α = 0.7", "alphas": [0.7, 0.7, 0.7, 0.7], "tag": "alpha_0.7"},
    {"name": "α = 0.9 (Selfish)", "alphas": [0.9, 0.9, 0.9, 0.9], "tag": "alpha_0.9"},
    {"name": "Heterogeneous Best (0.1, 0.2, 0.3, 0.5)", "alphas": [0.1, 0.2, 0.3, 0.5], "tag": "alpha_hetero"}
]

def run_step1_evaluation(
    alpha_configs: list = None,
    trace_indices: list = None,
    output_dir: str = None,
    save_plots: bool = True,
    verbose: bool = True
):
    """
    Executes Step 1: Fixed-Alpha Sweep on 20 Merton Demand Traces.
    Computes Base Paper metrics and system KPIs.
    """
    if alpha_configs is None:
        alpha_configs = DEFAULT_ALPHA_SWEEP
        
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        
    if verbose:
        print("=" * 80)
        print("STEP 1: ANCHOR BASELINES & BASE PAPER METRIC EVALUATION")
        print("=" * 80)
        print("Loading frozen policy from MADRL/results/phase1_weights/frozen_policy.pt...")
        
    actors = load_frozen_phase1_policy()
    sim = SupplyChainSimulation(actors)
    
    all_traces = load_evaluation_demand_traces()
    if trace_indices is not None:
        traces = [t for t in all_traces if t["trace_idx"] in trace_indices]
    else:
        traces = all_traces
        
    if verbose:
        print(f"Evaluating on {len(traces)} demand traces across {len(alpha_configs)} alpha configurations.\n")
        
    summary_by_config = []
    full_trace_breakdown = {}
    
    for cfg in alpha_configs:
        name = cfg["name"]
        alphas = cfg["alphas"]
        tag = cfg["tag"]
        
        if verbose:
            print(f"--> Running Configuration: {name} (alphas={alphas})")
            
        trace_metrics = []
        full_trace_breakdown[tag] = {
            "name": name,
            "alphas": alphas,
            "traces": []
        }
        
        for tr in traces:
            trace_idx = tr["trace_idx"]
            demand = tr["demand"]
            sim_res = sim.run_trace(demand, alphas, record_history=False)
            m = sim_res["metrics"]
            m["trace_idx"] = trace_idx
            trace_metrics.append(m)
            full_trace_breakdown[tag]["traces"].append(m)
            
        agg = aggregate_trace_metrics(trace_metrics)
        agg["config_name"] = name
        agg["alphas"] = alphas
        agg["tag"] = tag
        summary_by_config.append(agg)
        
        if verbose:
            print(f"    Per-Period Cost : {agg['per_period_cost']['mean']:>8.2f} ± {agg['per_period_cost']['std']:<6.2f}")
            print(f"    Avg Fluctuation : {agg['avg_fluctuation']['mean']:>8.3f} ± {agg['avg_fluctuation']['std']:<6.3f}")
            print(f"    Amplification % : {agg['bullwhip_amplification_pct']['mean']:>8.2f}% ± {agg['bullwhip_amplification_pct']['std']:<6.2f}%")
            print(f"    Service Level   : {agg['service_level']['mean']*100:>8.2f}% ± {agg['service_level']['std']*100:<6.2f}%")
            print(f"    Echelon F1..F4  : [{agg['echelon_fluctuations_mean']['F1_retailer']:.3f}, {agg['echelon_fluctuations_mean']['F2_warehouse']:.3f}, {agg['echelon_fluctuations_mean']['F3_distributor']:.3f}, {agg['echelon_fluctuations_mean']['F4_manufacturer']:.3f}]\n")

    # Save JSON results
    summary_path = os.path.join(output_dir, "step1_anchor_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_config, f, indent=2)
        
    breakdown_path = os.path.join(output_dir, "step1_trace_breakdown.json")
    with open(breakdown_path, "w", encoding="utf-8") as f:
        json.dump(full_trace_breakdown, f, indent=2)
        
    # Generate Markdown Comparison Table
    table_md_path = os.path.join(output_dir, "step1_comparison_table.md")
    md_content = generate_markdown_report(summary_by_config)
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    if verbose:
        print(md_content)
        
    # Generate Plots
    if save_plots:
        generate_step1_plots(summary_by_config, plots_dir)
        if verbose:
            print(f"\nPlots saved successfully to: {plots_dir}")
            
    return summary_by_config, full_trace_breakdown

def generate_markdown_report(summary_list: list) -> str:
    lines = []
    lines.append("# Step 1: Fixed-Alpha Baselines vs Base Paper Benchmarks")
    lines.append("")
    lines.append("Evaluation conducted over all 20 Merton Demand Traces (200 timesteps each).")
    lines.append("")
    lines.append("### Comprehensive Comparison Table")
    lines.append("")
    lines.append("| Policy / Configuration | Type | Per-Period Cost | Avg Fluct. (F) | Amplification (%) | Service Level (%) | Retailer Cost | Mfr Cost |")
    lines.append("|:-----------------------|:-----|----------------:|---------------:|------------------:|------------------:|--------------:|---------:|")
    
    # Base Paper entries
    for bp_name, bp_data in BASE_PAPER_BENCHMARKS.items():
        sl_str = "N/A" if bp_data["service_level"] is None else f"{bp_data['service_level']:.1f}%"
        lines.append(f"| **{bp_name}** | Specialist Benchmark | **{bp_data['per_period_cost']:.1f}** | **{bp_data['avg_fluctuation']:.3f}** | **{bp_data['amplification_pct']:.2f}%** | {sl_str} | - | - |")
        
    lines.append("|" + "-"*23 + "|" + "-"*6 + "|" + "-"*17 + "|" + "-"*16 + "|" + "-"*19 + "|" + "-"*19 + "|" + "-"*15 + "|" + "-"*10 + "|")
    
    # Sweep entries
    for item in summary_list:
        name = item["config_name"]
        cost_m = item["per_period_cost"]["mean"]
        cost_s = item["per_period_cost"]["std"]
        f_m = item["avg_fluctuation"]["mean"]
        f_s = item["avg_fluctuation"]["std"]
        amp_m = item["bullwhip_amplification_pct"]["mean"]
        amp_s = item["bullwhip_amplification_pct"]["std"]
        sl_m = item["service_level"]["mean"] * 100
        c_ret = item["echelon_per_period_costs_mean"]["retailer"]
        c_mfr = item["echelon_per_period_costs_mean"]["manufacturer"]
        
        lines.append(f"| Our Generalist: {name} | Static Policy | {cost_m:.2f} ± {cost_s:.1f} | {f_m:.3f} ± {f_s:.2f} | {amp_m:.2f}% ± {amp_s:.1f}% | {sl_m:.2f}% | {c_ret:.2f} | {c_mfr:.2f} |")
        
    lines.append("")
    lines.append("### Key Insights & Analysis")
    
    # Calculate alpha sensitivity
    costs = [item["per_period_cost"]["mean"] for item in summary_list if "α =" in item["config_name"]]
    if len(costs) >= 2:
        min_c = min(costs)
        max_c = max(costs)
        spread_pct = ((max_c - min_c) / min_c) * 100
        best_cfg = min(summary_list, key=lambda x: x["per_period_cost"]["mean"])
        lines.append(f"- **Best Static Alpha for Generalist Policy:** `{best_cfg['config_name']}` with Per-Period Cost `{best_cfg['per_period_cost']['mean']:.2f}`.")
        lines.append(f"- **Alpha Discrimination Spread:** {spread_pct:.1f}% difference between lowest and highest cost.")
        if spread_pct > 15:
            lines.append("- **Alpha Lever Assessment:** Strong lever (>15% cost spread). The policy actively differentiates cooperation levels.")
        else:
            lines.append("- **Alpha Lever Assessment:** Moderate/Weak lever (<15% cost spread).")
            
    lines.append("")
    return "\n".join(lines)

def generate_step1_plots(summary_list: list, plots_dir: str):
    # Filter uniform alphas
    uniform_items = [item for item in summary_list if "α =" in item["config_name"] and len(set(item["alphas"])) == 1]
    if not uniform_items:
        return
        
    alphas = [item["alphas"][0] for item in uniform_items]
    costs_mean = [item["per_period_cost"]["mean"] for item in uniform_items]
    costs_std = [item["per_period_cost"]["std"] for item in uniform_items]
    
    fluct_mean = [item["avg_fluctuation"]["mean"] for item in uniform_items]
    fluct_std = [item["avg_fluctuation"]["std"] for item in uniform_items]
    
    amp_mean = [item["bullwhip_amplification_pct"]["mean"] for item in uniform_items]
    amp_std = [item["bullwhip_amplification_pct"]["std"] for item in uniform_items]
    
    sl_mean = [item["service_level"]["mean"] * 100 for item in uniform_items]
    
    # 1. Cost vs Alpha Curve
    plt.figure(figsize=(8, 5), dpi=150)
    plt.errorbar(alphas, costs_mean, yerr=costs_std, fmt='-o', color='#1f77b4', ecolor='#aec7e8', elinewidth=2, capsize=4, label='Our Generalist Policy (Mean ± Std)')
    plt.axhline(y=34.0, color='#d62728', linestyle='--', linewidth=2, label='Base Paper HAPPO Specialist (~34.0)')
    plt.axhline(y=37.0, color='#ff7f0e', linestyle=':', linewidth=1.5, label='Base Paper PPO (~37.0)')
    plt.title("Per-Period System Cost vs. Incentive Parameter (α)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Cooperation / Incentive Level (α)", fontsize=11)
    plt.ylabel("Per-Period Total System Cost (h=1, b=1)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "alpha_cost_curve.png"))
    plt.close()
    
    # 2. Average Fluctuation vs Alpha Curve
    plt.figure(figsize=(8, 5), dpi=150)
    plt.errorbar(alphas, fluct_mean, yerr=fluct_std, fmt='-s', color='#2ca02c', ecolor='#98df8a', elinewidth=2, capsize=4, label='Our Generalist Policy (Mean ± Std)')
    plt.axhline(y=0.758, color='#d62728', linestyle='--', linewidth=2, label='Base Paper HAPPO Specialist (0.758)')
    plt.axhline(y=0.890, color='#ff7f0e', linestyle=':', linewidth=1.5, label='Base Paper PPO (0.890)')
    plt.title("Average Echelon Fluctuation (F) vs. Incentive Parameter (α)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Cooperation / Incentive Level (α)", fontsize=11)
    plt.ylabel("Average Demand Fluctuation F = (1/4) Σ F_i", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "alpha_fluctuation_curve.png"))
    plt.close()
    
    # 3. Bullwhip Amplification vs Alpha Curve
    plt.figure(figsize=(8, 5), dpi=150)
    plt.errorbar(alphas, amp_mean, yerr=amp_std, fmt='-^', color='#9467bd', ecolor='#c5b0d5', elinewidth=2, capsize=4, label='Our Generalist Policy (Mean ± Std)')
    plt.axhline(y=92.37, color='#d62728', linestyle='--', linewidth=2, label='Base Paper HAPPO Specialist (92.37%)')
    plt.axhline(y=122.8, color='#ff7f0e', linestyle=':', linewidth=1.5, label='Base Paper PPO (122.8%)')
    plt.title("Bullwhip Amplification (%) vs. Incentive Parameter (α)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Cooperation / Incentive Level (α)", fontsize=11)
    plt.ylabel("Amplification = (F_4 / F_1 - 1) × 100%", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "alpha_amplification_curve.png"))
    plt.close()
    
    # 4. Echelon Fluctuation Breakdown
    f1 = [item["echelon_fluctuations_mean"]["F1_retailer"] for item in uniform_items]
    f2 = [item["echelon_fluctuations_mean"]["F2_warehouse"] for item in uniform_items]
    f3 = [item["echelon_fluctuations_mean"]["F3_distributor"] for item in uniform_items]
    f4 = [item["echelon_fluctuations_mean"]["F4_manufacturer"] for item in uniform_items]
    
    x = np.arange(len(alphas))
    width = 0.18
    plt.figure(figsize=(9, 5), dpi=150)
    plt.bar(x - 1.5*width, f1, width, label='Echelon 1 (Retailer)', color='#1f77b4')
    plt.bar(x - 0.5*width, f2, width, label='Echelon 2 (Warehouse)', color='#ff7f0e')
    plt.bar(x + 0.5*width, f3, width, label='Echelon 3 (Distributor)', color='#2ca02c')
    plt.bar(x + 1.5*width, f4, width, label='Echelon 4 (Manufacturer)', color='#d62728')
    plt.xticks(x, [f"α={a}" for a in alphas], fontsize=10)
    plt.ylabel("Fluctuation Metric (F_i = CV of Orders)", fontsize=11)
    plt.title("Echelon-by-Echelon Fluctuation Across Alpha Sweep", fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "echelon_fluctuation_breakdown.png"))
    plt.close()

if __name__ == "__main__":
    run_step1_evaluation()
