import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root and step4 to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEP4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if STEP4_DIR not in sys.path:
    sys.path.insert(0, STEP4_DIR)

def generate_comparison_figures():
    s4_json = os.path.join(STEP4_DIR, "results", "step4_summary.json")
    p1_json = os.path.join(PROJECT_ROOT, "experiments", "step3_intra_episodic_controller", "results", "step3a_summary.json")
    plots_dir = os.path.join(STEP4_DIR, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isfile(s4_json):
        print(f"Results file not found: {s4_json}")
        return

    with open(s4_json, "r", encoding="utf-8") as f:
        step4_data = json.load(f)

    phase1_data = None
    if os.path.isfile(p1_json):
        with open(p1_json, "r", encoding="utf-8") as f:
            phase1_data = json.load(f)

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif'
    })

    # 1. Cost Comparison: Phase 1 (Static Training) vs Step 4 (Dynamic Retraining)
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["fixed_alpha_0.3", "rule_based", "random_alpha"]
    labels = ["Fixed Static (α=0.3)", "Rule-Based (N=50)", "Random Dynamic (N=50)"]
    if "llm_controller" in step4_data:
        strategies.append("llm_controller")
        labels.append("Gemini Intra (N=50)")

    x = np.arange(len(strategies))
    width = 0.35

    s4_means = [step4_data[s]["summary"]["per_period_cost"]["mean"] for s in strategies if s in step4_data]
    s4_stds = [step4_data[s]["summary"]["per_period_cost"]["std"] for s in strategies if s in step4_data]

    p1_means = []
    p1_stds = []
    if phase1_data:
        for s in strategies:
            if s in phase1_data:
                p1_means.append(phase1_data[s]["summary"]["per_period_cost"]["mean"])
                p1_stds.append(phase1_data[s]["summary"]["per_period_cost"]["std"])
            else:
                p1_means.append(0)
                p1_stds.append(0)

    if phase1_data:
        rects1 = ax.bar(x - width/2, p1_means, width, yerr=p1_stds, label='Phase 1 Policy (Static Training)', color='#94a3b8', capsize=5, alpha=0.9)
        rects2 = ax.bar(x + width/2, s4_means, width, yerr=s4_stds, label='Step 4 Policy (Dynamic Retraining)', color='#2563eb', capsize=5, alpha=0.9)
    else:
        rects2 = ax.bar(x, s4_means, width, yerr=s4_stds, label='Step 4 Dynamic Policy', color='#2563eb', capsize=5, alpha=0.9)

    ax.axhline(34.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Base Paper Specialist (34.0)')

    ax.set_ylabel('Per-Period Cost (Holding + Backlog)')
    ax.set_title('Per-Period Cost: Phase 1 (Static Training) vs. Step 4 (Dynamic Retraining)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add value annotations on bars
    if phase1_data:
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path1 = os.path.join(plots_dir, "step4_cost_comparison_vs_phase1.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved: {plot_path1}")

    # 2. Bullwhip Amplification Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    s4_bw = [step4_data[s]["summary"]["bullwhip_amplification_pct"]["mean"] for s in strategies if s in step4_data]
    s4_bw_std = [step4_data[s]["summary"]["bullwhip_amplification_pct"]["std"] for s in strategies if s in step4_data]

    p1_bw = []
    p1_bw_std = []
    if phase1_data:
        for s in strategies:
            if s in phase1_data:
                p1_bw.append(phase1_data[s]["summary"]["bullwhip_amplification_pct"]["mean"])
                p1_bw_std.append(phase1_data[s]["summary"]["bullwhip_amplification_pct"]["std"])
            else:
                p1_bw.append(0)
                p1_bw_std.append(0)

    if phase1_data:
        rects_bw1 = ax.bar(x - width/2, p1_bw, width, yerr=p1_bw_std, label='Phase 1 Policy (Static Training)', color='#94a3b8', capsize=5, alpha=0.9)
        rects_bw2 = ax.bar(x + width/2, s4_bw, width, yerr=s4_bw_std, label='Step 4 Dynamic Retrained Policy', color='#059669', capsize=5, alpha=0.9)
    else:
        rects_bw2 = ax.bar(x, s4_bw, width, yerr=s4_bw_std, label='Step 4 Dynamic Retrained Policy', color='#059669', capsize=5, alpha=0.9)

    ax.axhline(92.37, color='#dc2626', linestyle='--', linewidth=1.5, label='Base Paper Specialist (+92.4%)')
    ax.set_ylabel('Bullwhip Amplification (%)')
    ax.set_title('Bullwhip Amplification: Phase 1 vs. Step 4 Dynamic Retraining')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add value annotations
    if phase1_data:
        for rect in rects_bw1:
            h = rect.get_height()
            ax.annotate(f'{h:+.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, max(h, 0)),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')
    for rect in rects_bw2:
        h = rect.get_height()
        ax.annotate(f'{h:+.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, max(h, 0)),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, "step4_bullwhip_comparison_vs_phase1.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved: {plot_path2}")

    # 3. Static Alpha Sweep (0.1 to 0.9) on Dynamic Policy
    fig, ax = plt.subplots(figsize=(8, 5))
    alpha_keys = ["fixed_alpha_0.1", "fixed_alpha_0.3", "fixed_alpha_0.5", "fixed_alpha_0.7", "fixed_alpha_0.9"]
    alpha_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_costs = [step4_data[k]["summary"]["per_period_cost"]["mean"] for k in alpha_keys if k in step4_data]
    alpha_cost_stds = [step4_data[k]["summary"]["per_period_cost"]["std"] for k in alpha_keys if k in step4_data]

    ax.errorbar(alpha_vals[:len(alpha_costs)], alpha_costs, yerr=alpha_cost_stds, fmt='-o', color='#2563eb', ecolor='#93c5fd', capsize=5, linewidth=2, markersize=8, label='Step 4 Dynamic Policy Cost')
    ax.axhline(34.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Base Paper Specialist (34.0)')
    ax.set_xlabel('Fixed Alpha Parameter (α)')
    ax.set_ylabel('Per-Period Cost')
    ax.set_title('Step 4 Dynamic Policy Cost Across Static Alpha Spectrum')
    ax.set_xticks(alpha_vals)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path3 = os.path.join(plots_dir, "step4_static_alpha_spectrum.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"Saved: {plot_path3}")

    print("All comparison plots generated successfully.")

if __name__ == "__main__":
    generate_comparison_figures()
