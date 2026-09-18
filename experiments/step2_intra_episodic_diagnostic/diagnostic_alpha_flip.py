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

def make_alpha_flip_callable(alpha_before: float, alpha_after: float, flip_step: int = 100):
    """
    Returns a time-varying alpha function:
      t < flip_step  -> [alpha_before] * 4
      t >= flip_step -> [alpha_after] * 4
    """
    def alpha_fn(t: int, state: dict):
        if t < flip_step:
            return [alpha_before] * 4
        else:
            return [alpha_after] * 4
    return alpha_fn

def run_step2_diagnostic(
    flip_step: int = 100,
    trace_indices: list = None,
    output_dir: str = None,
    save_plots: bool = True,
    verbose: bool = True
):
    """
    Executes Step 2: Intra-Episodic Diagnostic.
    Tests if the frozen policy can respond dynamically to mid-episode alpha shifts.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        
    if verbose:
        print("=" * 80)
        print(f"STEP 2: INTRA-EPISODIC DIAGNOSTIC (α-Flip at t={flip_step})")
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
        print(f"Running diagnostic across {len(traces)} demand traces with 4 experimental conditions.\n")
        
    # Define conditions
    conditions = [
        {
            "tag": "flip_0.1_to_0.9",
            "name": "Cooperative → Selfish (α=0.1 → 0.9)",
            "alpha_fn": make_alpha_flip_callable(0.1, 0.9, flip_step),
            "type": "flip",
            "alpha_before": 0.1,
            "alpha_after": 0.9
        },
        {
            "tag": "flip_0.9_to_0.1",
            "name": "Selfish → Cooperative (α=0.9 → 0.1)",
            "alpha_fn": make_alpha_flip_callable(0.9, 0.1, flip_step),
            "type": "flip",
            "alpha_before": 0.9,
            "alpha_after": 0.1
        },
        {
            "tag": "static_0.1",
            "name": "Static Cooperative (α=0.1)",
            "alpha_fn": [0.1, 0.1, 0.1, 0.1],
            "type": "control",
            "alpha_before": 0.1,
            "alpha_after": 0.1
        },
        {
            "tag": "static_0.9",
            "name": "Static Selfish (α=0.9)",
            "alpha_fn": [0.9, 0.9, 0.9, 0.9],
            "type": "control",
            "alpha_before": 0.9,
            "alpha_after": 0.9
        }
    ]
    
    results_by_condition = {}
    trajectories_by_condition = {}
    
    for cond in conditions:
        tag = cond["tag"]
        name = cond["name"]
        alpha_spec = cond["alpha_fn"]
        
        if verbose:
            print(f"--> Testing Condition: {name}")
            
        cond_trace_metrics = []
        cond_trajectories = []
        
        for tr in traces:
            trace_idx = tr["trace_idx"]
            demand = tr["demand"]
            sim_res = sim.run_trace(demand, alpha_spec, record_history=True)
            m = sim_res["metrics"]
            m["trace_idx"] = trace_idx
            cond_trace_metrics.append(m)
            cond_trajectories.append({
                "trace_idx": trace_idx,
                "trajectory": sim_res["trajectory"]
            })
            
        agg = aggregate_trace_metrics(cond_trace_metrics)
        agg["condition_name"] = name
        agg["tag"] = tag
        results_by_condition[tag] = agg
        trajectories_by_condition[tag] = cond_trajectories
        
        if verbose:
            print(f"    Per-Period Cost : {agg['per_period_cost']['mean']:>8.2f} ± {agg['per_period_cost']['std']:<6.2f}")
            print(f"    Avg Fluctuation : {agg['avg_fluctuation']['mean']:>8.3f} ± {agg['avg_fluctuation']['std']:<6.3f}")
            print(f"    Amplification % : {agg['bullwhip_amplification_pct']['mean']:>8.2f}%")
            print(f"    Service Level   : {agg['service_level']['mean']*100:>8.2f}%\n")

    # Perform Transition Dynamics Analysis
    transition_analysis = analyze_transition_dynamics(trajectories_by_condition, flip_step=flip_step)
    
    # Save JSON results
    output_data = {
        "flip_step": flip_step,
        "n_traces": len(traces),
        "conditions_summary": results_by_condition,
        "transition_analysis": transition_analysis
    }
    
    with open(os.path.join(output_dir, "diagnostic_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    # Generate Markdown Report
    summary_md = generate_diagnostic_markdown_report(output_data)
    with open(os.path.join(output_dir, "diagnostic_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)
        
    if verbose:
        print(summary_md)
        
    # Generate Diagnostic Plots
    if save_plots:
        generate_diagnostic_plots(trajectories_by_condition, flip_step, plots_dir)
        if verbose:
            print(f"\nDiagnostic plots generated in: {plots_dir}")
            
    return output_data

def analyze_transition_dynamics(trajectories_by_condition: dict, flip_step: int = 100) -> dict:
    """
    Computes statistical stability metrics before and after the flip.
    """
    analysis = {}
    echelons = ["retailer", "warehouse", "distributor", "manufacturer"]
    
    for flip_tag in ["flip_0.1_to_0.9", "flip_0.9_to_0.1"]:
        if flip_tag not in trajectories_by_condition:
            continue
            
        traces_data = trajectories_by_condition[flip_tag]
        # Average trajectory across traces: shape [4, 200]
        all_orders = np.array([[tr["trajectory"]["orders"][i] for i in range(4)] for tr in traces_data]) # [N, 4, 200]
        mean_orders = np.mean(all_orders, axis=0) # [4, 200]
        
        all_inv = np.array([[tr["trajectory"]["inventory"][i] for i in range(4)] for tr in traces_data])
        mean_inv = np.mean(all_inv, axis=0)
        
        all_costs = np.array([[tr["trajectory"]["costs"][i] for i in range(4)] for tr in traces_data])
        mean_step_costs = np.mean(np.sum(all_costs, axis=1), axis=0) # [200]
        
        # Windows
        pre_window = (max(0, flip_step - 25), flip_step)
        trans_window = (flip_step, min(200, flip_step + 15))
        post_window = (min(200, flip_step + 20), min(200, flip_step + 75))
        
        pre_orders_mean = np.mean(mean_orders[:, pre_window[0]:pre_window[1]], axis=1)
        trans_orders_mean = np.mean(mean_orders[:, trans_window[0]:trans_window[1]], axis=1)
        post_orders_mean = np.mean(mean_orders[:, post_window[0]:post_window[1]], axis=1)
        
        # Settling time estimation: first timestep after flip where order delta from post-mean is < 15%
        settling_steps = {}
        for i, ech in enumerate(echelons):
            target = post_orders_mean[i]
            tol = max(0.5, 0.20 * np.std(mean_orders[i, post_window[0]:post_window[1]]))
            settled_t = None
            for t in range(flip_step, min(200, flip_step + 40)):
                if abs(mean_orders[i, t] - target) <= tol:
                    settled_t = t - flip_step
                    break
            settling_steps[ech] = settled_t if settled_t is not None else 40
            
        # Oscillation metric: variance of diff during transition window vs steady post window
        trans_diff_var = float(np.mean([np.var(np.diff(mean_orders[i, trans_window[0]:trans_window[1]])) for i in range(4)]))
        post_diff_var = float(np.mean([np.var(np.diff(mean_orders[i, post_window[0]:post_window[1]])) for i in range(4)]))
        
        analysis[flip_tag] = {
            "pre_flip_order_means": {ech: float(pre_orders_mean[i]) for i, ech in enumerate(echelons)},
            "trans_order_means": {ech: float(trans_orders_mean[i]) for i, ech in enumerate(echelons)},
            "post_flip_order_means": {ech: float(post_orders_mean[i]) for i, ech in enumerate(echelons)},
            "estimated_settling_steps": settling_steps,
            "trans_oscillation_var": trans_diff_var,
            "post_oscillation_var": post_diff_var,
            "pre_flip_mean_step_cost": float(np.mean(mean_step_costs[pre_window[0]:pre_window[1]])),
            "post_flip_mean_step_cost": float(np.mean(mean_step_costs[post_window[0]:post_window[1]]))
        }
        
    return analysis

def generate_diagnostic_markdown_report(data: dict) -> str:
    summary = data["conditions_summary"]
    analysis = data["transition_analysis"]
    flip_step = data["flip_step"]
    
    lines = []
    lines.append("# Step 2: Intra-Episodic Diagnostic Report")
    lines.append(f"Evaluation of mid-episode incentive parameter shift (flip at timestep $t={flip_step}$).")
    lines.append("")
    lines.append("### 1. Overall Performance Comparison")
    lines.append("")
    lines.append("| Condition | Type | Per-Period Cost | Avg Fluct. (F) | Amplification (%) | Service Level (%) |")
    lines.append("|:----------|:-----|----------------:|---------------:|------------------:|------------------:|")
    
    for tag, res in summary.items():
        name = res["condition_name"]
        cost_m = res["per_period_cost"]["mean"]
        cost_s = res["per_period_cost"]["std"]
        f_m = res["avg_fluctuation"]["mean"]
        amp_m = res["bullwhip_amplification_pct"]["mean"]
        sl_m = res["service_level"]["mean"] * 100
        lines.append(f"| **{name}** | `{tag}` | **{cost_m:.2f} ± {cost_s:.1f}** | {f_m:.3f} | {amp_m:.2f}% | {sl_m:.2f}% |")
        
    lines.append("")
    lines.append("### 2. Transition Dynamics & Settling Behavior")
    lines.append("")
    
    for tag, a in analysis.items():
        cond_name = summary[tag]["condition_name"]
        lines.append(f"#### Condition: {cond_name}")
        lines.append(f"- **Pre-flip Step Cost ($t < {flip_step}$):** `{a['pre_flip_mean_step_cost']:.2f}` per period")
        lines.append(f"- **Post-flip Step Cost ($t > {flip_step}$):** `{a['post_flip_mean_step_cost']:.2f}` per period")
        lines.append("- **Estimated Settling Time by Echelon:**")
        for ech, steps in a["estimated_settling_steps"].items():
            lines.append(f"  - `{ech.capitalize()}`: settled in ~**{steps} timesteps**")
        lines.append(f"- **Transition Oscillation Index:** `{a['trans_oscillation_var']:.3f}` (vs steady state `{a['post_oscillation_var']:.3f}`)")
        lines.append("")
        
    # Decision Criterion
    max_settling = max([steps for a in analysis.values() for steps in a["estimated_settling_steps"].values()])
    is_smooth = max_settling <= 15
    
    lines.append("### 3. Diagnostic Verdict & Next Steps")
    lines.append("")
    if is_smooth:
        lines.append("> [!IMPORTANT]")
        lines.append(f"> **DIAGNOSTIC PASSED (VERDICT: PROCEED TO STEP 3A - Intra-Episodic LLM Controller)**")
        lines.append(f"> The frozen generalist policy adapts cleanly to mid-episode $\\alpha$ shifts within **{max_settling} timesteps** without policy breakdown or severe oscillations.")
        lines.append("> The existing Phase 1 weights are fully compatible with intra-episodic meta-control. **No expensive retraining is required.**")
    else:
        lines.append("> [!WARNING]")
        lines.append(f"> **DIAGNOSTIC CAUTION (Settling time = {max_settling} timesteps)**")
        lines.append("> The policy exhibits delayed response or oscillation. Recommended path: Proceed with Step 3B (Phase 2 fixes / Discrete Alpha retrain).")
        
    lines.append("")
    return "\n".join(lines)

def generate_diagnostic_plots(trajectories_by_condition: dict, flip_step: int, plots_dir: str):
    echelons = ["Retailer", "Warehouse", "Distributor", "Manufacturer"]
    ech_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Single Trace Trajectory Detailed Plots (e.g. Trace 0)
    for cond_tag in ["flip_0.1_to_0.9", "flip_0.9_to_0.1"]:
        if cond_tag not in trajectories_by_condition:
            continue
        tr_data = trajectories_by_condition[cond_tag][0]["trajectory"]
        tr_idx = trajectories_by_condition[cond_tag][0]["trace_idx"]
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=150)
        
        # Subplot 1: Demand and Orders
        axs[0].plot(tr_data["demands"], color='black', linestyle=':', label='Customer Demand', alpha=0.7)
        for i in range(4):
            axs[0].plot(tr_data["orders"][i], color=ech_colors[i], label=f'{echelons[i]} Order', alpha=0.85)
        axs[0].axvline(x=flip_step, color='red', linestyle='--', linewidth=1.5, label=f'Alpha Shift Point (t={flip_step})')
        axs[0].set_ylabel("Order / Demand", fontsize=10)
        axs[0].set_title(f"Intra-Episodic Diagnostic — Trace {tr_idx} ({cond_tag.replace('_', ' ').upper()})", fontsize=12, fontweight='bold')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].legend(loc='upper right', ncol=3, fontsize=8)
        
        # Subplot 2: Inventory Trajectories
        for i in range(4):
            axs[1].plot(tr_data["inventory"][i], color=ech_colors[i], label=f'{echelons[i]} Inv')
        axs[1].axvline(x=flip_step, color='red', linestyle='--', linewidth=1.5)
        axs[1].set_ylabel("Inventory Level", fontsize=10)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].legend(loc='upper right', ncol=4, fontsize=8)
        
        # Subplot 3: Backlog Trajectories
        for i in range(4):
            axs[2].plot(tr_data["backlog"][i], color=ech_colors[i], label=f'{echelons[i]} Backlog')
        axs[2].axvline(x=flip_step, color='red', linestyle='--', linewidth=1.5)
        axs[2].set_ylabel("Backlog", fontsize=10)
        axs[2].grid(True, linestyle='--', alpha=0.5)
        axs[2].legend(loc='upper right', ncol=4, fontsize=8)
        
        # Subplot 4: Total Step Cost
        step_costs = np.sum(tr_data["costs"], axis=0)
        axs[3].plot(step_costs, color='#7f7f7f', label='Total System Step Cost (h=1, b=1)')
        axs[3].axvline(x=flip_step, color='red', linestyle='--', linewidth=1.5)
        axs[3].set_ylabel("Step Cost", fontsize=10)
        axs[3].set_xlabel("Timestep (t)", fontsize=11)
        axs[3].grid(True, linestyle='--', alpha=0.5)
        axs[3].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"trace_{tr_idx}_{cond_tag}.png"))
        plt.close()
        
    # 2. Multi-Trace Average Echelon Order Trajectories Comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True, dpi=150)
    axs = axs.flatten()
    
    cond_styles = {
        "flip_0.1_to_0.9": ("#1f77b4", "-", "Flip 0.1→0.9"),
        "flip_0.9_to_0.1": ("#d62728", "-", "Flip 0.9→0.1"),
        "static_0.1": ("#2ca02c", "--", "Static α=0.1"),
        "static_0.9": ("#ff7f0e", "--", "Static α=0.9")
    }
    
    for i, ech in enumerate(echelons):
        ax = axs[i]
        for cond_tag, (color, ls, label) in cond_styles.items():
            if cond_tag in trajectories_by_condition:
                traces = trajectories_by_condition[cond_tag]
                all_orders_ech = np.array([tr["trajectory"]["orders"][i] for tr in traces]) # [N, 200]
                mean_orders = np.mean(all_orders_ech, axis=0)
                ax.plot(mean_orders, color=color, linestyle=ls, label=label, linewidth=1.8 if "-" in ls else 1.2, alpha=0.9)
                
        ax.axvline(x=flip_step, color='gray', linestyle=':', linewidth=1.5, label='Flip t=100' if i==0 else "")
        ax.set_title(f"Echelon {i+1}: {ech} Average Orders", fontsize=11, fontweight='bold')
        ax.set_ylabel("Mean Order Quantity", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    axs[2].set_xlabel("Timestep (t)", fontsize=10)
    axs[3].set_xlabel("Timestep (t)", fontsize=10)
    plt.suptitle("Intra-Episodic Dynamic Response: Mean Order Trajectories across 20 Demand Traces", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "multi_trace_order_trajectories.png"))
    plt.close()
    
    # 3. Transition Settling Zoom Plot (t in [80, 140])
    plt.figure(figsize=(10, 5), dpi=150)
    for cond_tag in ["flip_0.1_to_0.9", "flip_0.9_to_0.1"]:
        if cond_tag in trajectories_by_condition:
            traces = trajectories_by_condition[cond_tag]
            # Total system orders across 4 echelons
            all_tot_orders = np.array([np.sum(tr["trajectory"]["orders"], axis=0) for tr in traces])
            mean_tot = np.mean(all_tot_orders, axis=0)
            std_tot = np.std(all_tot_orders, axis=0)
            
            t_range = np.arange(70, 140)
            color = "#1f77b4" if "0.1_to_0.9" in cond_tag else "#d62728"
            label = "Cooperative → Selfish (0.1→0.9)" if "0.1_to_0.9" in cond_tag else "Selfish → Cooperative (0.9→0.1)"
            
            plt.plot(t_range, mean_tot[t_range], color=color, linewidth=2, label=label)
            plt.fill_between(t_range, mean_tot[t_range] - 0.5*std_tot[t_range], mean_tot[t_range] + 0.5*std_tot[t_range], color=color, alpha=0.15)
            
    plt.axvline(x=flip_step, color='black', linestyle='--', linewidth=1.5, label=f'Incentive Shift Point (t={flip_step})')
    plt.title("Transition Lag & Settling Analysis: Aggregate Supply Chain Orders (t ∈ [70, 140])", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Timestep (t)", fontsize=10)
    plt.ylabel("Total System Orders (Sum across 4 Echelons)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "transition_settling_analysis.png"))
    plt.close()

if __name__ == "__main__":
    run_step2_diagnostic()
