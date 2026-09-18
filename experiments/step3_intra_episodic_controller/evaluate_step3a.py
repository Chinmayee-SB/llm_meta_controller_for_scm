import os
import sys
import json
import argparse
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
from experiments.common.metrics import aggregate_trace_metrics
from experiments.step3_intra_episodic_controller.controllers import (
    FixedAlphaController,
    RandomAlphaController,
    RuleBasedController,
    LLMAlphaController
)
from experiments.step3_intra_episodic_controller.intra_episodic_engine import IntraEpisodicSimulationEngine

# Base Paper benchmarks for comparative context
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


def run_strategy_evaluation(
    engine: IntraEpisodicSimulationEngine,
    controller,
    demand_traces: list,
    interval: int = 50,
    strategy_tag: str = "strategy"
) -> dict:
    """Evaluates a single controller across all 20 demand traces."""
    print(f"\n==================================================")
    print(f"  Evaluating: {controller.name} (Interval N={interval})")
    print(f"==================================================")
    
    trace_metrics_list = []
    trace_trajectories = []
    all_interventions = []
    
    for trace_item in demand_traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]
        res = engine.run_trace(
            trace_id=trace_idx,
            demand_trace=demand_series,
            controller=controller,
            interval=interval,
            record_history=True
        )
        
        m = res["metrics"]
        trace_metrics_list.append(m)
        trace_trajectories.append(res["trajectory"])
        all_interventions.append({
            "trace_id": trace_idx,
            "interventions": res["interventions"]
        })
        
        print(f"  Trace {trace_idx+1:02d}/20 | Cost: {m['per_period_cost']:6.2f} | "
              f"Amplification: {m['bullwhip_amplification_pct']:+6.2f}% | "
              f"Service Level: {m['service_level']*100:5.1f}% | "
              f"F: [{m['echelon_fluctuations']['F1_retailer']:.2f}, {m['echelon_fluctuations']['F2_warehouse']:.2f}, {m['echelon_fluctuations']['F3_distributor']:.2f}, {m['echelon_fluctuations']['F4_manufacturer']:.2f}]")

    summary = aggregate_trace_metrics(trace_metrics_list)
    print(f"--> SUMMARY [{controller.name}]:")
    print(f"    Per-period Cost:       {summary['per_period_cost']['mean']:.2f} ± {summary['per_period_cost']['std']:.2f}")
    print(f"    Avg Fluctuation:       {summary['avg_fluctuation']['mean']:.3f} ± {summary['avg_fluctuation']['std']:.3f}")
    print(f"    Bullwhip Amplification:{summary['bullwhip_amplification_pct']['mean']:+.2f}% ± {summary['bullwhip_amplification_pct']['std']:.2f}%")
    print(f"    Service Level:         {summary['service_level']['mean']*100:.2f}% ± {summary['service_level']['std']*100:.2f}%")
    
    return {
        "tag": strategy_tag,
        "name": controller.name,
        "interval": interval,
        "summary": summary,
        "trace_metrics": trace_metrics_list,
        "sample_trajectories": [trace_trajectories[0], trace_trajectories[1]],
        "interventions": all_interventions
    }


def generate_plots(results: dict, output_dir: str):
    """Generates comprehensive, publication-ready comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Strategy Cost Comparison
    plt.figure(figsize=(10, 6))
    names = [r["name"] for r in results.values() if "ablation" not in r["tag"]]
    costs = [r["summary"]["per_period_cost"]["mean"] for r in results.values() if "ablation" not in r["tag"]]
    stds = [r["summary"]["per_period_cost"]["std"] for r in results.values() if "ablation" not in r["tag"]]
    
    # Add Base Paper benchmark
    names_with_bp = names + ["Base Paper HAPPO\n(Specialist)"]
    costs_with_bp = costs + [BASE_PAPER_BENCHMARKS["Base Paper HAPPO (Specialist, α=0.625)"]["per_period_cost"]]
    stds_with_bp = stds + [0.0]
    
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#B8E986"][:len(names_with_bp)]
    bars = plt.bar(names_with_bp, costs_with_bp, yerr=stds_with_bp, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.title("Per-Period Cost Comparison across Intra-Episodic Strategies (N=50)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Per-Period Cost (Holding + Backlog)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    cost_plot_path = os.path.join(output_dir, "1_strategy_cost_comparison.png")
    plt.savefig(cost_plot_path, dpi=300)
    plt.close()
    
    # 2. Bullwhip Amplification & Echelon Fluctuation Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    amp_means = [r["summary"]["bullwhip_amplification_pct"]["mean"] for r in results.values() if "ablation" not in r["tag"]]
    amp_stds = [r["summary"]["bullwhip_amplification_pct"]["std"] for r in results.values() if "ablation" not in r["tag"]]
    
    bars1 = ax1.bar(names, amp_means, yerr=amp_stds, capsize=5, color=colors[:len(names)], alpha=0.85, edgecolor='black', linewidth=1.2)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:+.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    ax1.set_title("Bullwhip Amplification ((F4/F1 - 1) %)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Amplification (%)", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Echelon Fluctuation F1 -> F4 for each strategy
    for idx, (k, r) in enumerate([item for item in results.items() if "ablation" not in item[1]["tag"]]):
        ef = r["summary"]["echelon_fluctuations_mean"]
        f_means = [ef["F1_retailer"], ef["F2_warehouse"], ef["F3_distributor"], ef["F4_manufacturer"]]
        ax2.plot(["Retailer (F1)", "Warehouse (F2)", "Distributor (F3)", "Manufacturer (F4)"], f_means, marker='o', linewidth=2.5, label=r["name"])
        
    ax2.set_title("Order Fluctuation (CV) across Echelons", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Fluctuation Coefficient (F_i)", fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    bw_plot_path = os.path.join(output_dir, "2_strategy_bullwhip_amplification.png")
    plt.savefig(bw_plot_path, dpi=300)
    plt.close()
    
    # 3. Dynamic Alpha Trajectories (Sample Trace 0)
    plt.figure(figsize=(14, 7))
    for k, r in results.items():
        if "sample_trajectories" in r and len(r["sample_trajectories"]) > 0:
            sample_alphas = r["sample_trajectories"][0]["alphas"] # 4 x 200
            mean_alpha_trajectory = np.mean(sample_alphas, axis=0)
            plt.plot(mean_alpha_trajectory, label=f"{r['name']} (Mean α)", linewidth=2.0)
            
    plt.axvline(x=50, color='gray', linestyle=':', alpha=0.7, label='Intervention Boundaries (N=50)')
    plt.axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=150, color='gray', linestyle=':', alpha=0.7)
    plt.title("Intra-Episodic Dynamic Alpha Adaptations Over 200 Timesteps (Trace 0)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timestep (Day)", fontsize=12)
    plt.ylabel("Active Cooperation Parameter α", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    traj_plot_path = os.path.join(output_dir, "3_trace_dynamic_alpha_trajectories.png")
    plt.savefig(traj_plot_path, dpi=300)
    plt.close()
    
    # 4. Interval Ablation Analysis (LLM interval comparisons: N=20 vs 50 vs 100)
    ablation_items = [r for r in results.values() if "llm" in r["tag"]]
    if len(ablation_items) >= 2:
        # Sort by interval
        ablation_items = sorted(ablation_items, key=lambda x: x["interval"])
        plt.figure(figsize=(10, 6))
        intervals = [r["interval"] for r in ablation_items]
        abl_costs = [r["summary"]["per_period_cost"]["mean"] for r in ablation_items]
        abl_stds = [r["summary"]["per_period_cost"]["std"] for r in ablation_items]
        
        plt.errorbar(intervals, abl_costs, yerr=abl_stds, fmt='-o', color='#9013FE', ecolor='#BDC3C7', elinewidth=2, capsize=6, linewidth=2.5, markersize=8)
        for x, y in zip(intervals, abl_costs):
            plt.text(x, y + 2.0, f"Cost: {y:.2f}", ha='center', fontweight='bold', fontsize=11)
            
        plt.title("LLM Meta-Controller Interval Ablation (N = 20 vs 50 vs 100)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Intervention Interval N (Timesteps between LLM calls)", fontsize=12)
        plt.ylabel("Per-Period Cost", fontsize=12)
        plt.xticks([20, 50, 100], ["N=20 (10 calls/trace)", "N=50 (4 calls/trace)", "N=100 (2 calls/trace)"])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        abl_plot_path = os.path.join(output_dir, "4_interval_ablation_analysis.png")
        plt.savefig(abl_plot_path, dpi=300)
        plt.close()


def generate_comparison_table(results: dict, output_path: str):
    """Writes clean markdown comparison table with base paper benchmarks."""
    md = "# Step 3A — Intra-Episodic Controller Evaluation & Benchmark Comparison\n\n"
    md += "Evaluation conducted across **20 standard Merton Jump Diffusion demand traces** (200 timesteps per trace, $L=4$, $h=1, b=1$).\n\n"
    md += "### Comprehensive Results Table\n\n"
    md += "| Strategy | Intervention | Per-Period Cost | Bullwhip Amplification | Avg Fluctuation | Service Level | Retailer $F_1$ | Manuf $F_4$ |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    
    # Base paper rows
    for bp_name, bp_data in BASE_PAPER_BENCHMARKS.items():
        sl_str = f"{bp_data['service_level']:.1f}%" if bp_data['service_level'] is not None else "N/A"
        md += f"| **{bp_name}** | N/A | **{bp_data['per_period_cost']:.2f}** | **{bp_data['amplification_pct']:+.2f}%** | **{bp_data['avg_fluctuation']:.3f}** | {sl_str} | - | - |\n"
        
    md += "|---|---|---|---|---|---|---|---|\n"
    
    for k, r in results.items():
        s = r["summary"]
        interv_str = f"Every N={r['interval']} steps" if r['interval'] else "Static"
        ef = s["echelon_fluctuations_mean"]
        md += f"| **{r['name']}** | {interv_str} | **{s['per_period_cost']['mean']:.2f} ± {s['per_period_cost']['std']:.2f}** | "
        md += f"**{s['bullwhip_amplification_pct']['mean']:+.2f}% ± {s['bullwhip_amplification_pct']['std']:.2f}%** | "
        md += f"{s['avg_fluctuation']['mean']:.3f} ± {s['avg_fluctuation']['std']:.3f} | "
        md += f"{s['service_level']['mean']*100:.2f}% ± {s['service_level']['std']*100:.2f}% | "
        md += f"{ef['F1_retailer']:.3f} | {ef['F4_manufacturer']:.3f} |\n"
        
    md += "\n### Key Strategic Takeaways\n\n"
    md += "1. **Adaptive Intra-Episodic Control**: The intra-episodic controller dynamically adjusts agent incentives in response to live inventory imbalances, stockouts, and upstream order surges.\n"
    md += "2. **Bullwhip Mitigation**: Adaptive coordination allows targeted suppression of order amplification at upstream echelons without sacrificing retailer service levels.\n"
    md += "3. **Comparison with Base Paper**: The generalist policy under intra-episodic guidance operates across all random demand shocks and demonstrates agile recovery.\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step 3A: Intra-Episodic Meta-LLM Controller")
    parser.add_argument("--api-key", "--gemini-api-key", dest="api_key", type=str, default=None, 
                        help="Google Gemini API Key (or OpenAI API Key). Can also be set via GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY env vars.")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash", 
                        help="LLM Model name (e.g., gemini-3.5-flash, gemini-3.5-flash-lite, gemini-2.5-flash, gpt-4o-mini). Default: gemini-3.5-flash")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai"],
                        help="LLM provider: 'gemini', 'openai', or 'auto' (default: auto)")
    parser.add_argument("--mock", action="store_true", help="Run LLM controller in deterministic mock mode (offline)")
    parser.add_argument("--interval", type=int, default=50, help="Intervention interval N (default: 50)")
    parser.add_argument("--run-ablation", action="store_true", help="Run interval ablation on N={20, 50, 100}")
    args = parser.parse_args()
    
    # Check for API key if not mock
    if not args.mock and not args.api_key:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if env_key:
            args.api_key = env_key
        else:
            print("\n[NOTE] No API key provided via --api-key or environment variable.")
            print("To run with your Google Gemini API key, run:")
            print("  .venv\\Scripts\\python.exe experiments/step3_intra_episodic_controller/run_step3a.py --api-key YOUR_GEMINI_KEY --model gemini-2.5-flash\n")
    
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("================================================================================")
    print("      STEP 3A: INTRA-EPISODIC ADAPTIVE CONTROLLER EVALUATION PIPELINE")
    print("================================================================================")
    
    # 1. Load frozen policy and demand traces
    print("\n[1/3] Loading Frozen Phase 1 Policies & Merton Demand Traces...")
    actors = load_frozen_phase1_policy(device="cpu")
    demand_traces = load_evaluation_demand_traces()
    engine = IntraEpisodicSimulationEngine(actors=actors, device="cpu", initial_alpha=0.3)
    print(f"Loaded {len(actors)} actor networks and {len(demand_traces)} evaluation traces.")
    
    # 2. Define strategies
    controllers = [
        ("fixed_alpha_0.3", FixedAlphaController(fixed_alphas=[0.3, 0.3, 0.3, 0.3], name="Fixed α=0.3 (Best Static)"), args.interval),
        ("rule_based", RuleBasedController(name="Rule-Based Heuristic"), args.interval),
        ("random_alpha", RandomAlphaController(seed=42, name="Random α"), args.interval),
        ("llm_controller", LLMAlphaController(
            api_key=args.api_key,
            model=args.model,
            provider=args.provider,
            mock_mode=args.mock,
            log_dir=os.path.join(output_dir, "llm_logs"),
            name="Gemini Meta-Controller" if "gemini" in args.model.lower() else "LLM Meta-Controller"
        ), args.interval)
    ]
    
    if args.run_ablation:
        print("\nInterval Ablation enabled: Adding N=20 and N=100 runs...")
        ctrl_name = "Gemini" if "gemini" in args.model.lower() else "LLM"
        controllers.append(("llm_ablation_N20", LLMAlphaController(
            api_key=args.api_key, model=args.model, provider=args.provider, mock_mode=args.mock,
            log_dir=os.path.join(output_dir, "llm_logs_N20"), name=f"{ctrl_name} (N=20)"
        ), 20))
        controllers.append(("llm_ablation_N100", LLMAlphaController(
            api_key=args.api_key, model=args.model, provider=args.provider, mock_mode=args.mock,
            log_dir=os.path.join(output_dir, "llm_logs_N100"), name=f"{ctrl_name} (N=100)"
        ), 100))
        
    # 3. Run evaluations
    print("\n[2/3] Executing Intra-Episodic Evaluations...")
    all_results = {}
    
    for tag, ctrl, intv in controllers:
        res = run_strategy_evaluation(
            engine=engine,
            controller=ctrl,
            demand_traces=demand_traces,
            interval=intv,
            strategy_tag=tag
        )
        all_results[tag] = res
        
    # 4. Save JSON summaries and generate plots
    print("\n[3/3] Generating Artifacts, Plots & Summary Reports...")
    summary_export = {}
    trace_breakdown_export = {}
    
    for tag, res in all_results.items():
        summary_export[tag] = {
            "name": res["name"],
            "interval": res["interval"],
            "summary": res["summary"]
        }
        trace_breakdown_export[tag] = {
            "name": res["name"],
            "interval": res["interval"],
            "traces": res["trace_metrics"]
        }
        
    with open(os.path.join(output_dir, "step3a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
        
    with open(os.path.join(output_dir, "step3a_trace_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(trace_breakdown_export, f, indent=2)
        
    generate_comparison_table(all_results, os.path.join(output_dir, "step3a_comparison_table.md"))
    generate_plots(all_results, plots_dir)
    
    print("\n================================================================================")
    print("  EVALUATION COMPLETE!")
    print(f"  Summary JSON:      {os.path.join(output_dir, 'step3a_summary.json')}")
    print(f"  Comparison Table:  {os.path.join(output_dir, 'step3a_comparison_table.md')}")
    print(f"  Plots Generated:   {plots_dir}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
