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
from experiments.step5_llm_accumulated_memory.controllers_memory import (
    FixedAlphaController,
    RandomAlphaController,
    RuleBasedController,
    MemoryLLMAlphaController
)
from experiments.step5_llm_accumulated_memory.intra_episodic_memory_engine import IntraEpisodicMemorySimulationEngine

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
    engine: IntraEpisodicMemorySimulationEngine,
    controller,
    demand_traces: list,
    interval: int = 50,
    strategy_tag: str = "strategy",
    warmup_traces: int = 5
) -> dict:
    """Evaluates a controller across all demand traces (with warmup + eval split)."""
    print(f"\n==================================================")
    print(f"  Evaluating: {controller.name} (Interval N={interval})")
    print(f"==================================================")
    
    trace_metrics_list = []
    trace_trajectories = []
    all_interventions = []
    
    for trace_item in demand_traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]
        is_warmup = trace_idx < warmup_traces
        phase_str = f"[WARM-UP {trace_idx+1}/{warmup_traces}]" if is_warmup else f"[EVAL {trace_idx+1}/20]"
        
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
            "is_warmup": is_warmup,
            "interventions": res["interventions"]
        })
        
        print(f"  {phase_str:16s} Trace {trace_idx+1:02d}/20 | Cost: {m['per_period_cost']:6.2f} | "
              f"Bullwhip: {m['bullwhip_amplification_pct']:+6.2f}% | "
              f"SL: {m['service_level']*100:5.1f}% | "
              f"F: [{m['echelon_fluctuations']['F1_retailer']:.2f}, {m['echelon_fluctuations']['F2_warehouse']:.2f}, {m['echelon_fluctuations']['F3_distributor']:.2f}, {m['echelon_fluctuations']['F4_manufacturer']:.2f}]")

    # Metrics on ALL 20 traces
    summary_all = aggregate_trace_metrics(trace_metrics_list)
    
    # Metrics strictly on EVALUATION traces (Traces 6..20, indices warmup_traces..end)
    eval_metrics_list = trace_metrics_list[warmup_traces:]
    summary_eval = aggregate_trace_metrics(eval_metrics_list)
    
    print(f"\n--> SUMMARY [{controller.name}] (EVAL TRACES 6-20):")
    print(f"    Per-period Cost:       {summary_eval['per_period_cost']['mean']:.2f} ± {summary_eval['per_period_cost']['std']:.2f}")
    print(f"    Avg Fluctuation:       {summary_eval['avg_fluctuation']['mean']:.3f} ± {summary_eval['avg_fluctuation']['std']:.3f}")
    print(f"    Bullwhip Amplification:{summary_eval['bullwhip_amplification_pct']['mean']:+.2f}% ± {summary_eval['bullwhip_amplification_pct']['std']:.2f}%")
    print(f"    Service Level:         {summary_eval['service_level']['mean']*100:.2f}% ± {summary_eval['service_level']['std']*100:.2f}%")
    
    return {
        "tag": strategy_tag,
        "name": controller.name,
        "interval": interval,
        "warmup_traces": warmup_traces,
        "summary_eval": summary_eval,
        "summary_all": summary_all,
        "trace_metrics": trace_metrics_list,
        "eval_trace_metrics": eval_metrics_list,
        "sample_trajectories": [trace_trajectories[0], trace_trajectories[min(5, len(trace_trajectories)-1)]],
        "interventions": all_interventions
    }


def generate_plots(results: dict, output_dir: str):
    """Generates comprehensive, publication-ready comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cost Comparison on Evaluation Traces (Traces 6-20)
    plt.figure(figsize=(10, 6))
    names = [r["name"] for r in results.values()]
    costs = [r["summary_eval"]["per_period_cost"]["mean"] for r in results.values()]
    stds = [r["summary_eval"]["per_period_cost"]["std"] for r in results.values()]
    
    # Add Base Paper benchmark
    names_with_bp = names + ["Base Paper HAPPO\n(Specialist)"]
    costs_with_bp = costs + [BASE_PAPER_BENCHMARKS["Base Paper HAPPO (Specialist, α=0.625)"]["per_period_cost"]]
    stds_with_bp = stds + [0.0]
    
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#E91E63", "#9C27B0"][:len(names_with_bp)]
    bars = plt.bar(names_with_bp, costs_with_bp, yerr=stds_with_bp, capsize=5, color=colors, alpha=0.88, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.xticks(range(len(names_with_bp)), names_with_bp, rotation=15, ha='right', fontsize=10)
    plt.title("Per-Period Cost on Evaluation Traces (Traces 6–20, N=50)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Per-Period Cost (Holding + Backlog)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    cost_plot_path = os.path.join(output_dir, "1_memory_vs_baselines_cost.png")
    plt.savefig(cost_plot_path, dpi=300)
    plt.close()
    
    # 2. Trace-by-Trace Progression (Learning Curve from Trace 1 to 20)
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^', 'D', 'v']
    line_styles = ['--', '-.', ':', '-', '-']
    
    for idx, (tag, r) in enumerate(results.items()):
        trace_costs = [m["per_period_cost"] for m in r["trace_metrics"]]
        plt.plot(range(1, len(trace_costs)+1), trace_costs, label=r["name"], 
                 marker=markers[idx % len(markers)], linestyle=line_styles[idx % len(line_styles)],
                 linewidth=2.0, alpha=0.85)
        
    plt.axvline(x=5.5, color='red', linestyle='--', linewidth=1.8, label='Warm-up / Eval Boundary (Trace 5)')
    plt.title("Trace-by-Trace Cost Progression across 20 Traces (Warm-up vs Eval)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Merton Demand Trace (1 to 20)", fontsize=12)
    plt.ylabel("Per-Period Cost", fontsize=12)
    plt.xticks(range(1, 21))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    prog_plot_path = os.path.join(output_dir, "2_trace_by_trace_learning_curve.png")
    plt.savefig(prog_plot_path, dpi=300)
    plt.close()
    
    # 3. Bullwhip Amplification vs Service Level Comparison
    plt.figure(figsize=(11, 6))
    bw_vals = [r["summary_eval"]["bullwhip_amplification_pct"]["mean"] for r in results.values()]
    sl_vals = [r["summary_eval"]["service_level"]["mean"] * 100.0 for r in results.values()]
    strategy_labels = [r["name"] for r in results.values()]
    
    x = np.arange(len(strategy_labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(11, 6))
    rects1 = ax1.bar(x - width/2, bw_vals, width, label='Bullwhip Amplification (%)', color='#E67E22', alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Bullwhip Amplification (%) [Lower is Better]', color='#D35400', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_labels, rotation=15, ha='right', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, sl_vals, width, label='Service Level (%)', color='#27AE60', alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Service Level (%) [Higher is Better]', color='#1E8449', fontsize=12, fontweight='bold')
    ax2.set_ylim([90, 102])
    
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 1, f"+{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title("Bullwhip Amplification & Service Level on Evaluation Traces (Traces 6–20)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    bw_plot_path = os.path.join(output_dir, "3_bullwhip_and_service_level_comparison.png")
    plt.savefig(bw_plot_path, dpi=300)
    plt.close()
    
    # 4. Echelon Fluctuation Breakdown
    plt.figure(figsize=(10, 6))
    echelons = ["Retailer (F1)", "Warehouse (F2)", "Distributor (F3)", "Manufacturer (F4)"]
    for tag, r in results.items():
        eval_metrics = r["eval_trace_metrics"]
        f1_m = np.mean([m["echelon_fluctuations"]["F1_retailer"] for m in eval_metrics])
        f2_m = np.mean([m["echelon_fluctuations"]["F2_warehouse"] for m in eval_metrics])
        f3_m = np.mean([m["echelon_fluctuations"]["F3_distributor"] for m in eval_metrics])
        f4_m = np.mean([m["echelon_fluctuations"]["F4_manufacturer"] for m in eval_metrics])
        plt.plot(echelons, [f1_m, f2_m, f3_m, f4_m], marker='o', linewidth=2.2, label=r["name"])
        
    plt.title("Echelon Order Fluctuation (CV) Progression (Traces 6–20)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Order Fluctuation F_i = std(O) / mean(O)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    ech_plot_path = os.path.join(output_dir, "4_echelon_fluctuation_breakdown.png")
    plt.savefig(ech_plot_path, dpi=300)
    plt.close()
    
    print(f"\n[INFO] Generated 4 publication-ready plots in: {output_dir}")


def generate_comparison_table(results: dict, output_path: str):
    """Generates markdown summary table comparing evaluation trace performance."""
    md = "# Step 5: Intra-Episodic Meta-LLM with Accumulated Memory Benchmark\n\n"
    md += "Evaluation over 20 Merton Demand Traces (Warm-up on Traces 1–5, Official Evaluation on Traces 6–20, N=50, L=4, h=1, b=1).\n\n"
    md += "### Official Evaluation Benchmark Results (Traces 6–20, 15 Traces)\n\n"
    md += "| Strategy | Type | Per-Period Cost | Bullwhip Amplification | Avg Fluctuation | Service Level | Retailer F1 | Manuf F4 |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    
    # Base Paper benchmarks
    for bp_name, bp_data in BASE_PAPER_BENCHMARKS.items():
        md += f"| **{bp_name}** | Specialist Benchmark | **{bp_data['per_period_cost']:.2f}** | "
        md += f"+{bp_data['amplification_pct']:.2f}% | {bp_data['avg_fluctuation']:.3f} | N/A | — | — |\n"
        
    for tag, r in results.items():
        s = r["summary_eval"]
        eval_metrics = r["eval_trace_metrics"]
        f1_m = np.mean([m["echelon_fluctuations"]["F1_retailer"] for m in eval_metrics])
        f4_m = np.mean([m["echelon_fluctuations"]["F4_manufacturer"] for m in eval_metrics])
        
        md += f"| **{r['name']}** | Dynamic (N=50) | "
        md += f"**{s['per_period_cost']['mean']:.2f} ± {s['per_period_cost']['std']:.2f}** | "
        md += f"{s['bullwhip_amplification_pct']['mean']:+.2f}% ± {s['bullwhip_amplification_pct']['std']:.2f}% | "
        md += f"{s['avg_fluctuation']['mean']:.3f} ± {s['avg_fluctuation']['std']:.3f} | "
        md += f"{s['service_level']['mean']*100:.2f}% ± {s['service_level']['std']*100:.2f}% | "
        md += f"{f1_m:.3f} | {f4_m:.3f} |\n"
        
    md += "\n### Full 20-Trace Benchmark Results (Traces 1–20 Inclusive)\n\n"
    md += "| Strategy | Per-Period Cost (All 20) | Bullwhip Amp (All 20) | Service Level (All 20) |\n"
    md += "|---|---|---|---|\n"
    for tag, r in results.items():
        s_all = r["summary_all"]
        md += f"| **{r['name']}** | {s_all['per_period_cost']['mean']:.2f} ± {s_all['per_period_cost']['std']:.2f} | "
        md += f"{s_all['bullwhip_amplification_pct']['mean']:+.2f}% | {s_all['service_level']['mean']*100:.2f}% |\n"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[INFO] Comparison table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step 5: Intra-Episodic Meta-LLM Controller with Accumulated Memory")
    parser.add_argument("--api-key", "--gemini-api-key", dest="api_key", type=str, default=None,
                        help="Google Gemini API Key (or set GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest",
                        help="LLM model (e.g. gemini-flash-latest, gemini-flash-lite-latest). Default: gemini-flash-latest")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai"],
                        help="LLM provider: 'gemini', 'openai', or 'auto' (default: auto)")
    parser.add_argument("--mock", action="store_true", help="Run in deterministic mock mode (offline)")
    parser.add_argument("--interval", type=int, default=50, help="Intervention interval N (default: 50)")
    parser.add_argument("--warmup-traces", type=int, default=5, help="Number of warm-up traces (default: 5)")
    args = parser.parse_args()
    
    # Auto API key resolution
    if not args.mock and not args.api_key:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if env_key:
            args.api_key = env_key
            
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("================================================================================")
    print("      STEP 5: INTRA-EPISODIC META-LLM WITH ACCUMULATED MEMORY PIPELINE")
    print("================================================================================")
    
    # 1. Load frozen policy and demand traces
    print("\n[1/3] Loading Frozen Phase 1 Policies & Merton Demand Traces...")
    actors = load_frozen_phase1_policy(device="cpu")
    demand_traces = load_evaluation_demand_traces()
    engine = IntraEpisodicMemorySimulationEngine(actors=actors, device="cpu", initial_alpha=0.3)
    print(f"Loaded {len(actors)} actor networks and {len(demand_traces)} evaluation traces.")
    
    # 2. Define controllers to evaluate
    controllers_to_run = [
        ("fixed_alpha_0.3", FixedAlphaController(fixed_alphas=[0.3, 0.3, 0.3, 0.3], name="Fixed α=0.3 (Best Static)")),
        ("rule_based", RuleBasedController(name="Rule-Based Heuristic")),
        ("random_alpha", RandomAlphaController(seed=42, name="Random α Baseline")),
        ("llm_accumulated_memory", MemoryLLMAlphaController(
            api_key=args.api_key,
            model=args.model,
            provider=args.provider,
            mock_mode=args.mock,
            warmup_traces=args.warmup_traces,
            name="Gemini LLM (Accumulated Memory)"
        ))
    ]
    
    # 3. Execute evaluations
    print("\n[2/3] Executing Strategy Evaluations across 20 Traces (5 Warm-up + 15 Eval)...")
    all_results = {}
    for tag, ctrl in controllers_to_run:
        res = run_strategy_evaluation(
            engine=engine,
            controller=ctrl,
            demand_traces=demand_traces,
            interval=args.interval,
            strategy_tag=tag,
            warmup_traces=args.warmup_traces
        )
        all_results[tag] = res
        
    # 4. Export JSON results
    print("\n[3/3] Exporting Results and Generating Visualizations...")
    summary_export = {
        tag: {
            "name": res["name"],
            "interval": res["interval"],
            "warmup_traces": res["warmup_traces"],
            "summary_eval_traces_6_to_20": res["summary_eval"],
            "summary_all_traces_1_to_20": res["summary_all"]
        }
        for tag, res in all_results.items()
    }
    
    trace_breakdown_export = {
        tag: {
            "name": res["name"],
            "interval": res["interval"],
            "traces": res["trace_metrics"]
        }
        for tag, res in all_results.items()
    }
    
    with open(os.path.join(output_dir, "step5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
        
    with open(os.path.join(output_dir, "step5_trace_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(trace_breakdown_export, f, indent=2)
        
    # Generate tables and plots
    generate_comparison_table(all_results, os.path.join(output_dir, "step5_comparison_table.md"))
    generate_plots(all_results, plots_dir)
    
    # Copy plots to brain artifacts directory for immediate viewing
    brain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".gemini", "antigravity-ide", "brain", "65772b0b-78cf-4ded-9248-ca1196d2076d"))
    if os.path.exists(brain_dir):
        import shutil
        for pfile in ["1_memory_vs_baselines_cost.png", "2_trace_by_trace_learning_curve.png", "3_bullwhip_and_service_level_comparison.png", "4_echelon_fluctuation_breakdown.png"]:
            src = os.path.join(plots_dir, pfile)
            dst = os.path.join(brain_dir, f"step5_{pfile}")
            if os.path.exists(src):
                shutil.copy(src, dst)
                
    print("\n================================================================================")
    print("      STEP 5 EVALUATION COMPLETE — ALL RESULTS SAVED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
