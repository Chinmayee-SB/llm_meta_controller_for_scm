import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    RuleBasedController,
    RandomAlphaController,
    BaseController
)
from experiments.step3_intra_episodic_controller.intra_episodic_engine import IntraEpisodicSimulationEngine
from experiments.step3_intra_episodic_controller.evaluate_step3a import BASE_PAPER_BENCHMARKS, generate_comparison_table, generate_plots

class GeminiReplayController(BaseController):
    def __init__(self, log_dir, name="Gemini Meta-Controller"):
        super().__init__(name=name)
        self.log_dir = log_dir
        self.trace_id = 0
    def reset_episode(self, trace_id):
        self.trace_id = trace_id
    def decide_alphas(self, timestep, window_interval, window_data, current_alphas):
        fpath = os.path.join(self.log_dir, f"trace_{self.trace_id:02d}_t{timestep:03d}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d.get("parsed_alphas", current_alphas), d
        return current_alphas, {}

def main():
    actors = load_frozen_phase1_policy(device="cpu")
    demand_traces = load_evaluation_demand_traces()
    engine = IntraEpisodicSimulationEngine(actors=actors, device="cpu", initial_alpha=0.3)
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    controllers = [
        ("fixed_alpha_0.3", FixedAlphaController(fixed_alphas=[0.3, 0.3, 0.3, 0.3], name="Fixed α=0.3 (Best Static)"), 50),
        ("rule_based", RuleBasedController(name="Rule-Based Heuristic"), 50),
        ("random_alpha", RandomAlphaController(seed=42, name="Random α"), 50),
        ("llm_controller", GeminiReplayController(os.path.join(results_dir, "llm_logs"), name="Gemini Meta-Controller"), 50)
    ]
    
    all_results = {}
    for tag, ctrl, intv in controllers:
        trace_metrics = []
        trace_trajs = []
        all_interventions = []
        for t in demand_traces:
            res = engine.run_trace(
                trace_id=t["trace_idx"],
                demand_trace=t["demand"],
                controller=ctrl,
                interval=intv,
                record_history=True
            )
            trace_metrics.append(res["metrics"])
            trace_trajs.append(res["trajectory"])
            all_interventions.append({
                "trace_id": t["trace_idx"],
                "interventions": res["interventions"]
            })
        summary = aggregate_trace_metrics(trace_metrics)
        all_results[tag] = {
            "tag": tag,
            "name": ctrl.name,
            "interval": intv,
            "summary": summary,
            "trace_metrics": trace_metrics,
            "sample_trajectories": [trace_trajs[0], trace_trajs[1]],
            "interventions": all_interventions
        }
        print(f"Computed summary for {ctrl.name}: Cost = {summary['per_period_cost']['mean']:.2f} ± {summary['per_period_cost']['std']:.2f}")

    # Export JSON
    summary_export = {tag: {"name": res["name"], "interval": res["interval"], "summary": res["summary"]} for tag, res in all_results.items()}
    trace_breakdown_export = {tag: {"name": res["name"], "interval": res["interval"], "traces": res["trace_metrics"]} for tag, res in all_results.items()}
    
    with open(os.path.join(results_dir, "step3a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    with open(os.path.join(results_dir, "step3a_trace_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(trace_breakdown_export, f, indent=2)
        
    generate_comparison_table(all_results, os.path.join(results_dir, "step3a_comparison_table.md"))
    generate_plots(all_results, plots_dir)
    print("Done generating updated reports and plots!")

if __name__ == "__main__":
    main()
