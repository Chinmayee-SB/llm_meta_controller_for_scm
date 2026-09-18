import os
import sys
import json
import numpy as np
import torch
import warnings
from typing import List, Dict, Tuple, Optional, Union

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root, MADRL, and step4 to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
STEP4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)
if STEP4_DIR not in sys.path:
    sys.path.insert(0, STEP4_DIR)

from MADRL.config import get_config
from MADRL.algorithms.actor_critic import Actor
import gym

from experiments.common.demand_loader import load_evaluation_demand_traces
from experiments.common.metrics import aggregate_trace_metrics, compute_trace_metrics
from experiments.common.env_wrapper import SupplyChainSimulation
from experiments.step3_intra_episodic_controller.controllers import (
    FixedAlphaController,
    RandomAlphaController,
    RuleBasedController,
    LLMAlphaController
)
from experiments.step3_intra_episodic_controller.intra_episodic_engine import IntraEpisodicSimulationEngine

def load_dynamic_retrained_policy(weights_path: str = None, device: str = "cpu"):
    if weights_path is None:
        weights_path = os.path.join(STEP4_DIR, "weights", "dynamic_frozen_policy.pt")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Dynamic frozen policy not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, list):
        state_dicts = checkpoint
    elif isinstance(checkpoint, dict) and "actor_state_dicts" in checkpoint:
        state_dicts = checkpoint["actor_state_dicts"]
    else:
        raise ValueError(f"Unknown checkpoint format in {weights_path}")

    parser = get_config()
    args, _ = parser.parse_known_args([])
    args.num_agents = 4
    args.obs_dim = 8
    args.std_x_coef = 1.0
    args.std_y_coef = 0.5

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
    act_space = gym.spaces.Discrete(21)

    actors = []
    for i, state_dict in enumerate(state_dicts):
        actor = Actor(args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False
        actors.append(actor)

    return actors

def run_settling_diagnostic(actors, demand_traces, device="cpu"):
    """
    Evaluates order settling response when alpha shifts abruptly at t=100.
    Flip A: 0.1 -> 0.9 at t=100
    Flip B: 0.9 -> 0.1 at t=100
    """
    sim = SupplyChainSimulation(actors, device=device)
    results = {}

    for flip_name, (a_pre, a_post) in [("flip_0.1_to_0.9", (0.1, 0.9)), ("flip_0.9_to_0.1", (0.9, 0.1))]:
        trace_results = []
        for trace_item in demand_traces:
            d_idx = trace_item["trace_idx"]
            demand = trace_item["demand"]

            def dynamic_alpha_fn(t, state_dict, pre=a_pre, post=a_post):
                val = pre if t < 100 else post
                return [val, val, val, val]

            res = sim.run_trace(demand, dynamic_alpha_fn, record_history=True)
            traj = res["trajectory"]

            orders = np.array(traj["orders"])  # shape (4, 200)
            pre_order_mean = np.mean(orders[:, 85:100], axis=1)
            post_peak_order = np.max(orders[:, 100:115], axis=1)
            overshoot = float(np.mean(post_peak_order - pre_order_mean))

            trace_results.append({
                "trace_idx": d_idx,
                "metrics": res["metrics"],
                "overshoot": overshoot,
                "pre_order_mean": pre_order_mean.tolist(),
                "post_peak_order": post_peak_order.tolist()
            })

        avg_cost = float(np.mean([t["metrics"]["per_period_cost"] for t in trace_results]))
        avg_bw = float(np.mean([t["metrics"]["bullwhip_amplification_pct"] for t in trace_results]))
        avg_overshoot = float(np.mean([t["overshoot"] for t in trace_results]))

        results[flip_name] = {
            "avg_per_period_cost": avg_cost,
            "avg_bullwhip_amplification_pct": avg_bw,
            "avg_order_overshoot": avg_overshoot,
            "traces": trace_results
        }

    return results

def evaluate_dynamic_retrained_benchmark(device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    print("\n" + "=" * 70)
    print("  EVALUATING STEP 4 DYNAMIC ALPHA RETRAINED POLICY ON 20 MERTON TRACES")
    print("=" * 70)

    actors = load_dynamic_retrained_policy(device=device)
    demand_traces = load_evaluation_demand_traces()
    engine = IntraEpisodicSimulationEngine(actors=actors, device=device, initial_alpha=0.3)

    output_dir = os.path.join(STEP4_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    controllers = [
        ("fixed_alpha_0.1", FixedAlphaController(fixed_alphas=[0.1, 0.1, 0.1, 0.1], name="Fixed α=0.1"), 50),
        ("fixed_alpha_0.3", FixedAlphaController(fixed_alphas=[0.3, 0.3, 0.3, 0.3], name="Fixed α=0.3"), 50),
        ("fixed_alpha_0.5", FixedAlphaController(fixed_alphas=[0.5, 0.5, 0.5, 0.5], name="Fixed α=0.5"), 50),
        ("fixed_alpha_0.7", FixedAlphaController(fixed_alphas=[0.7, 0.7, 0.7, 0.7], name="Fixed α=0.7"), 50),
        ("fixed_alpha_0.9", FixedAlphaController(fixed_alphas=[0.9, 0.9, 0.9, 0.9], name="Fixed α=0.9"), 50),
        ("rule_based", RuleBasedController(name="Rule-Based Heuristic (N=50)"), 50),
        ("random_alpha", RandomAlphaController(seed=42, name="Random α (N=50)"), 50),
    ]

    all_results = {}
    summary_export = {}
    trace_breakdown_export = {}

    for tag, ctrl, interval in controllers:
        print(f"\nEvaluating: {ctrl.name} (Interval N={interval})...")
        trace_metrics_list = []
        trace_trajectories = []
        all_interventions = []

        for trace_item in demand_traces:
            trace_idx = trace_item["trace_idx"]
            demand = trace_item["demand"]
            res = engine.run_trace(
                trace_id=trace_idx,
                demand_trace=demand,
                controller=ctrl,
                interval=interval,
                record_history=True
            )
            trace_metrics_list.append(res["metrics"])
            trace_trajectories.append(res["trajectory"])
            all_interventions.append({
                "trace_id": trace_idx,
                "interventions": res["interventions"]
            })

        agg = aggregate_trace_metrics(trace_metrics_list)
        all_results[tag] = {
            "name": ctrl.name,
            "interval": interval,
            "summary": agg,
            "trace_metrics": trace_metrics_list
        }
        summary_export[tag] = {
            "name": ctrl.name,
            "interval": interval,
            "summary": agg
        }
        trace_breakdown_export[tag] = {
            "name": ctrl.name,
            "interval": interval,
            "traces": trace_metrics_list
        }
        print(f"  -> Cost: {agg['per_period_cost']['mean']:.2f} ± {agg['per_period_cost']['std']:.2f} | Bullwhip: {agg['bullwhip_amplification_pct']['mean']:+.2f}% | SL: {agg['service_level']['mean']*100:.2f}%")

    # Settling diagnostic
    print("\nEvaluating Dynamic Step-Shift Settling Diagnostic...")
    settling_diag = run_settling_diagnostic(actors, demand_traces, device=device)
    summary_export["settling_diagnostic"] = settling_diag
    print(f"  Flip 0.1 -> 0.9 Overshoot: {settling_diag['flip_0.1_to_0.9']['avg_order_overshoot']:.2f} | Cost: {settling_diag['flip_0.1_to_0.9']['avg_per_period_cost']:.2f}")
    print(f"  Flip 0.9 -> 0.1 Overshoot: {settling_diag['flip_0.9_to_0.1']['avg_order_overshoot']:.2f} | Cost: {settling_diag['flip_0.9_to_0.1']['avg_per_period_cost']:.2f}")

    # Save outputs
    summary_path = os.path.join(output_dir, "step4_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    breakdown_path = os.path.join(output_dir, "step4_trace_breakdown.json")
    with open(breakdown_path, "w", encoding="utf-8") as f:
        json.dump(trace_breakdown_export, f, indent=2)

    print(f"\nAll benchmark results saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Breakdown: {breakdown_path}")

    return summary_export

if __name__ == "__main__":
    evaluate_dynamic_retrained_benchmark()
