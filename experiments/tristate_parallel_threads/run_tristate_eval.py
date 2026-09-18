"""
run_tristate_eval.py

Evaluation script for the Tri-State MARL model. Evaluates trained policy checkpoints
under intra-episodic control (Fixed, Rule-Based, or Meta-LLM Supervisor) across the
20 frozen Merton demand traces according to the official Base Paper evaluation protocol.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.demand_loader import load_evaluation_demand_traces
from experiments.tristate_parallel_threads.tristate_intra_controller import (
    load_tristate_actors,
    TriStateIntraSimulationEngine,
    FixedTriStateController,
    RuleBasedTriStateController,
    LLMTriStateController
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tri-State MARL under Intra-Episodic Control")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint .pt file")
    parser.add_argument("--controller", type=str, default="fixed_cruiser",
                        choices=["fixed_cruiser", "fixed_shock", "fixed_backlog", "rule_based", "llm"],
                        help="Controller type to evaluate")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="LLM model name if using llm controller")
    parser.add_argument("--mock_llm", action="store_true", help="Run LLM in mock/heuristic fallback mode")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini or OpenAI API key")
    parser.add_argument("--interval", type=int, default=50, help="Intervention interval (timesteps)")
    parser.add_argument("--num_traces", type=int, default=20, help="Number of Merton traces to evaluate (max 20)")
    parser.add_argument("--trace_idx", type=int, default=-1, help="Run single 1-based trace index (e.g. 8 for Trace 8)")
    parser.add_argument("--demand_dir", type=str, default=None, help="Directory of evaluation demand traces")
    parser.add_argument("--use_generator_demand", action="store_true", help="Use training Merton generator distribution for evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.use_generator_demand and not args.demand_dir:
        args.demand_dir = os.path.join(PROJECT_ROOT, "test_data", "test_demand_merton_generated")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_path = args.checkpoint
    else:
        final_p = os.path.join(checkpoint_dir, "final_tristate_policy.pt")
        best_p = os.path.join(checkpoint_dir, "best_tristate_policy.pt")
        ckpt_path = final_p if os.path.exists(final_p) else best_p

    print("=" * 80)
    print(" TRI-STATE MARL EVALUATION PROTOCOL")
    print("=" * 80)
    print(f" Checkpoint:  {ckpt_path}")
    print(f" Controller:  {args.controller}")
    n_windows = 200 // args.interval
    print(f" Interval:    Every {args.interval} timesteps ({n_windows} windows per 200-step trace)")
    print(f" Device:      {device}")
    print("=" * 80)

    actors = load_tristate_actors(ckpt_path, device=device)
    engine = TriStateIntraSimulationEngine(actors, device=device, initial_alpha=0.375)

    # Initialize Controller
    if args.controller == "fixed_cruiser":
        controller = FixedTriStateController(fixed_alpha=0.375, name="Fixed Cruiser (α=0.375)")
    elif args.controller == "fixed_shock":
        controller = FixedTriStateController(fixed_alpha=0.10, name="Fixed Shock Absorber (α=0.10)")
    elif args.controller == "fixed_backlog":
        controller = FixedTriStateController(fixed_alpha=0.65, name="Fixed Backlog Clearer (α=0.65)")
    elif args.controller == "rule_based":
        controller = RuleBasedTriStateController(name="Rule-Based Tri-State")
    elif args.controller == "llm":
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        suffix = "_gen_demand" if args.demand_dir else ""
        llm_log_dir = os.path.join(os.path.dirname(__file__), "logs", f"llm_supervision_{ckpt_name}_N{args.interval}{suffix}")
        controller = LLMTriStateController(api_key=args.api_key, model=args.model, mock_mode=args.mock_llm, log_dir=llm_log_dir)
    else:
        raise ValueError(f"Unknown controller: {args.controller}")

    all_traces = load_evaluation_demand_traces(demand_dir=args.demand_dir)
    if args.trace_idx > 0:
        target_idx = args.trace_idx - 1
        eval_traces = [t for t in all_traces if t["trace_idx"] == target_idx]
    else:
        eval_traces = all_traces[:args.num_traces]

    trace_costs = []
    trace_f1 = []
    trace_f2 = []
    trace_f3 = []
    trace_f4 = []
    trace_avg_f = []
    trace_amp = []
    trace_sl = []
    trace_cfr = []

    print(f"\nEvaluating across {len(eval_traces)} frozen Merton demand traces...")
    print(f"{'Trace':<8}{'Per-Period Cost':<18}{'F1 (Ret)':<10}{'F2 (Whs)':<10}{'F3 (Dist)':<10}{'F4 (Mfr)':<10}{'Avg F':<10}{'Amp (%)':<12}{'SL (%)':<10}{'CFR (%)':<10}")
    print("-" * 102)

    try:
        for idx, trace_item in enumerate(eval_traces):
            trace_id = trace_item["trace_idx"]
            demand_series = trace_item["demand"]

            res = engine.run_trace(
                trace_id=trace_id,
                demand_trace=demand_series,
                controller=controller,
                interval=args.interval
            )

            c = res["per_period_cost"]
            f1, f2, f3, f4 = res["fluctuations"]
            f_avg = res["avg_fluctuation_F"]
            amp = res["bullwhip_amplification_pct"]
            sl = res["service_level"] * 100.0
            cfr = res.get("customer_fill_rate", 0.0) * 100.0

            trace_costs.append(c)
            trace_f1.append(f1)
            trace_f2.append(f2)
            trace_f3.append(f3)
            trace_f4.append(f4)
            trace_avg_f.append(f_avg)
            trace_amp.append(amp)
            trace_sl.append(sl)
            trace_cfr.append(cfr)

            amp_str = f"{amp:.2f}%"
            sl_str = f"{sl:.1f}%"
            cfr_str = f"{cfr:.1f}%"
            print(f"Trace {trace_id+1:<3} {c:<18.2f}{f1:<10.3f}{f2:<10.3f}{f3:<10.3f}{f4:<10.3f}{f_avg:<10.3f}{amp_str:<12}{sl_str:<10}{cfr_str:<10}")

    except RuntimeError as re_err:
        print("\n" + "!" * 80)
        print(f"[EVALUATION HALTED] {re_err}")
        print("Please provide a new API key to resume evaluation.")
        print("!" * 80)
        sys.exit(1)

    # Aggregate Benchmark Comparison Table
    mean_cost = float(np.mean(trace_costs))
    mean_f1 = float(np.mean(trace_f1))
    mean_f2 = float(np.mean(trace_f2))
    mean_f3 = float(np.mean(trace_f3))
    mean_f4 = float(np.mean(trace_f4))
    mean_avg_f = float(np.mean(trace_avg_f))
    mean_amp = float(np.mean(trace_amp))
    mean_sl = float(np.mean(trace_sl))
    mean_cfr = float(np.mean(trace_cfr))

    print("\n" + "=" * 70)
    print(" AGGREGATE EVALUATION RESULTS VS BASE PAPER BENCHMARKS")
    print("=" * 70)
    print(f"{'Metric':<32}{'Evaluated System':<20}{'Base Paper (HAPPO)':<18}")
    print("-" * 70)
    print(f"{'Per-Period Cost (Undiscounted)':<32}{mean_cost:<20.2f}{'~34.00':<18}")
    print(f"{'F1 (Retailer Fluctuation)':<32}{mean_f1:<20.3f}{'-':<18}")
    print(f"{'F2 (Wholesaler Fluctuation)':<32}{mean_f2:<20.3f}{'-':<18}")
    print(f"{'F3 (Distributor Fluctuation)':<32}{mean_f3:<20.3f}{'-':<18}")
    print(f"{'F4 (Manufacturer Fluctuation)':<32}{mean_f4:<20.3f}{'-':<18}")
    print(f"{'Average Fluctuation (F)':<32}{mean_avg_f:<20.3f}{'0.758':<18}")
    print(f"{'Bullwhip Amplification (A)':<32}{f'{mean_amp:.2f}%':<20}{'92.37%':<18}")
    print(f"{'Customer Service Level':<32}{f'{mean_sl:.1f}%':<20}{'-':<18}")
    print(f"{'Customer Fill Rate (CFR)':<32}{f'{mean_cfr:.1f}%':<20}{'-':<18}")
    print("=" * 70)

if __name__ == "__main__":
    main()
