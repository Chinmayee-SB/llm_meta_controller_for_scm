import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.step1_anchor_baselines.evaluate_base_paper_metrics import run_step1_evaluation

def main():
    parser = argparse.ArgumentParser(description="Step 1: Anchor Baseline Metrics & Base Paper Comparison")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 1 trace only")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--traces", type=int, nargs="+", default=None, help="Specific trace indices (e.g. 0 1 2)")
    args = parser.parse_args()

    trace_indices = [0] if args.smoke_test else args.traces
    run_step1_evaluation(
        trace_indices=trace_indices,
        save_plots=not args.no_plots,
        verbose=True
    )

if __name__ == "__main__":
    main()
