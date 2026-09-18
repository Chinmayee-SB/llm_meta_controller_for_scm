import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.step1_anchor_baselines.evaluate_base_paper_metrics import run_step1_evaluation
from experiments.step2_intra_episodic_diagnostic.diagnostic_alpha_flip import run_step2_diagnostic

def main():
    parser = argparse.ArgumentParser(description="Master Runner for SCM MARL Experiments")
    parser.add_argument("--step", type=str, default="all", choices=["1", "2", "3a", "all"], help="Which step to run: 1, 2, 3a, or all")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast test on single trace")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--mock", action="store_true", help="Run LLM controller in mock mode")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    args = parser.parse_args()

    trace_indices = [0] if args.smoke_test else None

    if args.step in ["1", "all"]:
        print("\n" + "="*80)
        print("EXECUTING STEP 1: ANCHOR BASELINES & BASE PAPER METRIC EVALUATION")
        print("="*80)
        run_step1_evaluation(trace_indices=trace_indices, save_plots=not args.no_plots, verbose=True)

    if args.step in ["2", "all"]:
        print("\n" + "="*80)
        print("EXECUTING STEP 2: INTRA-EPISODIC DIAGNOSTIC")
        print("="*80)
        run_step2_diagnostic(flip_step=100, trace_indices=trace_indices, save_plots=not args.no_plots, verbose=True)

    if args.step in ["3a", "all"]:
        print("\n" + "="*80)
        print("EXECUTING STEP 3A: INTRA-EPISODIC CONTROLLER EVALUATION")
        print("="*80)
        from experiments.step3_intra_episodic_controller.evaluate_step3a import main as run_step3a_main
        # Re-parse sys.argv for Step 3A
        run_step3a_main()

    print("\n" + "="*80)
    print("ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
