import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.step2_intra_episodic_diagnostic.diagnostic_alpha_flip import run_step2_diagnostic

def main():
    parser = argparse.ArgumentParser(description="Step 2: Intra-Episodic Diagnostic (Alpha Flip at Mid-Episode)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 1 trace only")
    parser.add_argument("--flip-step", type=int, default=100, help="Timestep at which alpha changes (default: 100)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--traces", type=int, nargs="+", default=None, help="Specific trace indices (e.g. 0 1 2)")
    args = parser.parse_args()

    trace_indices = [0] if args.smoke_test else args.traces
    run_step2_diagnostic(
        flip_step=args.flip_step,
        trace_indices=trace_indices,
        save_plots=not args.no_plots,
        verbose=True
    )

if __name__ == "__main__":
    main()
