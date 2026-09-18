import os
import sys

# Add project root and step4 to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEP4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if STEP4_DIR not in sys.path:
    sys.path.insert(0, STEP4_DIR)

from evaluation.evaluate_dynamic_policy import evaluate_dynamic_retrained_benchmark
from evaluation.generate_comparison_plots import generate_comparison_figures

def main():
    print("\n" + "=" * 70)
    print("  LAUNCHING COMPLETE STEP 4 EVALUATION & PLOTTING SUITE")
    print("=" * 70)
    evaluate_dynamic_retrained_benchmark()
    generate_comparison_figures()
    print("\nStep 4 Evaluation and Visual Analytics complete!")

if __name__ == "__main__":
    main()
