import os
import sys

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.step3b_interval_ablation.evaluate_step3b import main as run_step3b_main

def main():
    run_step3b_main()

if __name__ == "__main__":
    main()
