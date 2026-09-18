import os
import sys
import argparse

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.step3_intra_episodic_controller.evaluate_step3a import main as run_step3a_main

def main():
    run_step3a_main()

if __name__ == "__main__":
    main()
