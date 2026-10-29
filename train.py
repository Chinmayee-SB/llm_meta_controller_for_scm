"""
train.py

Root execution entrypoint for training policies.
Usage:
    python train.py --mode curriculum   # Trains Regime-Aware HAPPO (70/15/15 curriculum)
    python train.py --mode base         # Trains Reproduced Base Paper Specialist (alpha=0.375 fixed)
"""

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCM Policies")
    parser.add_argument("--mode", type=str, default="curriculum", choices=["curriculum", "base"],
                        help="Choose 'curriculum' (Regime-Aware) or 'base' (Base Paper Specialist)")
    args, unknown = parser.parse_known_args()

    # Pass remaining args to training scripts
    sys.argv = [sys.argv[0]] + unknown

    if args.mode == "curriculum":
        from regime_aware.train_curriculum import main
        main()
    else:
        from regime_aware.train_base_specialist import main
        main()
