"""
evaluate.py

Root execution entrypoint for evaluating the 280-trace benchmark.
Usage:
    python evaluate.py --controller fixed_cruiser
    python evaluate.py --controller base_specialist
    python evaluate.py --controller rule_based
    python evaluate.py --controller llm --model gpt-5.6-luna
"""

import sys
from regime_aware.evaluate_benchmark import main

if __name__ == "__main__":
    main()
