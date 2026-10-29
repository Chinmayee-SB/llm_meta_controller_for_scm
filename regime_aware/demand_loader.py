"""
demand_loader.py

Utility to load frozen evaluation demand traces in deterministic numeric order.
Supports both zero-padded ('demand_000.txt') and non-padded ('0.txt') trace naming conventions.
"""

import os
import glob
from typing import List, Dict, Optional


def load_evaluation_demand_traces(demand_dir: Optional[str] = None) -> List[Dict]:
    """
    Loads all demand trace text files from the specified directory in ascending numeric order.

    Returns:
        List of dictionaries with keys:
            - 'trace_idx': int
            - 'filename': str
            - 'filepath': str
            - 'demand': List[int]
    """
    if demand_dir is None:
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "test_data", "test_demand_merton"),
            "./test_data/test_demand_merton",
        ]
        for p in possible_dirs:
            if os.path.isdir(p):
                demand_dir = os.path.abspath(p)
                break

    if not demand_dir or not os.path.isdir(demand_dir):
        raise FileNotFoundError(f"Could not locate demand directory: {demand_dir}")

    files = glob.glob(os.path.join(demand_dir, "*.txt"))

    def get_num_key(fpath: str) -> int:
        base = os.path.splitext(os.path.basename(fpath))[0]
        digits = "".join(c for c in base if c.isdigit())
        try:
            return int(digits)
        except ValueError:
            return 999999

    files = sorted(files, key=get_num_key)

    traces = []
    for fpath in files:
        idx = get_num_key(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            demands = [int(x) for x in lines]
        traces.append({
            "trace_idx": idx,
            "filename": os.path.basename(fpath),
            "filepath": fpath,
            "demand": demands
        })

    return traces


if __name__ == "__main__":
    traces = load_evaluation_demand_traces()
    print(f"Loaded {len(traces)} traces successfully.")
