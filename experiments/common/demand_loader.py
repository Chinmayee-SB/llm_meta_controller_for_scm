import os
import glob
from typing import List, Dict

def load_evaluation_demand_traces(demand_dir: str = None) -> List[Dict]:
    """
    Loads all 20 Merton evaluation demand traces in numeric order (0.txt to 19.txt).
    Returns a list of dicts with:
      - 'trace_idx': int (0 to 19)
      - 'filename': str
      - 'demand': List[int] (200 timesteps)
    """
    if demand_dir is None:
        # Check standard locations
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "..", "MADRL", "test_data", "test_demand_merton"),
            os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test_demand_merton"),
            "./MADRL/test_data/test_demand_merton",
            "./test_data/test_demand_merton"
        ]
        for p in possible_dirs:
            if os.path.isdir(p):
                demand_dir = os.path.abspath(p)
                break
                
    if not demand_dir or not os.path.isdir(demand_dir):
        raise FileNotFoundError(f"Could not locate evaluation demand directory: {demand_dir}")

    files = glob.glob(os.path.join(demand_dir, "*.txt"))
    # Sort numerically by filename without extension
    def get_num_key(fpath):
        base = os.path.splitext(os.path.basename(fpath))[0]
        digits = ''.join(c for c in base if c.isdigit())
        try:
            return int(digits)
        except ValueError:
            return 9999
            
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
    t = load_evaluation_demand_traces()
    print(f"Loaded {len(t)} traces successfully.")
    for trace in t[:3]:
        print(f"Trace {trace['trace_idx']}: {len(trace['demand'])} steps, first 5: {trace['demand'][:5]}")
