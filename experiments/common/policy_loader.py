import os
import sys
import torch
import numpy as np
import gym

# Add project root and MADRL dir to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from MADRL.config import get_config
from MADRL.algorithms.actor_critic import Actor

def load_frozen_phase1_policy(weights_path: str = None, device: str = "cpu"):
    """
    Loads the 4 trained actor networks from Phase 1.
    Sets eval mode and requires_grad=False on all parameters.
    Returns:
        actors: List[Actor] (length 4: Retailer, Warehouse, Distributor, Manufacturer)
    """
    if weights_path is None:
        possible_paths = [
            os.path.join(PROJECT_ROOT, "MADRL", "results", "phase1_weights", "frozen_policy.pt"),
            os.path.join(PROJECT_ROOT, "results", "phase1_weights", "frozen_policy.pt"),
            "MADRL/results/phase1_weights/frozen_policy.pt"
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                weights_path = os.path.abspath(p)
                break
                
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Frozen policy checkpoint not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, list):
        state_dicts = checkpoint
    elif isinstance(checkpoint, dict) and "actor_state_dicts" in checkpoint:
        state_dicts = checkpoint["actor_state_dicts"]
    else:
        raise ValueError(f"Unknown checkpoint format in {weights_path}")

    parser = get_config()
    args, _ = parser.parse_known_args([])
    args.num_agents = 4
    args.obs_dim = 8
    args.std_x_coef = 1.0
    args.std_y_coef = 0.5
    
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
    act_space = gym.spaces.Discrete(21)
    
    actors = []
    for i, state_dict in enumerate(state_dicts):
        actor = Actor(args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False
        actors.append(actor)
        
    return actors

if __name__ == "__main__":
    actors = load_frozen_phase1_policy()
    print(f"Successfully loaded {len(actors)} actors.")
    total_params = sum(p.numel() for a in actors for p in a.parameters())
    print(f"Total frozen parameters: {total_params} across 4 agents.")
